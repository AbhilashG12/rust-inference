use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::embedder::EmbedderEngine;
use crate::tokenizer::Tokenizer;

const MAGIC: &[u8; 4] = b"RIDX";
const VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: usize,
    pub source_file: String,
    pub text: String,
}

pub struct DocumentStore {
    pub chunks: Vec<Chunk>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

   pub fn ingest_directory<P: AsRef<Path>>(
        &mut self,
        dir_path: P,
        tokenizer: &Tokenizer,
        chunk_tokens: usize,
        overlap_tokens: usize,
    ) {
        assert!(chunk_tokens > overlap_tokens, "Chunk size must be greater than overlap");

        let path = dir_path.as_ref();
        if !path.exists() || !path.is_dir() {
            println!(" Warning: Directory {:?} does not exist or is not a folder.", path);
            return;
        }

        let mut file_paths = Vec::new();
        Self::crawl_directory(path, &mut file_paths);
        println!(" Found {} files. Ingesting and chunking...", file_paths.len());

        let mut current_id = self.chunks.len();

        for file_path in file_paths {
            if let Ok(content) = fs::read_to_string(&file_path) {
                let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
                if content.trim().is_empty() {
                    continue;
                }
                
                let is_csv = file_name.to_lowercase().ends_with(".csv");
                if is_csv {
                    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
                    if let Some(header_line) = lines.next() {
                        let headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();
                        for row_line in lines {
                            let cols: Vec<&str> = row_line.split(',').map(|s| s.trim()).collect();
                            let pairs: Vec<String> = headers
                                .iter()
                                .zip(cols.iter())
                                .map(|(h, v)| format!("{} is {}", h.replace('_', " "), v))
                                .collect();

                            self.chunks.push(Chunk {
                                id: current_id,
                                source_file: file_name.clone(),
                                text: format!("From {}: {}.", file_name, pairs.join(", ")),
                            });
                            current_id += 1;
                        }
                    }
                    continue;
                }
                let document_chunks =
                    Self::sliding_window_chunk(&content, tokenizer, chunk_tokens, overlap_tokens);
                for text_chunk in document_chunks {
                    self.chunks.push(Chunk {
                        id: current_id,
                        source_file: file_name.clone(),
                        text: format!("[File: {}]\n{}", file_name, text_chunk),
                    });
                    current_id += 1;
                }
            }
        }

        println!(" Successfully ingested {} semantic chunks.", self.chunks.len());
    }

    fn crawl_directory(dir: &Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::crawl_directory(&path, files);
                } else if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "txt" || ext_str == "md" || ext_str == "csv" {
                        files.push(path);
                    }
                }
            }
        }
    }

    /// Chunks raw text directly while preserving newlines, casing, and CSV/Markdown structure.
    /// Keeps each chunk under ~90 words (~115 tokens) so MiniLM's 128-token limit never truncates it.
    /// Estimates subword token count accurately for both prose and dense CSV/code lines.
    fn estimate_tokens(line: &str) -> usize {
        let by_delims = line
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | '-' | '_' | '/' | ':' | '.'))
            .filter(|s| !s.is_empty())
            .count();
        let by_chars = (line.len() + 3) / 4;
        by_delims.max(by_chars)
    }

    fn sliding_window_chunk(
        text: &str,
        _tokenizer: &Tokenizer,
        chunk_tokens: usize,
        overlap_tokens: usize,
    ) -> Vec<String> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        let mut current_lines: Vec<&str> = Vec::new();
        let mut current_tok_count = 0;

        for line in lines {
            let line_toks = Self::estimate_tokens(line);

            if current_tok_count + line_toks > chunk_tokens && !current_lines.is_empty() {
                chunks.push(current_lines.join("\n").trim().to_string());

                let mut overlap_lines = Vec::new();
                let mut kept_toks = 0;
                for &prev in current_lines.iter().rev() {
                    let t = Self::estimate_tokens(prev);
                    if kept_toks + t > overlap_tokens && !overlap_lines.is_empty() {
                        break;
                    }
                    overlap_lines.push(prev);
                    kept_toks += t;
                }
                overlap_lines.reverse();
                current_lines = overlap_lines;
                current_tok_count = kept_toks;
            }

            current_lines.push(line);
            current_tok_count += line_toks;
        }

        if !current_lines.is_empty() {
            let last = current_lines.join("\n").trim().to_string();
            if !last.is_empty() {
                chunks.push(last);
            }
        }

        chunks
    }
}


#[derive(Debug)]
struct ScoredChunk {
    pub score: f32,
    pub chunk_idx: usize,
}

impl PartialEq for ScoredChunk {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}
impl Eq for ScoredChunk {}
impl PartialOrd for ScoredChunk {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for ScoredChunk {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct VectorStore {
    pub vectors: Vec<f32>,
    pub hidden_dim: usize,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            hidden_dim: 384,
        }
    }

    /// Parallelized indexing across all pinned physical cores.
    pub fn embed_store(&mut self, store: &DocumentStore, embedder: &EmbedderEngine) {
        println!(" Parallel-embedding {} chunks into a flat contiguous matrix...", store.chunks.len());

        let embedded: Vec<Vec<f32>> = store
            .chunks
            .par_iter()
            .map(|chunk| embedder.embed(&chunk.text))
            .collect();

        self.vectors.clear();
        self.vectors.reserve(store.chunks.len() * self.hidden_dim);
        for vec in embedded {
            self.vectors.extend_from_slice(&vec);
        }
        println!(" Vectorization complete.");
    }

    /// Flat-matrix dot-product search with cosine similarity thresholding.
    /// Returns (chunk_idx, score) pairs sorted highest-first.
    /// Hybrid Search: Dense Cosine Similarity + Exact Keyword Overlap Boost.
    pub fn search(
        &self,
        query: &str,
        doc_store: &DocumentStore,
        embedder: &EmbedderEngine,
        top_k: usize,
        min_score: f32,
    ) -> Vec<(usize, f32)> {
        let query_vec = embedder.embed(query);
        let mut heap = BinaryHeap::new();
        let num_chunks = self.vectors.len() / self.hidden_dim;

        // Extract meaningful query terms (len >= 4, lowercase, strip punctuation)
        let query_terms: Vec<String> = query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() >= 4 && !matches!(w.as_str(), "what" | "when" | "does" | "this" | "that" | "with" | "from" | "over" | "running"))
            .collect();

        for i in 0..num_chunks {
            let start = i * self.hidden_dim;
            let doc_vec = &self.vectors[start..start + self.hidden_dim];

            let mut dot_product = 0.0;
            for d in 0..self.hidden_dim {
                dot_product += query_vec[d] * doc_vec[d];
            }

            // Lexical boost for exact keyword hits
            let lexical_boost = if !query_terms.is_empty() {
                let chunk_lower = doc_store.chunks[i].text.to_lowercase();
                let hits = query_terms.iter().filter(|t| chunk_lower.contains(t.as_str())).count();
                0.20 * (hits as f32 / query_terms.len() as f32)
            } else {
                0.0
            };

            let hybrid_score = dot_product + lexical_boost;

            if hybrid_score >= min_score {
                heap.push(ScoredChunk {
                    score: hybrid_score,
                    chunk_idx: i,
                });
            }
        }

        let mut results = Vec::new();
        for _ in 0..top_k {
            if let Some(scored) = heap.pop() {
                results.push((scored.chunk_idx, scored.score));
            }
        }
        results
    }

    /// Zero-dependency binary serialization (.ridx) for instant cold boots.
    pub fn save_index<P: AsRef<Path>>(&self, doc_store: &DocumentStore, path: P) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(self.hidden_dim as u32).to_le_bytes())?;
        w.write_all(&(doc_store.chunks.len() as u32).to_le_bytes())?;

        for chunk in &doc_store.chunks {
            w.write_all(&(chunk.id as u64).to_le_bytes())?;

            let src_bytes = chunk.source_file.as_bytes();
            w.write_all(&(src_bytes.len() as u32).to_le_bytes())?;
            w.write_all(src_bytes)?;

            let txt_bytes = chunk.text.as_bytes();
            w.write_all(&(txt_bytes.len() as u32).to_le_bytes())?;
            w.write_all(txt_bytes)?;
        }

        for &val in &self.vectors {
            w.write_all(&val.to_le_bytes())?;
        }
        w.flush()
    }

    /// Loads DocumentStore + flat VectorStore directly from a .ridx binary file.
    pub fn load_index<P: AsRef<Path>>(path: P) -> io::Result<(DocumentStore, VectorStore)> {
        let mut r = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid .ridx magic header"));
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported .ridx version"));
        }

        r.read_exact(&mut buf4)?;
        let hidden_dim = u32::from_le_bytes(buf4) as usize;

        r.read_exact(&mut buf4)?;
        let num_chunks = u32::from_le_bytes(buf4) as usize;

        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            r.read_exact(&mut buf8)?;
            let id = u64::from_le_bytes(buf8) as usize;

            r.read_exact(&mut buf4)?;
            let src_len = u32::from_le_bytes(buf4) as usize;
            let mut src_buf = vec![0u8; src_len];
            r.read_exact(&mut src_buf)?;
            let source_file = String::from_utf8_lossy(&src_buf).into_owned();

            r.read_exact(&mut buf4)?;
            let txt_len = u32::from_le_bytes(buf4) as usize;
            let mut txt_buf = vec![0u8; txt_len];
            r.read_exact(&mut txt_buf)?;
            let text = String::from_utf8_lossy(&txt_buf).into_owned();

            chunks.push(Chunk {
                id,
                source_file,
                text,
            });
        }

        let total_floats = num_chunks * hidden_dim;
        let mut vectors = Vec::with_capacity(total_floats);
        for _ in 0..total_floats {
            r.read_exact(&mut buf4)?;
            vectors.push(f32::from_le_bytes(buf4));
        }

        Ok((
            DocumentStore { chunks },
            VectorStore { vectors, hidden_dim },
        ))
    }
}
