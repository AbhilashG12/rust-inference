use std::fs;
use std::path::{Path, PathBuf};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::tokenizer::Tokenizer;
use crate::embedder::EmbedderEngine;


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

    /// Recursively walks a directory, ingests valid text files, and semantically chunks them.
    pub fn ingest_directory<P: AsRef<Path>>(
        &mut self,
        dir_path: P,
        tokenizer: &Tokenizer,
        chunk_size: usize,
        overlap: usize,
    ) {
        assert!(chunk_size > overlap, "Chunk size must be greater than overlap");
        
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
            // Read file gracefully. Skip binaries or non-UTF8 files without crashing.
            if let Ok(content) = fs::read_to_string(&file_path) {
                let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
                
                // Only process files with content
                if content.trim().is_empty() { continue; }

                let document_chunks = Self::sliding_window_chunk(&content, tokenizer, chunk_size, overlap);
                
                for text_chunk in document_chunks {
                    self.chunks.push(Chunk {
                        id: current_id,
                        source_file: file_name.clone(),
                        text: text_chunk,
                    });
                    current_id += 1;
                }
            }
        }
        
        println!(" Successfully ingested {} semantic chunks into the Document Store.", self.chunks.len());
    }

    /// Pure standard-library recursive directory crawler. No third-party crates needed.
    fn crawl_directory(dir: &Path, files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::crawl_directory(&path, files);
                } else {
                    // Filter for readable text formats
                    if let Some(ext) = path.extension() {
                        let ext_str = ext.to_string_lossy().to_lowercase();
                        if ext_str == "txt" || ext_str == "md" || ext_str == "csv" {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    /// Slices text into overlapping windows using the embedder's actual tokenizer.
    fn sliding_window_chunk(
        text: &str,
        tokenizer: &Tokenizer,
        chunk_size: usize,
        overlap: usize,
    ) -> Vec<String> {
        // Tokenize the entire document
        let tokens = tokenizer.encode(text);
        let mut chunks = Vec::new();
        
        let mut start = 0;
        let step = chunk_size - overlap;

        while start < tokens.len() {
            let end = std::cmp::min(start + chunk_size, tokens.len());
            let chunk_tokens = &tokens[start..end];
            
            // Decode the token slice back into a string for the Embedder Engine
            let mut chunk_text = String::new();
            for &t in chunk_tokens {
                let word = tokenizer.decode(t);
                // Clean up WordPiece "##" artifacts if present, else just append
                if word.starts_with("##") {
                    chunk_text.push_str(&word[2..]);
                } else {
                    if !chunk_text.is_empty() {
                        chunk_text.push(' ');
                    }
                    chunk_text.push_str(&word);
                }
            }
            
            chunks.push(chunk_text.trim().to_string());

            if end == tokens.len() {
                break;
            }
            start += step;
        }

        chunks
    }
}
// ==========================================
// PHASE 3: VECTOR STORE & SEARCH (FLAT MATRIX)
// ==========================================

#[derive(Debug)]
struct ScoredChunk {
    pub score: f32,
    pub chunk_idx: usize,
}

// Priority Queue trait implementations
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
    /// A single, flat contiguous block of memory representing an [N, 384] matrix.
    /// This completely eliminates heap pointer chasing and maximizes L1 cache hits.
    pub vectors: Vec<f32>,
    pub hidden_dim: usize,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { 
            vectors: Vec::new(),
            hidden_dim: 384, // MiniLM vector dimension
        }
    }

    /// Passes every chunk through the Embedder Engine to compress its meaning into a vector.
    pub fn embed_store(&mut self, store: &DocumentStore, embedder: &EmbedderEngine) {
        println!(" Embedding {} chunks into a flat contiguous matrix...", store.chunks.len());
        self.vectors.clear();
        
        // Pre-allocate the exact exact block of RAM needed to prevent re-allocations
        self.vectors.reserve(store.chunks.len() * self.hidden_dim);
        
        for chunk in &store.chunks {
            // Embedder outputs an L2-Normalized vector
            let vector = embedder.embed(&chunk.text);
            // Append directly to the flat array
            self.vectors.extend_from_slice(&vector);
        }
        println!(" Vectorization complete.");
    }

    /// Blazing fast in-memory search over the flat matrix.
    pub fn search(&self, query: &str, embedder: &EmbedderEngine, top_k: usize) -> Vec<usize> {
        let query_vec = embedder.embed(query);
        let mut heap = BinaryHeap::new();

        let num_chunks = self.vectors.len() / self.hidden_dim;

        // Matrix-Vector Multiplication (GEMV) over a flat slice
        for i in 0..num_chunks {
            let start = i * self.hidden_dim;
            let doc_vec = &self.vectors[start .. start + self.hidden_dim];
            
            let mut dot_product = 0.0;
            
            // Because doc_vec and query_vec are contiguous memory slices, 
            // rustc/LLVM will auto-vectorize this loop into FMA AVX2 instructions.
            for d in 0..self.hidden_dim {
                dot_product += query_vec[d] * doc_vec[d];
            }
            
            heap.push(ScoredChunk { score: dot_product, chunk_idx: i });
        }

        let mut results = Vec::new();
        for _ in 0..top_k {
            if let Some(scored) = heap.pop() {
                results.push(scored.chunk_idx);
            }
        }
        results
    }
}
