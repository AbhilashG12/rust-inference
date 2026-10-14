use thiserror::Error;
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer: {0}")]
    LoadFailed(String),
}

pub struct Tokenizer {
    engine: HfTokenizer,
}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Self, TokenizerError> {
        let engine =
            HfTokenizer::from_file(path).map_err(|e| TokenizerError::LoadFailed(e.to_string()))?;
        Ok(Self { engine })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let formatted_text = text.replace("\\n", "\n");
        let encoding = self.engine.encode(formatted_text.as_str(), true).unwrap();
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, id: u32) -> String {
        self.engine.decode(&[id], false).unwrap()
    }
}
