//! Token counting utilities backed by the same tokenizer family used by modern OpenAI-compatible models.
//!
//! Budget decisions must be based on model tokens, not bytes. This module keeps
//! token accounting deterministic and shared across core/retrieve paths.

use std::sync::OnceLock;

use tiktoken_rs::CoreBPE;

use crate::{ContextError, Result};

static CL100K_BASE: OnceLock<std::result::Result<CoreBPE, String>> = OnceLock::new();

fn cl100k_base() -> Result<&'static CoreBPE> {
    CL100K_BASE
        .get_or_init(|| tiktoken_rs::cl100k_base().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| {
            ContextError::Unsupported(format!(
                "failed to initialize cl100k_base tokenizer: {error}"
            ))
        })
}

/// Count model tokens using the `cl100k_base` tokenizer.
///
/// Initialization errors are cached and propagated to every caller rather than
/// causing a process panic or silently switching to an inaccurate heuristic.
pub fn count_tokens(text: &str) -> Result<usize> {
    Ok(cl100k_base()?.encode_with_special_tokens(text).len())
}

/// Count tokens but keep a minimum charge for fixed overhead around retrieved items.
pub fn count_tokens_with_floor(text: &str, floor: usize) -> Result<usize> {
    Ok(count_tokens(text)?.max(floor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_tokens_with_model_tokenizer_not_bytes() -> Result<()> {
        assert_eq!(count_tokens("hello world")?, 2);
        assert!(count_tokens("你好，世界")? > 0);
        assert_ne!(
            count_tokens("fn main() { println!(\"hi\"); }")?,
            "fn main() { println!(\"hi\"); }".len() / 4
        );
        Ok(())
    }

    #[test]
    fn applies_floor_after_real_count() -> Result<()> {
        assert_eq!(count_tokens_with_floor("tiny", 50)?, 50);
        assert!(count_tokens_with_floor("a long enough sentence to tokenize", 1)? >= 1);
        Ok(())
    }

    #[test]
    fn tokenizer_initialization_result_is_stable() {
        let first = cl100k_base().map(std::ptr::from_ref);
        let second = cl100k_base().map(std::ptr::from_ref);
        assert_eq!(first.is_ok(), second.is_ok());
        if let (Err(first), Err(second)) = (first, second) {
            assert_eq!(first.to_string(), second.to_string());
        }
    }
}
