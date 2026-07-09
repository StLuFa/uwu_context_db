//! Token counting utilities backed by the same tokenizer family used by modern OpenAI-compatible models.
//!
//! Budget decisions must be based on model tokens, not bytes. This module keeps
//! token accounting deterministic and shared across core/retrieve paths.

use std::sync::OnceLock;

use tiktoken_rs::CoreBPE;

static CL100K_BASE: OnceLock<CoreBPE> = OnceLock::new();

/// Count model tokens using the `cl100k_base` tokenizer.
///
/// `cl100k_base` is the common tokenizer for `text-embedding-3-*`, GPT-4-class
/// OpenAI-compatible chat models, and most compatible providers. Keeping this in
/// core removes the previous byte-length heuristics from budget-sensitive paths.
pub fn count_tokens(text: &str) -> usize {
    let tokenizer = CL100K_BASE.get_or_init(|| {
        tiktoken_rs::cl100k_base().expect("cl100k_base tokenizer data must be embedded")
    });
    tokenizer.encode_with_special_tokens(text).len()
}

/// Count tokens but keep a minimum charge for fixed overhead around retrieved items.
pub fn count_tokens_with_floor(text: &str, floor: usize) -> usize {
    count_tokens(text).max(floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_tokens_with_model_tokenizer_not_bytes() {
        assert_eq!(count_tokens("hello world"), 2);
        assert!(count_tokens("你好，世界") > 0);
        assert_ne!(
            count_tokens("fn main() { println!(\"hi\"); }"),
            "fn main() { println!(\"hi\"); }".len() / 4
        );
    }

    #[test]
    fn applies_floor_after_real_count() {
        assert_eq!(count_tokens_with_floor("tiny", 50), 50);
        assert!(count_tokens_with_floor("a long enough sentence to tokenize", 1) >= 1);
    }
}
