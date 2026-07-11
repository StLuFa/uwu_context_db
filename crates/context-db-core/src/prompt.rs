//! Prompt optimization utilities shared by LLM callers.
//!
//! The optimizer is extractive by default: it preserves instructions, keeps
//! high-signal lines, and trims repeated/low-value evidence before the provider
//! request is built. LLM-based distillation can be layered on top by callers
//! through `LlmOpts::prompt_compression` without changing provider code.

use crate::tokenizer::count_tokens;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmTaskKind {
    #[default]
    General,
    Summary,
    Extraction,
    Deduplication,
    Arbitration,
    Merge,
    Synthesis,
    Reflection,
    Prediction,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheMode {
    Off,
    #[default]
    ProviderDefault,
    Force,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCompressionMode {
    Off,
    #[default]
    Extractive,
    Distill,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromptOptimization {
    pub cache_mode: PromptCacheMode,
    pub compression: PromptCompressionMode,
    pub compression_target_tokens: Option<usize>,
    pub cache_prefix_tokens: usize,
}

impl Default for PromptOptimization {
    fn default() -> Self {
        Self {
            cache_mode: PromptCacheMode::ProviderDefault,
            compression: PromptCompressionMode::Extractive,
            compression_target_tokens: Some(6_000),
            cache_prefix_tokens: 512,
        }
    }
}

impl PromptOptimization {
    pub fn no_cache(mut self) -> Self {
        self.cache_mode = PromptCacheMode::Off;
        self
    }

    pub fn force_cache(mut self) -> Self {
        self.cache_mode = PromptCacheMode::Force;
        self
    }

    pub fn no_compression(mut self) -> Self {
        self.compression = PromptCompressionMode::Off;
        self
    }

    pub fn target_tokens(mut self, target: usize) -> Self {
        self.compression_target_tokens = Some(target);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizedPrompt {
    pub text: String,
    pub original_tokens: usize,
    pub optimized_tokens: usize,
    pub compressed: bool,
}

pub fn optimize_prompt(prompt: &str, optimization: &PromptOptimization) -> OptimizedPrompt {
    let original_tokens = count_tokens(prompt);
    let target = optimization.compression_target_tokens.unwrap_or(usize::MAX);
    if optimization.compression == PromptCompressionMode::Off || original_tokens <= target {
        return OptimizedPrompt {
            text: prompt.to_string(),
            original_tokens,
            optimized_tokens: original_tokens,
            compressed: false,
        };
    }

    let text = extractive_compress(prompt, target);
    let optimized_tokens = count_tokens(&text);
    OptimizedPrompt {
        text,
        original_tokens,
        optimized_tokens,
        compressed: optimized_tokens < original_tokens,
    }
}

fn extractive_compress(prompt: &str, target_tokens: usize) -> String {
    let mut kept = Vec::new();
    let mut used = 0usize;
    let mut seen = std::collections::HashSet::new();

    for line in prompt.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if kept.last().is_some_and(|line: &String| !line.is_empty()) {
                kept.push(String::new());
            }
            continue;
        }

        let score = line_score(trimmed);
        if score == 0 {
            continue;
        }

        let normalized = normalize_line(trimmed);
        if !seen.insert(normalized) {
            continue;
        }

        let cost = count_tokens(trimmed).max(1);
        if used + cost > target_tokens && score < 4 {
            continue;
        }
        if used + cost > target_tokens && !kept.is_empty() {
            break;
        }

        kept.push(trimmed.to_string());
        used += cost;
    }

    if kept.is_empty() {
        prompt
            .chars()
            .take(target_tokens.saturating_mul(4))
            .collect()
    } else {
        kept.join("\n")
    }
}

fn line_score(line: &str) -> u8 {
    let lower = line.to_ascii_lowercase();
    if lower.starts_with("respond")
        || lower.starts_with("return")
        || lower.contains("json")
        || lower.contains("resolution")
        || lower.contains("evidence")
        || lower.contains("quality")
        || lower.contains("confidence")
        || lower.contains("conflict")
        || lower.contains("principle")
        || line.starts_with('#')
    {
        5
    } else if line.starts_with('-') || line.starts_with('*') || line.contains(':') {
        3
    } else if line.len() <= 240 {
        2
    } else {
        1
    }
}

fn normalize_line(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_repeated_prompt_lines() {
        let prompt = format!(
            "Return JSON only\n{}\nEvidence: alpha\nEvidence: alpha\nFinal instruction",
            "low value filler line that can be dropped\n".repeat(200)
        );
        let optimized = optimize_prompt(&prompt, &PromptOptimization::default().target_tokens(80));
        assert!(optimized.compressed);
        assert!(optimized.text.contains("Return JSON only"));
        assert!(optimized.text.contains("Evidence: alpha"));
        assert_eq!(optimized.text.matches("Evidence: alpha").count(), 1);
    }

    #[test]
    fn disabled_compression_returns_original() {
        let opts = PromptOptimization::default().no_compression();
        let optimized = optimize_prompt("hello world", &opts);
        assert_eq!(optimized.text, "hello world");
        assert!(!optimized.compressed);
    }
}
