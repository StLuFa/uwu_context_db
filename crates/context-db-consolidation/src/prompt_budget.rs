//! Shared, deterministic prompt budgeting for consolidation LLM flows.
//!
//! Callers provide semantic sections rather than one opaque string. Required
//! instructions are retained first; evidence is retained by priority and
//! truncated on UTF-8 boundaries until the configured token ceiling is met.

use agent_context_db_core::count_tokens;

use crate::ConfigError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptBudgetConfig {
    pub max_prompt_tokens: usize,
    pub min_section_tokens: usize,
}

impl PromptBudgetConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_prompt_tokens == 0 {
            return Err(ConfigError("max_prompt_tokens must be nonzero".into()));
        }
        if self.min_section_tokens > self.max_prompt_tokens {
            return Err(ConfigError(
                "min_section_tokens must not exceed max_prompt_tokens".into(),
            ));
        }
        Ok(())
    }
}

impl Default for PromptBudgetConfig {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 4_000,
            min_section_tokens: 32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PromptSection<'a> {
    pub label: &'a str,
    pub content: &'a str,
    /// Lower values are retained first. Equal priorities preserve input order.
    pub priority: u16,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BudgetedPrompt {
    pub text: String,
    pub original_tokens: usize,
    pub final_tokens: usize,
    pub retained_sections: usize,
    pub dropped_sections: usize,
    pub truncation_ratio: f32,
}

pub fn budget_prompt(
    config: PromptBudgetConfig,
    sections: &[PromptSection<'_>],
) -> Result<BudgetedPrompt, ConfigError> {
    config.validate()?;
    let original = render(
        sections
            .iter()
            .map(|section| (section.label, section.content)),
    );
    let original_tokens = tokens(&original)?;
    if original_tokens <= config.max_prompt_tokens {
        return Ok(BudgetedPrompt {
            text: original,
            original_tokens,
            final_tokens: original_tokens,
            retained_sections: sections.len(),
            dropped_sections: 0,
            truncation_ratio: 0.0,
        });
    }

    let mut ordered = sections.iter().enumerate().collect::<Vec<_>>();
    ordered.sort_by_key(|(index, section)| (!section.required, section.priority, *index));
    let mut retained: Vec<(&str, String)> = Vec::new();
    for (_, section) in ordered {
        let prefix = render(retained.iter().map(|(label, body)| (*label, body.as_str())));
        let prefix_tokens = tokens(&prefix)?;
        let remaining = config.max_prompt_tokens.saturating_sub(prefix_tokens);
        if remaining < config.min_section_tokens && !section.required {
            continue;
        }
        let candidate = truncate_to_fit(section.content, section.label, remaining)?;
        if !candidate.is_empty() {
            retained.push((section.label, candidate));
        }
    }

    let text = render(retained.iter().map(|(label, body)| (*label, body.as_str())));
    let final_tokens = tokens(&text)?;
    let retained_sections = retained.len();
    metrics::histogram!("uwu.consolidation.prompt.truncation_ratio")
        .record(1.0 - final_tokens as f64 / original_tokens.max(1) as f64);
    metrics::counter!("uwu.consolidation.prompt.dropped_sections")
        .increment(sections.len().saturating_sub(retained_sections) as u64);
    Ok(BudgetedPrompt {
        text,
        original_tokens,
        final_tokens,
        retained_sections,
        dropped_sections: sections.len().saturating_sub(retained_sections),
        truncation_ratio: 1.0 - final_tokens as f32 / original_tokens.max(1) as f32,
    })
}

fn render<'a>(sections: impl Iterator<Item = (&'a str, &'a str)>) -> String {
    sections
        .map(|(label, content)| format!("## {label}\n{content}"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn tokens(text: &str) -> Result<usize, ConfigError> {
    count_tokens(text).map_err(|error| ConfigError(format!("prompt tokenization failed: {error}")))
}

fn truncate_to_fit(content: &str, label: &str, budget: usize) -> Result<String, ConfigError> {
    if budget == 0 || tokens(&render(std::iter::once((label, ""))))? > budget {
        return Ok(String::new());
    }
    if tokens(&render(std::iter::once((label, content))))? <= budget {
        return Ok(content.to_string());
    }
    let boundaries = content
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(content.len()))
        .collect::<Vec<_>>();
    let (mut low, mut high) = (0usize, boundaries.len());
    while low + 1 < high {
        let mid = (low + high) / 2;
        if tokens(&render(std::iter::once((
            label,
            &content[..boundaries[mid]],
        ))))?
            <= budget
        {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(content[..boundaries[low]].trim_end().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_required_and_high_priority_sections_within_exact_budget() {
        let config = PromptBudgetConfig {
            max_prompt_tokens: 70,
            min_section_tokens: 4,
        };
        let result = budget_prompt(
            config,
            &[
                PromptSection {
                    label: "Instructions",
                    content: "Return valid JSON with a result field.",
                    priority: 0,
                    required: true,
                },
                PromptSection {
                    label: "Primary evidence",
                    content: &"important evidence ".repeat(80),
                    priority: 1,
                    required: false,
                },
                PromptSection {
                    label: "Secondary evidence",
                    content: &"secondary material ".repeat(80),
                    priority: 10,
                    required: false,
                },
            ],
        )
        .unwrap();
        assert!(result.final_tokens <= config.max_prompt_tokens);
        assert!(result.text.contains("Return valid JSON"));
        assert!(result.text.contains("Primary evidence"));
        assert!(result.truncation_ratio > 0.0);
    }

    #[test]
    fn rejects_invalid_budget() {
        assert!(
            budget_prompt(
                PromptBudgetConfig {
                    max_prompt_tokens: 0,
                    min_section_tokens: 0
                },
                &[]
            )
            .is_err()
        );
    }
}
