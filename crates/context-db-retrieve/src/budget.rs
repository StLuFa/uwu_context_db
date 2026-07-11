//! Token-budget loading based on a multiple-choice knapsack.
//!
//! Each retrieval hit can be loaded at L0/L1/L2 with different token costs and
//! value. The optimizer chooses the globally best mix under the requested token
//! budget instead of greedily truncating by relevance.

use agent_context_db_core::{ContentLevel, ContentPayload, count_tokens_with_floor};

use crate::RetrievalHit;

const L0_FLOOR: usize = 1;
const L1_FLOOR: usize = 50;
const L2_FLOOR: usize = 100;

#[derive(Debug, Clone)]
pub struct BudgetLoadPlan {
    pub hits: Vec<RetrievalHit>,
    pub tokens_used: usize,
}

#[derive(Debug, Clone)]
pub struct LevelAllocation {
    pub uri: agent_context_db_core::ContextUri,
    pub level: ContentLevel,
    pub tokens: usize,
}

#[derive(Debug, Clone)]
struct Choice {
    hit_index: usize,
    level: ContentLevel,
    cost: usize,
    value: f32,
    payload: ContentPayload,
}

/// Select hits and content levels with a multiple-choice knapsack optimizer.
pub fn load_hits_within_budget(hits: Vec<RetrievalHit>, budget: usize) -> BudgetLoadPlan {
    if budget == 0 || hits.is_empty() {
        return BudgetLoadPlan {
            hits: Vec::new(),
            tokens_used: 0,
        };
    }

    let choices_by_hit: Vec<Vec<Choice>> = hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| choices_for_hit(idx, hit, budget))
        .collect();

    let selected = select_choices(&choices_by_hit, budget);
    let mut tokens_used = 0usize;
    let mut loaded = Vec::with_capacity(selected.len());

    for choice in selected {
        let mut hit = hits[choice.hit_index].clone();
        hit.level = choice.level;
        hit.content = choice.payload;
        tokens_used += choice.cost;
        loaded.push(hit);
    }

    loaded.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uri.to_string().cmp(&b.uri.to_string()))
    });

    BudgetLoadPlan {
        hits: loaded,
        tokens_used,
    }
}

/// Allocate only URI levels for callers that will load content later.
pub fn allocate_hit_levels(hits: &[RetrievalHit], budget: usize) -> Vec<LevelAllocation> {
    let choices_by_hit: Vec<Vec<Choice>> = hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| choices_for_hit(idx, hit, budget))
        .collect();

    select_choices(&choices_by_hit, budget)
        .into_iter()
        .map(|choice| LevelAllocation {
            uri: hits[choice.hit_index].uri.clone(),
            level: choice.level,
            tokens: choice.cost,
        })
        .collect()
}

fn select_choices(choices_by_hit: &[Vec<Choice>], budget: usize) -> Vec<Choice> {
    let mut dp = vec![0.0f32; budget + 1];
    let mut paths: Vec<Vec<Choice>> = vec![Vec::new(); budget + 1];

    for choices in choices_by_hit {
        let mut next = dp.clone();
        let mut next_paths = paths.clone();

        for used in 0..=budget {
            for choice in choices {
                if choice.cost == 0 || used + choice.cost > budget {
                    continue;
                }
                let value = dp[used] + choice.value;
                let target = used + choice.cost;
                if value > next[target] {
                    next[target] = value;
                    let mut path = paths[used].clone();
                    path.push(choice.clone());
                    next_paths[target] = path;
                }
            }
        }

        dp = next;
        paths = next_paths;
    }

    let best_budget = (0..=budget)
        .max_by(|a, b| {
            dp[*a]
                .partial_cmp(&dp[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    paths.swap_remove(best_budget)
}

fn choices_for_hit(idx: usize, hit: &RetrievalHit, budget: usize) -> Vec<Choice> {
    let mut choices = Vec::new();
    push_text_choices(&mut choices, idx, hit, budget);
    if choices.is_empty() {
        let cost = fallback_cost(&hit.content);
        if cost <= budget {
            choices.push(Choice {
                hit_index: idx,
                level: hit.level,
                cost,
                value: value_for(hit, hit.level, cost),
                payload: hit.content.clone(),
            });
        }
    }
    choices
}

fn push_text_choices(choices: &mut Vec<Choice>, idx: usize, hit: &RetrievalHit, budget: usize) {
    let ContentPayload::Text {
        sparse,
        dense,
        full,
    } = &hit.content
    else {
        return;
    };

    let l0_cost = count_tokens_with_floor(sparse, L0_FLOOR);
    if l0_cost <= budget {
        choices.push(Choice {
            hit_index: idx,
            level: ContentLevel::L0,
            cost: l0_cost,
            value: value_for(hit, ContentLevel::L0, l0_cost),
            payload: text_payload(sparse, sparse, sparse),
        });
    }

    if !dense.trim().is_empty() {
        let l1_cost = count_tokens_with_floor(dense, L1_FLOOR);
        if l1_cost <= budget {
            choices.push(Choice {
                hit_index: idx,
                level: ContentLevel::L1,
                cost: l1_cost,
                value: value_for(hit, ContentLevel::L1, l1_cost),
                payload: text_payload(sparse, dense, dense),
            });
        }
    }

    if !full.trim().is_empty() {
        let l2_cost = count_tokens_with_floor(full, L2_FLOOR);
        if l2_cost <= budget {
            choices.push(Choice {
                hit_index: idx,
                level: ContentLevel::L2,
                cost: l2_cost,
                value: value_for(hit, ContentLevel::L2, l2_cost),
                payload: text_payload(sparse, dense, full),
            });
        }
    }
}

fn text_payload(sparse: &str, dense: &str, full: &str) -> ContentPayload {
    ContentPayload::Text {
        sparse: sparse.to_string(),
        dense: dense.to_string(),
        full: full.to_string(),
    }
}

fn fallback_cost(content: &ContentPayload) -> usize {
    match content {
        ContentPayload::Text { sparse, .. } => count_tokens_with_floor(sparse, L2_FLOOR),
        ContentPayload::Image { .. } => L0_FLOOR,
        ContentPayload::Audio { transcript, .. } => count_tokens_with_floor(transcript, L1_FLOOR),
        ContentPayload::Structured { summary, .. } => count_tokens_with_floor(summary, L1_FLOOR),
        ContentPayload::Composite { summary, .. } => count_tokens_with_floor(summary, L1_FLOOR),
    }
}

fn value_for(hit: &RetrievalHit, level: ContentLevel, cost: usize) -> f32 {
    let level_gain = match level {
        ContentLevel::L0 => 1.0,
        ContentLevel::L1 => 1.45,
        ContentLevel::L2 => 1.85,
    };
    let quality = hit.metadata.quality_score.unwrap_or(0.75).clamp(0.05, 1.0);
    let content_type_gain = if hit.content_type.is_some() {
        1.05
    } else {
        1.0
    };
    let cost_penalty = (cost.max(1) as f32).ln_1p() * 0.01;
    (hit.relevance.max(0.0) * quality * content_type_gain * level_gain) - cost_penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, ContextMeta, ContextUri};

    fn hit(uri: &str, relevance: f32, sparse: &str, dense: &str, full: &str) -> RetrievalHit {
        let metadata = ContextMeta {
            quality_score: Some(1.0),
            ..Default::default()
        };
        RetrievalHit {
            uri: ContextUri::parse(uri).unwrap(),
            level: ContentLevel::L0,
            content: ContentPayload::Text {
                sparse: sparse.into(),
                dense: dense.into(),
                full: full.into(),
            },
            relevance,
            parent_chain: vec![],
            content_type: Some(ContentType::Fact),
            metadata,
            created_at: None,
            updated_at: None,
        }
    }

    #[test]
    fn knapsack_can_choose_multiple_l0_over_single_large_hit() {
        let hits = vec![
            hit(
                "uwu://t/agent/a/fact/a",
                0.95,
                "alpha",
                &"large ".repeat(120),
                &"larger ".repeat(300),
            ),
            hit("uwu://t/agent/a/fact/b", 0.9, "beta", "beta", "beta"),
            hit("uwu://t/agent/a/fact/c", 0.88, "gamma", "gamma", "gamma"),
        ];

        let plan = load_hits_within_budget(hits, 20);
        assert!(plan.tokens_used <= 20);
        assert!(plan.hits.len() >= 2);
    }

    #[test]
    fn allocate_levels_respects_budget() {
        let hits = vec![hit(
            "uwu://t/agent/a/fact/a",
            0.8,
            "alpha",
            "alpha",
            "alpha",
        )];
        let plan = allocate_hit_levels(&hits, 10);
        assert_eq!(plan.len(), 1);
        assert!(plan[0].tokens <= 10);
    }
}
