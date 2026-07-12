//! Token-budget loading based on a multiple-choice knapsack.
//!
//! Each retrieval hit can be loaded at L0/L1/L2 with different token costs and
//! value. The optimizer chooses the globally best mix under the requested token
//! budget instead of greedily truncating by relevance.

use agent_context_db_core::{ContentLevel, ContentPayload, count_tokens_with_floor};

use crate::{RetrievalHit, TokenBudgetConfig};

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
pub fn load_hits_within_budget(
    hits: Vec<RetrievalHit>,
    budget: usize,
    config: TokenBudgetConfig,
) -> agent_context_db_core::Result<BudgetLoadPlan> {
    if budget == 0 || hits.is_empty() {
        return Ok(BudgetLoadPlan {
            hits: Vec::new(),
            tokens_used: 0,
        });
    }

    let choices_by_hit: Vec<Vec<Choice>> = hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| choices_for_hit(idx, hit, budget, config))
        .collect::<agent_context_db_core::Result<_>>()?;

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

    Ok(BudgetLoadPlan {
        hits: loaded,
        tokens_used,
    })
}

/// Allocate only URI levels for callers that will load content later.
pub fn allocate_hit_levels(
    hits: &[RetrievalHit],
    budget: usize,
    config: TokenBudgetConfig,
) -> agent_context_db_core::Result<Vec<LevelAllocation>> {
    let choices_by_hit: Vec<Vec<Choice>> = hits
        .iter()
        .enumerate()
        .map(|(idx, hit)| choices_for_hit(idx, hit, budget, config))
        .collect::<agent_context_db_core::Result<_>>()?;

    Ok(select_choices(&choices_by_hit, budget)
        .into_iter()
        .map(|choice| LevelAllocation {
            uri: hits[choice.hit_index].uri.clone(),
            level: choice.level,
            tokens: choice.cost,
        })
        .collect())
}

#[derive(Clone, Copy)]
enum Predecessor {
    Carry,
    Choice {
        previous_budget: usize,
        index: usize,
    },
}

fn select_choices(choices_by_hit: &[Vec<Choice>], budget: usize) -> Vec<Choice> {
    let mut scores = vec![0.0f32; budget + 1];
    let mut predecessors = Vec::with_capacity(choices_by_hit.len());

    for choices in choices_by_hit {
        let mut next = scores.clone();
        let mut layer = vec![Predecessor::Carry; budget + 1];

        for (used, score) in scores.iter().copied().enumerate() {
            for (index, choice) in choices.iter().enumerate() {
                if choice.cost == 0 || used + choice.cost > budget {
                    continue;
                }
                let value = score + choice.value;
                let target = used + choice.cost;
                if value > next[target] {
                    next[target] = value;
                    layer[target] = Predecessor::Choice {
                        previous_budget: used,
                        index,
                    };
                }
            }
        }

        scores = next;
        predecessors.push(layer);
    }

    let mut current_budget = (0..=budget)
        .max_by(|a, b| {
            scores[*a]
                .partial_cmp(&scores[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut selected = Vec::new();
    for (hit_index, layer) in predecessors.iter().enumerate().rev() {
        if let Predecessor::Choice {
            previous_budget,
            index,
        } = layer[current_budget]
        {
            selected.push(choices_by_hit[hit_index][index].clone());
            current_budget = previous_budget;
        }
    }
    selected.reverse();
    selected
}

fn choices_for_hit(
    idx: usize,
    hit: &RetrievalHit,
    budget: usize,
    config: TokenBudgetConfig,
) -> agent_context_db_core::Result<Vec<Choice>> {
    let mut choices = Vec::new();
    push_text_choices(&mut choices, idx, hit, budget, config)?;
    if choices.is_empty() {
        let cost = fallback_cost(&hit.content, config)?;
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
    Ok(choices)
}

fn push_text_choices(
    choices: &mut Vec<Choice>,
    idx: usize,
    hit: &RetrievalHit,
    budget: usize,
    config: TokenBudgetConfig,
) -> agent_context_db_core::Result<()> {
    let ContentPayload::Text {
        sparse,
        dense,
        full,
    } = &hit.content
    else {
        return Ok(());
    };

    let l0_cost = count_tokens_with_floor(sparse, config.l0_floor)?;
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
        let l1_cost = count_tokens_with_floor(dense, config.l1_floor)?;
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
        let l2_cost = count_tokens_with_floor(full, config.l2_floor)?;
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
    Ok(())
}

fn text_payload(sparse: &str, dense: &str, full: &str) -> ContentPayload {
    ContentPayload::Text {
        sparse: sparse.to_string(),
        dense: dense.to_string(),
        full: full.to_string(),
    }
}

fn fallback_cost(
    content: &ContentPayload,
    config: TokenBudgetConfig,
) -> agent_context_db_core::Result<usize> {
    match content {
        ContentPayload::Text { sparse, .. } => count_tokens_with_floor(sparse, config.l2_floor),
        ContentPayload::Image { .. } => Ok(config.l0_floor),
        ContentPayload::Audio { transcript, .. } => {
            count_tokens_with_floor(transcript, config.l1_floor)
        }
        ContentPayload::Structured { summary, .. } => {
            count_tokens_with_floor(summary, config.l1_floor)
        }
        ContentPayload::Composite { summary, .. } => {
            count_tokens_with_floor(summary, config.l1_floor)
        }
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

    fn select_choices_with_full_paths(
        choices_by_hit: &[Vec<Choice>],
        budget: usize,
    ) -> Vec<Choice> {
        let mut scores = vec![0.0f32; budget + 1];
        let mut paths = vec![Vec::new(); budget + 1];
        for choices in choices_by_hit {
            let mut next = scores.clone();
            let mut next_paths = paths.clone();
            for used in 0..=budget {
                for choice in choices {
                    if choice.cost == 0 || used + choice.cost > budget {
                        continue;
                    }
                    let target = used + choice.cost;
                    let value = scores[used] + choice.value;
                    if value > next[target] {
                        next[target] = value;
                        next_paths[target] = paths[used].clone();
                        next_paths[target].push(choice.clone());
                    }
                }
            }
            scores = next;
            paths = next_paths;
        }
        let best = (0..=budget)
            .max_by(|a, b| {
                scores[*a]
                    .partial_cmp(&scores[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        paths.swap_remove(best)
    }

    #[test]
    fn predecessor_dp_is_equivalent_to_full_path_dp() {
        let hits = [
            hit(
                "uwu://t/agent/a/fact/a",
                0.91,
                "a",
                "a dense",
                "a full text",
            ),
            hit(
                "uwu://t/agent/a/fact/b",
                0.73,
                "b",
                "b dense",
                "b full text",
            ),
            hit(
                "uwu://t/agent/a/fact/c",
                0.52,
                "c",
                "c dense",
                "c full text",
            ),
        ];
        let config = TokenBudgetConfig {
            l0_floor: 1,
            l1_floor: 3,
            l2_floor: 5,
        };
        let choices = hits
            .iter()
            .enumerate()
            .map(|(index, hit)| choices_for_hit(index, hit, 12, config).unwrap())
            .collect::<Vec<_>>();

        for budget in 0..=12 {
            let actual = select_choices(&choices, budget);
            let expected = select_choices_with_full_paths(&choices, budget);
            let signature = |selected: &[Choice]| {
                selected
                    .iter()
                    .map(|choice| (choice.hit_index, choice.level, choice.cost))
                    .collect::<Vec<_>>()
            };
            assert_eq!(signature(&actual), signature(&expected), "budget={budget}");
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

        let plan = load_hits_within_budget(hits, 20, TokenBudgetConfig::default()).unwrap();
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
        let plan = allocate_hit_levels(&hits, 10, TokenBudgetConfig::default()).unwrap();
        assert_eq!(plan.len(), 1);
        assert!(plan[0].tokens <= 10);
    }
}
