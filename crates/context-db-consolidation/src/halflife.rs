//! 半衰期预测 — LLM 驱动的知识有效期预测 + 到期复审。
//!
//! 理论根基：放射性衰变 + Anki SM-2 间隔重复。
//! 巩固时让 LLM 评估 domain volatility / specificity / technological context。

use crate::quality::QualityRoute;
use agent_context_db_core::{
    ConsolidationMeta, ConsolidationStatus, ContentType, ContextEntry, ContextUri, LineageEntry,
    LlmClient, LlmOpts, MvccVersion, StateScope,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::Arc;

/// 半衰期预测结果。
#[derive(Debug, Clone)]
pub struct HalfLifePrediction {
    pub half_life_days: f64,
    pub confidence: f32,
    pub reasoning: String,
}

/// LLM 驱动的半衰期预测器。
pub struct HalfLifePredictor {
    llm: Arc<dyn LlmClient>,
}

#[derive(Debug, Deserialize)]
struct LlmHalfLifeResponse {
    half_life_days: f64,
    confidence: f32,
    reasoning: String,
}

impl HalfLifePredictor {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// 预测知识的半衰期。
    ///
    /// 策略：
    /// 1. 优先用 LLM 评估 domain volatility + specificity + tech context
    /// 2. LLM 不可用时回退到启发性规则
    pub async fn predict(&self, content: &str, _domain_hint: Option<&str>) -> HalfLifePrediction {
        let prompt = format!(
            r#"Predict the knowledge half-life (in days) for this insight:

"{content}"

Consider these factors:
- Domain volatility: framework-specific knowledge decays faster than mathematical truths
- Specificity: specific API calls/function names change fast, general principles slow
- Technological context: is this tied to a specific version/era?

Return a JSON object with these fields:
{{"half_life_days": <number, -1 for near-infinite timeless truths>,
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining the decision>"}}"#
        );

        let opts = LlmOpts {
            max_tokens: Some(256),
            temperature: Some(0.0),
            ..Default::default()
        };

        match self.llm.complete(&prompt, &opts).await {
            Ok(response) => {
                // Try to parse LLM JSON response
                if let Ok(parsed) = serde_json::from_str::<LlmHalfLifeResponse>(&response) {
                    return HalfLifePrediction {
                        half_life_days: parsed.half_life_days.max(1.0),
                        confidence: parsed.confidence.clamp(0.0, 1.0),
                        reasoning: parsed.reasoning,
                    };
                }
                // Parse failed — fall through to heuristic
            }
            Err(_) => {
                // LLM unavailable — fall through to heuristic
            }
        }

        // Heuristic fallback
        Self::heuristic_predict(content)
    }

    /// 启发性规则回退（不依赖 LLM）。
    fn heuristic_predict(content: &str) -> HalfLifePrediction {
        let has_code = content.contains('(') && content.contains(')')
            || content.contains("::")
            || content.contains("fn ");
        let has_principle = content.contains("原则")
            || content.contains("principle")
            || content.contains("always")
            || content.contains("never")
            || content.contains("定义");

        let (days, reasoning) = if has_code && !has_principle {
            (
                60.0,
                "contains specific API/function references — likely version-dependent",
            )
        } else if has_principle {
            (365.0, "contains general principles — slow to change")
        } else {
            (180.0, "mixed content — moderate decay")
        };

        HalfLifePrediction {
            half_life_days: days,
            confidence: 0.6,
            reasoning: reasoning.to_string(),
        }
    }

    /// 查找已过半衰期的条目。
    pub fn find_expired(&self, created_at: DateTime<Utc>, half_life_days: f64) -> bool {
        Self::is_expired_at(created_at, half_life_days, Utc::now())
    }

    pub fn is_expired_at(
        created_at: DateTime<Utc>,
        half_life_days: f64,
        now: DateTime<Utc>,
    ) -> bool {
        let age_days = (now - created_at).num_hours().max(0) as f64 / 24.0;
        age_days > half_life_days.max(0.1)
    }
}

// ===========================================================================
// 主动间隔重复复习调度器
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewAction {
    Rehearse,
    Revalidate,
    ForgetCandidate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewTask {
    pub uri: ContextUri,
    pub content_type: Option<ContentType>,
    pub action: ReviewAction,
    pub route: QualityRoute,
    pub due_score: f32,
    pub age_days: f64,
    pub half_life_days: f64,
    pub stability_days: f64,
    pub reinforcements: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReviewMemoryState {
    pub half_life_days: f64,
    pub stability_days: f64,
    pub reinforcements: u32,
    pub last_reviewed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewOutcome {
    pub uri: ContextUri,
    pub action: ReviewAction,
    pub route: QualityRoute,
    pub next_stability_days: f64,
    pub reinforcements: u32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SpacedRepetitionConfig {
    pub due_threshold: f32,
    pub revalidate_threshold: f32,
    pub forget_threshold: f32,
    pub max_tasks: usize,
    pub default_half_life_days: f64,
    pub default_stability_days: f64,
}

impl Default for SpacedRepetitionConfig {
    fn default() -> Self {
        Self {
            due_threshold: 0.82,
            revalidate_threshold: 1.05,
            forget_threshold: 1.75,
            max_tasks: 128,
            default_half_life_days: 30.0,
            default_stability_days: 14.0,
        }
    }
}

pub struct SpacedRepetitionScheduler {
    config: SpacedRepetitionConfig,
}

impl Default for SpacedRepetitionScheduler {
    fn default() -> Self {
        Self::new(SpacedRepetitionConfig::default())
    }
}

impl SpacedRepetitionScheduler {
    pub fn new(config: SpacedRepetitionConfig) -> Self {
        Self { config }
    }

    pub fn plan(&self, entries: &[ContextEntry], now: DateTime<Utc>) -> Vec<ReviewTask> {
        let mut tasks = entries
            .iter()
            .filter_map(|entry| self.task_for_entry(entry, now))
            .collect::<Vec<_>>();
        tasks.sort_by(|a, b| {
            b.due_score
                .partial_cmp(&a.due_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.uri.as_str().cmp(b.uri.as_str()))
        });
        tasks.truncate(self.config.max_tasks);
        tasks
    }

    pub fn apply_adoption(
        &self,
        entry: &mut ContextEntry,
        task: &ReviewTask,
        now: DateTime<Utc>,
    ) -> ReviewOutcome {
        let (next_stability_days, reinforcements) =
            reinforce_on_adoption(task.stability_days, task.reinforcements);
        upsert_review_state(
            entry,
            ReviewMemoryState {
                half_life_days: task.half_life_days.max(next_stability_days),
                stability_days: next_stability_days,
                reinforcements,
                last_reviewed_at: now,
            },
        );
        entry.updated_at = now;
        entry
            .metadata
            .tags
            .retain(|tag| !tag.starts_with("quality:review:"));
        entry.metadata.tags.push("quality:review:adopted".into());
        ReviewOutcome {
            uri: entry.uri.clone(),
            action: ReviewAction::Rehearse,
            route: QualityRoute::Rehearse,
            next_stability_days,
            reinforcements,
            tags: entry.metadata.tags.clone(),
        }
    }

    fn task_for_entry(&self, entry: &ContextEntry, now: DateTime<Utc>) -> Option<ReviewTask> {
        if !matches!(
            entry.metadata.state_scope,
            Some(StateScope::Long) | Some(StateScope::Mid)
        ) {
            return None;
        }

        let state = review_state(entry).unwrap_or_else(|| ReviewMemoryState {
            half_life_days: entry
                .metadata
                .consolidation
                .as_ref()
                .and_then(|meta| meta.half_life_days)
                .unwrap_or(self.config.default_half_life_days),
            stability_days: self.config.default_stability_days,
            reinforcements: 0,
            last_reviewed_at: entry.updated_at,
        });
        let half_life_days = state.half_life_days.max(0.1);
        let age_days = (now - state.last_reviewed_at).num_hours().max(0) as f64 / 24.0;
        let quality = entry.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
        let due_score = (age_days / half_life_days) as f32;
        if due_score < self.config.due_threshold && quality >= 0.35 {
            return None;
        }

        let (action, route, reason) = if due_score >= self.config.forget_threshold && quality < 0.30
        {
            (
                ReviewAction::ForgetCandidate,
                QualityRoute::ForgetCandidate,
                format!(
                    "overdue {:.2}x half-life with low quality {:.2}",
                    due_score, quality
                ),
            )
        } else if due_score >= self.config.revalidate_threshold || quality < 0.45 {
            (
                ReviewAction::Revalidate,
                QualityRoute::Revalidate,
                format!(
                    "due {:.2}x half-life or quality {:.2} requires revalidation",
                    due_score, quality
                ),
            )
        } else {
            (
                ReviewAction::Rehearse,
                QualityRoute::Rehearse,
                format!("approaching half-life at {:.2}x", due_score),
            )
        };

        Some(ReviewTask {
            uri: entry.uri.clone(),
            content_type: entry.content_type(),
            action,
            route,
            due_score,
            age_days,
            half_life_days,
            stability_days: state.stability_days.max(0.1),
            reinforcements: state.reinforcements,
            reason,
        })
    }
}

pub fn review_state(entry: &ContextEntry) -> Option<ReviewMemoryState> {
    entry.metadata.custom_field("spaced_repetition")
}

pub fn upsert_review_state(entry: &mut ContextEntry, state: ReviewMemoryState) {
    let _ = entry.metadata.set_custom_field("spaced_repetition", &state);
    let meta = entry
        .metadata
        .consolidation
        .get_or_insert_with(|| ConsolidationMeta {
            source: "spaced-repetition".to_string(),
            generation: 0,
            status: ConsolidationStatus::InProgress,
            patch_count: 0,
            lineage: vec![],
            evidence_uris: vec![],
            corroboration: 0,
            half_life_days: Some(state.half_life_days),
            entangled_with: vec![],
        });
    meta.half_life_days = Some(state.half_life_days);
    meta.lineage.push(LineageEntry {
        version: MvccVersion(0),
        timestamp: state.last_reviewed_at,
        change_summary: format!(
            "spaced repetition reviewed: stability {:.2}d, reinforcements {}",
            state.stability_days, state.reinforcements
        ),
    });
}

// ===========================================================================
// Anki SM-2 风格的 retrieval-induced revival（强化学习式）
// ===========================================================================

/// 检索命中且被采纳时，重置 stability 并延长有效期。
///
/// SM-2 风格: stability_new = stability * (1 + reinforcements * 0.5)
pub fn reinforce_on_adoption(current_stability: f64, reinforcements: u32) -> (f64, u32) {
    let safe_stability = current_stability.max(0.1);
    let new_stability = safe_stability * (1.0 + reinforcements as f64 * 0.5).max(1.15);
    (new_stability, reinforcements.saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContextEntry, TenantId};
    use chrono::Duration;
    use uuid::Uuid;

    fn long_entry(age_days: i64, half_life_days: f64, quality: f32) -> ContextEntry {
        let now = Utc::now();
        let mut entry = ContextEntry::new_text(
            ContextUri::parse("uwu://t/a/memory/fact/review-target").unwrap(),
            TenantId(Uuid::nil()),
            "stable fact",
        );
        entry.metadata.content_type = Some(ContentType::Fact);
        entry.metadata.state_scope = Some(StateScope::Long);
        entry.metadata.quality_score = Some(quality);
        entry.updated_at = now - Duration::days(age_days);
        upsert_review_state(
            &mut entry,
            ReviewMemoryState {
                half_life_days,
                stability_days: half_life_days,
                reinforcements: 1,
                last_reviewed_at: now - Duration::days(age_days),
            },
        );
        entry
    }

    #[test]
    fn scheduler_orders_due_reviews_and_routes_revalidate() {
        let now = Utc::now();
        let scheduler = SpacedRepetitionScheduler::default();
        let entries = vec![long_entry(45, 30.0, 0.7), long_entry(5, 30.0, 0.8)];

        let tasks = scheduler.plan(&entries, now);

        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].action, ReviewAction::Revalidate);
        assert_eq!(tasks[0].route, QualityRoute::Revalidate);
        assert!(tasks[0].due_score > 1.0);
    }

    #[test]
    fn adoption_extends_stability_and_records_state() {
        let now = Utc::now();
        let scheduler = SpacedRepetitionScheduler::default();
        let mut entry = long_entry(29, 30.0, 0.8);
        let task = scheduler.plan(&[entry.clone()], now).remove(0);

        let outcome = scheduler.apply_adoption(&mut entry, &task, now);
        let state = review_state(&entry).unwrap();

        assert_eq!(outcome.route, QualityRoute::Rehearse);
        assert_eq!(state.reinforcements, 2);
        assert!(state.stability_days > task.stability_days);
        assert!(
            entry
                .metadata
                .tags
                .contains(&"quality:review:adopted".to_string())
        );
    }

    #[test]
    fn low_quality_overdue_memory_becomes_forget_candidate() {
        let now = Utc::now();
        let scheduler = SpacedRepetitionScheduler::default();
        let entries = vec![long_entry(90, 30.0, 0.2)];

        let tasks = scheduler.plan(&entries, now);

        assert_eq!(tasks[0].action, ReviewAction::ForgetCandidate);
        assert_eq!(tasks[0].route, QualityRoute::ForgetCandidate);
    }
}
