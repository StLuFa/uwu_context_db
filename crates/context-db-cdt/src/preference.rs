//! CognitivePreferenceExtractor — 三层偏好信号提取。

use crate::{CognitiveDelta, CognitivePreferencePair, PreferenceSource, TrajectorySummary};
use agent_context_db_core::{ContentType, ContextUri, Result};

/// 认知偏好提取器。
pub struct CognitivePreferenceExtractor;

impl CognitivePreferenceExtractor {
    pub fn new() -> Self {
        Self
    }

    /// 从成功/失败 trajectory 对比中提取偏好对。
    pub fn extract_pairs(
        summaries: &[(TrajectorySummary, bool)], // (trajectory, success)
    ) -> Vec<CognitivePreferencePair> {
        let mut pairs = Vec::new();
        let successes: Vec<_> = summaries.iter().filter(|(_, s)| *s).collect();
        let failures: Vec<_> = summaries.iter().filter(|(_, s)| !*s).collect();

        for (chosen, _) in &successes {
            for (rejected, _) in &failures {
                let delta = CognitiveDelta {
                    contradiction_diff: rejected.contradictions as i32
                        - chosen.contradictions as i32,
                    confidence_diff: chosen.avg_confidence - rejected.avg_confidence,
                    evidence_diff: 0,
                    knowledge_graph_growth: 0,
                };

                pairs.push(CognitivePreferencePair {
                    chosen: (*chosen).clone(),
                    rejected: (*rejected).clone(),
                    preference_source: PreferenceSource::TaskOutcome,
                    confidence: chosen.avg_confidence,
                    cognitive_delta: delta,
                });
            }
        }
        pairs
    }
}
