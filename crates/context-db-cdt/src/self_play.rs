//! CognitiveSelfPlay — 策略分离自我对弈训练。

use crate::policy_value::{CognitiveState, CognitiveValue, PolicyModule, ValueModule};
use crate::{CognitiveDelta, CognitivePreferencePair, PreferenceSource};
use agent_context_db_core::{ContextUri, Result};

/// 自我对弈结果。
#[derive(Debug, Clone)]
pub struct SelfPlayResult {
    pub preference_pair: CognitivePreferencePair,
    pub cognitive_value: CognitiveValue,
}

/// 认知自我对弈 — 生成"问题-尝试-评估"循环。
pub struct CognitiveSelfPlay {
    policy: PolicyModule,
    value: ValueModule,
}

impl CognitiveSelfPlay {
    pub fn new() -> Self {
        Self {
            policy: PolicyModule::new(),
            value: ValueModule::new(),
        }
    }

    /// 执行 N 轮自我对弈，产生偏好对。
    pub async fn self_train(
        &self,
        rounds: usize,
        initial_state: &CognitiveState,
    ) -> Vec<SelfPlayResult> {
        let mut results = Vec::new();
        let mut state = initial_state.clone();

        for _ in 0..rounds {
            // Policy 生成候选动作
            let candidates = self.policy.generate(&state);

            // 模拟执行并评估（简化：用 confidence 估计效果）
            let mut valued: Vec<_> = candidates
                .iter()
                .map(|c| {
                    let mut s = state.clone();
                    s.avg_confidence = (s.avg_confidence + c.confidence * 0.1).min(1.0);
                    (c, self.value.evaluate(&s))
                })
                .collect();

            // 认知胜负判定：composite 最高 = chosen，最低 = rejected
            valued.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
            let (best_c, best_v) = &valued[0];
            let (worst_c, worst_v) = &valued[valued.len() - 1];

            results.push(SelfPlayResult {
                preference_pair: CognitivePreferencePair {
                    chosen: crate::TrajectorySummary {
                        task_id: "self-play".into(),
                        task_description: best_c.description.clone(),
                        success: true,
                        steps: 1,
                        contradictions: 0,
                        avg_confidence: best_c.confidence,
                    },
                    rejected: crate::TrajectorySummary {
                        task_id: "self-play".into(),
                        task_description: worst_c.description.clone(),
                        success: false,
                        steps: 2,
                        contradictions: 1,
                        avg_confidence: worst_c.confidence,
                    },
                    preference_source: PreferenceSource::KnowledgeConsistency,
                    confidence: best_v.composite,
                    cognitive_delta: CognitiveDelta {
                        contradiction_diff: -1,
                        confidence_diff: best_v.composite - worst_v.composite,
                        evidence_diff: 0,
                        knowledge_graph_growth: 0,
                    },
                },
                cognitive_value: best_v.clone(),
            });

            state = state.clone(); // 保持状态简化
        }
        results
    }
}
