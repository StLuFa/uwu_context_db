//! CognitiveSelfPlay — 策略分离自我对弈训练。

use crate::CognitivePreferencePair;
use crate::policy_value::{CognitiveState, CognitiveValue, ValueModule};
use crate::tree_search::CognitiveTreeSearch;
use agent_context_db_core::LlmClient;
use std::sync::Arc;

/// 自我对弈结果。
#[derive(Debug, Clone)]
pub struct SelfPlayResult {
    pub preference_pair: CognitivePreferencePair,
    pub cognitive_value: CognitiveValue,
    pub state_before: CognitiveState,
    pub state_after: CognitiveState,
}

/// 认知自我对弈 — 生成"问题-尝试-评估"循环。
pub struct CognitiveSelfPlay {
    search: CognitiveTreeSearch,
    value: ValueModule,
}

impl CognitiveSelfPlay {
    pub fn new() -> Self {
        Self {
            search: CognitiveTreeSearch::new(4, 4).with_simulations(48),
            value: ValueModule::new(),
        }
    }

    pub fn with_search(mut self, search: CognitiveTreeSearch) -> Self {
        self.search = search;
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.search = self.search.with_llm(llm);
        self
    }

    /// 执行 N 轮 LATS 自我对弈，产生可喂给 DPO 的偏好对。
    pub async fn self_train(
        &self,
        rounds: usize,
        initial_state: &CognitiveState,
    ) -> Vec<SelfPlayResult> {
        let mut results = Vec::new();
        let mut state = initial_state.clone();

        for _round in 0..rounds {
            let state_before = state.clone();
            let tree = self.search.search(&state).await;
            let Some(preference_pair) = CognitiveTreeSearch::extract_pair(&tree) else {
                break;
            };
            let state_after = tree
                .best_path
                .last()
                .map(|node| node.state.clone())
                .unwrap_or_else(|| state.clone());
            let cognitive_value = self.value.evaluate(&state_after);

            results.push(SelfPlayResult {
                preference_pair,
                cognitive_value: cognitive_value.clone(),
                state_before,
                state_after: state_after.clone(),
            });

            if state_after == state {
                break;
            }
            state = state_after;
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::ContextUri;

    fn uri(s: &str) -> ContextUri {
        ContextUri::parse(s).unwrap()
    }

    #[tokio::test]
    async fn self_play_advances_state_with_best_action() {
        let state = CognitiveState {
            graph_density: 0.2,
            recent_errors: vec![uri("uwu://t/agent/a/memories/error/e1")],
            active_hypotheses: vec![uri("uwu://t/agent/a/memories/hypothesis/h1")],
            avg_confidence: 0.3,
        };
        let search = CognitiveTreeSearch::new(3, 3)
            .with_simulations(16)
            .with_rollout_depth(2);
        let play = CognitiveSelfPlay::new().with_search(search);
        let results = play.self_train(3, &state).await;
        assert_eq!(results.len(), 3);
        assert!(results[0].state_after.avg_confidence >= results[0].state_before.avg_confidence);
        assert!(results.last().unwrap().state_after.avg_confidence >= state.avg_confidence);
        assert!(
            results
                .iter()
                .all(|result| result.preference_pair.chosen.steps > 1)
        );
        assert!(results.iter().all(|result| {
            result.preference_pair.chosen.avg_confidence
                >= result.preference_pair.rejected.avg_confidence
        }));
    }
}
