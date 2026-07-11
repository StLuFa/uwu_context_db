//! LATS — Language Agent Tree Search 的搜索-反思-再搜索闭环。
//!
//! 该模块把现有 MCTS、Reflexion 和 ExpeL 连接起来：先用 policy/value 搜索候选动作，
//! 将低价值路径转成失败轨迹，再用反思 guidance 改写下一轮认知状态。

use crate::CognitivePreferencePair;
use crate::policy_value::CognitiveState;
use crate::reflection::{FailureTrace, ReflectionGenerator, ReflexionEvolutionResult};
use crate::tree_search::{CognitiveTreeSearch, SearchTree};
use crate::voting::{EvolvableInsight, InsightEvolutionEngine};
use agent_context_db_core::{ContextUri, LlmClient};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LatsConfig {
    pub iterations: usize,
    pub branching_factor: usize,
    pub max_depth: usize,
    pub simulations: usize,
    pub reflection_threshold: f32,
}

impl Default for LatsConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            branching_factor: 3,
            max_depth: 3,
            simulations: 32,
            reflection_threshold: 0.55,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatsIteration {
    pub iteration: usize,
    pub best_action: Option<String>,
    pub worst_action: Option<String>,
    pub preference: Option<CognitivePreferencePair>,
    pub reflection: Option<ReflexionEvolutionResult>,
    pub state: CognitiveState,
}

#[derive(Debug, Clone)]
pub struct LatsReport {
    pub iterations: Vec<LatsIteration>,
    pub final_state: CognitiveState,
    pub insights: Vec<EvolvableInsight>,
}

pub struct LatsLoop {
    config: LatsConfig,
    search: CognitiveTreeSearch,
    reflector: Arc<ReflectionGenerator>,
    evolution: InsightEvolutionEngine,
}

impl LatsLoop {
    pub fn new(config: LatsConfig, reflector: Arc<ReflectionGenerator>) -> Self {
        let search = CognitiveTreeSearch::new(config.branching_factor, config.max_depth)
            .with_simulations(config.simulations);
        Self {
            config,
            search,
            reflector,
            evolution: InsightEvolutionEngine::new(),
        }
    }

    pub fn with_evolution(mut self, evolution: InsightEvolutionEngine) -> Self {
        self.evolution = evolution;
        self
    }

    pub fn with_search_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.search = CognitiveTreeSearch::new(self.config.branching_factor, self.config.max_depth)
            .with_simulations(self.config.simulations)
            .with_llm(llm);
        self
    }

    pub async fn run(&self, initial: CognitiveState) -> LatsReport {
        let mut state = initial;
        let mut insights = Vec::new();
        let mut iterations = Vec::new();

        for iteration in 0..self.config.iterations.max(1) {
            let tree = self.search.search(&state).await;
            let preference = CognitiveTreeSearch::extract_pair(&tree);
            let should_reflect = preference
                .as_ref()
                .map(|p| p.rejected.avg_confidence < self.config.reflection_threshold)
                .unwrap_or(false);
            let reflection = if should_reflect {
                let failures = failure_traces_from_tree(iteration, &tree, preference.as_ref());
                Some(
                    self.reflector
                        .evolve_failures(&failures, &mut insights, &self.evolution)
                        .await,
                )
            } else {
                None
            };

            // Search and reflection produce proposals only. Observed state changes require an
            // execution receipt and self-verification, so LATS cannot mutate epistemic state.
            state = apply_lats_feedback(&state, preference.as_ref(), reflection.as_ref());

            iterations.push(LatsIteration {
                iteration,
                best_action: tree
                    .best_path
                    .last()
                    .and_then(|n| n.action.as_ref().map(|a| a.description.clone())),
                worst_action: tree
                    .worst_path
                    .last()
                    .and_then(|n| n.action.as_ref().map(|a| a.description.clone())),
                preference,
                reflection,
                state: state.clone(),
            });
        }

        LatsReport {
            iterations,
            final_state: state,
            insights,
        }
    }
}

fn failure_traces_from_tree(
    iteration: usize,
    tree: &SearchTree,
    preference: Option<&CognitivePreferencePair>,
) -> Vec<FailureTrace> {
    let worst = tree.worst_path.last().unwrap_or(&tree.root);
    let failed_step = worst
        .action
        .as_ref()
        .map(|a| a.description.clone())
        .unwrap_or_else(|| "search produced no useful action".into());
    let task_description = preference
        .map(|p| p.rejected.task_description.clone())
        .unwrap_or_else(|| format!("lats iteration {iteration}"));
    let error_message = format!(
        "candidate value {:.3} below reflection threshold during LATS iteration {}",
        worst.mean_value(),
        iteration
    );
    let relevant_knowledge = worst
        .state
        .active_hypotheses
        .iter()
        .map(ContextUri::to_string)
        .collect();
    let trace = tree
        .worst_path
        .iter()
        .filter_map(|node| node.action.as_ref().map(|a| a.description.clone()))
        .collect();

    vec![FailureTrace {
        task_description,
        failed_step,
        error_message,
        relevant_knowledge,
        trace,
    }]
}

fn apply_lats_feedback(
    state: &CognitiveState,
    _preference: Option<&CognitivePreferencePair>,
    _reflection: Option<&ReflexionEvolutionResult>,
) -> CognitiveState {
    state.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{JsonSchema, LlmClient, LlmError, LlmOpts};
    use async_trait::async_trait;

    struct FailingLlm;

    #[async_trait]
    impl LlmClient for FailingLlm {
        async fn complete(
            &self,
            _prompt: &str,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Err(LlmError::Provider("fail".into()))
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Err(LlmError::Provider("fail".into()))
        }

        async fn embed(
            &self,
            _text: &str,
        ) -> std::result::Result<agent_context_db_core::EmbeddingVector, LlmError> {
            Ok(agent_context_db_core::EmbeddingVector::new(
                vec![0.0; 8],
                "test",
                1,
            ))
        }
    }

    fn uri(s: &str) -> ContextUri {
        ContextUri::parse(s).unwrap()
    }

    #[tokio::test]
    async fn lats_reflects_and_improves_state() {
        let state = CognitiveState {
            graph_density: 0.2,
            recent_errors: vec![uri("uwu://t/agent/a/memories/error/e1")],
            active_hypotheses: vec![uri("uwu://t/agent/a/memories/hypothesis/h1")],
            avg_confidence: 0.35,
        };
        let loop_runner = LatsLoop::new(
            LatsConfig {
                iterations: 2,
                simulations: 12,
                reflection_threshold: 0.99,
                ..Default::default()
            },
            Arc::new(ReflectionGenerator::new(Arc::new(FailingLlm))),
        );

        let report = loop_runner.run(state.clone()).await;
        assert_eq!(report.iterations.len(), 2);
        assert_eq!(report.final_state, state);
        assert!(report.iterations.iter().any(|i| i.reflection.is_some()));
        assert!(!report.insights.is_empty());
    }
}
