//! 树搜索 — AlphaZero/LATS 风格的 policy/value 分离搜索。

use crate::policy_value::{
    ActionCandidate, CognitiveState, CognitiveValue, PolicyModule, ValueModule,
};
use crate::{CognitiveDelta, CognitivePreferencePair, PreferenceSource};
use agent_context_db_core::{LlmClient, LlmOpts};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// 搜索树节点。
#[derive(Debug, Clone)]
pub struct SearchNode {
    pub state: CognitiveState,
    pub action: Option<ActionCandidate>,
    pub value: CognitiveValue,
    pub children: Vec<SearchNode>,
    pub depth: usize,
    /// MCTS visit count.
    pub visits: usize,
    /// Accumulated value from simulations.
    pub value_sum: f32,
    /// Policy prior assigned when this node was expanded.
    pub prior: f32,
}

impl SearchNode {
    pub fn mean_value(&self) -> f32 {
        if self.visits == 0 {
            self.value.composite
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

/// 搜索树。
pub struct SearchTree {
    pub root: SearchNode,
    pub best_path: Vec<SearchNode>,
    pub worst_path: Vec<SearchNode>,
}

/// 认知树搜索 — 生成多条候选 trajectory，选最优/最差作为偏好对。
pub struct CognitiveTreeSearch {
    policy: PolicyModule,
    value: ValueModule,
    llm: Option<Arc<dyn LlmClient>>,
    branching_factor: usize,
    max_depth: usize,
    simulations: usize,
    exploration_c: f32,
    rollout_depth: usize,
}

impl CognitiveTreeSearch {
    pub fn new(branching_factor: usize, max_depth: usize) -> Self {
        Self {
            policy: PolicyModule::new(),
            value: ValueModule::new(),
            llm: None,
            branching_factor: branching_factor.max(1),
            max_depth: max_depth.max(1),
            simulations: 32,
            exploration_c: 1.4,
            rollout_depth: 2,
        }
    }

    pub fn with_simulations(mut self, simulations: usize) -> Self {
        self.simulations = simulations.max(1);
        self
    }

    pub fn with_exploration(mut self, exploration_c: f32) -> Self {
        self.exploration_c = exploration_c.max(0.01);
        self
    }

    pub fn with_rollout_depth(mut self, rollout_depth: usize) -> Self {
        self.rollout_depth = rollout_depth;
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 搜索 — LLM/policy expansion + UCT selection + value rollout + backup。
    pub async fn search(&self, initial: &CognitiveState) -> SearchTree {
        let mut root = SearchNode {
            state: initial.clone(),
            action: None,
            value: self.value.evaluate(initial),
            children: vec![],
            depth: 0,
            visits: 0,
            value_sum: 0.0,
            prior: 1.0,
        };

        for _ in 0..self.simulations {
            self.simulate(&mut root).await;
        }

        let best_path = self.path_by(
            |nodes| {
                nodes
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        score_for_path(a)
                            .partial_cmp(&score_for_path(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
            },
            &root,
        );
        let worst_path = self.path_by(
            |nodes| {
                nodes
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        score_for_path(a)
                            .partial_cmp(&score_for_path(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
            },
            &root,
        );

        SearchTree {
            root,
            best_path,
            worst_path,
        }
    }

    fn simulate<'a>(
        &'a self,
        node: &'a mut SearchNode,
    ) -> Pin<Box<dyn Future<Output = f32> + Send + 'a>> {
        Box::pin(async move {
            let value = if node.depth >= self.max_depth {
                self.value.evaluate(&node.state).composite
            } else {
                if node.children.is_empty() {
                    self.expand(node).await;
                    let rollout = self.rollout_value(&node.state).await;
                    node.visits += 1;
                    node.value_sum += rollout;
                    return rollout;
                }

                let Some(idx) = self.select_child(node) else {
                    return self.value.evaluate(&node.state).composite;
                };
                let Some(child) = node.children.get_mut(idx) else {
                    return self.value.evaluate(&node.state).composite;
                };
                self.simulate(child).await
            };

            node.visits += 1;
            node.value_sum += value;
            value
        })
    }

    async fn expand(&self, node: &mut SearchNode) {
        if node.depth >= self.max_depth {
            return;
        }
        let candidates = self.expand_actions(&node.state, node.depth).await;
        node.children = candidates
            .into_iter()
            .take(self.branching_factor)
            .map(|action| {
                let state = self.policy.predict(&node.state, &action);
                let value = self.value.evaluate(&state);
                SearchNode {
                    state,
                    value,
                    prior: action.prior,
                    action: Some(action),
                    children: vec![],
                    depth: node.depth + 1,
                    visits: 0,
                    value_sum: 0.0,
                }
            })
            .collect();
    }

    async fn expand_actions(&self, state: &CognitiveState, depth: usize) -> Vec<ActionCandidate> {
        if let Some(llm) = &self.llm
            && let Ok(actions) = llm_expand_actions(llm.as_ref(), state, depth).await
            && !actions.is_empty()
        {
            return self.policy.prioritize(actions);
        }
        self.policy.generate(state)
    }

    async fn rollout_value(&self, state: &CognitiveState) -> f32 {
        let mut current = state.clone();
        let mut total = self.value.evaluate(&current).composite;
        let mut weight = 1.0;
        let mut norm = 1.0;

        for depth in 0..self.rollout_depth {
            let actions = self.expand_actions(&current, depth).await;
            let Some(best) = actions.into_iter().max_by(|a, b| {
                a.prior
                    .partial_cmp(&b.prior)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                break;
            };
            current = self.policy.predict(&current, &best);
            weight *= 0.72;
            total += self.value.evaluate(&current).composite * weight;
            norm += weight;
        }

        (total / norm).clamp(0.0, 1.0)
    }

    fn select_child(&self, node: &SearchNode) -> Option<usize> {
        let parent_visits = node.visits.max(1) as f32;
        node.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let sa = puct_score(a, parent_visits, self.exploration_c);
                let sb = puct_score(b, parent_visits, self.exploration_c);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    fn path_by<F>(&self, choose: F, root: &SearchNode) -> Vec<SearchNode>
    where
        F: Fn(&[SearchNode]) -> Option<usize>,
    {
        let mut path = vec![root.clone()];
        let mut node = root;
        while !node.children.is_empty() {
            let Some(idx) = choose(&node.children) else {
                break;
            };
            let Some(child) = node.children.get(idx) else {
                break;
            };
            node = child;
            path.push(node.clone());
        }
        path
    }

    /// 从搜索树提取偏好对（最优路径 vs 最差路径）。
    pub fn extract_pair(tree: &SearchTree) -> Option<CognitivePreferencePair> {
        let best = tree.best_path.last().or_else(|| best_leaf(&tree.root))?;
        let worst = tree.worst_path.last().or_else(|| worst_leaf(&tree.root))?;
        if best.depth == worst.depth
            && (best.mean_value() - worst.mean_value()).abs() < f32::EPSILON
        {
            return None;
        }

        let best_desc = describe_path(&tree.best_path, "optimal reasoning path");
        let worst_desc = describe_path(&tree.worst_path, "suboptimal reasoning path");
        let confidence = path_value(&tree.best_path) - path_value(&tree.worst_path);

        Some(CognitivePreferencePair {
            chosen: crate::TrajectorySummary {
                task_id: "lats".into(),
                task_description: best_desc,
                success: false,
                steps: tree.best_path.len().saturating_sub(1).max(best.depth),
                contradictions: best.state.recent_errors.len(),
                avg_confidence: path_value(&tree.best_path),
            },
            rejected: crate::TrajectorySummary {
                task_id: "lats".into(),
                task_description: worst_desc,
                success: false,
                steps: tree.worst_path.len().saturating_sub(1).max(worst.depth),
                contradictions: worst.state.recent_errors.len().max(1),
                avg_confidence: path_value(&tree.worst_path),
            },
            preference_source: PreferenceSource::Simulation,
            confidence: confidence.abs().clamp(0.0, 1.0),
            cognitive_delta: CognitiveDelta {
                contradiction_diff: worst.state.recent_errors.len() as i32
                    - best.state.recent_errors.len() as i32,
                confidence_diff: confidence,
                evidence_diff: best.state.active_hypotheses.len() as i32
                    - worst.state.active_hypotheses.len() as i32,
                knowledge_graph_growth: ((best.state.graph_density - worst.state.graph_density)
                    * 100.0) as i32,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct LlmActionCandidate {
    description: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    prior: f32,
    #[serde(default)]
    expected_effects: Vec<String>,
}

async fn llm_expand_actions(
    llm: &dyn LlmClient,
    state: &CognitiveState,
    depth: usize,
) -> agent_context_db_core::Result<Vec<ActionCandidate>> {
    let prompt = format!(
        r#"Generate candidate next reasoning actions for Language Agent Tree Search.
Return JSON only, as an array:
[
  {{"description":"...", "confidence":0.0, "prior":0.0, "expected_effects":["reduce errors", "increase evidence"]}}
]

State:
- depth: {depth}
- graph_density: {graph_density:.3}
- recent_errors: {errors}
- active_hypotheses: {hypotheses}
- avg_confidence: {confidence:.3}

Prefer actions that concretely reduce contradictions, validate hypotheses, improve evidence coverage, or link knowledge nodes."#,
        graph_density = state.graph_density,
        errors = state.recent_errors.len(),
        hypotheses = state.active_hypotheses.len(),
        confidence = state.avg_confidence,
    );
    let response = llm.complete(&prompt, &LlmOpts::default()).await?;
    let json = extract_json_array(&response)?;
    let raw = serde_json::from_str::<Vec<LlmActionCandidate>>(json)?;
    Ok(raw
        .into_iter()
        .filter(|action| !action.description.trim().is_empty())
        .map(|action| {
            ActionCandidate::new(
                action.description,
                if action.confidence > 0.0 {
                    action.confidence
                } else {
                    0.5
                },
                if action.prior > 0.0 {
                    action.prior
                } else {
                    0.25
                },
                normalize_effects(action.expected_effects),
            )
        })
        .collect())
}

fn normalize_effects(effects: Vec<String>) -> Vec<String> {
    if effects.is_empty() {
        return vec!["increase evidence".into()];
    }
    effects
}

fn extract_json_array(text: &str) -> agent_context_db_core::Result<&str> {
    let trimmed = text.trim();
    if let Some(after) = trimmed.strip_prefix("```json")
        && let Some((json, _)) = after.split_once("```")
    {
        return Ok(json.trim());
    }
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return Ok(trimmed);
    }
    Err(agent_context_db_core::ContextError::Llm(
        agent_context_db_core::LlmError::Provider(
            "LATS action response must be a JSON array".into(),
        ),
    ))
}

fn puct_score(node: &SearchNode, parent_visits: f32, exploration_c: f32) -> f32 {
    let q = node.mean_value();
    let u = exploration_c * node.prior * parent_visits.sqrt() / (1.0 + node.visits as f32);
    q + u
}

fn score_for_path(node: &SearchNode) -> f32 {
    node.mean_value() + node.visits as f32 * 0.001 + node.depth as f32 * 0.0005
}

fn path_value(path: &[SearchNode]) -> f32 {
    if path.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut weight = 1.0;
    let mut norm = 0.0;
    for node in path.iter().skip(1) {
        total += node.mean_value() * weight;
        norm += weight;
        weight *= 0.85;
    }
    if norm <= f32::EPSILON {
        path.last().map(SearchNode::mean_value).unwrap_or(0.0)
    } else {
        (total / norm).clamp(0.0, 1.0)
    }
}

fn describe_path(path: &[SearchNode], fallback: &str) -> String {
    let steps = path
        .iter()
        .filter_map(|node| {
            node.action
                .as_ref()
                .map(|action| action.description.clone())
        })
        .collect::<Vec<_>>();
    if steps.is_empty() {
        fallback.into()
    } else {
        steps.join(" -> ")
    }
}

fn best_leaf(root: &SearchNode) -> Option<&SearchNode> {
    if root.children.is_empty() {
        return Some(root);
    }
    root.children.iter().filter_map(best_leaf).max_by(|a, b| {
        score_for_path(a)
            .partial_cmp(&score_for_path(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn worst_leaf(root: &SearchNode) -> Option<&SearchNode> {
    if root.children.is_empty() {
        return Some(root);
    }
    root.children.iter().filter_map(worst_leaf).min_by(|a, b| {
        score_for_path(a)
            .partial_cmp(&score_for_path(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{JsonSchema, LlmError};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ExpandingLlm {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for ExpandingLlm {
        async fn complete(
            &self,
            _prompt: &str,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(r#"[
                {"description":"verify contradiction with retrieved evidence","confidence":0.91,"prior":0.7,"expected_effects":["reduce errors","increase evidence"]},
                {"description":"speculate without evidence","confidence":0.25,"prior":0.3,"expected_effects":[]}
            ]"#
                .into())
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            self.complete(_prompt, _opts).await
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

    fn uri(s: &str) -> agent_context_db_core::ContextUri {
        agent_context_db_core::ContextUri::parse(s).unwrap()
    }

    #[tokio::test]
    async fn mcts_expands_root_and_extracts_pair() {
        let state = CognitiveState {
            graph_density: 0.2,
            recent_errors: vec![uri("uwu://t/agent/a/memories/error/e1")],
            active_hypotheses: vec![uri("uwu://t/agent/a/memories/hypothesis/h1")],
            avg_confidence: 0.4,
        };
        let search = CognitiveTreeSearch::new(3, 3).with_simulations(24);
        let tree = search.search(&state).await;
        assert!(tree.root.visits > 0);
        assert!(!tree.root.children.is_empty());
        assert!(!tree.best_path.is_empty());
        assert!(!tree.worst_path.is_empty());
        assert!(tree.best_path.len() > 1);
        assert!(tree.root.children.iter().any(|child| child.visits > 0));
        let pair = CognitiveTreeSearch::extract_pair(&tree).unwrap();
        assert!(pair.confidence >= 0.0);
        assert!(pair.chosen.task_description.contains(" -> ") || pair.chosen.steps > 1);
    }

    #[tokio::test]
    async fn lats_uses_llm_expansion_and_value_backup() {
        let state = CognitiveState {
            graph_density: 0.2,
            recent_errors: vec![uri("uwu://t/agent/a/memories/error/e1")],
            active_hypotheses: vec![uri("uwu://t/agent/a/memories/hypothesis/h1")],
            avg_confidence: 0.35,
        };
        let llm = Arc::new(ExpandingLlm {
            calls: AtomicUsize::new(0),
        });
        let search = CognitiveTreeSearch::new(2, 3)
            .with_simulations(16)
            .with_rollout_depth(2)
            .with_llm(llm.clone());
        let tree = search.search(&state).await;

        assert!(llm.calls.load(Ordering::SeqCst) > 0);
        assert!(tree.root.children.iter().any(|child| {
            child
                .action
                .as_ref()
                .unwrap()
                .description
                .contains("retrieved evidence")
        }));
        assert!(tree.root.visits > 0);
        let projected = &tree.best_path.last().unwrap().state;
        assert_eq!(projected.avg_confidence, state.avg_confidence);
        assert_eq!(projected.recent_errors, state.recent_errors);
        assert_eq!(projected.active_hypotheses, state.active_hypotheses);
    }
}
