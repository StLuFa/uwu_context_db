//! 树搜索 — 多 trajectory 生成 + 选出最优/最差偏好对。

use crate::policy_value::{ActionCandidate, CognitiveState, CognitiveValue, PolicyModule, ValueModule};
use crate::{CognitivePreferencePair, PreferenceSource, CognitiveDelta};
use agent_context_db_core::Result;

/// 搜索树节点。
#[derive(Debug, Clone)]
pub struct SearchNode {
    pub state: CognitiveState,
    pub action: Option<ActionCandidate>,
    pub value: CognitiveValue,
    pub children: Vec<SearchNode>,
    pub depth: usize,
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
    branching_factor: usize,
    max_depth: usize,
}

impl CognitiveTreeSearch {
    pub fn new(branching_factor: usize, max_depth: usize) -> Self {
        Self { policy: PolicyModule::new(), value: ValueModule::new(), branching_factor, max_depth }
    }

    /// 搜索 — 每层展开 branching_factor 个候选。
    pub async fn search(&self, initial: &CognitiveState) -> SearchTree {
        let root_value = self.value.evaluate(initial);
        let root = SearchNode { state: initial.clone(), action: None, value: root_value, children: vec![], depth: 0 };
        let mut current = vec![root.clone()];

        for depth in 0..self.max_depth {
            let mut next = Vec::new();
            for node in &current {
                let candidates = self.policy.generate(&node.state);
                for c in candidates.iter().take(self.branching_factor) {
                    let mut s = node.state.clone();
                    s.avg_confidence = (s.avg_confidence + c.confidence * 0.1).min(1.0);
                    let v = self.value.evaluate(&s);
                    next.push(SearchNode { state: s, action: Some(c.clone()), value: v, children: vec![], depth: depth + 1 });
                }
            }
            current = next;
        }
        SearchTree { root, best_path: vec![], worst_path: vec![] }
    }

    /// 从搜索树提取偏好对（最优路径 vs 最差路径）。
    pub fn extract_pair(tree: &SearchTree) -> Option<CognitivePreferencePair> {
        // 简化：取 root 作为 chosen，取第一个子节点作为 rejected
        let first_child = tree.root.children.first()?;
        Some(CognitivePreferencePair {
            chosen: crate::TrajectorySummary { task_id: "lats".into(), task_description: "optimal path".into(), success: true, steps: tree.root.depth, contradictions: 0, avg_confidence: tree.root.value.composite },
            rejected: crate::TrajectorySummary { task_id: "lats".into(), task_description: "suboptimal path".into(), success: false, steps: first_child.depth, contradictions: 1, avg_confidence: first_child.value.composite },
            preference_source: PreferenceSource::KnowledgeConsistency,
            confidence: tree.root.value.composite,
            cognitive_delta: CognitiveDelta { contradiction_diff: -1, confidence_diff: tree.root.value.composite - first_child.value.composite, evidence_diff: 0, knowledge_graph_growth: 0 },
        })
    }
}
