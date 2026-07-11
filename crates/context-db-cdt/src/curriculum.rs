//! CurriculumGenerator — 主动课程：知识图谱拓扑前沿 + ZPD 排序。

use crate::TrainingGoal;
use agent_context_db_core::{ContentType, ContextUri, GraphRelation, GraphStore, Result};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// 课程生成器。
pub struct CurriculumGenerator {
    pub exploration_ratio: f32,
    pub zpd_difficulty: f32,
    graph: Option<Arc<dyn GraphStore>>,
    bootstrap_frontier: Vec<FrontierNode>,
}

/// 知识图谱前沿节点。
#[derive(Debug, Clone)]
pub struct FrontierNode {
    pub uri: ContextUri,
    pub difficulty: f32,
    pub prerequisite_count: usize,
    pub expected_knowledge: String,
    pub content_type: Option<ContentType>,
    pub zpd_score: f32,
}

impl CurriculumGenerator {
    pub fn new(exploration_ratio: f32) -> Self {
        Self {
            exploration_ratio,
            zpd_difficulty: 0.6,
            graph: None,
            bootstrap_frontier: Vec::new(),
        }
    }

    pub fn with_graph(mut self, graph: Arc<dyn GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    pub fn with_zpd_difficulty(mut self, difficulty: f32) -> Self {
        self.zpd_difficulty = difficulty.clamp(0.0, 1.0);
        self
    }

    pub fn with_bootstrap_frontier(mut self, frontier: Vec<FrontierNode>) -> Self {
        self.bootstrap_frontier = self.sort_frontier(frontier);
        self
    }

    pub fn with_bootstrap_targets(mut self, targets: Vec<ContextUri>) -> Self {
        self.bootstrap_frontier = self.sort_frontier(
            targets
                .into_iter()
                .enumerate()
                .map(|(idx, uri)| {
                    let difficulty = (0.35 + idx as f32 * 0.08).clamp(0.05, 0.95);
                    FrontierNode {
                        expected_knowledge: expected_knowledge_for(&uri, 0, difficulty),
                        content_type: infer_content_type(&uri),
                        zpd_score: self.zpd_score(difficulty),
                        uri,
                        difficulty,
                        prerequisite_count: 0,
                    }
                })
                .collect(),
        );
        self
    }

    /// 生成下一个训练目标。
    pub async fn next_goal(&self, known_uris: &[ContextUri]) -> Result<TrainingGoal> {
        if !known_uris.is_empty() {
            let confidence: HashMap<ContextUri, f32> =
                known_uris.iter().cloned().map(|uri| (uri, 0.85)).collect();
            let frontier = self.find_graph_frontier(&confidence).await?;
            if let Some(node) = frontier.into_iter().next() {
                let prerequisites = self.prerequisites_for(&node.uri, known_uris).await?;
                return Ok(TrainingGoal {
                    target_node: node.uri,
                    difficulty: node.difficulty,
                    prerequisite_skills: prerequisites,
                    expected_new_knowledge: node.expected_knowledge,
                });
            }
        }

        if let Some(node) = self.bootstrap_frontier.first() {
            return Ok(TrainingGoal {
                target_node: node.uri.clone(),
                difficulty: node.difficulty,
                prerequisite_skills: vec![],
                expected_new_knowledge: node.expected_knowledge.clone(),
            });
        }

        Err(agent_context_db_core::ContextError::Unsupported(
            "no curriculum frontier available; provide known_uris, graph neighbors, or bootstrap targets".into(),
        ))
    }

    /// Validate legacy known-node input without pretending known nodes are unknown frontier nodes.
    /// A real frontier requires graph neighbors or explicit bootstrap targets.
    pub fn find_frontier(&self, known: &HashMap<String, f32>) -> Result<Vec<FrontierNode>> {
        for uri in known.keys() {
            ContextUri::parse(uri)?;
        }
        Ok(Vec::new())
    }

    /// 用 GraphStore 做真实 frontier 扩展：从已掌握节点出发，找未学邻居。
    pub async fn find_graph_frontier(
        &self,
        known: &HashMap<ContextUri, f32>,
    ) -> Result<Vec<FrontierNode>> {
        let Some(graph) = &self.graph else {
            let fallback: HashMap<String, f32> = known
                .iter()
                .map(|(uri, confidence)| (uri.as_str().to_string(), *confidence))
                .collect();
            return self.find_frontier(&fallback);
        };

        let known_set: HashSet<&str> = known.keys().map(|uri| uri.as_str()).collect();
        let mut candidate_prereqs: HashMap<ContextUri, usize> = HashMap::new();
        let mut best_confidence: HashMap<ContextUri, f32> = HashMap::new();

        for (uri, confidence) in known {
            if *confidence < 0.7 {
                continue;
            }
            let neighbors = graph.neighbors(uri, None).await?;
            for neighbor in neighbors {
                if known_set.contains(neighbor.as_str()) {
                    continue;
                }
                *candidate_prereqs.entry(neighbor.clone()).or_insert(0) += 1;
                best_confidence
                    .entry(neighbor)
                    .and_modify(|c| *c = c.max(*confidence))
                    .or_insert(*confidence);
            }
        }

        let mut frontier = Vec::new();
        for (uri, prerequisite_count) in candidate_prereqs {
            let centrality = graph.centrality(&uri).await?.clamp(0.0, 1.0);
            let base_confidence = best_confidence.get(&uri).copied().unwrap_or(0.75);
            let difficulty = estimate_difficulty(base_confidence, prerequisite_count, centrality);
            frontier.push(FrontierNode {
                expected_knowledge: expected_knowledge_for(&uri, prerequisite_count, difficulty),
                content_type: infer_content_type(&uri),
                zpd_score: self.zpd_score(difficulty),
                uri,
                difficulty,
                prerequisite_count,
            });
        }

        Ok(self.sort_frontier(frontier))
    }

    pub fn zpd_score(&self, difficulty: f32) -> f32 {
        let distance = (difficulty - self.zpd_difficulty).abs();
        (1.0 - distance / 1.0).clamp(0.0, 1.0)
    }

    fn sort_frontier(&self, mut frontier: Vec<FrontierNode>) -> Vec<FrontierNode> {
        frontier.sort_by(|a, b| {
            b.zpd_score
                .partial_cmp(&a.zpd_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.prerequisite_count.cmp(&a.prerequisite_count))
        });
        frontier
    }

    async fn prerequisites_for(
        &self,
        target: &ContextUri,
        known_uris: &[ContextUri],
    ) -> Result<Vec<ContextUri>> {
        let Some(graph) = &self.graph else {
            return Ok(vec![]);
        };
        let known: HashSet<&str> = known_uris.iter().map(|uri| uri.as_str()).collect();
        let edges = graph
            .batch_traverse(
                std::slice::from_ref(target),
                &[GraphRelation::DerivedFrom, GraphRelation::EvidenceOf],
                1,
            )
            .await
            .unwrap_or_default();
        Ok(edges
            .into_iter()
            .flat_map(|(from, to, _)| [from, to])
            .filter(|uri| known.contains(uri.as_str()))
            .collect())
    }
}

fn estimate_difficulty(base_confidence: f32, prerequisite_count: usize, centrality: f32) -> f32 {
    let novelty = 1.0 - base_confidence.clamp(0.0, 1.0);
    let prereq_load = (prerequisite_count as f32 * 0.08).min(0.3);
    let topology_load = centrality * 0.25;
    (0.25 + novelty * 0.35 + prereq_load + topology_load).clamp(0.05, 0.95)
}

fn infer_content_type(uri: &ContextUri) -> Option<ContentType> {
    let s = uri.as_str();
    if s.contains("/skill/") {
        Some(ContentType::Skill)
    } else if s.contains("/procedure/") {
        Some(ContentType::Procedure)
    } else if s.contains("/error/") {
        Some(ContentType::Error)
    } else if s.contains("/fact/") {
        Some(ContentType::Fact)
    } else if s.contains("/hypothesis/") {
        Some(ContentType::Hypothesis)
    } else {
        None
    }
}

fn expected_knowledge_for(uri: &ContextUri, prerequisite_count: usize, difficulty: f32) -> String {
    format!(
        "learn frontier node {} with {} prerequisite(s), estimated difficulty {:.2}",
        uri.as_str(),
        prerequisite_count,
        difficulty
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{GraphRelation, GraphStore};
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockGraph {
        edges: Mutex<Vec<(ContextUri, ContextUri)>>,
    }

    #[async_trait]
    impl GraphStore for MockGraph {
        async fn add_edge(
            &self,
            from: &ContextUri,
            to: &ContextUri,
            _kind: GraphRelation,
        ) -> Result<()> {
            self.edges.lock().unwrap().push((from.clone(), to.clone()));
            Ok(())
        }

        async fn remove_edge(&self, _from: &ContextUri, _to: &ContextUri) -> Result<()> {
            Ok(())
        }

        async fn neighbors(
            &self,
            uri: &ContextUri,
            _kind: Option<GraphRelation>,
        ) -> Result<Vec<ContextUri>> {
            Ok(self
                .edges
                .lock()
                .unwrap()
                .iter()
                .filter(|(from, _)| from == uri)
                .map(|(_, to)| to.clone())
                .collect())
        }

        async fn batch_traverse(
            &self,
            seeds: &[ContextUri],
            _kinds: &[GraphRelation],
            _max_hops: usize,
        ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
            let seed_set: HashSet<&str> = seeds.iter().map(|uri| uri.as_str()).collect();
            Ok(self
                .edges
                .lock()
                .unwrap()
                .iter()
                .filter(|(_, to)| seed_set.contains(to.as_str()))
                .map(|(from, to)| (to.clone(), from.clone(), GraphRelation::DerivedFrom))
                .collect())
        }

        async fn centrality(&self, _uri: &ContextUri) -> Result<f32> {
            Ok(0.5)
        }
    }

    fn uri(s: &str) -> ContextUri {
        ContextUri::parse(s).unwrap()
    }

    #[tokio::test]
    async fn graph_frontier_excludes_known_and_sorts_by_zpd() {
        let graph = Arc::new(MockGraph::default());
        graph
            .add_edge(
                &uri("uwu://t/agent/a/memories/skill/base"),
                &uri("uwu://t/agent/a/memories/skill/next"),
                GraphRelation::DerivedFrom,
            )
            .await
            .unwrap();
        let curriculum = CurriculumGenerator::new(0.2).with_graph(graph);
        let known = HashMap::from([(uri("uwu://t/agent/a/memories/skill/base"), 0.9)]);
        let frontier = curriculum.find_graph_frontier(&known).await.unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].content_type, Some(ContentType::Skill));
        assert!(frontier[0].zpd_score > 0.5);
    }

    #[tokio::test]
    async fn next_goal_uses_graph_frontier() {
        let graph = Arc::new(MockGraph::default());
        let base = uri("uwu://t/agent/a/memories/skill/base");
        let next = uri("uwu://t/agent/a/memories/skill/next");
        graph
            .add_edge(&base, &next, GraphRelation::DerivedFrom)
            .await
            .unwrap();
        let curriculum = CurriculumGenerator::new(0.2).with_graph(graph);
        let goal = curriculum.next_goal(&[base]).await.unwrap();
        assert_eq!(goal.target_node, next);
        assert!(!goal.expected_new_knowledge.is_empty());
    }

    #[test]
    fn no_graph_does_not_relabel_known_nodes_as_frontier() {
        let curriculum = CurriculumGenerator::new(0.2);
        let known = HashMap::from([("uwu://t/agent/a/memories/skill/base".to_string(), 0.9)]);
        assert!(curriculum.find_frontier(&known).unwrap().is_empty());
        let invalid = HashMap::from([("not a uri".to_string(), 0.9)]);
        assert!(curriculum.find_frontier(&invalid).is_err());
    }

    #[tokio::test]
    async fn next_goal_errors_without_frontier_or_bootstrap_targets() {
        let curriculum = CurriculumGenerator::new(0.2);
        let err = curriculum.next_goal(&[]).await.unwrap_err();
        assert!(err.to_string().contains("no curriculum frontier"));
    }

    #[tokio::test]
    async fn next_goal_uses_configured_bootstrap_target() {
        let target = uri("uwu://t/agent/a/memories/skill/bootstrap");
        let curriculum = CurriculumGenerator::new(0.2).with_bootstrap_targets(vec![target.clone()]);
        let goal = curriculum.next_goal(&[]).await.unwrap();
        assert_eq!(goal.target_node, target);
        assert!(goal.expected_new_knowledge.contains("frontier node"));
    }
}
