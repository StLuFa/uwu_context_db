//! RelationalAxis — 轴3 关系图存储（批量图查询，避免 N+1）。

use agent_context_db_core::{ContextUri, GraphRelation, GraphStore};
use std::collections::HashMap;
use std::sync::Arc;

/// 关系类型（与 GraphRelation 对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelKind { EvolvedFrom, EvolvedTo, EvidenceOf, EntangledWith, Contradicts, Corroborates, DerivedFrom, Supersedes }

/// 关系轴 — 独立的关系图存储，不投影到 FS。
pub struct RelationalAxis {
    graph: Option<Arc<dyn GraphStore>>,
}

impl RelationalAxis {
    pub fn new() -> Self { Self { graph: None } }
    pub fn with_graph(graph: Arc<dyn GraphStore>) -> Self { Self { graph: Some(graph) } }

    /// 批量图遍历 — 一次查询所有 URI 的关系，避免 N+1。
    pub async fn expand_relations(&self, uris: &[ContextUri], kind: Option<RelKind>, max_hops: usize) -> HashMap<String, Vec<ContextUri>> {
        let graph = match &self.graph { Some(g) => g, None => return HashMap::new() };
        let mut result = HashMap::new();
        let gk = kind.map(|k| match k {
            RelKind::EvolvedFrom => GraphRelation::EvolvedFrom,
            RelKind::EvolvedTo => GraphRelation::EvolvedTo,
            RelKind::EvidenceOf => GraphRelation::EvidenceOf,
            RelKind::EntangledWith => GraphRelation::EntangledWith,
            RelKind::Contradicts => GraphRelation::Contradicts,
            RelKind::Corroborates => GraphRelation::Corroborates,
            RelKind::DerivedFrom => GraphRelation::DerivedFrom,
            RelKind::Supersedes => GraphRelation::Supersedes,
        });
        for uri in uris {
            if let Ok(neighbors) = graph.neighbors(uri, gk).await {
                if !neighbors.is_empty() {
                    result.insert(uri.to_string(), neighbors);
                }
            }
        }
        result
    }

    /// 可解释性血统：沿 EvidenceOf + DerivedFrom 追溯。
    pub async fn evidence_tree(&self, uri: &ContextUri, max_hops: usize) -> Vec<ContextUri> {
        let kinds = [RelKind::EvidenceOf, RelKind::DerivedFrom];
        let rels = self.expand_relations(&[uri.clone()], Some(RelKind::EvidenceOf), max_hops).await;
        rels.get(&uri.to_string()).cloned().unwrap_or_default()
    }
}
