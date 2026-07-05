//! 联想检索 — 沿 evolved_from/to 图 1-2 跳扩展检索结果。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, FsOps, GraphStore, Result};
use std::sync::Arc;

/// 联想检索器 — 在向量召回基础上，沿联想图扩展。
pub struct AssociativeRetriever {
    fs: Arc<dyn FsOps>,
    graph: Option<Arc<dyn GraphStore>>,
    max_hops: usize,
    decay_factor: f32,
}

impl AssociativeRetriever {
    pub fn new(fs: Arc<dyn FsOps>, graph: Option<Arc<dyn GraphStore>>) -> Self {
        Self {
            fs,
            graph,
            max_hops: 2,
            decay_factor: 0.5,
        }
    }

    /// 沿联想图扩展检索命中。
    pub async fn expand(
        &self,
        base_hits: &[crate::RetrievalHit],
    ) -> Result<Vec<crate::RetrievalHit>> {
        let graph = match &self.graph {
            Some(g) => g,
            None => return Ok(vec![]),
        };

        let mut expanded = Vec::new();
        for hit in base_hits {
            let neighbors = graph.neighbors(&hit.uri, None).await?;
            for (hop, neighbor_uri) in neighbors.iter().enumerate() {
                if hop >= self.max_hops {
                    break;
                }
                let weight = self.decay_factor.powi(hop as i32);
                if weight < 0.1 {
                    continue;
                }
                if let Ok(payload) = self.fs.read(neighbor_uri, ContentLevel::L0).await {
                    expanded.push(crate::RetrievalHit {
                        uri: neighbor_uri.clone(),
                        level: ContentLevel::L0,
                        content: payload,
                        relevance: hit.relevance * weight,
                        parent_chain: vec![hit.uri.clone()],
                        content_type: None,
                    });
                }
            }
        }
        Ok(expanded)
    }
}

/// 简化版 RetrievalHit — 避免循环依赖。
#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub uri: ContextUri,
    pub level: ContentLevel,
    pub content: ContentPayload,
    pub relevance: f32,
    pub parent_chain: Vec<ContextUri>,
    pub content_type: Option<agent_context_db_core::ContentType>,
}
