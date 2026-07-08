//! 向量索引适配器：桥接 core 的 `VectorIndex` trait 到 uwu_database 的 `VectorStore`。
//!
//! 薄适配层，只做类型映射，不引入业务逻辑。

use agent_context_db_core::{ContextError, ContextUri, IndexHit, IndexPoint, Result, VectorIndex};
use async_trait::async_trait;

/// 将 uwu_database::VectorStore 适配为 context-db 的 VectorIndex。
///
/// - `IndexPoint.uri` ↔ `Record.id`
/// - `IndexPoint.payload` ↔ `Record.metadata`
/// - `IndexHit.payload` ↔ `Match.metadata`
pub struct UwuVectorIndex {
    inner: std::sync::Arc<dyn uwu_database::VectorStore>,
}

impl UwuVectorIndex {
    pub fn new(vs: std::sync::Arc<dyn uwu_database::VectorStore>) -> Self {
        Self { inner: vs }
    }
}

#[async_trait]
impl VectorIndex for UwuVectorIndex {
    async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()> {
        let record = uwu_database::Record {
            id: point.uri.to_string(),
            vector: point.vector,
            metadata: serde_json::from_value(point.payload).unwrap_or_default(),
        };
        self.inner
            .upsert(collection, &[record])
            .await
            .map_err(|e| ContextError::Storage(format!("vector upsert: {e}")))
    }

    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>> {
        let filter_map: Option<std::collections::HashMap<String, serde_json::Value>> =
            filter.and_then(|v| serde_json::from_value(v).ok());

        let q = uwu_database::Query {
            vector: &query,
            top_k,
            filter: filter_map.as_ref(),
        };
        let matches = self
            .inner
            .search(collection, q)
            .await
            .map_err(|e| ContextError::Storage(format!("vector search: {e}")))?;

        Ok(matches
            .into_iter()
            .filter_map(|m| {
                // 底层存的是有效的 uwu:// URI；解析失败则跳过（记 debug）
                let uri = ContextUri::parse(&m.id).ok()?;
                Some(IndexHit {
                    uri,
                    score: m.score,
                    payload: serde_json::to_value(m.metadata).unwrap_or_default(),
                })
            })
            .collect())
    }

    async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()> {
        self.inner
            .delete(collection, &[uri.to_string()])
            .await
            .map_err(|e| ContextError::Storage(format!("vector delete: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 IndexPoint ↔ Record 的映射方向一致性。
    #[test]
    fn point_to_record_field_mapping() {
        let uri = ContextUri::parse("uwu://t/x").unwrap();
        let point = IndexPoint {
            uri: uri.clone(),
            vector: vec![1.0, 0.0],
            embedding_model_id: None,
            embedding_dim: None,
            embedding_version: None,
            payload: serde_json::json!({"k": "v"}),
        };
        let record = uwu_database::Record {
            id: point.uri.to_string(),
            vector: point.vector.clone(),
            metadata: serde_json::from_value(point.payload.clone()).unwrap(),
        };
        assert_eq!(record.id, "uwu://t/x");
        assert_eq!(record.vector, vec![1.0, 0.0]);
        assert_eq!(record.metadata.get("k").and_then(|v| v.as_str()), Some("v"));
    }
}
