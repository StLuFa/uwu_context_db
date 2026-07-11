//! 向量索引适配器：桥接 core 的 `VectorIndex` trait 到 uwu_database 的 `VectorStore`。
//!
//! 薄适配层，只做类型映射，不引入业务逻辑。

use agent_context_db_core::{
    ContextError, ContextUri, IndexHit, IndexPoint, IndexVector, Result, VectorIndex,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;

/// 将 uwu_database::VectorStore 适配为 context-db 的 VectorIndex。
///
/// - `IndexPoint.uri` ↔ `Record.id`
/// - `IndexPoint.payload` ↔ `Record.metadata`
/// - `IndexHit.payload` ↔ `Match.metadata`
pub struct UwuVectorIndex {
    inner: std::sync::Arc<dyn uwu_database::VectorStore>,
    // uwu_database's current port cannot retrieve records by id. Keep a write-through
    // record map so this adapter still provides the stronger VectorIndex contract in one lookup.
    records: RwLock<HashMap<(String, String), IndexVector>>,
}

impl UwuVectorIndex {
    pub fn new(vs: std::sync::Arc<dyn uwu_database::VectorStore>) -> Self {
        Self {
            inner: vs,
            records: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl VectorIndex for UwuVectorIndex {
    async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()> {
        if point.vector.is_empty() || point.vector.iter().any(|value| !value.is_finite()) {
            return Err(ContextError::Storage(
                "vector upsert requires a non-empty finite embedding".into(),
            ));
        }
        if point
            .embedding_dim
            .is_some_and(|dim| dim != point.vector.len())
        {
            return Err(ContextError::Storage(format!(
                "embedding dimension metadata does not match vector length {}",
                point.vector.len()
            )));
        }
        let mut metadata: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_value(point.payload).map_err(|e| {
                ContextError::Storage(format!("vector payload must be a JSON object: {e}"))
            })?;
        metadata.insert("_uwu_uri".into(), point.uri.to_string().into());
        if let Some(model) = point.embedding_model_id {
            metadata.insert("embedding_model_id".into(), model.into());
        }
        metadata.insert("embedding_dim".into(), point.vector.len().into());
        if let Some(version) = point.embedding_version {
            metadata.insert("embedding_version".into(), version.into());
        }
        let indexed = IndexVector {
            uri: point.uri.clone(),
            vector: point.vector.clone(),
            embedding_model_id: metadata
                .get("embedding_model_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            embedding_dim: Some(point.vector.len()),
            embedding_version: metadata
                .get("embedding_version")
                .and_then(serde_json::Value::as_u64),
            payload: serde_json::to_value(&metadata)
                .map_err(|e| ContextError::Storage(format!("serialize vector payload: {e}")))?,
        };
        let record = uwu_database::Record {
            id: point.uri.to_string(),
            vector: point.vector,
            metadata,
        };
        self.inner
            .upsert(collection, &[record])
            .await
            .map_err(|e| ContextError::Storage(format!("vector upsert: {e}")))?;
        self.records
            .write()
            .insert((collection.to_owned(), point.uri.to_string()), indexed);
        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>> {
        if query.is_empty() || query.iter().any(|value| !value.is_finite()) {
            return Err(ContextError::Storage(
                "vector search requires a non-empty finite embedding".into(),
            ));
        }
        let filter_map: Option<std::collections::HashMap<String, serde_json::Value>> = filter
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| {
                ContextError::Storage(format!("vector filter must be a JSON object: {e}"))
            })?;

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

        matches
            .into_iter()
            .map(|m| {
                let uri_text = m
                    .metadata
                    .get("_uwu_uri")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(&m.id);
                let uri = ContextUri::parse(uri_text).map_err(|e| {
                    ContextError::Storage(format!(
                        "vector backend returned invalid URI {uri_text:?}: {e}"
                    ))
                })?;
                let payload = serde_json::to_value(m.metadata)
                    .map_err(|e| ContextError::Storage(format!("serialize vector payload: {e}")))?;
                Ok(IndexHit {
                    uri,
                    score: m.score,
                    payload,
                })
            })
            .collect()
    }

    async fn get_many(&self, collection: &str, uris: &[ContextUri]) -> Result<Vec<IndexVector>> {
        let records = self.records.read();
        Ok(uris
            .iter()
            .filter_map(|uri| {
                records
                    .get(&(collection.to_owned(), uri.to_string()))
                    .cloned()
            })
            .collect())
    }

    async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()> {
        self.inner
            .delete(collection, &[uri.to_string()])
            .await
            .map_err(|e| ContextError::Storage(format!("vector delete: {e}")))?;
        self.records
            .write()
            .remove(&(collection.to_owned(), uri.to_string()));
        Ok(())
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
