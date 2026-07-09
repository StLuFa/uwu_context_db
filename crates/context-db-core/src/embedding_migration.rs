//! Embedding model migration planning.
//!
//! This module keeps vector-space upgrades explicit: old and new embedding
//! models must not silently share the same index collection.

use crate::{ContextUri, IndexPoint, LlmClient, Result, VectorIndex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelVersion {
    pub model_id: String,
    pub dim: usize,
    pub version: u64,
}

impl EmbeddingModelVersion {
    pub fn new(model_id: impl Into<String>, dim: usize, version: u64) -> Self {
        Self {
            model_id: model_id.into(),
            dim,
            version,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingMigrationPhase {
    Reindex,
    DualWrite,
    Cutover,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingMigrationPlan {
    pub source: EmbeddingModelVersion,
    pub target: EmbeddingModelVersion,
    pub source_collection: String,
    pub target_collection: String,
    pub phase: EmbeddingMigrationPhase,
}

impl EmbeddingMigrationPlan {
    pub fn new(
        source: EmbeddingModelVersion,
        target: EmbeddingModelVersion,
        collection_prefix: impl Into<String>,
    ) -> Self {
        let collection_prefix = collection_prefix.into();
        Self {
            source_collection: collection_name(&collection_prefix, &source),
            target_collection: collection_name(&collection_prefix, &target),
            source,
            target,
            phase: EmbeddingMigrationPhase::Reindex,
        }
    }

    pub fn advance(&mut self, phase: EmbeddingMigrationPhase) {
        self.phase = phase;
    }

    pub fn action_for_point(&self, point: &IndexPoint) -> EmbeddingMigrationAction {
        let Some(model_id) = &point.embedding_model_id else {
            return EmbeddingMigrationAction::Reembed {
                uri: point.uri.clone(),
                target_collection: self.target_collection.clone(),
            };
        };
        let dim = point.embedding_dim.unwrap_or(point.vector.len());
        let version = point.embedding_version.unwrap_or(0);

        if model_id == &self.target.model_id
            && dim == self.target.dim
            && version == self.target.version
        {
            return EmbeddingMigrationAction::Skip;
        }

        match self.phase {
            EmbeddingMigrationPhase::Reindex => EmbeddingMigrationAction::Reembed {
                uri: point.uri.clone(),
                target_collection: self.target_collection.clone(),
            },
            EmbeddingMigrationPhase::DualWrite => EmbeddingMigrationAction::DualWrite {
                uri: point.uri.clone(),
                source_collection: self.source_collection.clone(),
                target_collection: self.target_collection.clone(),
            },
            EmbeddingMigrationPhase::Cutover | EmbeddingMigrationPhase::Complete => {
                EmbeddingMigrationAction::Reembed {
                    uri: point.uri.clone(),
                    target_collection: self.target_collection.clone(),
                }
            }
        }
    }

    pub fn should_search_collection(&self) -> Vec<String> {
        match self.phase {
            EmbeddingMigrationPhase::Reindex => vec![self.source_collection.clone()],
            EmbeddingMigrationPhase::DualWrite => {
                vec![
                    self.target_collection.clone(),
                    self.source_collection.clone(),
                ]
            }
            EmbeddingMigrationPhase::Cutover | EmbeddingMigrationPhase::Complete => {
                vec![self.target_collection.clone()]
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingMigrationAction {
    Skip,
    Reembed {
        uri: ContextUri,
        target_collection: String,
    },
    DualWrite {
        uri: ContextUri,
        source_collection: String,
        target_collection: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingMigrationReport {
    pub scanned: usize,
    pub skipped: usize,
    pub reembedded: usize,
    pub dual_written: usize,
    pub deleted_from_source: usize,
}

pub struct EmbeddingMigrationExecutor<'a, I, L> {
    pub index: &'a I,
    pub llm: &'a L,
    pub plan: EmbeddingMigrationPlan,
}

impl<'a, I, L> EmbeddingMigrationExecutor<'a, I, L>
where
    I: VectorIndex,
    L: LlmClient,
{
    pub fn new(index: &'a I, llm: &'a L, plan: EmbeddingMigrationPlan) -> Self {
        Self { index, llm, plan }
    }

    pub async fn run<F>(
        &self,
        points: &[IndexPoint],
        mut text_for_uri: F,
    ) -> Result<EmbeddingMigrationReport>
    where
        F: FnMut(&ContextUri) -> Option<String> + Send,
    {
        let mut report = EmbeddingMigrationReport::default();
        let mut pending = Vec::new();
        let mut texts = Vec::new();

        for point in points {
            report.scanned += 1;
            match self.plan.action_for_point(point) {
                EmbeddingMigrationAction::Skip => report.skipped += 1,
                EmbeddingMigrationAction::Reembed {
                    target_collection, ..
                } => {
                    let text =
                        text_for_uri(&point.uri).unwrap_or_else(|| point.payload.to_string());
                    texts.push(text);
                    pending.push(PendingEmbeddingWrite {
                        point,
                        target_collection,
                        write_source_collection: None,
                        delete_source_after_write: matches!(
                            self.plan.phase,
                            EmbeddingMigrationPhase::Cutover | EmbeddingMigrationPhase::Complete
                        ),
                    });
                    report.reembedded += 1;
                }
                EmbeddingMigrationAction::DualWrite {
                    source_collection,
                    target_collection,
                    ..
                } => {
                    let text =
                        text_for_uri(&point.uri).unwrap_or_else(|| point.payload.to_string());
                    texts.push(text);
                    pending.push(PendingEmbeddingWrite {
                        point,
                        target_collection,
                        write_source_collection: Some(source_collection),
                        delete_source_after_write: false,
                    });
                    report.dual_written += 1;
                }
            }
        }

        let embeddings = self.llm.embed_batch(&texts).await?;
        if embeddings.len() != pending.len() {
            return Err(crate::ContextError::Storage(format!(
                "embedding provider returned {} vectors for {} migration writes",
                embeddings.len(),
                pending.len()
            )));
        }

        for (write, embedding) in pending.into_iter().zip(embeddings) {
            let migrated = IndexPoint::from_embedding(
                write.point.uri.clone(),
                embedding.clone(),
                write.point.payload.clone(),
            );
            self.index
                .upsert(&write.target_collection, migrated)
                .await?;
            if let Some(source_collection) = write.write_source_collection {
                let source_point = IndexPoint::from_embedding(
                    write.point.uri.clone(),
                    embedding,
                    write.point.payload.clone(),
                );
                self.index.upsert(&source_collection, source_point).await?;
            }
            if write.delete_source_after_write {
                self.index
                    .delete(&self.plan.source_collection, &write.point.uri)
                    .await?;
                report.deleted_from_source += 1;
            }
        }

        Ok(report)
    }

    pub async fn dual_write_text(
        &self,
        uri: ContextUri,
        text: &str,
        payload: serde_json::Value,
    ) -> Result<()> {
        let embedding = self.llm.embed(text).await?;
        let target_point =
            IndexPoint::from_embedding(uri.clone(), embedding.clone(), payload.clone());
        self.index
            .upsert(&self.plan.target_collection, target_point)
            .await?;

        if matches!(self.plan.phase, EmbeddingMigrationPhase::DualWrite) {
            let source_point = IndexPoint::from_embedding(uri, embedding, payload);
            self.index
                .upsert(&self.plan.source_collection, source_point)
                .await?;
        }
        Ok(())
    }
}

struct PendingEmbeddingWrite<'p> {
    point: &'p IndexPoint,
    target_collection: String,
    write_source_collection: Option<String>,
    delete_source_after_write: bool,
}

fn collection_name(prefix: &str, version: &EmbeddingModelVersion) -> String {
    format!(
        "{}__{}__d{}__v{}",
        prefix,
        sanitize_collection_part(&version.model_id),
        version.dim,
        version.version
    )
}

fn sanitize_collection_part(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingVector, IndexHit, JsonSchema, LlmError, LlmOpts};
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::collections::HashMap;

    fn point(model_id: Option<&str>, dim: Option<usize>, version: Option<u64>) -> IndexPoint {
        IndexPoint {
            uri: ContextUri::parse("uwu://t/agent/a/memories/fact/f1").unwrap(),
            vector: vec![0.0; dim.unwrap_or(3)],
            embedding_model_id: model_id.map(str::to_string),
            embedding_dim: dim,
            embedding_version: version,
            payload: serde_json::Value::Null,
        }
    }

    #[derive(Default)]
    struct MemoryIndex {
        points: Mutex<HashMap<String, Vec<IndexPoint>>>,
        deleted: Mutex<Vec<(String, ContextUri)>>,
    }

    #[async_trait]
    impl VectorIndex for MemoryIndex {
        async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()> {
            self.points
                .lock()
                .entry(collection.to_string())
                .or_default()
                .push(point);
            Ok(())
        }

        async fn search(
            &self,
            collection: &str,
            _query: Vec<f32>,
            top_k: usize,
            _filter: Option<serde_json::Value>,
        ) -> Result<Vec<IndexHit>> {
            Ok(self
                .points
                .lock()
                .get(collection)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(top_k)
                .map(|point| IndexHit {
                    uri: point.uri,
                    score: 1.0,
                    payload: point.payload,
                })
                .collect())
        }

        async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()> {
            self.deleted
                .lock()
                .push((collection.to_string(), uri.clone()));
            Ok(())
        }
    }

    struct MockLlm;

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete(
            &self,
            _prompt: &str,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok(String::new())
        }

        async fn embed(&self, text: &str) -> std::result::Result<EmbeddingVector, LlmError> {
            Ok(EmbeddingVector::new(
                vec![text.len() as f32; 4],
                "new-model",
                2,
            ))
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }
    }

    #[test]
    fn migration_reembeds_missing_or_old_metadata() {
        let plan = EmbeddingMigrationPlan::new(
            EmbeddingModelVersion::new("old-model", 3, 1),
            EmbeddingModelVersion::new("new/model", 4, 2),
            "memories",
        );
        assert!(matches!(
            plan.action_for_point(&point(None, None, None)),
            EmbeddingMigrationAction::Reembed { .. }
        ));
        assert!(matches!(
            plan.action_for_point(&point(Some("old-model"), Some(3), Some(1))),
            EmbeddingMigrationAction::Reembed { .. }
        ));
        assert!(matches!(
            plan.action_for_point(&point(Some("new/model"), Some(4), Some(2))),
            EmbeddingMigrationAction::Skip
        ));
    }

    #[tokio::test]
    async fn executor_reembeds_points_into_target_collection() {
        let index = MemoryIndex::default();
        let llm = MockLlm;
        let plan = EmbeddingMigrationPlan::new(
            EmbeddingModelVersion::new("old-model", 3, 1),
            EmbeddingModelVersion::new("new-model", 4, 2),
            "memories",
        );
        let executor = EmbeddingMigrationExecutor::new(&index, &llm, plan.clone());
        let points = vec![point(Some("old-model"), Some(3), Some(1))];

        let report = executor
            .run(&points, |uri| Some(format!("source text for {uri}")))
            .await
            .unwrap();

        assert_eq!(report.scanned, 1);
        assert_eq!(report.reembedded, 1);
        let stored = index.points.lock();
        let migrated = stored
            .get(&plan.target_collection)
            .unwrap()
            .first()
            .unwrap();
        assert_eq!(migrated.embedding_model_id.as_deref(), Some("new-model"));
        assert_eq!(migrated.embedding_dim, Some(4));
        assert_eq!(migrated.embedding_version, Some(2));
    }

    #[tokio::test]
    async fn executor_cutover_deletes_migrated_source_points() {
        let index = MemoryIndex::default();
        let llm = MockLlm;
        let mut plan = EmbeddingMigrationPlan::new(
            EmbeddingModelVersion::new("old-model", 3, 1),
            EmbeddingModelVersion::new("new-model", 4, 2),
            "memories",
        );
        plan.advance(EmbeddingMigrationPhase::Cutover);
        let executor = EmbeddingMigrationExecutor::new(&index, &llm, plan.clone());
        let points = vec![point(Some("old-model"), Some(3), Some(1))];

        let report = executor
            .run(&points, |_| Some("source text".into()))
            .await
            .unwrap();

        assert_eq!(report.reembedded, 1);
        assert_eq!(report.deleted_from_source, 1);
        assert_eq!(index.deleted.lock()[0].0, plan.source_collection);
    }

    #[test]
    fn dual_write_searches_new_then_old_collection() {
        let mut plan = EmbeddingMigrationPlan::new(
            EmbeddingModelVersion::new("old-model", 3, 1),
            EmbeddingModelVersion::new("new-model", 4, 2),
            "memories",
        );
        plan.advance(EmbeddingMigrationPhase::DualWrite);
        assert_eq!(plan.should_search_collection().len(), 2);
        assert!(matches!(
            plan.action_for_point(&point(Some("old-model"), Some(3), Some(1))),
            EmbeddingMigrationAction::DualWrite { .. }
        ));
    }
}
