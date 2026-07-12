//! Multimodal embedding-space migration, evaluation, and durable atomic cutover.

use crate::{
    ContextError, ContextUri, EmbeddingInput, EmbeddingSpaceId, EncodedEmbedding, Modality,
    MultimodalEncoder, Result, SpaceCheckedVectorIndex, VectorIndex,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::PathBuf, sync::Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingMigrationPhase {
    Reindex,
    DualWrite,
    Evaluating,
    Cutover,
    Complete,
}
impl EmbeddingMigrationPhase {
    fn can_transition(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Reindex, Self::DualWrite)
                | (Self::DualWrite, Self::Evaluating)
                | (Self::Evaluating, Self::Cutover)
                | (Self::Cutover, Self::Complete)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityCollections {
    pub source: String,
    pub target: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigrationEvaluationGate {
    pub min_recall_at_k: f32,
    pub min_cosine_margin: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingMigrationState {
    pub id: String,
    pub generation: u64,
    pub source_space: EmbeddingSpaceId,
    pub target_space: EmbeddingSpaceId,
    pub collections: BTreeMap<Modality, ModalityCollections>,
    pub phase: EmbeddingMigrationPhase,
    pub gate: MigrationEvaluationGate,
    pub evaluation: Option<PairedRetrievalMetrics>,
}

pub trait MigrationStateStore: Send + Sync {
    fn load(&self, id: &str) -> Result<Option<EmbeddingMigrationState>>;
    fn compare_and_swap(
        &self,
        expected_generation: u64,
        next: &EmbeddingMigrationState,
    ) -> Result<()>;
}

/// Durable JSON state store: write, fsync, atomic rename, then fsync parent directory.
pub struct FileMigrationStateStore {
    directory: PathBuf,
    lock: Mutex<()>,
}
impl FileMigrationStateStore {
    pub fn new(directory: impl Into<PathBuf>) -> Result<Self> {
        let directory = directory.into();
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            lock: Mutex::new(()),
        })
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        if id.is_empty()
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        {
            return Err(ContextError::Storage("invalid migration id".into()));
        }
        Ok(self.directory.join(format!("{id}.json")))
    }
}
impl MigrationStateStore for FileMigrationStateStore {
    fn load(&self, id: &str) -> Result<Option<EmbeddingMigrationState>> {
        let path = self.path(id)?;
        match fs::read(path) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    fn compare_and_swap(
        &self,
        expected_generation: u64,
        next: &EmbeddingMigrationState,
    ) -> Result<()> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| ContextError::Storage("migration state lock poisoned".into()))?;
        let path = self.path(&next.id)?;
        let current = match fs::read(&path) {
            Ok(bytes) => Some(serde_json::from_slice::<EmbeddingMigrationState>(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let actual = current.as_ref().map_or(0, |s| s.generation);
        if actual != expected_generation || next.generation != expected_generation + 1 {
            return Err(ContextError::VersionConflict(format!(
                "migration generation expected {expected_generation}, actual {actual}, next {}",
                next.generation
            )));
        }
        let temp = self
            .directory
            .join(format!(".{}.{}.tmp", next.id, next.generation));
        let bytes = serde_json::to_vec(next)?;
        let mut file = fs::File::create(&temp)?;
        use std::io::Write;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temp, &path)?;
        fs::File::open(&self.directory)?.sync_all()?;
        Ok(())
    }
}

pub struct AtomicCutover<S> {
    store: S,
}
impl<S: MigrationStateStore> AtomicCutover<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }
    pub fn transition(
        &self,
        id: &str,
        expected_generation: u64,
        next_phase: EmbeddingMigrationPhase,
    ) -> Result<EmbeddingMigrationState> {
        let mut state = self
            .store
            .load(id)?
            .ok_or_else(|| ContextError::NotFound(id.into()))?;
        if state.generation != expected_generation || !state.phase.can_transition(next_phase) {
            return Err(ContextError::VersionConflict(
                "invalid or stale migration transition".into(),
            ));
        }
        if next_phase == EmbeddingMigrationPhase::Cutover {
            let metrics = state.evaluation.as_ref().ok_or_else(|| {
                ContextError::VersionConflict("cutover requires evaluation".into())
            })?;
            if metrics.recall_at_k < state.gate.min_recall_at_k
                || metrics.cosine_margin < state.gate.min_cosine_margin
            {
                return Err(ContextError::VersionConflict(
                    "evaluation gate failed".into(),
                ));
            }
        }
        state.phase = next_phase;
        state.generation += 1;
        self.store.compare_and_swap(expected_generation, &state)?;
        Ok(state)
    }
}

#[derive(Debug, Clone)]
pub struct MigrationItem {
    pub uri: ContextUri,
    pub input: EmbeddingInput,
    pub payload: serde_json::Value,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingMigrationReport {
    pub encoded: usize,
    pub written: usize,
    pub by_modality: BTreeMap<Modality, usize>,
}

pub struct MultimodalMigrationExecutor<'a, I, E> {
    pub index: &'a I,
    pub encoder: &'a E,
    pub state: &'a EmbeddingMigrationState,
}
impl<'a, I: VectorIndex, E: MultimodalEncoder> MultimodalMigrationExecutor<'a, I, E> {
    pub async fn run(&self, items: &[MigrationItem]) -> Result<EmbeddingMigrationReport> {
        self.encoder.space().validate()?;
        if self.encoder.space() != &self.state.target_space {
            return Err(ContextError::Unsupported(
                "migration encoder target space mismatch".into(),
            ));
        }
        let capabilities = self.encoder.capabilities();
        for item in items {
            item.input.validate()?;
            if !capabilities.modalities.contains(&item.input.modality()) {
                return Err(ContextError::Unsupported(format!(
                    "encoder lacks {:?} capability",
                    item.input.modality()
                )));
            }
            if !self.state.collections.contains_key(&item.input.modality()) {
                return Err(ContextError::Storage(
                    "missing modality collection mapping".into(),
                ));
            }
        }
        let mut report = EmbeddingMigrationReport::default();
        for chunk in items.chunks(capabilities.max_batch_size.max(1)) {
            let inputs: Vec<_> = chunk.iter().map(|item| item.input.clone()).collect();
            let embeddings = self.encoder.encode_batch(&inputs).await?;
            if embeddings.len() != chunk.len() {
                return Err(ContextError::Storage(
                    "encoder result count mismatch".into(),
                ));
            }
            report.encoded += embeddings.len();
            for (item, embedding) in chunk.iter().zip(embeddings) {
                let modality = item.input.modality();
                let collections = &self.state.collections[&modality];
                let target = SpaceCheckedVectorIndex::new(
                    self.index,
                    &collections.target,
                    self.state.target_space.clone(),
                )?;
                target
                    .upsert(item.uri.clone(), embedding, item.payload.clone())
                    .await?;
                report.written += 1;
                *report.by_modality.entry(modality).or_default() += 1;
            }
        }
        Ok(report)
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalPair {
    pub query: EncodedEmbedding,
    pub positive: ContextUri,
    pub positive_embedding: EncodedEmbedding,
    pub negatives: Vec<EncodedEmbedding>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedRetrievalMetrics {
    pub k: usize,
    pub pairs: usize,
    pub recall_at_k: f32,
    pub cosine_margin: f32,
}
pub async fn evaluate_paired_retrieval<I: VectorIndex>(
    index: &I,
    collection: &str,
    space: &EmbeddingSpaceId,
    pairs: &[RetrievalPair],
    k: usize,
) -> Result<PairedRetrievalMetrics> {
    if pairs.is_empty() || k == 0 {
        return Err(ContextError::Unsupported(
            "evaluation requires pairs and k > 0".into(),
        ));
    }
    let checked = SpaceCheckedVectorIndex::new(index, collection, space.clone())?;
    let mut recalled = 0usize;
    let mut margin = 0.0;
    for pair in pairs {
        pair.query.ensure_space(space)?;
        pair.positive_embedding.ensure_space(space)?;
        for negative in &pair.negatives {
            negative.ensure_space(space)?;
        }
        let hits = checked.search(pair.query.clone(), k, None).await?;
        if hits.iter().any(|hit| hit.uri == pair.positive) {
            recalled += 1;
        }
        let positive = cosine(&pair.query.values, &pair.positive_embedding.values)?;
        let hardest = pair
            .negatives
            .iter()
            .map(|negative| cosine(&pair.query.values, &negative.values))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(-1.0_f32, f32::max);
        margin += positive - hardest;
    }
    Ok(PairedRetrievalMetrics {
        k,
        pairs: pairs.len(),
        recall_at_k: recalled as f32 / pairs.len() as f32,
        cosine_margin: margin / pairs.len() as f32,
    })
}
fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() || a.is_empty() {
        return Err(ContextError::Unsupported(
            "cosine dimension mismatch".into(),
        ));
    }
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let an = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let bn = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if an == 0.0 || bn == 0.0 {
        return Err(ContextError::Unsupported("cosine zero vector".into()));
    }
    Ok(dot / (an * bn))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingNormalization, IndexHit, IndexPoint};
    use async_trait::async_trait;
    use parking_lot::Mutex as ParkingMutex;
    use std::collections::HashMap;
    fn space(checkpoint: &str) -> EmbeddingSpaceId {
        EmbeddingSpaceId {
            model: "clip".into(),
            checkpoint: checkpoint.into(),
            preprocess: "p1".into(),
            dim: 2,
            normalization: EmbeddingNormalization::None,
        }
    }
    #[derive(Default)]
    struct Index {
        points: ParkingMutex<HashMap<String, Vec<IndexPoint>>>,
    }
    #[async_trait]
    impl VectorIndex for Index {
        async fn upsert(&self, c: &str, p: IndexPoint) -> Result<()> {
            self.points.lock().entry(c.into()).or_default().push(p);
            Ok(())
        }
        async fn search(
            &self,
            c: &str,
            q: Vec<f32>,
            k: usize,
            _: Option<serde_json::Value>,
        ) -> Result<Vec<IndexHit>> {
            let mut p = self.points.lock().get(c).cloned().unwrap_or_default();
            p.sort_by(|a, b| {
                cosine(&q, &b.vector)
                    .unwrap()
                    .total_cmp(&cosine(&q, &a.vector).unwrap())
            });
            Ok(p.into_iter()
                .take(k)
                .map(|p| IndexHit {
                    uri: p.uri,
                    score: 1.0,
                    payload: p.payload,
                })
                .collect())
        }
        async fn delete(&self, _: &str, _: &ContextUri) -> Result<()> {
            Ok(())
        }
    }
    struct Encoder {
        space: EmbeddingSpaceId,
    }
    #[async_trait]
    impl MultimodalEncoder for Encoder {
        fn space(&self) -> &EmbeddingSpaceId {
            &self.space
        }
        fn capabilities(&self) -> crate::EncoderCapabilities {
            crate::EncoderCapabilities {
                modalities: [Modality::Text, Modality::Image, Modality::Audio]
                    .into_iter()
                    .collect(),
                batch: true,
                max_batch_size: 2,
            }
        }
        async fn encode_batch(&self, i: &[EmbeddingInput]) -> Result<Vec<EncodedEmbedding>> {
            i.iter()
                .map(|input| {
                    EncodedEmbedding::new(
                        self.space.clone(),
                        match input.modality() {
                            Modality::Text => vec![1.0, 0.0],
                            Modality::Image => vec![0.0, 1.0],
                            Modality::Audio => vec![0.5, 0.5],
                        },
                    )
                })
                .collect()
        }
    }
    fn uri(id: &str) -> ContextUri {
        ContextUri::parse(format!("uwu://t/agent/a/memory/fact/{id}")).unwrap()
    }
    #[tokio::test]
    async fn migrates_all_modalities_and_evaluates() {
        let index = Index::default();
        let target = space("v2");
        let state = EmbeddingMigrationState {
            id: "m1".into(),
            generation: 1,
            source_space: space("v1"),
            target_space: target.clone(),
            collections: [Modality::Text, Modality::Image, Modality::Audio]
                .into_iter()
                .map(|m| {
                    (
                        m,
                        ModalityCollections {
                            source: format!("old-{m:?}"),
                            target: format!("new-{m:?}"),
                        },
                    )
                })
                .collect(),
            phase: EmbeddingMigrationPhase::Reindex,
            gate: MigrationEvaluationGate {
                min_recall_at_k: 1.0,
                min_cosine_margin: 0.1,
            },
            evaluation: None,
        };
        let items = vec![
            MigrationItem {
                uri: uri("t"),
                input: EmbeddingInput::Text {
                    text: "real".into(),
                },
                payload: serde_json::Value::Null,
            },
            MigrationItem {
                uri: uri("i"),
                input: EmbeddingInput::Image {
                    bytes: vec![1, 2],
                    mime_type: "image/png".into(),
                },
                payload: serde_json::Value::Null,
            },
            MigrationItem {
                uri: uri("a"),
                input: EmbeddingInput::Audio {
                    bytes: vec![3, 4],
                    mime_type: "audio/wav".into(),
                },
                payload: serde_json::Value::Null,
            },
        ];
        let report = MultimodalMigrationExecutor {
            index: &index,
            encoder: &Encoder {
                space: target.clone(),
            },
            state: &state,
        }
        .run(&items)
        .await
        .unwrap();
        assert_eq!(report.written, 3);
        let q = EncodedEmbedding::new(target.clone(), vec![1.0, 0.0]).unwrap();
        let metrics = evaluate_paired_retrieval(
            &index,
            "new-Text",
            &target,
            &[RetrievalPair {
                query: q.clone(),
                positive: uri("t"),
                positive_embedding: q,
                negatives: vec![EncodedEmbedding::new(target.clone(), vec![0.0, 1.0]).unwrap()],
            }],
            1,
        )
        .await
        .unwrap();
        assert_eq!(metrics.recall_at_k, 1.0);
        assert!(metrics.cosine_margin > 0.9)
    }
    #[test]
    fn durable_cutover_is_cas_and_gate_protected() {
        let dir = std::env::temp_dir().join(format!("uwu-mm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = FileMigrationStateStore::new(&dir).unwrap();
        let mut state = EmbeddingMigrationState {
            id: "m".into(),
            generation: 1,
            source_space: space("v1"),
            target_space: space("v2"),
            collections: BTreeMap::new(),
            phase: EmbeddingMigrationPhase::Evaluating,
            gate: MigrationEvaluationGate {
                min_recall_at_k: 0.9,
                min_cosine_margin: 0.1,
            },
            evaluation: Some(PairedRetrievalMetrics {
                k: 1,
                pairs: 1,
                recall_at_k: 1.0,
                cosine_margin: 0.5,
            }),
        };
        store
            .compare_and_swap(0, &{
                state.generation = 1;
                state.clone()
            })
            .unwrap();
        let state = AtomicCutover::new(store)
            .transition("m", 1, EmbeddingMigrationPhase::Cutover)
            .unwrap();
        assert_eq!(state.phase, EmbeddingMigrationPhase::Cutover);
        assert_eq!(state.generation, 2);
        fs::remove_dir_all(dir).unwrap()
    }
    #[tokio::test]
    async fn rejects_space_mismatch() {
        let index = Index::default();
        let result = evaluate_paired_retrieval(
            &index,
            "x",
            &space("v1"),
            &[RetrievalPair {
                query: EncodedEmbedding::new(space("v2"), vec![1.0, 0.0]).unwrap(),
                positive: uri("x"),
                positive_embedding: EncodedEmbedding::new(space("v1"), vec![1.0, 0.0]).unwrap(),
                negatives: vec![],
            }],
            1,
        )
        .await;
        assert!(result.is_err())
    }
}
