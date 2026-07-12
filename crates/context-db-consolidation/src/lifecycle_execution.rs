//! Durable execution of destructive lifecycle decisions.
//!
//! A job is persisted before any external mutation.  Each completed phase is
//! checkpointed, so a process restart resumes at the first incomplete phase.
//! Archive/cold-storage writes an immutable manifest before live references are
//! removed; `restore` replays that manifest in the opposite direction.

use agent_context_db_core::{
    BlobRef, BlobStore, ContentLevel, ContentPart, ContentPayload, ContentStore, ContextEntry,
    ContextError, ContextUri, GraphRelation, GraphStore, IndexPoint, LifecycleAction, PageRequest,
    Result, StateScope, UriCategory, VectorIndex,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleJobState {
    Pending,
    Running,
    Retryable,
    Succeeded,
    PermanentlyFailed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleOperation {
    Archive,
    Delete,
    ColdStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetacogTier {
    Hot,
    Cold,
}

/// Deterministic heat routing for metacognitive state. Mid-horizon observations
/// are cold; immediate state and durable learned rules remain directly available.
pub fn metacog_tier(entry: &ContextEntry) -> Result<MetacogTier> {
    if entry.uri.category() != UriCategory::Metacog {
        return Err(ContextError::InvalidUri(format!(
            "heat routing is only defined for metacog entries: {}",
            entry.uri
        )));
    }
    match entry.metadata.state_scope {
        Some(StateScope::Mid) => Ok(MetacogTier::Cold),
        Some(StateScope::Short | StateScope::Long) => Ok(MetacogTier::Hot),
        None => Err(ContextError::Storage(format!(
            "metacog entry {} has no state scope",
            entry.uri
        ))),
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LifecycleCheckpoint {
    pub manifest: bool,
    pub vector: bool,
    pub graph: bool,
    pub content: bool,
    pub blobs: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleAudit {
    pub at: DateTime<Utc>,
    pub state: LifecycleJobState,
    pub message: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleJob {
    pub id: Uuid,
    pub idempotency_key: String,
    pub uri: ContextUri,
    pub operation: LifecycleOperation,
    pub state: LifecycleJobState,
    pub attempts: u32,
    pub max_attempts: u32,
    pub next_attempt_at: DateTime<Utc>,
    pub checkpoint: LifecycleCheckpoint,
    pub last_error: Option<String>,
    pub audit: Vec<LifecycleAudit>,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub lease_owner: Option<Uuid>,
    #[serde(default)]
    pub lease_until: Option<DateTime<Utc>>,
    #[serde(default)]
    pub restore_checkpoint: LifecycleCheckpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveManifest {
    pub job_id: Uuid,
    pub entry: ContextEntry,
    pub blobs: Vec<(BlobRef, Vec<u8>)>,
    pub edges: Vec<(ContextUri, ContextUri, GraphRelation)>,
    pub vector: Option<IndexPoint>,
    pub created_at: DateTime<Utc>,
}

#[async_trait]
pub trait LifecycleGraphStore: GraphStore {
    /// Returns every incoming and outgoing edge touching `uri`, preserving direction and kind.
    async fn edges_for(
        &self,
        uri: &ContextUri,
    ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>>;
}

#[async_trait]
pub trait LifecycleBlobStore: BlobStore {
    /// Removes one reference. Implementations with shared blobs must retain data until unreferenced.
    async fn delete(&self, blob_ref: &BlobRef) -> Result<()>;
}

#[async_trait]
pub trait LifecycleVectorStore: VectorIndex {
    /// Returns the exact point needed to recreate this URI, or `None` when it is not indexed.
    async fn get_point(&self, collection: &str, uri: &ContextUri) -> Result<Option<IndexPoint>>;
}

#[async_trait]
pub trait ColdStorage: Send + Sync {
    async fn put_manifest(&self, key: &str, manifest: &ArchiveManifest) -> Result<()>;
    async fn get_manifest(&self, key: &str) -> Result<ArchiveManifest>;
}

/// Filesystem cold store using atomic rename. Existing keys are immutable and
/// compared byte-for-byte, making a repeated migration safe.
pub struct FileColdStorage {
    root: PathBuf,
    lock: Mutex<()>,
}
impl FileColdStorage {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            lock: Mutex::new(()),
        }
    }
    fn path(&self, key: &str) -> PathBuf {
        self.root
            .join(format!("{}.json", blake3::hash(key.as_bytes()).to_hex()))
    }
}
#[async_trait]
impl ColdStorage for FileColdStorage {
    async fn put_manifest(&self, key: &str, manifest: &ArchiveManifest) -> Result<()> {
        let _guard = self.lock.lock().await;
        std::fs::create_dir_all(&self.root)?;
        let bytes = serde_json::to_vec(manifest)?;
        let path = self.path(key);
        if path.exists() {
            if std::fs::read(path)? == bytes {
                return Ok(());
            }
            return Err(ContextError::VersionConflict(format!(
                "cold manifest key collision: {key}"
            )));
        }
        atomic_write(&path, &bytes)
    }
    async fn get_manifest(&self, key: &str) -> Result<ArchiveManifest> {
        Ok(serde_json::from_slice(&std::fs::read(self.path(key))?)?)
    }
}

/// Durable JSON journal. A single atomic snapshot is intentionally used: jobs
/// are small, while rename + fsync gives deterministic crash recovery without
/// requiring callers to provision another database.
pub struct FileLifecycleJournal {
    path: PathBuf,
    jobs: Mutex<BTreeMap<String, LifecycleJob>>,
}
impl FileLifecycleJournal {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let jobs = if path.exists() {
            serde_json::from_slice(&std::fs::read(&path)?)?
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            path,
            jobs: Mutex::new(jobs),
        })
    }
    fn lock_path(&self) -> PathBuf {
        self.path.with_extension("lock")
    }
    fn transact<T>(
        &self,
        f: impl FnOnce(&mut BTreeMap<String, LifecycleJob>) -> Result<T>,
    ) -> Result<T> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.lock_path())?;
        lock.lock_exclusive()?;
        let mut jobs = if self.path.exists() {
            serde_json::from_slice(&std::fs::read(&self.path)?)?
        } else {
            BTreeMap::new()
        };
        let result = f(&mut jobs)?;
        atomic_write(&self.path, &serde_json::to_vec(&jobs)?)?;
        lock.unlock()?;
        Ok(result)
    }
    pub async fn upsert(&self, job: LifecycleJob) -> Result<LifecycleJob> {
        let result = self.transact(|jobs| {
            if let Some(existing) = jobs.get(&job.idempotency_key) {
                return Ok(existing.clone());
            }
            jobs.insert(job.idempotency_key.clone(), job.clone());
            Ok(job)
        })?;
        *self.jobs.lock().await = self.load()?;
        Ok(result)
    }
    fn load(&self) -> Result<BTreeMap<String, LifecycleJob>> {
        if self.path.exists() {
            Ok(serde_json::from_slice(&std::fs::read(&self.path)?)?)
        } else {
            Ok(BTreeMap::new())
        }
    }
    async fn update(&self, job: &mut LifecycleJob) -> Result<()> {
        self.transact(|jobs| {
            let current = jobs
                .get(&job.idempotency_key)
                .ok_or_else(|| ContextError::NotFound(job.idempotency_key.clone()))?;
            if current.revision != job.revision {
                return Err(ContextError::VersionConflict(format!(
                    "stale lifecycle revision for {}",
                    job.idempotency_key
                )));
            }
            job.revision += 1;
            jobs.insert(job.idempotency_key.clone(), job.clone());
            Ok(())
        })
    }
    async fn claim(
        &self,
        key: &str,
        owner: Uuid,
        now: DateTime<Utc>,
    ) -> Result<Option<LifecycleJob>> {
        self.transact(|jobs| {
            let Some(job) = jobs.get_mut(key) else {
                return Ok(None);
            };
            if job.next_attempt_at > now
                || matches!(
                    job.state,
                    LifecycleJobState::Succeeded | LifecycleJobState::PermanentlyFailed
                )
                || job.lease_until.is_some_and(|until| until > now)
            {
                return Ok(None);
            }
            job.lease_owner = Some(owner);
            job.lease_until = Some(now + chrono::Duration::seconds(30));
            job.revision += 1;
            Ok(Some(job.clone()))
        })
    }
    pub async fn recoverable(&self, now: DateTime<Utc>) -> Vec<LifecycleJob> {
        self.load()
            .unwrap_or_default()
            .into_values()
            .filter(|j| {
                matches!(
                    j.state,
                    LifecycleJobState::Pending
                        | LifecycleJobState::Running
                        | LifecycleJobState::Retryable
                ) && j.next_attempt_at <= now
                    && j.lease_until.is_none_or(|until| until <= now)
            })
            .collect()
    }
    pub async fn get(&self, key: &str) -> Option<LifecycleJob> {
        self.load().ok()?.get(key).cloned()
    }
}

pub struct LifecycleActionExecutor {
    journal: Arc<FileLifecycleJournal>,
    content: Arc<dyn ContentStore>,
    graph: Arc<dyn LifecycleGraphStore>,
    vector: Arc<dyn LifecycleVectorStore>,
    blobs: Arc<dyn LifecycleBlobStore>,
    cold: Arc<dyn ColdStorage>,
    collection: String,
    key_locks: Mutex<BTreeMap<String, Arc<Mutex<()>>>>,
}
impl LifecycleActionExecutor {
    pub fn new(
        journal: Arc<FileLifecycleJournal>,
        content: Arc<dyn ContentStore>,
        graph: Arc<dyn LifecycleGraphStore>,
        vector: Arc<dyn LifecycleVectorStore>,
        blobs: Arc<dyn LifecycleBlobStore>,
        cold: Arc<dyn ColdStorage>,
        collection: impl Into<String>,
    ) -> Self {
        Self {
            journal,
            content,
            graph,
            vector,
            blobs,
            cold,
            collection: collection.into(),
            key_locks: Mutex::new(BTreeMap::new()),
        }
    }
    pub async fn submit(&self, uri: ContextUri, action: LifecycleAction) -> Result<LifecycleJob> {
        let operation = match action {
            LifecycleAction::Archive => LifecycleOperation::Archive,
            LifecycleAction::Delete => LifecycleOperation::Delete,
            LifecycleAction::Downgrade {
                to_level: ContentLevel::L0 | ContentLevel::L1,
            } => LifecycleOperation::ColdStorage,
            _ => {
                return Err(ContextError::Unsupported(
                    "lifecycle action has no destructive execution".into(),
                ));
            }
        };
        let key = format!("{}:{operation:?}", uri);
        let now = Utc::now();
        let job = LifecycleJob {
            id: Uuid::new_v4(),
            idempotency_key: key,
            uri,
            operation,
            state: LifecycleJobState::Pending,
            attempts: 0,
            max_attempts: 5,
            next_attempt_at: now,
            checkpoint: Default::default(),
            last_error: None,
            audit: vec![LifecycleAudit {
                at: now,
                state: LifecycleJobState::Pending,
                message: "accepted".into(),
            }],
            revision: 0,
            lease_owner: None,
            lease_until: None,
            restore_checkpoint: Default::default(),
        };
        self.journal.upsert(job).await
    }
    pub async fn route_metacog(&self, entry: &ContextEntry) -> Result<Option<LifecycleJob>> {
        match metacog_tier(entry)? {
            MetacogTier::Hot => Ok(None),
            MetacogTier::Cold => self
                .submit(
                    entry.uri.clone(),
                    LifecycleAction::Downgrade {
                        to_level: ContentLevel::L1,
                    },
                )
                .await
                .map(Some),
        }
    }

    pub async fn run_pending(&self) -> Result<Vec<LifecycleJob>> {
        let mut out = Vec::new();
        let owner = Uuid::new_v4();
        for j in self.journal.recoverable(Utc::now()).await {
            let Some(current) = self
                .journal
                .claim(&j.idempotency_key, owner, Utc::now())
                .await?
            else {
                continue;
            };
            out.push(self.execute(current).await?);
        }
        Ok(out)
    }

    async fn lock_for(&self, key: &str) -> Arc<Mutex<()>> {
        let mut locks = self.key_locks.lock().await;
        locks
            .entry(key.to_owned())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
    async fn execute(&self, mut job: LifecycleJob) -> Result<LifecycleJob> {
        job.state = LifecycleJobState::Running;
        job.attempts += 1;
        audit(&mut job, "attempt started");
        // The running checkpoint must be durable before any destructive backend operation.
        self.journal.update(&mut job).await?;
        let result = self.execute_phases(&mut job).await;
        match result {
            Ok(()) => {
                job.state = LifecycleJobState::Succeeded;
                job.last_error = None;
                audit(&mut job, "all references committed");
            }
            Err(e) => {
                job.last_error = Some(e.to_string());
                job.state = if job.attempts >= job.max_attempts {
                    LifecycleJobState::PermanentlyFailed
                } else {
                    LifecycleJobState::Retryable
                };
                job.next_attempt_at =
                    Utc::now() + chrono::Duration::seconds(1_i64 << job.attempts.min(10));
                audit(&mut job, &format!("execution failed: {e}"));
            }
        }
        // Release the durable lease as part of the final CAS transition.
        job.lease_owner = None;
        job.lease_until = None;
        // Never report a state transition that failed to reach durable storage.
        self.journal.update(&mut job).await?;
        Ok(job)
    }
    async fn execute_phases(&self, job: &mut LifecycleJob) -> Result<()> {
        let key = job.idempotency_key.clone();
        if !job.checkpoint.manifest {
            let entry = self.read_entry(&job.uri).await?;
            let blobs = collect_blobs(&entry.payload);
            let mut blob_data = Vec::with_capacity(blobs.len());
            for b in blobs {
                blob_data.push((b.clone(), self.blobs.get(&b).await?));
            }
            let edges = self.edges(&job.uri).await?;
            let vector = self.vector.get_point(&self.collection, &job.uri).await?;
            self.cold
                .put_manifest(
                    &key,
                    &ArchiveManifest {
                        job_id: job.id,
                        entry,
                        blobs: blob_data,
                        edges,
                        vector,
                        created_at: Utc::now(),
                    },
                )
                .await?;
            job.checkpoint.manifest = true;
            self.journal.update(job).await?;
        }
        if !job.checkpoint.vector {
            self.vector.delete(&self.collection, &job.uri).await?;
            job.checkpoint.vector = true;
            self.journal.update(job).await?;
        }
        if !job.checkpoint.graph {
            let edges = self.cold.get_manifest(&key).await?.edges;
            for (from, to, _) in edges {
                self.graph.remove_edge(&from, &to).await?;
            }
            job.checkpoint.graph = true;
            self.journal.update(job).await?;
        }
        if !job.checkpoint.content {
            self.content.delete(&job.uri).await?;
            job.checkpoint.content = true;
            self.journal.update(job).await?;
        }
        if !job.checkpoint.blobs {
            let blobs = self.cold.get_manifest(&key).await?.blobs;
            for (blob, _) in blobs {
                self.blobs.delete(&blob).await?;
            }
            job.checkpoint.blobs = true;
            self.journal.update(job).await?;
        }
        Ok(())
    }
    async fn read_entry(&self, uri: &ContextUri) -> Result<ContextEntry> {
        self.content
            .scan_by_prefix(&uri.to_string(), PageRequest::new(1))
            .await?
            .into_iter()
            .find(|e| e.uri == *uri)
            .ok_or_else(|| ContextError::NotFound(uri.to_string()))
    }
    async fn edges(
        &self,
        uri: &ContextUri,
    ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
        self.graph.edges_for(uri).await
    }
    pub async fn restore(&self, key: &str) -> Result<()> {
        let lock = self.lock_for(key).await;
        let _guard = lock.lock().await;
        let m = self.cold.get_manifest(key).await?;
        if let Some(job) = self.journal.get(key).await
            && job.state != LifecycleJobState::Succeeded
        {
            return Err(ContextError::VersionConflict(format!(
                "cannot restore incomplete lifecycle job {key}"
            )));
        }
        let mut job = self
            .journal
            .get(key)
            .await
            .ok_or_else(|| ContextError::NotFound(key.into()))?;
        if !job.restore_checkpoint.blobs {
            for (r, data) in &m.blobs {
                let written = self.blobs.put(data, &r.mime_type).await?;
                if written.hash != r.hash {
                    return Err(ContextError::Storage("restored blob hash mismatch".into()));
                }
            }
            job.restore_checkpoint.blobs = true;
            self.journal.update(&mut job).await?;
        }
        if !job.restore_checkpoint.content {
            match self.read_entry(&m.entry.uri).await {
                Ok(existing) if serde_json::to_vec(&existing)? != serde_json::to_vec(&m.entry)? => {
                    return Err(ContextError::VersionConflict(format!(
                        "hot entry changed since cold snapshot: {}",
                        m.entry.uri
                    )));
                }
                Ok(_) => {}
                Err(ContextError::NotFound(_)) => {
                    self.content.write(m.entry.clone()).await?;
                }
                Err(error) => return Err(error),
            }
            job.restore_checkpoint.content = true;
            self.journal.update(&mut job).await?;
        }
        if !job.restore_checkpoint.graph {
            for (from, to, kind) in &m.edges {
                self.graph.add_edge(from, to, *kind).await?;
            }
            job.restore_checkpoint.graph = true;
            self.journal.update(&mut job).await?;
        }
        if !job.restore_checkpoint.vector {
            if let Some(point) = m.vector {
                self.vector.upsert(&self.collection, point).await?;
            }
            job.restore_checkpoint.vector = true;
            self.journal.update(&mut job).await?;
        }
        Ok(())
    }
}

fn collect_blobs(payload: &ContentPayload) -> Vec<BlobRef> {
    let mut out = Vec::new();
    match payload {
        ContentPayload::Image { raw, .. } | ContentPayload::Audio { raw, .. } => {
            out.push(raw.clone())
        }
        ContentPayload::Structured {
            schema: Some(s), ..
        } => {
            if let Some(b) = &s.blob {
                out.push(b.clone())
            }
        }
        ContentPayload::Composite { parts, .. } => {
            for p in parts {
                match p {
                    ContentPart::Text(v) | ContentPart::Image(v) | ContentPart::Audio(v) => {
                        out.extend(collect_blobs(v))
                    }
                    ContentPart::Reference(_) => {}
                }
            }
        }
        _ => {}
    }
    out
}
fn audit(job: &mut LifecycleJob, message: &str) {
    job.audit.push(LifecycleAudit {
        at: Utc::now(),
        state: job.state,
        message: message.into(),
    });
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{
        ContentHash, ContentType, IndexHit, MediaType, MvccVersion, Page, SchemaRef, TenantId,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct MemoryPorts {
        entries: Mutex<HashMap<String, ContextEntry>>,
        edges: Mutex<Vec<(ContextUri, ContextUri, GraphRelation)>>,
        vectors: Mutex<HashMap<(String, String), IndexPoint>>,
        blobs: Mutex<HashMap<String, (BlobRef, Vec<u8>)>>,
        fail_vector_deletes: AtomicUsize,
    }

    impl MemoryPorts {
        async fn contains_entry(&self, uri: &ContextUri) -> bool {
            self.entries.lock().await.contains_key(&uri.to_string())
        }
        async fn contains_blob(&self, blob: &BlobRef) -> bool {
            self.blobs.lock().await.contains_key(&blob.hash.0)
        }
    }

    #[async_trait]
    impl ContentStore for MemoryPorts {
        async fn read(&self, uri: &ContextUri, _level: ContentLevel) -> Result<ContentPayload> {
            self.entries
                .lock()
                .await
                .get(&uri.to_string())
                .map(|e| e.payload.clone())
                .ok_or_else(|| ContextError::NotFound(uri.to_string()))
        }
        async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
            self.entries
                .lock()
                .await
                .insert(entry.uri.to_string(), entry);
            Ok(MvccVersion(1))
        }
        async fn delete(&self, uri: &ContextUri) -> Result<()> {
            self.entries.lock().await.remove(&uri.to_string());
            Ok(())
        }
        async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
            let mut entries = self.entries.lock().await;
            let mut entry = entries
                .remove(&from.to_string())
                .ok_or_else(|| ContextError::NotFound(from.to_string()))?;
            entry.uri = to.clone();
            entries.insert(to.to_string(), entry);
            Ok(())
        }
        async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
            for entry in entries {
                self.write(entry.clone()).await?;
            }
            Ok(vec![MvccVersion(1); entries.len()])
        }
        async fn scan_by_prefix(
            &self,
            prefix: &str,
            page: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            let mut values: Vec<_> = self
                .entries
                .lock()
                .await
                .values()
                .filter(|e| e.uri.to_string().starts_with(prefix))
                .cloned()
                .collect();
            values.sort_by(|a, b| a.uri.cmp(&b.uri));
            values.truncate(page.effective_limit());
            Ok(Page::new(values, None))
        }
        async fn scan_by_type(
            &self,
            prefix: &str,
            kind: ContentType,
            page: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            let page = self.scan_by_prefix(prefix, page).await?;
            Ok(Page::new(
                page.items
                    .into_iter()
                    .filter(|e| e.metadata.content_type == Some(kind))
                    .collect(),
                None,
            ))
        }
    }

    #[async_trait]
    impl GraphStore for MemoryPorts {
        async fn add_edge(
            &self,
            from: &ContextUri,
            to: &ContextUri,
            kind: GraphRelation,
        ) -> Result<()> {
            let edge = (from.clone(), to.clone(), kind);
            let mut edges = self.edges.lock().await;
            if !edges.contains(&edge) {
                edges.push(edge);
            }
            Ok(())
        }
        async fn remove_edge(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
            self.edges
                .lock()
                .await
                .retain(|(f, t, _)| f != from || t != to);
            Ok(())
        }
        async fn outgoing_neighbors(
            &self,
            uri: &ContextUri,
            kind: Option<GraphRelation>,
        ) -> Result<Vec<ContextUri>> {
            Ok(self
                .edges
                .lock()
                .await
                .iter()
                .filter(|(f, _, k)| f == uri && kind.is_none_or(|v| v == *k))
                .map(|(_, t, _)| t.clone())
                .collect())
        }
        async fn batch_traverse(
            &self,
            seeds: &[ContextUri],
            kinds: &[GraphRelation],
            _max_hops: usize,
        ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
            Ok(self
                .edges
                .lock()
                .await
                .iter()
                .filter(|(f, _, k)| seeds.contains(f) && (kinds.is_empty() || kinds.contains(k)))
                .cloned()
                .collect())
        }
        async fn centrality(&self, uri: &ContextUri) -> Result<f32> {
            Ok(self
                .edges
                .lock()
                .await
                .iter()
                .filter(|(f, t, _)| f == uri || t == uri)
                .count() as f32)
        }
    }

    #[async_trait]
    impl LifecycleGraphStore for MemoryPorts {
        async fn edges_for(
            &self,
            uri: &ContextUri,
        ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
            Ok(self
                .edges
                .lock()
                .await
                .iter()
                .filter(|(f, t, _)| f == uri || t == uri)
                .cloned()
                .collect())
        }
    }

    #[async_trait]
    impl VectorIndex for MemoryPorts {
        async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()> {
            self.vectors
                .lock()
                .await
                .insert((collection.into(), point.uri.to_string()), point);
            Ok(())
        }
        async fn search(
            &self,
            _collection: &str,
            _query: Vec<f32>,
            _top_k: usize,
            _filter: Option<serde_json::Value>,
        ) -> Result<Vec<IndexHit>> {
            Ok(vec![])
        }
        async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()> {
            if self
                .fail_vector_deletes
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                return Err(ContextError::Storage("injected vector failure".into()));
            }
            self.vectors
                .lock()
                .await
                .remove(&(collection.into(), uri.to_string()));
            Ok(())
        }
    }

    #[async_trait]
    impl LifecycleVectorStore for MemoryPorts {
        async fn get_point(
            &self,
            collection: &str,
            uri: &ContextUri,
        ) -> Result<Option<IndexPoint>> {
            Ok(self
                .vectors
                .lock()
                .await
                .get(&(collection.into(), uri.to_string()))
                .cloned())
        }
    }

    #[async_trait]
    impl BlobStore for MemoryPorts {
        async fn put(&self, data: &[u8], mime_type: &str) -> Result<BlobRef> {
            let blob = BlobRef {
                hash: ContentHash(blake3::hash(data).to_hex().to_string()),
                size: data.len(),
                mime_type: mime_type.into(),
            };
            self.blobs
                .lock()
                .await
                .insert(blob.hash.0.clone(), (blob.clone(), data.to_vec()));
            Ok(blob)
        }
        async fn get(&self, blob: &BlobRef) -> Result<Vec<u8>> {
            self.blobs
                .lock()
                .await
                .get(&blob.hash.0)
                .map(|(_, bytes)| bytes.clone())
                .ok_or_else(|| ContextError::NotFound(blob.hash.0.clone()))
        }
        async fn dedup_check(&self, hash: &ContentHash) -> Result<bool> {
            Ok(self.blobs.lock().await.contains_key(&hash.0))
        }
    }

    #[async_trait]
    impl LifecycleBlobStore for MemoryPorts {
        async fn delete(&self, blob: &BlobRef) -> Result<()> {
            self.blobs.lock().await.remove(&blob.hash.0);
            Ok(())
        }
    }

    struct Fixture {
        root: PathBuf,
        ports: Arc<MemoryPorts>,
        journal: Arc<FileLifecycleJournal>,
        executor: LifecycleActionExecutor,
        uri: ContextUri,
        incoming: ContextUri,
        outgoing: ContextUri,
        blob: BlobRef,
        entry: ContextEntry,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    async fn fixture() -> Fixture {
        let root = std::env::temp_dir().join(format!("context-db-lifecycle-{}", Uuid::new_v4()));
        let ports = Arc::new(MemoryPorts::default());
        let tenant = TenantId(Uuid::new_v4());
        let uri = ContextUri::parse(format!("uwu://{}/agent/test/memory/item", tenant.0)).unwrap();
        let incoming =
            ContextUri::parse(format!("uwu://{}/agent/test/memory/incoming", tenant.0)).unwrap();
        let outgoing =
            ContextUri::parse(format!("uwu://{}/agent/test/memory/outgoing", tenant.0)).unwrap();
        let blob = ports
            .put(b"schema-data", "application/schema+json")
            .await
            .unwrap();
        let mut entry = ContextEntry::new_text(uri.clone(), tenant, "complete entry");
        entry.media_type = MediaType::Binary;
        entry.payload = ContentPayload::Structured {
            summary: "summary".into(),
            schema: Some(SchemaRef::json_schema(blob.clone())),
            data: serde_json::json!({"complete": true}),
        };
        ports.write(entry.clone()).await.unwrap();
        ports
            .add_edge(&incoming, &uri, GraphRelation::EvidenceOf)
            .await
            .unwrap();
        ports
            .add_edge(&uri, &outgoing, GraphRelation::DerivedFrom)
            .await
            .unwrap();
        ports
            .upsert(
                "memory",
                IndexPoint {
                    uri: uri.clone(),
                    vector: vec![0.25, 0.75],
                    embedding_model_id: Some("test-model".into()),
                    embedding_dim: Some(2),
                    embedding_version: Some(7),
                    payload: serde_json::json!({"kind": "memory"}),
                },
            )
            .await
            .unwrap();
        let journal = Arc::new(FileLifecycleJournal::open(root.join("journal.json")).unwrap());
        let cold = Arc::new(FileColdStorage::new(root.join("cold")));
        let executor = LifecycleActionExecutor::new(
            journal.clone(),
            ports.clone(),
            ports.clone(),
            ports.clone(),
            ports.clone(),
            cold,
            "memory",
        );
        Fixture {
            root,
            ports,
            journal,
            executor,
            uri,
            incoming,
            outgoing,
            blob,
            entry,
        }
    }

    #[test]
    fn metacog_heat_model_is_namespace_specific_and_deterministic() {
        let tenant = TenantId(Uuid::new_v4());
        let mut entry = ContextEntry::new_text(
            ContextUri::parse(format!("uwu://{}/metacog/reflection/item", tenant.0)).unwrap(),
            tenant,
            "reflection",
        );
        entry.metadata.state_scope = Some(StateScope::Mid);
        assert_eq!(metacog_tier(&entry).unwrap(), MetacogTier::Cold);
        entry.metadata.state_scope = Some(StateScope::Long);
        assert_eq!(metacog_tier(&entry).unwrap(), MetacogTier::Hot);
        entry.uri = ContextUri::parse(format!("uwu://{}/state/runtime/item", tenant.0)).unwrap();
        assert!(metacog_tier(&entry).is_err());
    }

    #[tokio::test]
    async fn submit_persists_pending_job() {
        let f = fixture().await;
        let job = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Archive)
            .await
            .unwrap();
        assert_eq!(job.state, LifecycleJobState::Pending);
        assert_eq!(
            f.journal.get(&job.idempotency_key).await.unwrap().id,
            job.id
        );
        assert!(f.root.join("journal.json").is_file());
    }

    #[tokio::test]
    async fn duplicate_submit_is_idempotent() {
        let f = fixture().await;
        let first = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Archive)
            .await
            .unwrap();
        let second = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Archive)
            .await
            .unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(second.audit.len(), 1);
    }

    #[tokio::test]
    async fn failed_step_becomes_retryable_with_checkpoint_and_audit() {
        let f = fixture().await;
        f.ports.fail_vector_deletes.store(1, Ordering::SeqCst);
        let submitted = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Archive)
            .await
            .unwrap();
        let result = f.executor.run_pending().await.unwrap().pop().unwrap();
        assert_eq!(result.state, LifecycleJobState::Retryable);
        assert!(result.checkpoint.manifest);
        assert!(!result.checkpoint.vector);
        assert!(
            result
                .last_error
                .as_deref()
                .unwrap()
                .contains("injected vector failure")
        );
        assert_eq!(
            f.journal
                .get(&submitted.idempotency_key)
                .await
                .unwrap()
                .attempts,
            1
        );
    }

    #[tokio::test]
    async fn retry_resumes_and_succeeds() {
        let f = fixture().await;
        f.ports.fail_vector_deletes.store(1, Ordering::SeqCst);
        let _submitted = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Archive)
            .await
            .unwrap();
        let failed = f.executor.run_pending().await.unwrap().pop().unwrap();
        let mut due = failed;
        due.next_attempt_at = Utc::now() - chrono::Duration::seconds(1);
        f.journal.update(&mut due).await.unwrap();
        let retried = f.executor.run_pending().await.unwrap().pop().unwrap();
        assert_eq!(retried.state, LifecycleJobState::Succeeded);
        assert_eq!(retried.attempts, 2);
        assert!(retried.checkpoint.blobs);
        assert!(!f.ports.contains_entry(&f.uri).await);
    }

    #[tokio::test]
    async fn reopened_journal_recovers_pending_job() {
        let f = fixture().await;
        let submitted = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Archive)
            .await
            .unwrap();
        let reopened = Arc::new(FileLifecycleJournal::open(f.root.join("journal.json")).unwrap());
        let cold = Arc::new(FileColdStorage::new(f.root.join("cold")));
        let executor = LifecycleActionExecutor::new(
            reopened.clone(),
            f.ports.clone(),
            f.ports.clone(),
            f.ports.clone(),
            f.ports.clone(),
            cold,
            "memory",
        );
        let recovered = executor.run_pending().await.unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].id, submitted.id);
        assert_eq!(recovered[0].state, LifecycleJobState::Succeeded);
        assert_eq!(
            reopened
                .get(&submitted.idempotency_key)
                .await
                .unwrap()
                .state,
            LifecycleJobState::Succeeded
        );
    }

    #[tokio::test]
    async fn restore_recovers_complete_entry_vector_bidirectional_graph_and_blob() {
        let f = fixture().await;
        let submitted = f
            .executor
            .submit(f.uri.clone(), LifecycleAction::Delete)
            .await
            .unwrap();
        let completed = f.executor.run_pending().await.unwrap().pop().unwrap();
        assert_eq!(completed.state, LifecycleJobState::Succeeded);
        assert!(!f.ports.contains_entry(&f.uri).await);
        assert!(!f.ports.contains_blob(&f.blob).await);
        assert!(f.ports.edges.lock().await.is_empty());
        assert!(f.ports.get_point("memory", &f.uri).await.unwrap().is_none());

        f.executor
            .restore(&submitted.idempotency_key)
            .await
            .unwrap();
        let restored = f
            .ports
            .entries
            .lock()
            .await
            .get(&f.uri.to_string())
            .cloned()
            .unwrap();
        assert_eq!(
            serde_json::to_value(&restored.payload).unwrap(),
            serde_json::to_value(&f.entry.payload).unwrap()
        );
        assert_eq!(restored.media_type, f.entry.media_type);
        assert!(f.ports.contains_blob(&f.blob).await);
        let point = f.ports.get_point("memory", &f.uri).await.unwrap().unwrap();
        assert_eq!(point.vector, vec![0.25, 0.75]);
        assert_eq!(point.embedding_version, Some(7));
        let edges = f.ports.edges_for(&f.uri).await.unwrap();
        assert!(edges.contains(&(f.incoming.clone(), f.uri.clone(), GraphRelation::EvidenceOf)));
        assert!(edges.contains(&(
            f.uri.clone(),
            f.outgoing.clone(),
            GraphRelation::DerivedFrom
        )));

        // Repeated and concurrent on-demand reads converge without duplicate graph state.
        let key = submitted.idempotency_key.clone();
        let (a, b) = tokio::join!(f.executor.restore(&key), f.executor.restore(&key));
        a.unwrap();
        b.unwrap();
        assert_eq!(f.ports.edges_for(&f.uri).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn metacog_cold_to_hot_closed_loop() {
        let mut f = fixture().await;
        f.entry.uri = ContextUri::parse(format!(
            "uwu://{}/metacog/reflection/item",
            f.entry.tenant.0
        ))
        .unwrap();
        f.entry.metadata.state_scope = Some(StateScope::Mid);
        f.uri = f.entry.uri.clone();
        f.ports.write(f.entry.clone()).await.unwrap();
        let job = f.executor.route_metacog(&f.entry).await.unwrap().unwrap();
        let completed = f.executor.run_pending().await.unwrap();
        assert_eq!(completed[0].state, LifecycleJobState::Succeeded);
        assert!(!f.ports.contains_entry(&f.uri).await);
        f.executor.restore(&job.idempotency_key).await.unwrap();
        assert!(f.ports.contains_entry(&f.uri).await);
    }
}
