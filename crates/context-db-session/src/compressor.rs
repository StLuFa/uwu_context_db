//! 持久化会话压缩状态机及正式语义编排。

use crate::{
    CommitTaskId, DoneMarker, FailureMetadata, MemoryChange, MemoryDiff, SessionHandle, TaskStatus,
};
use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentRepo, ContextEntry, ContextError, ContextMeta, ContextUri,
    FsOps, MediaType, MvccVersion, Result, TenantId,
};
use agent_context_db_parse::{CandidateAction, MemoryExtractor, SemanticProcessor};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, sync::Arc};

pub trait SessionTaskStore: ContentRepo + FsOps {}
impl<T: ContentRepo + FsOps> SessionTaskStore for T {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskRecord {
    task_id: CommitTaskId,
    session: SessionHandle,
    archive_uri: ContextUri,
    status: TaskStatus,
    attempt: u32,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

pub struct SessionCompressorImpl {
    store: Arc<dyn SessionTaskStore>,
    extractor: Arc<dyn MemoryExtractor>,
    semantic: Arc<dyn SemanticProcessor>,
    active: Mutex<HashSet<CommitTaskId>>,
}

impl SessionCompressorImpl {
    pub fn new(
        store: Arc<dyn SessionTaskStore>,
        extractor: Arc<dyn MemoryExtractor>,
        semantic: Arc<dyn SemanticProcessor>,
    ) -> Self {
        Self {
            store,
            extractor,
            semantic,
            active: Mutex::new(HashSet::new()),
        }
    }

    pub fn archive_file_uri(session: &SessionHandle) -> ContextUri {
        session
            .archive_dir
            .join(&session.compression_index.to_string())
            .join("messages.jsonl")
    }

    fn state_uri(session: &SessionHandle, task_id: CommitTaskId) -> ContextUri {
        session
            .archive_dir
            .join("tasks")
            .join(&format!("{}.json", task_id.0))
    }

    fn diff_uri(session: &SessionHandle) -> ContextUri {
        session
            .archive_dir
            .join(&session.compression_index.to_string())
            .join("memory_diff.json")
    }

    fn done_uri(session: &SessionHandle) -> ContextUri {
        session
            .archive_dir
            .join(&session.compression_index.to_string())
            .join(".done")
    }

    fn task_id(session: &SessionHandle) -> CommitTaskId {
        let key = format!(
            "{}:{}:{}",
            session.session_id, session.compression_index, session.archive_dir
        );
        CommitTaskId(uuid::Uuid::new_v5(
            &uuid::Uuid::NAMESPACE_URL,
            key.as_bytes(),
        ))
    }

    pub async fn enqueue(&self, session: &SessionHandle) -> Result<CommitTaskId> {
        let task_id = Self::task_id(session);
        if self
            .load_record(&Self::state_uri(session, task_id))
            .await
            .is_ok()
        {
            return Ok(task_id);
        }
        let archive_uri = Self::archive_file_uri(session);
        self.store
            .write(text_entry(archive_uri.clone(), archive_payload(session)?))
            .await?;
        let now = chrono::Utc::now();
        let record = TaskRecord {
            task_id,
            session: session.clone(),
            archive_uri,
            status: TaskStatus::Pending,
            attempt: 0,
            created_at: now,
            updated_at: now,
        };
        self.save_record(&record).await?;
        Ok(task_id)
    }

    pub async fn run(&self, session: &SessionHandle) -> Result<DoneMarker> {
        let task_id = self.enqueue(session).await?;
        self.process(session, task_id).await
    }

    pub async fn retry(
        &self,
        session: &SessionHandle,
        task_id: CommitTaskId,
    ) -> Result<DoneMarker> {
        self.process(session, task_id).await
    }

    pub async fn recover(&self, session: &SessionHandle) -> Result<DoneMarker> {
        let task_id = Self::task_id(session);
        self.process(session, task_id).await
    }

    pub async fn poll(&self, session: &SessionHandle, task_id: CommitTaskId) -> Result<TaskStatus> {
        Ok(self
            .load_record(&Self::state_uri(session, task_id))
            .await?
            .status)
    }

    async fn process(&self, session: &SessionHandle, task_id: CommitTaskId) -> Result<DoneMarker> {
        let state_uri = Self::state_uri(session, task_id);
        let mut record = self.load_record(&state_uri).await?;
        if let TaskStatus::Done(done) = record.status {
            return Ok(done);
        }
        if !self.active.lock().insert(task_id) {
            return Err(ContextError::VersionConflict(
                "compression task is already processing".into(),
            ));
        }
        let result = self.process_inner(&mut record).await;
        self.active.lock().remove(&task_id);
        result
    }

    async fn process_inner(&self, record: &mut TaskRecord) -> Result<DoneMarker> {
        record.attempt += 1;
        record.updated_at = chrono::Utc::now();
        record.status = TaskStatus::Processing {
            attempt: record.attempt,
            started_at: record.updated_at,
        };
        self.save_record(record).await?;

        let result = self.execute_pipeline(record).await;
        match result {
            Ok(done) => {
                // All artifacts are durable before this single persisted terminal transition.
                record.status = TaskStatus::Done(done.clone());
                record.updated_at = done.finished_at;
                self.save_record(record).await?;
                Ok(done)
            }
            Err(error) => {
                record.updated_at = chrono::Utc::now();
                record.status = TaskStatus::Failed(FailureMetadata {
                    message: error.to_string(),
                    attempt: record.attempt,
                    failed_at: record.updated_at,
                    retryable: true,
                });
                self.save_record(record).await?;
                Err(error)
            }
        }
    }

    async fn execute_pipeline(&self, record: &TaskRecord) -> Result<DoneMarker> {
        let candidates = self.extractor.extract(&record.archive_uri).await?;
        let decisions = self.extractor.deduplicate(candidates).await?;
        let mut diff = MemoryDiff::default();
        for decision in decisions {
            match decision.action {
                CandidateAction::Create => {
                    let target_uri = decision
                        .candidate
                        .source_uri
                        .join(&format!("memories/{}", uuid::Uuid::new_v4()));
                    let abstract_text = self.semantic.generate_abstract(&target_uri).await?;
                    diff.adds.push(MemoryChange {
                        uri: target_uri,
                        content_type: decision.candidate.content_type,
                        before: None,
                        after: Some(serde_json::json!({"abstract": abstract_text})),
                        reason: decision.reason,
                    });
                }
                CandidateAction::Merge => diff.updates.push(MemoryChange {
                    uri: decision.merge_target.ok_or_else(|| {
                        ContextError::Storage("validated merge decision lost merge target".into())
                    })?,
                    content_type: decision.candidate.content_type,
                    before: None,
                    after: Some(serde_json::json!({"merged": true})),
                    reason: decision.reason,
                }),
                CandidateAction::Skip => {}
            }
        }
        let parent = record
            .archive_uri
            .parent()
            .ok_or_else(|| ContextError::InvalidUri("archive has no parent".into()))?;
        let overview = self.semantic.aggregate_upward(&parent).await?;
        if overview.trim().is_empty() {
            return Err(ContextError::Storage(
                "semantic processor returned an empty overview".into(),
            ));
        }
        let diff_uri = Self::diff_uri(&record.session);
        self.store
            .write(text_entry(diff_uri.clone(), serde_json::to_string(&diff)?))
            .await?;
        let done = DoneMarker {
            task_id: record.task_id,
            finished_at: chrono::Utc::now(),
            abstract_uri: record.archive_uri.clone(),
            overview_uri: parent,
            memory_diff_uri: diff_uri,
        };
        self.store
            .write(text_entry(done.overview_uri.clone(), overview))
            .await?;
        self.store
            .write(text_entry(
                Self::done_uri(&record.session),
                serde_json::to_string(&done)?,
            ))
            .await?;
        Ok(done)
    }

    async fn load_record(&self, uri: &ContextUri) -> Result<TaskRecord> {
        let payload = self.store.read(uri, ContentLevel::L2).await?;
        let text = payload_text(payload);
        serde_json::from_str(&text).map_err(Into::into)
    }

    async fn save_record(&self, record: &TaskRecord) -> Result<()> {
        let uri = Self::state_uri(&record.session, record.task_id);
        self.store
            .write(text_entry(uri, serde_json::to_string(record)?))
            .await?;
        Ok(())
    }
}

fn payload_text(payload: ContentPayload) -> String {
    match payload {
        ContentPayload::Text { full, .. } => full,
        _ => String::new(),
    }
}

fn text_entry(uri: ContextUri, text: String) -> ContextEntry {
    let now = chrono::Utc::now();
    ContextEntry {
        uri,
        tenant: TenantId(uuid::Uuid::nil()),
        payload: ContentPayload::Text {
            sparse: text.clone(),
            dense: text.clone(),
            full: text,
        },
        media_type: MediaType::Text,
        metadata: ContextMeta::default(),
        mvcc_version: MvccVersion(0),
        created_at: now,
        updated_at: now,
        derivation: None,
    }
}

fn archive_payload(session: &SessionHandle) -> Result<String> {
    let mut jsonl = String::new();
    for message in &session.messages {
        jsonl.push_str(&serde_json::to_string(message)?);
        jsonl.push('\n');
    }
    let compressed = zstd::encode_all(jsonl.as_bytes(), 3).map_err(ContextError::Io)?;
    Ok(format!("zstd+base64:{}", BASE64.encode(compressed)))
}

#[cfg(test)]
mod tests;
