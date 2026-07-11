use super::*;
use crate::{Role, SessionMessage};
use agent_context_db_core::{
    ContentType, DirEntry, FindPattern, GrepHit, Page, PageRequest, TreeNode,
};
use agent_context_db_parse::{CandidateAction, DedupDecision, MemoryCandidate};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::{Barrier, Notify};

#[derive(Default)]
struct PersistentStore {
    entries: Mutex<HashMap<String, ContextEntry>>,
    fail_suffix_once: Mutex<Option<String>>,
}

impl PersistentStore {
    fn fail_once(&self, suffix: &str) {
        *self.fail_suffix_once.lock() = Some(suffix.into());
    }
}

#[async_trait]
impl ContentRepo for PersistentStore {
    async fn write(&self, mut entry: ContextEntry) -> Result<MvccVersion> {
        let fail = self
            .fail_suffix_once
            .lock()
            .as_ref()
            .is_some_and(|suffix| entry.uri.to_string().ends_with(suffix));
        if fail {
            self.fail_suffix_once.lock().take();
            return Err(ContextError::Storage(
                "injected output write failure".into(),
            ));
        }
        entry.mvcc_version = MvccVersion(1);
        self.entries.lock().insert(entry.uri.to_string(), entry);
        Ok(MvccVersion(1))
    }
    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.entries.lock().remove(&uri.to_string());
        Ok(())
    }
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        let mut entries = self.entries.lock();
        let entry = entries
            .remove(&from.to_string())
            .ok_or_else(|| ContextError::NotFound(from.to_string()))?;
        entries.insert(
            to.to_string(),
            ContextEntry {
                uri: to.clone(),
                ..entry
            },
        );
        Ok(())
    }
}

#[async_trait]
impl FsOps for PersistentStore {
    async fn ls(&self, _: &ContextUri, _: PageRequest) -> Result<Page<DirEntry>> {
        Ok(Page::new(vec![], None))
    }
    async fn find(&self, _: &FindPattern, _: PageRequest) -> Result<Page<ContextUri>> {
        Ok(Page::new(vec![], None))
    }
    async fn grep(&self, _: &str, _: &ContextUri) -> Result<Vec<GrepHit>> {
        Ok(vec![])
    }
    async fn tree(&self, root: &ContextUri, _: usize, _: PageRequest) -> Result<Page<TreeNode>> {
        Ok(Page::new(
            vec![TreeNode {
                uri: root.clone(),
                is_dir: true,
                children: vec![],
            }],
            None,
        ))
    }
    async fn read(&self, uri: &ContextUri, _: ContentLevel) -> Result<ContentPayload> {
        self.entries
            .lock()
            .get(&uri.to_string())
            .map(|entry| entry.payload.clone())
            .ok_or_else(|| ContextError::NotFound(uri.to_string()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FailStep {
    None,
    Extract,
    Deduplicate,
    Abstract,
    Aggregate,
}

struct TestPipeline {
    fail: Mutex<FailStep>,
    calls: AtomicUsize,
    entered: Option<Arc<Barrier>>,
    release: Option<Arc<Notify>>,
    block_once: AtomicBool,
}

impl TestPipeline {
    fn new(fail: FailStep) -> Self {
        Self {
            fail: Mutex::new(fail),
            calls: AtomicUsize::new(0),
            entered: None,
            release: None,
            block_once: AtomicBool::new(false),
        }
    }
    fn blocking(entered: Arc<Barrier>, release: Arc<Notify>) -> Self {
        Self {
            fail: Mutex::new(FailStep::None),
            calls: AtomicUsize::new(0),
            entered: Some(entered),
            release: Some(release),
            block_once: AtomicBool::new(true),
        }
    }
    fn succeed(&self) {
        *self.fail.lock() = FailStep::None;
    }
    fn fail_if(&self, step: FailStep) -> Result<()> {
        if *self.fail.lock() == step {
            Err(ContextError::Storage("pipeline failure".into()))
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl MemoryExtractor for TestPipeline {
    async fn extract(&self, archive: &ContextUri) -> Result<Vec<MemoryCandidate>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.block_once.swap(false, Ordering::SeqCst) {
            self.entered.as_ref().unwrap().wait().await;
            self.release.as_ref().unwrap().notified().await;
        }
        self.fail_if(FailStep::Extract)?;
        Ok(vec![MemoryCandidate {
            content_type: ContentType::Fact,
            content: "memory".into(),
            source_uri: archive.clone(),
            confidence: 1.0,
        }])
    }
    async fn deduplicate(&self, candidates: Vec<MemoryCandidate>) -> Result<Vec<DedupDecision>> {
        self.fail_if(FailStep::Deduplicate)?;
        Ok(candidates
            .into_iter()
            .map(|candidate| DedupDecision {
                candidate,
                action: CandidateAction::Create,
                merge_target: None,
                reason: "new".into(),
            })
            .collect())
    }
}

#[async_trait]
impl SemanticProcessor for TestPipeline {
    async fn generate_abstract(&self, _: &ContextUri) -> Result<String> {
        self.fail_if(FailStep::Abstract)?;
        Ok("abstract".into())
    }
    async fn generate_overview(&self, _: &ContextUri) -> Result<String> {
        Ok("overview".into())
    }
    async fn aggregate_upward(&self, _: &ContextUri) -> Result<String> {
        self.fail_if(FailStep::Aggregate)?;
        Ok("overview".into())
    }
    async fn multimodal_to_text(&self, _: &ContextUri) -> Result<(String, String)> {
        Ok(("abstract".into(), "overview".into()))
    }
}

fn session() -> SessionHandle {
    SessionHandle {
        session_id: uuid::Uuid::new_v4(),
        user_id: "user".into(),
        agent_id: "agent".into(),
        messages: vec![SessionMessage {
            role: Role::User,
            content: "hello".into(),
            timestamp: chrono::Utc::now(),
            metadata: serde_json::Value::Null,
        }],
        compression_index: 1,
        archive_dir: ContextUri::parse("uwu://tenant/sessions/session/archive").unwrap(),
    }
}

fn compressor(store: Arc<PersistentStore>, pipeline: Arc<TestPipeline>) -> SessionCompressorImpl {
    SessionCompressorImpl::new(store, pipeline.clone(), pipeline)
}

#[tokio::test]
async fn success_transitions_pending_processing_done() {
    let store = Arc::new(PersistentStore::default());
    let pipeline = Arc::new(TestPipeline::new(FailStep::None));
    let compressor = compressor(store, pipeline);
    let session = session();
    let task = compressor.enqueue(&session).await.unwrap();
    assert_eq!(
        compressor.poll(&session, task).await.unwrap(),
        TaskStatus::Pending
    );
    let done = compressor.retry(&session, task).await.unwrap();
    assert!(
        matches!(compressor.poll(&session, task).await.unwrap(), TaskStatus::Done(value) if value == done)
    );
}

#[tokio::test]
async fn every_pipeline_step_failure_persists_metadata() {
    for step in [
        FailStep::Extract,
        FailStep::Deduplicate,
        FailStep::Abstract,
        FailStep::Aggregate,
    ] {
        let store = Arc::new(PersistentStore::default());
        let pipeline = Arc::new(TestPipeline::new(step));
        let compressor = compressor(store, pipeline);
        let session = session();
        let task = compressor.enqueue(&session).await.unwrap();
        assert!(compressor.retry(&session, task).await.is_err());
        assert!(
            matches!(compressor.poll(&session, task).await.unwrap(), TaskStatus::Failed(m) if m.attempt == 1 && m.retryable && m.message.contains("pipeline failure"))
        );
    }
}

#[tokio::test]
async fn retry_increments_attempt_and_succeeds() {
    let store = Arc::new(PersistentStore::default());
    let pipeline = Arc::new(TestPipeline::new(FailStep::Extract));
    let compressor = compressor(store, pipeline.clone());
    let session = session();
    let task = compressor.enqueue(&session).await.unwrap();
    assert!(compressor.retry(&session, task).await.is_err());
    pipeline.succeed();
    compressor.retry(&session, task).await.unwrap();
    let record = compressor
        .load_record(&SessionCompressorImpl::state_uri(&session, task))
        .await
        .unwrap();
    assert_eq!(record.attempt, 2);
}

#[tokio::test]
async fn repeated_run_is_idempotent() {
    let store = Arc::new(PersistentStore::default());
    let pipeline = Arc::new(TestPipeline::new(FailStep::None));
    let compressor = compressor(store, pipeline.clone());
    let session = session();
    assert_eq!(
        compressor.run(&session).await.unwrap(),
        compressor.run(&session).await.unwrap()
    );
    assert_eq!(pipeline.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rebuilt_compressor_polls_and_recovers_persisted_task() {
    let store = Arc::new(PersistentStore::default());
    let pipeline = Arc::new(TestPipeline::new(FailStep::None));
    let session = session();
    let task = compressor(store.clone(), pipeline.clone())
        .enqueue(&session)
        .await
        .unwrap();
    let rebuilt = compressor(store, pipeline);
    assert_eq!(
        rebuilt.poll(&session, task).await.unwrap(),
        TaskStatus::Pending
    );
    rebuilt.recover(&session).await.unwrap();
    assert!(matches!(
        rebuilt.poll(&session, task).await.unwrap(),
        TaskStatus::Done(_)
    ));
}

#[tokio::test]
async fn output_write_failure_never_becomes_done() {
    let store = Arc::new(PersistentStore::default());
    let pipeline = Arc::new(TestPipeline::new(FailStep::None));
    let compressor = compressor(store.clone(), pipeline);
    let session = session();
    let task = compressor.enqueue(&session).await.unwrap();
    store.fail_once("memory_diff.json");
    assert!(compressor.retry(&session, task).await.is_err());
    assert!(
        matches!(compressor.poll(&session, task).await.unwrap(), TaskStatus::Failed(m) if m.attempt == 1)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_run_executes_pipeline_once() {
    let store = Arc::new(PersistentStore::default());
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Notify::new());
    let pipeline = Arc::new(TestPipeline::blocking(entered.clone(), release.clone()));
    let compressor = Arc::new(compressor(store, pipeline.clone()));
    let session = session();
    compressor.enqueue(&session).await.unwrap();
    let c = compressor.clone();
    let s = session.clone();
    let first = tokio::spawn(async move { c.run(&s).await });
    entered.wait().await;
    assert!(matches!(
        compressor.run(&session).await,
        Err(ContextError::VersionConflict(_))
    ));
    release.notify_one();
    first.await.unwrap().unwrap();
    assert_eq!(pipeline.calls.load(Ordering::SeqCst), 1);
}
