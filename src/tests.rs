use super::*;
use agent_context_db_core::*;
use agent_context_db_retrieve::{Predicate, SortKey};
use agent_context_db_testkit::{MemoryContextStore, MemoryVersionStore};
use async_trait::async_trait;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Default)]
struct Index;
#[async_trait]
impl VectorIndex for Index {
    async fn upsert(&self, _: &str, _: IndexPoint) -> agent_context_db_core::Result<()> {
        Ok(())
    }
    async fn search(
        &self,
        _: &str,
        _: Vec<f32>,
        _: usize,
        _: Option<serde_json::Value>,
    ) -> agent_context_db_core::Result<Vec<IndexHit>> {
        Ok(vec![])
    }
    async fn delete(&self, _: &str, _: &ContextUri) -> agent_context_db_core::Result<()> {
        Ok(())
    }
}

struct Gate {
    allow: bool,
    calls: AtomicUsize,
}
impl Gate {
    fn new(allow: bool) -> Self {
        Self {
            allow,
            calls: AtomicUsize::new(0),
        }
    }
}
impl ExecutionGate for Gate {
    fn version(&self) -> u64 {
        1
    }
    fn preflight(
        &self,
        _: &ExecutionContext,
        request: ExecutionRequest,
    ) -> std::result::Result<PolicyDecision, PolicyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(PolicyDecision {
            allowed: self.allow,
            required: false,
            policy_version: 1,
            request,
            response: None,
            selected_rule: None,
            audit: vec![],
        })
    }
    fn postflight(
        &self,
        _: &ExecutionContext,
        _: &PolicyDecision,
        response: ExecutionResponse,
    ) -> std::result::Result<ExecutionResponse, PolicyError> {
        Ok(response)
    }
}

#[derive(Default)]
struct Interactive;
#[async_trait]
impl version::InteractiveVersionStore for Interactive {
    async fn begin_cherry_pick(
        &self,
        _: &ContextUri,
        _: &version::CommitId,
        _: &version::BranchName,
        _: version::ConflictSessionPersistence,
    ) -> version::Result<version::ConflictSession> {
        Err(version::VersionError::NotFound("unused".into()))
    }
    async fn begin_rebase(
        &self,
        _: &ContextUri,
        _: &version::BranchName,
        _: &version::BranchName,
        _: version::ConflictSessionPersistence,
    ) -> version::Result<version::ConflictSession> {
        Err(version::VersionError::NotFound("unused".into()))
    }
    async fn continue_conflict_session(
        &self,
        _: version::ConflictSession,
        _: version::ConflictResolutionSet,
    ) -> version::Result<Vec<version::CommitId>> {
        Ok(vec![])
    }
    async fn load_conflict_session(
        &self,
        _: &version::ConflictSessionId,
    ) -> version::Result<version::ConflictSession> {
        Err(version::VersionError::NotFound("unused".into()))
    }
    async fn continue_conflict_session_by_id(
        &self,
        _: &version::ConflictSessionId,
        _: version::ConflictResolutionSet,
    ) -> version::Result<Vec<version::CommitId>> {
        Ok(vec![])
    }
    async fn abort_conflict_session(&self, _: &version::ConflictSessionId) -> version::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct Lifecycle {
    routed: Mutex<Vec<String>>,
    restored: Mutex<Vec<String>>,
}
#[async_trait]
impl LifecyclePort for Lifecycle {
    async fn route_after_write(&self, entry: &ContextEntry) -> agent_context_db_core::Result<()> {
        self.routed.lock().unwrap().push(entry.uri.to_string());
        Ok(())
    }
    async fn restore_before_read(&self, uri: &ContextUri) -> agent_context_db_core::Result<()> {
        self.restored.lock().unwrap().push(uri.to_string());
        Ok(())
    }
}
struct Executor(AtomicUsize);
#[async_trait]
impl TypedExecutor for Executor {
    async fn execute(
        &self,
        ctx: &ExecutionContext,
        _: ExecutionRequest,
    ) -> agent_context_db_core::Result<ExecutionResponse> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ExecutionResponse::new(format!(
            "{}:{}",
            ctx.tenant_id, ctx.request_id
        )))
    }
}
struct Federation(Mutex<Option<ExecutionContext>>);
#[async_trait]
impl FederationGateway for Federation {
    async fn execute(
        &self,
        ctx: &ExecutionContext,
        _: ExecutionRequest,
    ) -> agent_context_db_core::Result<ExecutionResponse> {
        *self.0.lock().unwrap() = Some(ctx.clone());
        Ok(ExecutionResponse::new("federated"))
    }
}
struct Reactions(Mutex<Vec<Reaction>>);
impl ReactionSink for Reactions {
    fn emit(&self, r: Reaction) {
        self.0.lock().unwrap().push(r);
    }
}
struct Guard(AtomicUsize);
#[async_trait]
impl RuntimeGuard for Guard {
    async fn shutdown(&self) -> agent_context_db_core::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
struct FailAudit;
impl AuditSink for FailAudit {
    fn record(&self, _: AuditEvent) -> std::result::Result<(), String> {
        Err("audit unavailable".into())
    }
}

fn ctx(name: &str) -> RequestContext {
    RequestContext::new(
        TenantIdentity::new(TenantId(Uuid::nil()), name).unwrap(),
        "actor-7",
        "request-9",
        Instant::now() + Duration::from_secs(5),
        CancellationToken::default(),
    )
    .unwrap()
}
fn uri(tenant: &str, suffix: &str) -> ContextUri {
    ContextUri::parse(format!("uwu://{tenant}/agent/a/{suffix}")).unwrap()
}
fn entry(tenant: &str, suffix: &str) -> ContextEntry {
    ContextEntry::new_text(uri(tenant, suffix), TenantId(Uuid::nil()), "payload")
}

struct Fixture {
    db: ContextDb,
    gate: Arc<Gate>,
    lifecycle: Arc<Lifecycle>,
    executor: Arc<Executor>,
    federation: Arc<Federation>,
    reactions: Arc<Reactions>,
    guard: Arc<Guard>,
}
fn fixture(allow: bool, audit: Arc<dyn AuditSink>) -> Fixture {
    let store = Arc::new(MemoryContextStore::new());
    let gate = Arc::new(Gate::new(allow));
    let lifecycle = Arc::new(Lifecycle::default());
    let executor = Arc::new(Executor(AtomicUsize::new(0)));
    let federation = Arc::new(Federation(Mutex::new(None)));
    let reactions = Arc::new(Reactions(Mutex::new(vec![])));
    let guard = Arc::new(Guard(AtomicUsize::new(0)));
    let parts = ContextDbParts {
        fs: store.clone(),
        content: store.clone(),
        content_store: store,
        vector: Arc::new(Index),
        watch: Arc::new(WatchHub::new(8)),
        versions: Arc::new(
            MemoryVersionStore::new(version::VersionAnalysisConfig::default()).unwrap(),
        ),
        interactive_versions: Arc::new(Interactive),
        gate: gate.clone(),
        lifecycle: Some(lifecycle.clone()),
        tool_executor: Some(executor.clone()),
        skill_executor: Some(executor.clone()),
        llm: None,
        reactions: Some(reactions.clone()),
        audit,
        wasm: None,
        federation: Some(federation.clone()),
        runtime_guards: vec![guard.clone()],
    };
    Fixture {
        db: ContextDbBuilder::injected(parts).build().unwrap(),
        gate,
        lifecycle,
        executor,
        federation,
        reactions,
        guard,
    }
}
fn ok_fixture() -> Fixture {
    fixture(true, Arc::new(BoundedAuditSink::new(32).unwrap()))
}

#[tokio::test]
async fn content_tenant_mismatch_rejected() {
    let f = ok_fixture();
    assert!(matches!(
        f.db.read(&ctx("a"), &uri("b", "state/mid/x"), ContentLevel::L0)
            .await,
        Err(FacadeError::TenantViolation(_))
    ));
    assert!(matches!(
        f.db.write(&ctx("a"), entry("b", "state/mid/x")).await,
        Err(FacadeError::TenantViolation(_))
    ));
}
#[tokio::test]
async fn retrieval_deadline_is_enforced() {
    let f = ok_fixture();
    let c = RequestContext::new(
        TenantIdentity::new(TenantId(Uuid::nil()), "a").unwrap(),
        "x",
        "r",
        Instant::now() - Duration::from_millis(1),
        CancellationToken::default(),
    )
    .unwrap();
    assert!(matches!(
        f.db.retrieve(
            &c,
            &Query::Find {
                scope: None,
                predicate: Predicate::default(),
                budget: 1,
                order: SortKey::Relevance,
                expand: None,
            },
            &RetrieveContext::default(),
        )
        .await,
        Err(FacadeError::DeadlineExceeded)
    ));
}
#[test]
fn watch_prefix_tenant_mismatch_rejected() {
    let f = ok_fixture();
    assert!(matches!(
        f.db.watch(
            &ctx("a"),
            WatchOptions {
                prefix: Some("uwu://b/".into()),
                ..Default::default()
            }
        ),
        Err(FacadeError::TenantViolation(_))
    ));
}
#[tokio::test]
async fn version_scope_tenant_mismatch_rejected() {
    let f = ok_fixture();
    assert!(matches!(
        f.db.versions(&ctx("a"))
            .log(&uri("b", "state/mid/x"), &version::LogOpts::default())
            .await,
        Err(FacadeError::TenantViolation(_))
    ));
}
#[tokio::test]
async fn cancelled_request_short_circuits_content() {
    let f = ok_fixture();
    let token = CancellationToken::default();
    token.cancel();
    let c = RequestContext::new(
        TenantIdentity::new(TenantId(Uuid::nil()), "a").unwrap(),
        "x",
        "r",
        Instant::now() + Duration::from_secs(1),
        token,
    )
    .unwrap();
    assert!(matches!(
        f.db.write(&c, entry("a", "state/mid/x")).await,
        Err(FacadeError::Cancelled)
    ));
}
#[tokio::test]
async fn successful_write_routes_lifecycle() {
    let f = ok_fixture();
    f.db.write(&ctx("a"), entry("a", "state/mid/x"))
        .await
        .unwrap();
    assert_eq!(
        f.lifecycle.routed.lock().unwrap().as_slice(),
        &["uwu://a/agent/a/state/mid/x"]
    );
}
#[tokio::test]
async fn metacog_restore_occurs_before_read() {
    let f = ok_fixture();
    let e = entry("a", "metacog/mid/x");
    f.db.write(&ctx("a"), e.clone()).await.unwrap();
    f.db.read(&ctx("a"), &e.uri, ContentLevel::L0)
        .await
        .unwrap();
    assert_eq!(
        f.lifecycle.restored.lock().unwrap().as_slice(),
        &[e.uri.to_string()]
    );
}
#[tokio::test]
async fn audit_failure_propagates() {
    let f = fixture(true, Arc::new(FailAudit));
    assert!(matches!(
        f.db.write(&ctx("a"), entry("a", "state/mid/x")).await,
        Err(FacadeError::CommittedWithPostCommitFailure { .. })
    ));
}
#[test]
fn bounded_audit_evicts_oldest() {
    let s = BoundedAuditSink::new(2).unwrap();
    for op in ["one", "two", "three"] {
        s.record(AuditEvent {
            tenant: "a".into(),
            actor: "x".into(),
            request_id: "r".into(),
            operation: op.into(),
            allowed: true,
            detail: String::new(),
        })
        .unwrap();
    }
    let e = s.snapshot();
    assert_eq!(e.len(), 2);
    assert_eq!(e[0].operation, "two");
    assert_eq!(e[1].operation, "three");
}
#[tokio::test]
async fn reaction_has_request_and_actor_attribution() {
    let f = ok_fixture();
    f.db.write(&ctx("a"), entry("a", "state/mid/x"))
        .await
        .unwrap();
    let r = f.reactions.0.lock().unwrap();
    assert_eq!(r[0].execution_id, "request-9");
    assert_eq!(r[0].attributions[0].cause_id, "actor-7:content.write");
}
#[tokio::test]
async fn denied_tool_never_reaches_executor() {
    let f = fixture(false, Arc::new(BoundedAuditSink::new(8).unwrap()));
    assert!(matches!(
        f.db.execute_tool(&ctx("a"), "x", ExecutionRequest::new("x", "y"))
            .await,
        Err(FacadeError::PolicyDenied(_))
    ));
    assert_eq!(f.executor.0.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn tool_and_skill_both_pass_gate() {
    let f = ok_fixture();
    f.db.execute_tool(&ctx("a"), "t", ExecutionRequest::new("t", "x"))
        .await
        .unwrap();
    f.db.execute_skill(&ctx("a"), "s", ExecutionRequest::new("s", "x"))
        .await
        .unwrap();
    assert_eq!(f.gate.calls.load(Ordering::SeqCst), 2);
    assert_eq!(f.executor.0.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn llm_gate_denial_precedes_configuration_lookup() {
    let f = fixture(false, Arc::new(BoundedAuditSink::new(8).unwrap()));
    assert!(matches!(
        f.db.llm_complete(&ctx("a"), "secret", &LlmOpts::default())
            .await,
        Err(FacadeError::PolicyDenied(_))
    ));
    assert_eq!(f.gate.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn federation_receives_tenant_request_and_actor() {
    let f = ok_fixture();
    let out =
        f.db.federate(&ctx("a"), ExecutionRequest::new("remote", "x"))
            .await
            .unwrap();
    assert_eq!(out.content, "federated");
    let c = f.federation.0.lock().unwrap().clone().unwrap();
    assert_eq!(
        (c.tenant_id, c.actor_id, c.request_id),
        ("a".into(), "actor-7".into(), "request-9".into())
    );
}
#[tokio::test]
async fn wasm_gateway_has_no_runtime_and_unconfigured_is_safe() {
    let f = ok_fixture();
    let err = f.db.wasm_install(&ctx("a"), "m", vec![]).await.unwrap_err();
    assert!(matches!(err, FacadeError::NotConfigured("wasm")));
}
#[tokio::test]
async fn shutdown_is_idempotent_across_clones() {
    let f = ok_fixture();
    let clone = f.db.clone();
    f.db.shutdown().await.unwrap();
    clone.shutdown().await.unwrap();
    assert_eq!(f.guard.0.load(Ordering::SeqCst), 1);
}
