//! M1 检索管线集成测试 —— 用 `MemoryContextStore`（纯内存 `FsOps`）验证完整检索链路，
//! 无 PG / Qdrant 依赖，满足 ARCHITECTURE.md M1 验收标准。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentRepo, ContextEntry, ContextUri, FsOps, MemoryClass,
    TenantId,
};
use agent_context_db_retrieve::{
    ContextRetriever, RetrieveContext, RuleBasedIntentAnalyzer, RuleBasedPlanner, ScoreReranker,
};
use agent_context_db_testkit::MemoryContextStore;
use std::sync::Arc;
use uuid::Uuid;

fn tenant() -> TenantId {
    TenantId(Uuid::nil())
}

fn uri(s: &str) -> ContextUri {
    ContextUri::parse(s).unwrap()
}

fn entry(uri_str: &str, abstract_: &str, class: MemoryClass) -> ContextEntry {
    let mut e = ContextEntry::new_text(uri(uri_str), tenant(), abstract_);
    e.metadata.memory_class = Some(class);
    e
}

fn retriever(store: Arc<MemoryContextStore>) -> ContextRetriever {
    let fs: Arc<dyn FsOps> = store;
    ContextRetriever::new(
        fs,
        None,
        Arc::new(RuleBasedPlanner::new("t1", "a1")),
        Arc::new(ScoreReranker { keep: 10 }),
    )
    .with_intent_analyzer(Arc::new(RuleBasedIntentAnalyzer::new("t1", "a1")))
}

/// 构建一个包含多种记忆的 MemoryContextStore。
async fn seed_store() -> MemoryContextStore {
    let store = MemoryContextStore::new();

    // user preferences
    store
        .write(entry(
            "uwu://t1/agent/a1/memories/preferences/p1",
            "prefers dark mode and monospace fonts",
            MemoryClass::Preferences,
        ))
        .await
        .unwrap();

    // agent cases
    store
        .write(entry(
            "uwu://t1/agent/a1/memories/cases/c1",
            "solved null pointer bug by adding null check in parse_input",
            MemoryClass::Cases,
        ))
        .await
        .unwrap();

    store
        .write(entry(
            "uwu://t1/agent/a1/memories/cases/c2",
            "fixed memory leak in websocket handler by using Weak refs",
            MemoryClass::Cases,
        ))
        .await
        .unwrap();

    // skills
    store
        .write(entry(
            "uwu://t1/agent/a1/memories/skills/s1",
            "deploy using: docker build -t app . && docker push",
            MemoryClass::Skills,
        ))
        .await
        .unwrap();

    // events
    store
        .write(entry(
            "uwu://t1/agent/a1/memories/events/e1",
            "production outage on 2025-03-15 caused by expired TLS cert",
            MemoryClass::Events,
        ))
        .await
        .unwrap();

    // tools
    store
        .write(entry(
            "uwu://t1/agent/a1/memories/tools/t1",
            "kubectl get pods -n production --sort-by=.metadata.creationTimestamp",
            MemoryClass::Tools,
        ))
        .await
        .unwrap();

    store
}

fn ctx() -> RetrieveContext {
    RetrieveContext {
        user_id: Some("t1".into()),
        agent_id: Some("a1".into()),
        budget_tokens: None,
        prefer_level: ContentLevel::L0,
        trace_enabled: true,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn retrieve_event_recall_returns_events() {
    let store = Arc::new(seed_store().await);
    let retriever = retriever(store.clone());

    let result = retriever
        .retrieve("what happened during the outage?", &ctx())
        .await
        .unwrap();

    // 验证 trace 记录了意图分析
    assert!(!result.trace.steps.is_empty());
    assert!(result.trace.steps.iter().any(|step| matches!(
        step,
        agent_context_db_retrieve::TraceStep::IntentAnalysis {
            decision: Some(decision),
            ..
        } if decision.primary == agent_context_db_retrieve::IntentKind::EventRecall
    )));

    // 命中应包含 event 条目
    let has_event = result
        .hits
        .iter()
        .any(|h| matches!(&h.content, ContentPayload::Text { sparse, .. } if sparse.contains("TLS")));
    assert!(has_event, "should find the TLS cert outage event");
}

#[tokio::test]
async fn retrieve_howto_targets_skills() {
    let store = Arc::new(seed_store().await);
    let retriever = retriever(store.clone());

    let result = retriever
        .retrieve("how to deploy the app?", &ctx())
        .await
        .unwrap();

    let has_skill = result
        .hits
        .iter()
        .any(|h| matches!(&h.content, ContentPayload::Text { sparse, .. } if sparse.contains("docker")));
    assert!(has_skill, "should find the docker deploy skill");
}

#[tokio::test]
async fn retrieve_semantic_search_finds_preferences() {
    let store = Arc::new(seed_store().await);
    let retriever = retriever(store.clone());

    let result = retriever
        .retrieve("what does the user prefer?", &ctx())
        .await
        .unwrap();

    let has_pref = result
        .hits
        .iter()
        .any(|h| matches!(&h.content, ContentPayload::Text { sparse, .. } if sparse.contains("dark mode")));
    assert!(has_pref, "should find the dark mode preference");
}

#[tokio::test]
async fn retrieve_typed_calls_through_to_retrieve() {
    let store = Arc::new(seed_store().await);
    let retriever = retriever(store.clone());

    let result = retriever.retrieve("bug fix", &ctx()).await.unwrap();

    let has_case = result
        .hits
        .iter()
        .any(|h| matches!(&h.content, ContentPayload::Text { sparse, .. } if sparse.contains("null pointer")));
    assert!(has_case, "should find the null pointer case");
}

#[tokio::test]
async fn trace_is_populated_when_enabled() {
    let store = Arc::new(seed_store().await);
    let retriever = retriever(store.clone());

    let result = retriever
        .retrieve("dark mode font preference", &ctx())
        .await
        .unwrap();

    // trace 至少包含 IntentAnalysis, InitialLocate, Rerank
    let kinds: Vec<String> = result
        .trace
        .steps
        .iter()
        .map(|s| format!("{:?}", s))
        .collect();
    assert!(
        kinds.iter().any(|k| k.contains("IntentAnalysis")),
        "trace should have IntentAnalysis step, got: {:?}",
        kinds
    );
    assert!(
        kinds.iter().any(|k| k.contains("Rerank")),
        "trace should have Rerank step"
    );
}

#[tokio::test]
async fn empty_store_returns_empty_result() {
    let store = Arc::new(MemoryContextStore::new());
    let retriever = retriever(store.clone());

    let result = retriever.retrieve("anything", &ctx()).await.unwrap();
    assert!(result.hits.is_empty());
}
