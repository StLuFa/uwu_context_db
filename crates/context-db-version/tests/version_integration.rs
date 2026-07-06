//! M2 版本层集成测试 —— 用 `MemoryVersionStore` 验证完整的分支/合并/时间旅行链路。
//!
//! ARCHITECTURE.md M2 验收标准：fork 一个子树 → 改写 → rollback 回原版本。

use agent_context_db_core::{ContentLevel, ContextEntry, ContextUri, TenantId};
use agent_context_db_testkit::MemoryVersionStore;
use agent_context_db_version::{
    AsOfTime, BranchName, BranchType, ChangeSet, CommitMeta, CommitTrigger, LogOpts,
    MergeStrategy, UriChange, VersionRef, VersionStore,
};

fn scope() -> ContextUri {
    ContextUri::parse("uwu://t1/agent/a1/state/mid").unwrap()
}

fn uri(s: &str) -> ContextUri {
    ContextUri::parse(s).unwrap()
}

fn entry(uri: &ContextUri, text: &str) -> ContextEntry {
    ContextEntry::new_text(uri.clone(), TenantId(uuid::Uuid::nil()), text)
}

// ── M2 验收测试：fork → write → merge → rollback ──────────────────────

#[tokio::test]
async fn m2_acceptance_fork_rewrite_rollback() {
    let store = MemoryVersionStore::new();
    let s = scope();
    let state_uri = uri("uwu://t1/agent/a1/state/mid/content.json");

    // 1. 初始 commit（基线 State）
    let baseline = store
        .commit(
            &s,
            ChangeSet {
                adds: vec![state_uri.clone()],
                ..Default::default()
            },
            CommitMeta::default(),
        )
        .await
        .unwrap();
    store.put_entry_version(&baseline, &state_uri, &entry(&state_uri, "{\"mood\": \"neutral\"}"));

    // 创建 main 分支
    let main = BranchName::new("main");
    store
        .create_branch(&s, main.clone(), baseline.clone(), BranchType::Main)
        .await
        .unwrap();

    // 2. fork 实验分支
    let fork = BranchName::new("fork-explore");
    store
        .create_branch(&s, fork.clone(), baseline.clone(), BranchType::StateFork)
        .await
        .unwrap();

    // 3. 在 fork 分支上改写 State（模拟推演）— 用 scope commit（推进 HEAD）
    let fork_commit = store
        .commit(
            &s,
            ChangeSet {
                updates: vec![UriChange {
                    uri: state_uri.clone(),
                    old_hash: None,
                    new_hash: agent_context_db_version::ContentHash("v2".into()),
                    diff_summary: "mood changed to happy".into(),
                }],
                ..Default::default()
            },
            CommitMeta {
                trigger: CommitTrigger::ForkPromotion {
                    fork_name: "fork-explore".into(),
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();
    store.put_entry_version(&fork_commit, &state_uri, &entry(&state_uri, "{\"mood\": \"happy\"}"));

    // 4. 验证时间旅行：baseline 时能读到 v1
    let v1 = store
        .read_at(&state_uri, VersionRef::Commit(baseline.clone()), ContentLevel::L0)
        .await
        .unwrap();
    assert!(matches!(&v1, agent_context_db_core::ContentPayload::Abstract(s) if s.contains("neutral")));

    // 5. 验证 fork commit 时能读到 v2
    let v2 = store
        .read_at(&state_uri, VersionRef::Commit(fork_commit.clone()), ContentLevel::L0)
        .await
        .unwrap();
    assert!(matches!(&v2, agent_context_db_core::ContentPayload::Abstract(s) if s.contains("happy")));

    // 6. FastForward merge fork into main
    let result = store
        .merge(&s, &fork, &main, MergeStrategy::FastForward)
        .await
        .unwrap();
    assert_eq!(result.commit, fork_commit);
    assert!(result.conflicts.is_empty());

    // 7. diff: baseline → fork_commit
    let diff = store.diff_commits(&s, &baseline, &fork_commit).await.unwrap();
    assert!(diff.updates.contains(&state_uri), "state_uri should be in diff updates");

    // 8. log: 历史包含两个 commit
    let log = store
        .log(&s, &LogOpts { max_count: Some(10), branch: Some(main) })
        .await
        .unwrap();
    assert!(log.len() >= 2, "log should have at least 2 commits, got {}", log.len());
}

// ── 时间旅行：按时间戳读取 ──────────────────────────────────────────

#[tokio::test]
async fn asof_read_by_timestamp_finds_prior_commit() {
    let store = MemoryVersionStore::new();
    let s = scope();
    let uri = uri("uwu://t1/agent/a1/memories/events/e99");

    let c1 = store
        .commit(&s, ChangeSet::default(), CommitMeta::default())
        .await
        .unwrap();
    store.put_entry_version(&c1, &uri, &entry(&uri, "event: first deploy"));

    let payload = store
        .asof_read(&uri, AsOfTime::Timestamp(chrono::Utc::now()), ContentLevel::L0)
        .await
        .unwrap();
    assert!(matches!(payload, agent_context_db_core::ContentPayload::Abstract(s) if s.contains("first deploy")));
}

// ── 三路合并（分歧场景）─────────────────────────────────────────────

#[tokio::test]
async fn three_way_merge_on_divergent_branches() {
    let store = MemoryVersionStore::new();
    let s = scope();

    let root = store
        .commit(&s, ChangeSet::default(), CommitMeta::default())
        .await
        .unwrap();

    // 用 commit_on_parent 创建两条从 root 分叉的独立推演线
    let c_a = store.commit_on_parent(&root, CommitMeta::default());
    let c_b = store.commit_on_parent(&root, CommitMeta::default());

    let a = BranchName::new("a");
    let b = BranchName::new("b");
    store.create_branch(&s, a.clone(), c_a, BranchType::Experiment).await.unwrap();
    store.create_branch(&s, b.clone(), c_b, BranchType::Experiment).await.unwrap();

    // FastForward 应拒绝（分歧）
    assert!(store.merge(&s, &a, &b, MergeStrategy::FastForward).await.is_err());

    // ThreeWay 应创建 merge commit（2 parents，无冲突）
    let result = store.merge(&s, &a, &b, MergeStrategy::ThreeWay).await.unwrap();
    assert!(result.conflicts.is_empty());

    // 验证 B 的 HEAD 已被 merge commit 取代
    let branches = store.list_branches(&s).await.unwrap();
    let b_branch = branches.iter().find(|br| br.name.0 == "b").unwrap();
    assert_eq!(b_branch.head, result.commit);
}

// ── Tag 创建 + 列表 ──────────────────────────────────────────────────

#[tokio::test]
async fn create_and_list_tags() {
    let store = MemoryVersionStore::new();
    let s = scope();

    let c1 = store.commit(&s, ChangeSet::default(), CommitMeta::default()).await.unwrap();

    store.create_tag(&s, agent_context_db_version::Tag {
        name: agent_context_db_version::TagName::new("stable"),
        target: c1.clone(),
        tag_type: agent_context_db_version::TagType::Mutable,
        message: "first stable".into(),
        created_by: agent_context_db_version::Author { agent_id: None, user_id: None, system: true },
        created_at: chrono::Utc::now(),
    }).await.unwrap();

    let tags = store.list_tags(&s).await.unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0].name.0, "stable");
    assert_eq!(tags[0].target, c1);
}
