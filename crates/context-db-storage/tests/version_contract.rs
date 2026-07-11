use std::sync::Arc;

use agent_context_db_core::{ContentLevel, ContextEntry, ContextUri, TenantId};
use agent_context_db_storage::{
    PgVersionStore, SqliteVersionStore, context_db_migrations, migrate_sqlite,
};
use agent_context_db_testkit::MemoryVersionStore;
use agent_context_db_version::{
    BranchName, BranchType, ChangeSet, CommitId, CommitMeta, ConflictStrategy, ContentHash,
    MergeStrategy, UriChange, VersionRef, VersionStore,
};
use uwu_database::config::{
    CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend, VectorBackend,
    VectorConfig,
};

fn config(backend: SqlBackend, url: String) -> RuntimeConfig {
    RuntimeConfig {
        deploy: DeployConfig::default(),
        database: DbConfig {
            backend,
            url,
            max_connections: 4,
            min_connections: 0,
            acquire_timeout_secs: 10,
            idle_timeout_secs: 60,
            max_lifetime_secs: 300,
            test_before_acquire: false,
            statement_cache_capacity: 100,
            application_name: Some("version-contract".into()),
        },
        cache: CacheConfig {
            backend: CacheBackend::None,
            url: None,
            capacity: 0,
        },
        vector: VectorConfig {
            backend: VectorBackend::Memory,
            url: None,
            api_key: None,
        },
    }
}

async fn stores() -> Vec<(&'static str, Arc<dyn VersionStore>)> {
    let mut stores: Vec<(&str, Arc<dyn VersionStore>)> =
        vec![("memory", Arc::new(MemoryVersionStore::new()))];
    let db = uwu_database::Database::connect(&config(SqlBackend::Sqlite, "sqlite::memory:".into()))
        .await
        .unwrap();
    migrate_sqlite(&db.pool).await.unwrap();
    stores.push((
        "sqlite",
        Arc::new(SqliteVersionStore::new(Arc::new(db.pool))),
    ));

    if let Ok(url) = std::env::var("DATABASE_URL") {
        let db = uwu_database::Database::connect(&config(SqlBackend::Postgres, url))
            .await
            .unwrap();
        let mut migrator = uwu_database::Migrator::new();
        for migration in context_db_migrations() {
            migrator = migrator.add(migration);
        }
        migrator.up(&db.pool).await.unwrap();
        stores.push(("postgres", Arc::new(PgVersionStore::new(Arc::new(db.pool)))));
    }
    stores
}

fn scope(case: &str) -> ContextUri {
    ContextUri::parse(format!("uwu://contract/agent/{case}/state/mid")).unwrap()
}
fn item(scope: &ContextUri, name: &str) -> ContextUri {
    ContextUri::parse(format!("{scope}/{name}")).unwrap()
}
fn entry(uri: &ContextUri, text: &str) -> ContextEntry {
    ContextEntry::new_text(uri.clone(), TenantId(uuid::Uuid::nil()), text)
}
fn add(uri: &ContextUri, text: &str) -> ChangeSet {
    ChangeSet {
        adds: vec![entry(uri, text)],
        ..Default::default()
    }
}
fn update(uri: &ContextUri, text: &str) -> ChangeSet {
    ChangeSet {
        updates: vec![UriChange {
            uri: uri.clone(),
            old_hash: None,
            new_hash: ContentHash(text.into()),
            diff_summary: text.into(),
            entry: entry(uri, text),
        }],
        ..Default::default()
    }
}
async fn branch(store: &dyn VersionStore, scope: &ContextUri, name: &str) -> CommitId {
    store
        .list_branches(scope)
        .await
        .unwrap()
        .into_iter()
        .find(|b| b.name.as_str() == name)
        .unwrap()
        .head
}
async fn text(store: &dyn VersionStore, uri: &ContextUri, commit: CommitId) -> Option<String> {
    store
        .read_at(uri, VersionRef::Commit(commit), ContentLevel::L0)
        .await
        .ok()
        .map(|p| p.sparse_text().to_string())
}

async fn merge_contract(store: Arc<dyn VersionStore>, case: &str) {
    let s = scope(case);
    let a = item(&s, "a");
    let b = item(&s, "b");
    let root = store
        .commit(
            &s,
            ChangeSet {
                adds: vec![entry(&a, "base-a"), entry(&b, "base-b")],
                ..Default::default()
            },
            CommitMeta::default(),
        )
        .await
        .unwrap();
    let main = BranchName::new("main");
    let feature = BranchName::new("feature");
    store
        .create_branch(&s, main.clone(), root.clone(), BranchType::Main)
        .await
        .unwrap();
    store
        .create_branch(&s, feature.clone(), root, BranchType::Experiment)
        .await
        .unwrap();
    store.switch_head(&s, &feature).await.unwrap();
    store
        .commit(
            &s,
            ChangeSet {
                deletes: vec![a.clone()],
                updates: update(&b, "feature-b").updates,
                ..Default::default()
            },
            CommitMeta::default(),
        )
        .await
        .unwrap();
    store.switch_head(&s, &main).await.unwrap();
    store
        .commit(
            &s,
            add(&item(&s, "main-only"), "main").clone(),
            CommitMeta::default(),
        )
        .await
        .unwrap();
    let merged = store
        .merge(&s, &feature, &main, MergeStrategy::ThreeWay)
        .await
        .unwrap();
    assert!(
        merged.conflicts.is_empty(),
        "{case}: independent changes conflict"
    );
    assert_eq!(
        text(store.as_ref(), &a, merged.commit.clone()).await,
        None,
        "{case}: delete did not propagate"
    );
    assert_eq!(
        text(store.as_ref(), &b, merged.commit.clone())
            .await
            .as_deref(),
        Some("feature-b")
    );

    let conflict_root = merged.commit;
    let left = BranchName::new("left");
    let right = BranchName::new("right");
    store
        .create_branch(
            &s,
            left.clone(),
            conflict_root.clone(),
            BranchType::Experiment,
        )
        .await
        .unwrap();
    store
        .create_branch(&s, right.clone(), conflict_root, BranchType::Experiment)
        .await
        .unwrap();
    store.switch_head(&s, &left).await.unwrap();
    store
        .commit(&s, update(&b, "left"), CommitMeta::default())
        .await
        .unwrap();
    store.switch_head(&s, &right).await.unwrap();
    store
        .commit(&s, update(&b, "right"), CommitMeta::default())
        .await
        .unwrap();
    let before = branch(store.as_ref(), &s, "right").await;
    let result = store
        .merge(&s, &left, &right, MergeStrategy::ThreeWay)
        .await
        .unwrap();
    assert_eq!(
        result.conflicts,
        vec![b.clone()],
        "{case}: conflict classification"
    );
    assert_eq!(
        text(store.as_ref(), &b, result.commit).await.as_deref(),
        Some("right")
    );
    assert_ne!(branch(store.as_ref(), &s, "right").await, before);
}

async fn rewrite_contract(store: Arc<dyn VersionStore>, case: &str) {
    let s = scope(case);
    let u = item(&s, "value");
    let root = store
        .commit(&s, add(&u, "root"), CommitMeta::default())
        .await
        .unwrap();
    let main = BranchName::new("main");
    let feature = BranchName::new("feature");
    store
        .create_branch(&s, main.clone(), root.clone(), BranchType::Main)
        .await
        .unwrap();
    store
        .create_branch(&s, feature.clone(), root.clone(), BranchType::Experiment)
        .await
        .unwrap();
    store.switch_head(&s, &feature).await.unwrap();
    let f1 = store
        .commit(&s, update(&u, "f1"), CommitMeta::default())
        .await
        .unwrap();
    let f2 = store
        .commit(&s, add(&item(&s, "extra"), "f2"), CommitMeta::default())
        .await
        .unwrap();
    store.switch_head(&s, &main).await.unwrap();
    let picked = store
        .cherry_pick(&s, &f2, &main, ConflictStrategy::Fail)
        .await
        .unwrap();
    assert_eq!(
        branch(store.as_ref(), &s, "main").await,
        picked,
        "{case}: cherry ref"
    );

    let bad = CommitId::new();
    let before = branch(store.as_ref(), &s, "main").await;
    assert!(
        store
            .cherry_pick(&s, &bad, &main, ConflictStrategy::Fail)
            .await
            .is_err()
    );
    assert_eq!(
        branch(store.as_ref(), &s, "main").await,
        before,
        "{case}: failed cherry changed ref"
    );

    let rebased = store
        .rebase(&s, &feature, &main, ConflictStrategy::Theirs)
        .await
        .unwrap();
    assert!(!rebased.is_empty());
    assert_eq!(
        branch(store.as_ref(), &s, "feature").await,
        rebased.last().unwrap().clone(),
        "{case}: rebase ref"
    );

    let tip_before = branch(store.as_ref(), &s, "feature").await;
    assert!(store.squash(&s, vec![f2, f1], "bad order").await.is_err());
    assert_eq!(
        branch(store.as_ref(), &s, "feature").await,
        tip_before,
        "{case}: failed squash changed ref"
    );

    let detached_scope = scope(&format!("{case}-detached"));
    let du = item(&detached_scope, "v");
    let d1 = store
        .commit(&detached_scope, add(&du, "one"), CommitMeta::default())
        .await
        .unwrap();
    let detached = BranchName::new("detached-main");
    store
        .create_branch(
            &detached_scope,
            detached.clone(),
            d1.clone(),
            BranchType::Main,
        )
        .await
        .unwrap();
    store
        .commit(&detached_scope, update(&du, "two"), CommitMeta::default())
        .await
        .unwrap();
    assert_eq!(
        branch(store.as_ref(), &detached_scope, "detached-main").await,
        d1,
        "{case}: detached commit advanced branch"
    );
}

#[tokio::test]
async fn merge_contract_memory_sqlite_and_optional_postgres() {
    for (name, store) in stores().await {
        merge_contract(store, &format!("merge-{name}-{}", uuid::Uuid::new_v4())).await;
    }
}
#[tokio::test]
async fn rewrite_contract_memory_sqlite_and_optional_postgres() {
    for (name, store) in stores().await {
        rewrite_contract(store, &format!("rewrite-{name}-{}", uuid::Uuid::new_v4())).await;
    }
}
