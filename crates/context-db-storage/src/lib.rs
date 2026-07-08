//! # agent-context-db-storage (L7 存储层)
//!
//! 双层存储的**适配器 + 装配根**：
//! - [`PgContextStore`]：基于 `uwu_database::DbPool` 的内容层适配器，实现全部四个窄端口。
//! - [`UwuVectorIndex`]：将 `uwu_database::VectorStore` 适配为 core 的 `VectorIndex`。
//! - [`ContextDbService`]：composition root —— 唯一同时持有内容层与索引层的地方。
//! - `IndexPoint` / `IndexHit` / `VectorIndex` 从 `agent_context_db_core` re-export。
//!
//! ## 解耦约束
//!
//! - 后端具体类型只在此层出现；上层（retrieve/session/parse）只依赖 core 窄端口。
//! - PG 适配器通过 `uwu_database` 基础库注入连接池与向量后端，不自行管理连接。

pub mod migrations;
pub mod perf;
pub mod pg;
pub mod pg_version;
pub mod uwu_cache_adapter;
pub mod vector_index;

// Re-export core vector types（来源统一）
pub use agent_context_db_core::{
    AclProtectedStore, IndexHit, IndexPoint, PathAcl, Principal, VectorIndex, WatchHub,
    WatchSource, WatchableStore,
};

pub use migrations::context_db_migrations;
pub use perf::{
    BatchWriteBuffer, DedupStore, WalEntry, WriteAheadLogger, compress, content_hash, decompress,
};
pub use pg::PgContextStore;
pub use pg_version::PgVersionStore;
pub use vector_index::UwuVectorIndex;

use agent_context_db_core::Result;
use async_trait::async_trait;
use std::sync::Arc;

// ===========================================================================
// 装配根：唯一持有内容层 + 索引层的地方
// ===========================================================================

/// Composition root。内容层用任意 `ContextStore`（PG 或 Memory），
/// 索引层用 `VectorIndex`。上层拿到的是它暴露的窄端口克隆。
pub struct ContextDbService<S> {
    content: Arc<S>,
    index: Arc<dyn VectorIndex>,
}

impl<S> ContextDbService<S>
where
    S: agent_context_db_core::FsOps + agent_context_db_core::ContentRepo + 'static,
{
    /// 通用构造器：注入任意实现了 ContextStore 的内容层 + VectorIndex。
    pub fn new(content: Arc<S>, index: Arc<dyn VectorIndex>) -> Self {
        Self { content, index }
    }

    /// 交出内容层的只读寻址窄端口（供检索层使用）。
    pub fn fs_ops(&self) -> Arc<S> {
        self.content.clone()
    }

    /// 交出内容层的写端口。
    pub fn content_repo(&self) -> Arc<S> {
        self.content.clone()
    }

    /// 交出索引层端口。
    pub fn vector_index(&self) -> Arc<dyn VectorIndex> {
        self.index.clone()
    }
}

impl<S> ContextDbService<S>
where
    S: WatchSource + 'static,
{
    /// 交出 CDC/watch 端口。
    pub fn watch_source(&self) -> Arc<S> {
        self.content.clone()
    }
}

impl<S> Clone for ContextDbService<S> {
    fn clone(&self) -> Self {
        Self {
            content: self.content.clone(),
            index: self.index.clone(),
        }
    }
}

// ===========================================================================
// uwu_database 便捷构造器
// ===========================================================================

/// 使用 `uwu_database::Database` 快速构造带 ACL 的 `ContextDbService`。
///
/// 内容层自动从 `DbPool` 构建 `PgContextStore` 并包上 `AclProtectedStore`；
/// 向量层优先使用 `Database.vector`，无配置时回退到空实现。
///
/// 首次调用会自动应用 context-db 的 SQL 迁移。
pub async fn service_from_uwu_db(
    db: uwu_database::Database,
    acl: Arc<PathAcl>,
    principal: Principal,
) -> Result<ContextDbService<WatchableStore<AclProtectedStore<PgContextStore>>>> {
    // 1. 应用迁移
    let mut migrator = uwu_database::Migrator::new();
    for m in context_db_migrations() {
        migrator = migrator.add(m);
    }
    migrator.up(&db.pool).await.map_err(|e| {
        agent_context_db_core::ContextError::Storage(format!("migration failed: {e}"))
    })?;

    // 2. 构造带 ACL 的内容层适配器
    let content = Arc::new(WatchableStore::new(
        AclProtectedStore::new(PgContextStore::new(Arc::new(db.pool)), acl, principal),
        Arc::new(WatchHub::default()),
    ));

    // 3. 构造向量层适配器（无配置时回退空实现）
    let index: Arc<dyn VectorIndex> = match db.vector {
        Some(vs) => Arc::new(UwuVectorIndex::new(vs)),
        None => Arc::new(NoopVectorIndex::new()),
    };

    Ok(ContextDbService::new(content, index))
}

/// 空实现：无向量后端时的降级方案。
///
/// 所有操作仅首次 warn（E.4: OnceLock 去重），静默返回空结果。
struct NoopVectorIndex {
    warned: std::sync::OnceLock<()>,
}

impl NoopVectorIndex {
    fn new() -> Self {
        Self {
            warned: std::sync::OnceLock::new(),
        }
    }
    fn warn_once(&self, msg: &str) {
        self.warned.get_or_init(|| {
            tracing::warn!("{msg}");
        });
    }
}

#[async_trait]
impl VectorIndex for NoopVectorIndex {
    async fn upsert(&self, _collection: &str, _point: IndexPoint) -> Result<()> {
        self.warn_once("NoopVectorIndex: no vector backend configured — operations are no-ops");
        Ok(())
    }
    async fn search(
        &self,
        _collection: &str,
        _query: Vec<f32>,
        _top_k: usize,
        _filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>> {
        self.warn_once("NoopVectorIndex: no vector backend configured — operations are no-ops");
        Ok(vec![])
    }
    async fn delete(
        &self,
        _collection: &str,
        _uri: &agent_context_db_core::ContextUri,
    ) -> Result<()> {
        Ok(())
    }
}

// ===========================================================================
// PG 集成测试（service_from_uwu_db 全链路）
// ===========================================================================

#[cfg(test)]
mod pg_tests {
    use super::*;
    use agent_context_db_core::{AclRule, Permissions};
    use uwu_database::Database;
    use uwu_database::config::{
        CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend,
        VectorBackend, VectorConfig,
    };

    fn pg_url() -> Option<String> {
        std::env::var("DATABASE_URL").ok()
    }

    fn require_pg() -> String {
        pg_url().expect("SKIP: DATABASE_URL not set")
    }

    fn full_acl() -> Arc<PathAcl> {
        let acl = Arc::new(PathAcl::new());
        acl.add_rule(AclRule {
            path_pattern: "uwu://".into(),
            principal: Principal::User("test".into()),
            permissions: Permissions::full(),
            priority: 1,
        });
        acl
    }

    fn test_principal() -> Principal {
        Principal::User("test".into())
    }

    fn test_cfg() -> RuntimeConfig {
        RuntimeConfig {
            deploy: DeployConfig::default(),
            database: DbConfig {
                backend: SqlBackend::Postgres,
                url: pg_url().unwrap(),
                max_connections: 2,
                min_connections: 0,
                acquire_timeout_secs: 5,
                idle_timeout_secs: 60,
                max_lifetime_secs: 300,
                test_before_acquire: false,
                statement_cache_capacity: 100,
                application_name: Some("ctx_svc_test".into()),
            },
            cache: CacheConfig {
                backend: CacheBackend::None,
                capacity: 0,
                url: None,
            },
            vector: VectorConfig {
                backend: VectorBackend::Memory,
                url: None,
                api_key: None,
            },
        }
    }

    #[tokio::test]
    async fn test_service_from_uwu_db_assembles() {
        let _url = require_pg();
        let cfg = test_cfg();
        let db = Database::connect(&cfg).await.unwrap();

        let service = service_from_uwu_db(db, full_acl(), test_principal())
            .await
            .unwrap();

        // 验证各端口可用
        let _fs = service.fs_ops();
        let _repo = service.content_repo();
        let _idx = service.vector_index();

        // 验证 Clone
        let _clone = service.clone();
    }

    #[tokio::test]
    async fn test_service_from_uwu_db_migration_idempotent() {
        let _url = require_pg();
        let cfg = test_cfg();

        // 第一次：创建表
        let db1 = Database::connect(&cfg).await.unwrap();
        let svc1 = service_from_uwu_db(db1, full_acl(), test_principal())
            .await
            .unwrap();
        drop(svc1);

        // 第二次：不应报错（表已存在）
        let db2 = Database::connect(&cfg).await.unwrap();
        let svc2 = service_from_uwu_db(db2, full_acl(), test_principal())
            .await
            .unwrap();

        // 验证 NoopVectorIndex 也可正常使用
        let idx = svc2.vector_index();
        let uri = agent_context_db_core::ContextUri::parse("uwu://memory/test/noop").unwrap();
        idx.delete("nonexistent", &uri).await.unwrap();
        let results = idx.search("nonexistent", vec![1.0], 5, None).await.unwrap();
        assert!(results.is_empty());
    }
}
