//! # agent-context-db-storage (L7 存储层)
//!
//! 双层存储的**适配器 + 装配根**：
//! - [`SqliteContextStore`]：默认嵌入式内容后端。
//! - [`PgContextStore`]：可选 PostgreSQL 内容后端。
//! - [`UwuVectorIndex`]：将 `uwu_database::VectorStore` 适配为 core 的 `VectorIndex`；默认由 Qdrant Edge 提供。
//! - [`ContextDbService`]：composition root，唯一同时持有内容层与索引层的地方。
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
pub mod sqlite;
pub mod uwu_cache_adapter;
pub mod vector_index;

// Re-export core vector types（来源统一）
pub use agent_context_db_core::{
    AclProtectedStore, IndexHit, IndexPoint, PathAcl, Principal, SemanticWriteDedupStore,
    VectorIndex, WatchHub, WatchSource, WatchableStore,
};

pub use migrations::context_db_migrations;
pub use perf::{
    BatchWriteBuffer, DedupStore, WalEntry, WriteAheadLogger, compress, content_hash, decompress,
};
pub use pg::PgContextStore;
pub use pg_version::PgVersionStore;
pub use sqlite::{SqliteContextStore, migrate_sqlite};
pub use vector_index::UwuVectorIndex;

use agent_context_db_core::{
    ContentRepo, ContentStore, ContextEntry, ContextError, ContextUri, DirEntry, FindPattern,
    FsOps, GrepHit, MvccVersion, Result, TreeNode,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Runtime-selected SQL content backend.
#[derive(Clone)]
pub enum SqlContextStore {
    Sqlite(SqliteContextStore),
    Postgres(PgContextStore),
}

#[async_trait]
impl FsOps for SqlContextStore {
    async fn ls(&self, dir: &ContextUri) -> Result<Vec<DirEntry>> {
        match self {
            Self::Sqlite(store) => store.ls(dir).await,
            Self::Postgres(store) => store.ls(dir).await,
        }
    }
    async fn find(&self, pattern: &FindPattern) -> Result<Vec<ContextUri>> {
        match self {
            Self::Sqlite(store) => store.find(pattern).await,
            Self::Postgres(store) => store.find(pattern).await,
        }
    }
    async fn grep(&self, pattern: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        match self {
            Self::Sqlite(store) => store.grep(pattern, scope).await,
            Self::Postgres(store) => store.grep(pattern, scope).await,
        }
    }
    async fn tree(&self, root: &ContextUri, depth: usize) -> Result<TreeNode> {
        match self {
            Self::Sqlite(store) => store.tree(root, depth).await,
            Self::Postgres(store) => store.tree(root, depth).await,
        }
    }
    async fn read(
        &self,
        uri: &ContextUri,
        level: agent_context_db_core::ContentLevel,
    ) -> Result<agent_context_db_core::ContentPayload> {
        match self {
            Self::Sqlite(store) => FsOps::read(store, uri, level).await,
            Self::Postgres(store) => FsOps::read(store, uri, level).await,
        }
    }
}

#[async_trait]
impl ContentStore for SqlContextStore {
    async fn read(
        &self,
        uri: &ContextUri,
        level: agent_context_db_core::ContentLevel,
    ) -> Result<agent_context_db_core::ContentPayload> {
        match self {
            Self::Sqlite(store) => ContentStore::read(store, uri, level).await,
            Self::Postgres(store) => ContentStore::read(store, uri, level).await,
        }
    }
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        match self {
            Self::Sqlite(store) => ContentStore::write(store, entry).await,
            Self::Postgres(store) => ContentStore::write(store, entry).await,
        }
    }
    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        match self {
            Self::Sqlite(store) => ContentStore::delete(store, uri).await,
            Self::Postgres(store) => ContentStore::delete(store, uri).await,
        }
    }
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        match self {
            Self::Sqlite(store) => ContentStore::rename(store, from, to).await,
            Self::Postgres(store) => ContentStore::rename(store, from, to).await,
        }
    }
    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        match self {
            Self::Sqlite(store) => ContentStore::batch_write(store, entries).await,
            Self::Postgres(store) => ContentStore::batch_write(store, entries).await,
        }
    }
    async fn scan_by_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<ContextEntry>> {
        match self {
            Self::Sqlite(store) => ContentStore::scan_by_prefix(store, prefix, limit).await,
            Self::Postgres(store) => ContentStore::scan_by_prefix(store, prefix, limit).await,
        }
    }
}

#[async_trait]
impl ContentRepo for SqlContextStore {
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        match self {
            Self::Sqlite(store) => ContentRepo::write(store, entry).await,
            Self::Postgres(store) => ContentRepo::write(store, entry).await,
        }
    }
    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        match self {
            Self::Sqlite(store) => ContentRepo::delete(store, uri).await,
            Self::Postgres(store) => ContentRepo::delete(store, uri).await,
        }
    }
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        match self {
            Self::Sqlite(store) => ContentRepo::rename(store, from, to).await,
            Self::Postgres(store) => ContentRepo::rename(store, from, to).await,
        }
    }
    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        match self {
            Self::Sqlite(store) => ContentRepo::batch_write(store, entries).await,
            Self::Postgres(store) => ContentRepo::batch_write(store, entries).await,
        }
    }
}

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

/// 使用 `uwu_database::Database` 构造带 ACL 的服务。
///
/// SQLite 与 PostgreSQL 都在这里选择并迁移；向量后端必须由调用方通过
/// `Database::connect_with_vector` 初始化，避免配置错误时静默降级为无索引。
pub async fn service_from_uwu_db(
    db: uwu_database::Database,
    acl: Arc<PathAcl>,
    principal: Principal,
) -> Result<
    ContextDbService<WatchableStore<SemanticWriteDedupStore<AclProtectedStore<SqlContextStore>>>>,
> {
    let store = match db.pool.backend() {
        uwu_database::SqlBackend::Sqlite => {
            migrate_sqlite(&db.pool).await?;
            SqlContextStore::Sqlite(SqliteContextStore::try_new(Arc::new(db.pool.clone()))?)
        }
        uwu_database::SqlBackend::Postgres => {
            let mut migrator = uwu_database::Migrator::new();
            for migration in context_db_migrations() {
                migrator = migrator.add(migration);
            }
            migrator
                .up(&db.pool)
                .await
                .map_err(|e| ContextError::Storage(format!("postgres migration failed: {e}")))?;
            SqlContextStore::Postgres(PgContextStore::new(Arc::new(db.pool.clone())))
        }
        backend => {
            return Err(ContextError::Storage(format!(
                "unsupported context-db SQL backend: {backend:?}"
            )));
        }
    };

    let content = Arc::new(WatchableStore::new(
        SemanticWriteDedupStore::new(AclProtectedStore::new(store, acl, principal)),
        Arc::new(WatchHub::default()),
    ));
    let vector = db.vector.ok_or_else(|| {
        ContextError::Storage(
            "vector backend is not initialized; use Database::connect_with_vector".into(),
        )
    })?;
    let index: Arc<dyn VectorIndex> = Arc::new(UwuVectorIndex::new(vector));
    Ok(ContextDbService::new(content, index))
}

/// Convert context-db storage settings into an `uwu_database` runtime configuration.
pub fn runtime_config(
    config: &agent_context_db_core::config::StorageConfig,
) -> uwu_database::RuntimeConfig {
    let sql_backend = match config.backend {
        agent_context_db_core::config::SqlStorageBackend::Sqlite => {
            uwu_database::SqlBackend::Sqlite
        }
        agent_context_db_core::config::SqlStorageBackend::Postgres => {
            uwu_database::SqlBackend::Postgres
        }
    };
    let vector_backend = match config.vector_backend {
        agent_context_db_core::config::VectorStorageBackend::QdrantEdge => {
            uwu_database::VectorBackend::QdrantEdge
        }
        agent_context_db_core::config::VectorStorageBackend::Memory => {
            uwu_database::VectorBackend::Memory
        }
    };
    uwu_database::RuntimeConfig {
        deploy: uwu_database::config::DeployConfig::default(),
        database: uwu_database::DbConfig {
            backend: sql_backend,
            url: config.database_url.clone(),
            max_connections: config.max_connections.try_into().unwrap_or(u32::MAX),
            min_connections: 0,
            acquire_timeout_secs: 5,
            idle_timeout_secs: 600,
            max_lifetime_secs: 1800,
            test_before_acquire: false,
            statement_cache_capacity: 100,
            application_name: Some("uwu-context-db".into()),
        },
        cache: uwu_database::CacheConfig {
            backend: uwu_database::CacheBackend::Memory,
            url: None,
            capacity: 10_000,
        },
        vector: uwu_database::VectorConfig {
            backend: vector_backend,
            url: config.vector_url.clone(),
            api_key: None,
        },
    }
}

/// Default embedded runtime: SQLite for content and Qdrant Edge for vectors.
pub fn default_runtime_config() -> uwu_database::RuntimeConfig {
    runtime_config(&agent_context_db_core::config::StorageConfig::default())
}

/// Connect and assemble the default embedded database in one call.
pub async fn default_embedded_service(
    acl: Arc<PathAcl>,
    principal: Principal,
) -> Result<
    ContextDbService<WatchableStore<SemanticWriteDedupStore<AclProtectedStore<SqlContextStore>>>>,
> {
    let db = uwu_database::Database::connect_with_vector(&default_runtime_config())
        .await
        .map_err(|e| ContextError::Storage(format!("connect embedded database failed: {e}")))?;
    service_from_uwu_db(db, acl, principal).await
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
        let db = Database::connect_with_vector(&cfg).await.unwrap();

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
        let db1 = Database::connect_with_vector(&cfg).await.unwrap();
        let svc1 = service_from_uwu_db(db1, full_acl(), test_principal())
            .await
            .unwrap();
        drop(svc1);

        // 第二次：不应报错（表已存在）
        let db2 = Database::connect_with_vector(&cfg).await.unwrap();
        let svc2 = service_from_uwu_db(db2, full_acl(), test_principal())
            .await
            .unwrap();

        // 向量后端已实际装配，而不是静默 no-op。
        let _idx = svc2.vector_index();
    }
}
