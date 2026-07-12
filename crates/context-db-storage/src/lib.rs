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
use agent_context_db_core::{Page, PageRequest};

pub mod graph;
pub mod migrations;
pub mod outbox;
pub mod perf;
pub mod pg;
pub mod pg_version;
pub mod sqlite;
pub mod sqlite_version;
pub mod uwu_cache_adapter;
pub mod vector_index;

// Re-export core vector types（来源统一）
pub use agent_context_db_core::{
    AclProtectedStore, IndexHit, IndexPoint, PathAcl, Principal, SemanticWriteDedupStore,
    VectorIndex, WatchHub, WatchSource, WatchableStore,
};

pub use graph::{BatchWriteConfig, GraphCentralityConfig};
pub use migrations::context_db_migrations;
pub use perf::{
    BatchWriteBuffer, DedupStore, WalEntry, WriteAheadLogger, compress, content_hash, decompress,
};
pub use pg::PgContextStore;
pub use pg_version::PgVersionStore;
pub use sqlite::{SqliteContextStore, migrate_sqlite};
pub use sqlite_version::SqliteVersionStore;
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
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        match self {
            Self::Sqlite(store) => store.ls(dir, page).await,
            Self::Postgres(store) => store.ls(dir, page).await,
        }
    }
    async fn find(&self, pattern: &FindPattern, page: PageRequest) -> Result<Page<ContextUri>> {
        match self {
            Self::Sqlite(store) => store.find(pattern, page).await,
            Self::Postgres(store) => store.find(pattern, page).await,
        }
    }
    async fn grep(&self, pattern: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        match self {
            Self::Sqlite(store) => store.grep(pattern, scope).await,
            Self::Postgres(store) => store.grep(pattern, scope).await,
        }
    }
    async fn tree(
        &self,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        match self {
            Self::Sqlite(store) => store.tree(root, depth, page).await,
            Self::Postgres(store) => store.tree(root, depth, page).await,
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
    async fn scan_by_prefix(&self, prefix: &str, page: PageRequest) -> Result<Page<ContextEntry>> {
        match self {
            Self::Sqlite(store) => ContentStore::scan_by_prefix(store, prefix, page).await,
            Self::Postgres(store) => ContentStore::scan_by_prefix(store, prefix, page).await,
        }
    }
    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: agent_context_db_core::ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        match self {
            Self::Sqlite(store) => {
                ContentStore::scan_by_type(store, prefix, content_type, page).await
            }
            Self::Postgres(store) => {
                ContentStore::scan_by_type(store, prefix, content_type, page).await
            }
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
    version: Arc<dyn agent_context_db_version::VersionStore>,
    interactive_version: Arc<dyn agent_context_db_version::InteractiveVersionStore>,
    _outbox_runtime: Option<Arc<outbox::OutboxRuntime>>,
}

impl<S> ContextDbService<S>
where
    S: agent_context_db_core::FsOps + agent_context_db_core::ContentRepo + 'static,
{
    /// 通用构造器：注入任意实现了 ContextStore 的内容层 + VectorIndex。
    pub fn new(
        content: Arc<S>,
        index: Arc<dyn VectorIndex>,
        version: Arc<dyn agent_context_db_version::VersionStore>,
        interactive_version: Arc<dyn agent_context_db_version::InteractiveVersionStore>,
    ) -> Self {
        Self {
            content,
            index,
            version,
            interactive_version,
            _outbox_runtime: None,
        }
    }

    fn with_outbox_runtime(mut self, runtime: outbox::OutboxRuntime) -> Self {
        self._outbox_runtime = Some(Arc::new(runtime));
        self
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

    pub fn version_store(&self) -> Arc<dyn agent_context_db_version::VersionStore> {
        self.version.clone()
    }

    pub fn interactive_version_store(
        &self,
    ) -> Arc<dyn agent_context_db_version::InteractiveVersionStore> {
        self.interactive_version.clone()
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
            version: self.version.clone(),
            interactive_version: self.interactive_version.clone(),
            _outbox_runtime: self._outbox_runtime.clone(),
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
    contradiction_detector: Arc<dyn agent_context_db_version::ContradictionDetector>,
    centrality_config: GraphCentralityConfig,
    version_analysis_config: agent_context_db_version::VersionAnalysisConfig,
) -> Result<
    ContextDbService<WatchableStore<SemanticWriteDedupStore<AclProtectedStore<SqlContextStore>>>>,
> {
    let pool = Arc::new(db.pool.clone());
    let (store, version, interactive_version): (
        SqlContextStore,
        Arc<dyn agent_context_db_version::VersionStore>,
        Arc<dyn agent_context_db_version::InteractiveVersionStore>,
    ) = match db.pool.backend() {
        uwu_database::SqlBackend::Sqlite => {
            migrate_sqlite(&db.pool).await?;
            let versions = Arc::new(
                SqliteVersionStore::new(pool.clone(), version_analysis_config.clone())
                    .map_err(|error| {
                        ContextError::Storage(format!(
                            "construct sqlite version store failed: {error}"
                        ))
                    })?
                    .with_contradiction_detector(contradiction_detector.clone()),
            );
            (
                SqlContextStore::Sqlite(SqliteContextStore::try_new(
                    pool.clone(),
                    centrality_config,
                )?),
                versions.clone(),
                versions,
            )
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
            let versions = Arc::new(
                PgVersionStore::new(pool.clone(), version_analysis_config.clone())
                    .map_err(|error| {
                        ContextError::Storage(format!(
                            "construct postgres version store failed: {error}"
                        ))
                    })?
                    .with_contradiction_detector(contradiction_detector.clone()),
            );
            (
                SqlContextStore::Postgres(PgContextStore::new(pool.clone(), centrality_config)?),
                versions.clone(),
                versions,
            )
        }
        backend => {
            return Err(ContextError::Storage(format!(
                "unsupported context-db SQL backend: {backend:?}"
            )));
        }
    };

    let content = Arc::new(WatchableStore::new(
        SemanticWriteDedupStore::new(AclProtectedStore::new(store, acl, principal))?,
        Arc::new(WatchHub::default()),
    ));
    let vector = db.vector.ok_or_else(|| {
        ContextError::Storage(
            "vector backend is not initialized; use Database::connect_with_vector".into(),
        )
    })?;
    let index: Arc<dyn VectorIndex> = Arc::new(UwuVectorIndex::new(vector));
    let runtime = outbox::start_worker(pool, index.clone(), outbox::OutboxConfig::default());
    Ok(
        ContextDbService::new(content, index, version, interactive_version)
            .with_outbox_runtime(runtime),
    )
}

pub(crate) fn max_connections(
    config: &agent_context_db_core::config::StorageConfig,
) -> Result<u32> {
    if config.max_connections == 0 {
        return Err(ContextError::Storage(
            "invalid storage configuration: max_connections must be greater than zero".into(),
        ));
    }
    u32::try_from(config.max_connections).map_err(|_| {
        ContextError::Storage(format!(
            "invalid storage configuration: max_connections ({}) exceeds u32::MAX",
            config.max_connections
        ))
    })
}

/// Convert context-db storage settings into an `uwu_database` runtime configuration.
pub fn runtime_config(
    config: &agent_context_db_core::config::StorageConfig,
) -> Result<uwu_database::RuntimeConfig> {
    let max_connections = max_connections(config)?;
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
    Ok(uwu_database::RuntimeConfig {
        deploy: uwu_database::config::DeployConfig::default(),
        database: uwu_database::DbConfig {
            backend: sql_backend,
            url: config.database_url.clone(),
            max_connections,
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
    })
}

/// Default embedded runtime: SQLite for content and Qdrant Edge for vectors.
pub fn default_runtime_config() -> Result<uwu_database::RuntimeConfig> {
    runtime_config(&agent_context_db_core::config::StorageConfig::default())
}

/// Connect and assemble the default embedded database in one call.
pub async fn default_embedded_service(
    acl: Arc<PathAcl>,
    principal: Principal,
    contradiction_detector: Arc<dyn agent_context_db_version::ContradictionDetector>,
) -> Result<
    ContextDbService<WatchableStore<SemanticWriteDedupStore<AclProtectedStore<SqlContextStore>>>>,
> {
    let config = default_runtime_config()?;
    let db = uwu_database::Database::connect_with_vector(&config)
        .await
        .map_err(|e| ContextError::Storage(format!("connect embedded database failed: {e}")))?;
    service_from_uwu_db(
        db,
        acl,
        principal,
        contradiction_detector,
        GraphCentralityConfig::default(),
        agent_context_db_version::VersionAnalysisConfig::default(),
    )
    .await
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use crate::pg::PgStoreConfig;
    use agent_context_db_core::config::StorageConfig;

    #[test]
    fn max_connections_accepts_boundaries() {
        let config = StorageConfig {
            max_connections: 1,
            ..Default::default()
        };
        assert_eq!(runtime_config(&config).unwrap().database.max_connections, 1);
        assert_eq!(
            PgStoreConfig::from_uwu_config(&config)
                .unwrap()
                .max_connections,
            1
        );

        let maximum = StorageConfig {
            max_connections: u32::MAX as usize,
            ..Default::default()
        };
        assert_eq!(
            runtime_config(&maximum).unwrap().database.max_connections,
            u32::MAX
        );
        assert_eq!(
            PgStoreConfig::from_uwu_config(&maximum)
                .unwrap()
                .max_connections,
            u32::MAX
        );
    }

    #[test]
    fn max_connections_rejects_zero() {
        let config = StorageConfig {
            max_connections: 0,
            ..Default::default()
        };
        assert!(runtime_config(&config).is_err());
        assert!(PgStoreConfig::from_uwu_config(&config).is_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn max_connections_rejects_values_exceeding_u32() {
        let config = StorageConfig {
            max_connections: u32::MAX as usize + 1,
            ..Default::default()
        };
        assert!(runtime_config(&config).is_err());
        assert!(PgStoreConfig::from_uwu_config(&config).is_err());
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

    struct TestContradictionDetector;

    #[async_trait::async_trait]
    impl agent_context_db_version::ContradictionDetector for TestContradictionDetector {
        async fn contradiction_confidence(
            &self,
            _left: &str,
            _right: &str,
        ) -> agent_context_db_version::Result<f32> {
            Ok(0.0)
        }
    }

    fn test_detector() -> Arc<dyn agent_context_db_version::ContradictionDetector> {
        Arc::new(TestContradictionDetector)
    }

    fn test_cfg(url: String) -> RuntimeConfig {
        RuntimeConfig {
            deploy: DeployConfig::default(),
            database: DbConfig {
                backend: SqlBackend::Postgres,
                url,
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
        let Some(url) = pg_url() else { return };
        let cfg = test_cfg(url);
        let db = Database::connect_with_vector(&cfg).await.unwrap();

        let service = service_from_uwu_db(
            db,
            full_acl(),
            test_principal(),
            test_detector(),
            GraphCentralityConfig::default(),
            agent_context_db_version::VersionAnalysisConfig::default(),
        )
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
        let Some(url) = pg_url() else { return };
        let cfg = test_cfg(url);

        // 第一次：创建表
        let db1 = Database::connect_with_vector(&cfg).await.unwrap();
        let svc1 = service_from_uwu_db(
            db1,
            full_acl(),
            test_principal(),
            test_detector(),
            GraphCentralityConfig::default(),
            agent_context_db_version::VersionAnalysisConfig::default(),
        )
        .await
        .unwrap();
        drop(svc1);

        // 第二次：不应报错（表已存在）
        let db2 = Database::connect_with_vector(&cfg).await.unwrap();
        let svc2 = service_from_uwu_db(
            db2,
            full_acl(),
            test_principal(),
            test_detector(),
            GraphCentralityConfig::default(),
            agent_context_db_version::VersionAnalysisConfig::default(),
        )
        .await
        .unwrap();

        // 向量后端已实际装配，而不是静默 no-op。
        let _idx = svc2.vector_index();
    }
}
