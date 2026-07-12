//! SQLite content adapter used by the default embedded deployment.
//!
//! The adapter stores the complete [`ContextEntry`] JSON alongside indexed columns. This keeps
//! every payload and metadata field lossless while allowing URI, tenant, type, and text queries to
//! remain efficient. Writes and their MVCC snapshots share one transaction.
use agent_context_db_core::{Page, PageRequest};

use agent_context_db_core::{
    BlobRef, BlobStore, BrowsingOps, ContentHash, ContentLevel, ContentPayload, ContentRepo,
    ContentStore, ContentType, ContextDiff, ContextEntry, ContextError, ContextUri, DirEntry,
    FindPattern, FsOps, GraphRelation, GraphStore, GrepHit, MvccVersion, Result, StateScope,
    TenantId, TenantOps, TreeNode, VersionEntry, VersionOps, sanitize_entry_for_write,
};
use async_trait::async_trait;
use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;
use uwu_database::DbPool;

use crate::graph::{
    BatchWriteConfig, GraphCentralityConfig, incremental_pagerank_scores, pagerank_scores,
};
use crate::outbox::{
    IndexMutation, collection_from_entry, enqueue_sqlite, point_from_entry, upsert_mutation,
};

#[derive(Clone)]
pub struct SqliteContextStore {
    pool: SqlitePool,
    read_cache: Option<Arc<dyn agent_context_db_core::ReadCache>>,
    centrality_config: GraphCentralityConfig,
    batch_config: BatchWriteConfig,
}

impl SqliteContextStore {
    pub fn try_new(pool: Arc<DbPool>, centrality_config: GraphCentralityConfig) -> Result<Self> {
        let pool = pool
            .as_sqlite()
            .map_err(|e| {
                ContextError::Storage(format!("SqliteContextStore requires sqlite backend: {e}"))
            })?
            .clone();
        Ok(Self {
            pool,
            read_cache: None,
            centrality_config,
            batch_config: BatchWriteConfig::default(),
        })
    }

    pub fn with_cache(mut self, cache: Arc<dyn agent_context_db_core::ReadCache>) -> Self {
        self.read_cache = Some(cache);
        self
    }

    pub fn with_batch_config(mut self, config: BatchWriteConfig) -> Self {
        self.batch_config = config;
        self
    }

    fn sqlite_pool(&self) -> &SqlitePool {
        &self.pool
    }

    fn storage_err(op: &str, error: impl std::error::Error + 'static) -> ContextError {
        tracing::error!(op, error = ?agent_context_db_core::ErrorReport::from_error(&error), "sqlite storage operation failed");
        ContextError::Storage(format!("{op} failed: {error}"))
    }

    fn dir_prefix(dir: &ContextUri) -> String {
        format!("{}/", dir.to_string().trim_end_matches('/'))
    }

    fn escape_like(value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    }

    fn prefix_pattern(prefix: &str) -> String {
        format!("{}%", Self::escape_like(prefix))
    }

    /// Translate the public glob syntax to a SQL LIKE pattern while preserving literal LIKE
    /// metacharacters. `*` and `?` are the only glob metacharacters supported by FindPattern.
    fn glob_pattern(glob: &str) -> String {
        let mut pattern = String::with_capacity(glob.len());
        for ch in glob.chars() {
            match ch {
                '*' => pattern.push('%'),
                '?' => pattern.push('_'),
                '\\' | '%' | '_' => {
                    pattern.push('\\');
                    pattern.push(ch);
                }
                _ => pattern.push(ch),
            }
        }
        pattern
    }

    fn version_to_sql(version: MvccVersion) -> Result<i64> {
        i64::try_from(version.0).map_err(|_| {
            ContextError::VersionConflict(format!(
                "version {} exceeds the SQLite INTEGER range",
                version.0
            ))
        })
    }

    fn version_from_sql(version: i64) -> Result<MvccVersion> {
        u64::try_from(version).map(MvccVersion).map_err(|_| {
            ContextError::Storage(format!("invalid negative SQLite version {version}"))
        })
    }

    async fn write_in_tx(
        tx: &mut Transaction<'_, Sqlite>,
        entry: &ContextEntry,
    ) -> Result<MvccVersion> {
        entry.validate_tenant_binding()?;
        let mut entry = sanitize_entry_for_write(entry)?;
        let uri = entry.uri.to_string();
        let tenant = entry.tenant.0.to_string();
        let content_type = entry
            .metadata
            .content_type
            .unwrap_or(ContentType::Evidence)
            .as_path_segment();
        let state_scope = entry.metadata.state_scope.map(state_scope_name);
        let (l0, l1, l2) = payload_levels(&entry.payload);
        let tags_json = serde_json::to_string(&entry.metadata.tags)?;
        let custom_json = serde_json::to_string(&entry.metadata.custom)?;
        let now = chrono::Utc::now();
        entry.updated_at = now;

        let mvcc: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO context_entries
                (uri, tenant_id, l0_abstract, l1_overview, l2_full_text, content_type,
                 state_scope, tags_json, custom_json, entry_json, mvcc_version,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', 1, ?, ?)
            ON CONFLICT(uri) DO UPDATE SET
                tenant_id = excluded.tenant_id,
                l0_abstract = excluded.l0_abstract,
                l1_overview = excluded.l1_overview,
                l2_full_text = excluded.l2_full_text,
                content_type = excluded.content_type,
                state_scope = excluded.state_scope,
                tags_json = excluded.tags_json,
                custom_json = excluded.custom_json,
                mvcc_version = context_entries.mvcc_version + 1,
                updated_at = excluded.updated_at
            RETURNING mvcc_version
            "#,
        )
        .bind(&uri)
        .bind(&tenant)
        .bind(&l0)
        .bind(&l1)
        .bind(&l2)
        .bind(content_type)
        .bind(state_scope)
        .bind(&tags_json)
        .bind(&custom_json)
        .bind(entry.created_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| Self::storage_err("write entry", e))?;

        let version = Self::version_from_sql(mvcc)?;
        entry.mvcc_version = version;
        let entry_json = serde_json::to_string(&entry)?;
        sqlx::query("UPDATE context_entries SET entry_json = ? WHERE uri = ?")
            .bind(&entry_json)
            .bind(&uri)
            .execute(&mut **tx)
            .await
            .map_err(|e| Self::storage_err("store entry snapshot", e))?;

        sqlx::query(
            "INSERT INTO context_versions (uri, mvcc_version, l0_abstract, entry_json, created_at) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&uri)
        .bind(mvcc)
        .bind(&l0)
        .bind(&entry_json)
        .bind(now.to_rfc3339())
        .execute(&mut **tx)
        .await
        .map_err(|e| Self::storage_err("write version", e))?;

        Ok(version)
    }

    async fn bump_graph_revision(
        tx: &mut Transaction<'_, Sqlite>,
        scope: &str,
        operation: &str,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<u64> {
        let revision: i64 = sqlx::query_scalar("INSERT INTO context_graph_revisions(scope, revision) VALUES (?, 1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1 RETURNING revision")
            .bind(scope).fetch_one(&mut **tx).await.map_err(|e| Self::storage_err("bump graph revision", e))?;
        sqlx::query("INSERT INTO context_graph_mutations(scope, revision, operation, from_uri, to_uri, created_at) VALUES (?,?,?,?,?,?)")
            .bind(scope).bind(revision).bind(operation).bind(from).bind(to).bind(chrono::Utc::now().to_rfc3339())
            .execute(&mut **tx).await.map_err(|e| Self::storage_err("log graph mutation", e))?;
        u64::try_from(revision).map_err(|_| ContextError::Storage("negative graph revision".into()))
    }

    async fn load_entry(&self, uri: &ContextUri) -> Result<ContextEntry> {
        let json: Option<String> =
            sqlx::query_scalar("SELECT entry_json FROM context_entries WHERE uri = ?")
                .bind(uri.to_string())
                .fetch_optional(self.sqlite_pool())
                .await
                .map_err(|e| Self::storage_err("load entry", e))?;
        json.map(|value| serde_json::from_str(&value))
            .transpose()?
            .ok_or_else(|| ContextError::NotFound(uri.to_string()))
    }
}

/// Applies the SQLite schema. Every statement is idempotent, so startup can safely call this on
/// every process launch without maintaining a second migration state machine.
pub async fn migrate_sqlite(pool: &DbPool) -> Result<()> {
    let sqlite = pool
        .as_sqlite()
        .map_err(|e| ContextError::Storage(format!("sqlite migration requires sqlite: {e}")))?;
    let statements = [
        "PRAGMA foreign_keys = ON",
        r#"CREATE TABLE IF NOT EXISTS context_entries (
            uri TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            l0_abstract TEXT NOT NULL,
            l1_overview TEXT,
            l2_full_text TEXT NOT NULL DEFAULT '',
            content_type TEXT NOT NULL DEFAULT 'evidence',
            state_scope TEXT,
            tags_json TEXT NOT NULL DEFAULT '[]',
            custom_json TEXT NOT NULL DEFAULT '{}',
            entry_json TEXT NOT NULL,
            mvcc_version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"#,
        "CREATE INDEX IF NOT EXISTS idx_ctx_sqlite_tenant ON context_entries (tenant_id)",
        "CREATE INDEX IF NOT EXISTS idx_ctx_sqlite_uri ON context_entries (uri)",
        "CREATE INDEX IF NOT EXISTS idx_ctx_sqlite_type_uri ON context_entries (content_type, uri)",
        r#"CREATE TABLE IF NOT EXISTS context_versions (
            uri TEXT NOT NULL,
            mvcc_version INTEGER NOT NULL,
            l0_abstract TEXT NOT NULL,
            entry_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (uri, mvcc_version)
        )"#,
        "CREATE INDEX IF NOT EXISTS idx_ctx_sqlite_versions ON context_versions (uri, mvcc_version DESC)",
        r#"CREATE TABLE IF NOT EXISTS context_relations (
            from_uri TEXT NOT NULL,
            to_uri TEXT NOT NULL,
            relation_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (from_uri, to_uri, relation_kind)
        )"#,
        "CREATE INDEX IF NOT EXISTS idx_ctx_sqlite_rel_to ON context_relations (to_uri, relation_kind)",
        r#"CREATE TABLE IF NOT EXISTS context_index_outbox (
            id TEXT PRIMARY KEY,
            mutation_json TEXT NOT NULL,
            uri TEXT,
            mvcc_version INTEGER,
            status TEXT NOT NULL CHECK (status IN ('pending','processing','done','failed','dead')),
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at TEXT NOT NULL,
            lease_until TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        )"#,
        "CREATE INDEX IF NOT EXISTS idx_context_index_outbox_ready ON context_index_outbox (status, available_at, created_at)",
        r#"CREATE TABLE IF NOT EXISTS context_graph_revisions (scope TEXT PRIMARY KEY, revision INTEGER NOT NULL CHECK(revision >= 0))"#,
        r#"CREATE TABLE IF NOT EXISTS context_graph_mutations (scope TEXT NOT NULL, revision INTEGER NOT NULL, operation TEXT NOT NULL, from_uri TEXT, to_uri TEXT, created_at TEXT NOT NULL, PRIMARY KEY(scope, revision))"#,
        r#"CREATE TABLE IF NOT EXISTS context_graph_centrality (scope TEXT NOT NULL, revision INTEGER NOT NULL, algorithm_config TEXT NOT NULL, uri TEXT NOT NULL, score REAL NOT NULL, PRIMARY KEY(scope, revision, algorithm_config, uri))"#,
        r#"CREATE TABLE IF NOT EXISTS context_blobs (
            content_hash TEXT PRIMARY KEY,
            data BLOB NOT NULL,
            mime_type TEXT NOT NULL,
            size INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS version_commits (id TEXT PRIMARY KEY, scope TEXT NOT NULL, tree_hash TEXT NOT NULL, author_json TEXT NOT NULL, message TEXT NOT NULL, timestamp TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}')"#,
        "CREATE INDEX IF NOT EXISTS idx_sqlite_version_commits_scope_time ON version_commits(scope, timestamp DESC)",
        r#"CREATE TABLE IF NOT EXISTS version_commit_parents (commit_id TEXT NOT NULL REFERENCES version_commits(id) ON DELETE CASCADE, parent_id TEXT NOT NULL REFERENCES version_commits(id) ON DELETE RESTRICT, ordinal INTEGER NOT NULL CHECK(ordinal >= 0), PRIMARY KEY(commit_id, parent_id), UNIQUE(commit_id, ordinal), CHECK(commit_id <> parent_id))"#,
        "CREATE INDEX IF NOT EXISTS idx_sqlite_version_parents_child ON version_commit_parents(parent_id)",
        r#"CREATE TABLE IF NOT EXISTS version_branches (scope TEXT NOT NULL, name TEXT NOT NULL, head TEXT NOT NULL, branch_type TEXT NOT NULL, lifecycle_json TEXT NOT NULL DEFAULT '{}', created_from TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(scope, name))"#,
        r#"CREATE TABLE IF NOT EXISTS version_tags (scope TEXT NOT NULL, name TEXT NOT NULL, target TEXT NOT NULL, tag_type TEXT NOT NULL, message TEXT, timestamp TEXT NOT NULL, condition_expr TEXT, PRIMARY KEY(scope, name))"#,
        r#"CREATE TABLE IF NOT EXISTS version_entry_deltas (commit_id TEXT NOT NULL REFERENCES version_commits(id) ON DELETE CASCADE, uri TEXT NOT NULL, op TEXT NOT NULL, entry_json TEXT, rename_from TEXT, PRIMARY KEY(commit_id, uri))"#,
        "CREATE INDEX IF NOT EXISTS idx_sqlite_version_deltas_uri ON version_entry_deltas(uri, commit_id)",
        r#"CREATE TABLE IF NOT EXISTS version_heads (scope TEXT PRIMARY KEY, commit_id TEXT NOT NULL REFERENCES version_commits(id) ON DELETE RESTRICT, attached_branch TEXT, FOREIGN KEY(scope, attached_branch) REFERENCES version_branches(scope, name) ON DELETE RESTRICT)"#,
        r#"CREATE TABLE IF NOT EXISTS version_commit_checkpoints (commit_id TEXT PRIMARY KEY REFERENCES version_commits(id) ON DELETE CASCADE, snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL, last_accessed_at TEXT NOT NULL, access_count INTEGER NOT NULL DEFAULT 0)"#,
        "CREATE INDEX IF NOT EXISTS idx_sqlite_version_checkpoints_heat ON version_commit_checkpoints(last_accessed_at DESC, access_count DESC, created_at DESC)",
        r#"CREATE TABLE IF NOT EXISTS version_conflict_sessions (id TEXT PRIMARY KEY, scope TEXT NOT NULL, operation_json TEXT NOT NULL, session_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT)"#,
        "CREATE INDEX IF NOT EXISTS idx_sqlite_conflict_sessions_scope_status ON version_conflict_sessions(scope, status, updated_at DESC)",
    ];
    for statement in statements {
        sqlx::query(statement)
            .execute(sqlite)
            .await
            .map_err(|e| SqliteContextStore::storage_err("sqlite migration", e))?;
    }
    Ok(())
}

#[async_trait]
impl ContentRepo for SqliteContextStore {
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let uri = entry.uri.clone();
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin write", e))?;
        let version = Self::write_in_tx(&mut tx, &entry).await?;
        if let Some(mutation) = upsert_mutation(&entry, version)? {
            enqueue_sqlite(&mut tx, &mutation).await?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit write", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(&uri).await;
        }
        Ok(version)
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin delete", e))?;
        let affected = sqlx::query("DELETE FROM context_entries WHERE uri = ?")
            .bind(uri.to_string())
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("delete", e))?;
        if affected.rows_affected() == 0 {
            return Err(ContextError::NotFound(uri.to_string()));
        }
        let collection = sqlx::query_scalar::<_, String>(
            "SELECT entry_json FROM context_versions WHERE uri = ? ORDER BY mvcc_version DESC LIMIT 1",
        )
        .bind(uri.to_string())
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("read deleted collection", e))?
        .and_then(|json| serde_json::from_str::<ContextEntry>(&json).ok())
        .map(|entry| collection_from_entry(&entry))
        .unwrap_or_else(|| crate::outbox::DEFAULT_COLLECTION.to_owned());
        enqueue_sqlite(
            &mut tx,
            &IndexMutation::Delete {
                collection,
                uri: uri.clone(),
            },
        )
        .await?;
        sqlx::query("DELETE FROM context_versions WHERE uri = ?")
            .bind(uri.to_string())
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("delete versions", e))?;
        let graph_changed =
            sqlx::query("DELETE FROM context_relations WHERE from_uri = ? OR to_uri = ?")
                .bind(uri.to_string())
                .bind(uri.to_string())
                .execute(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("delete relations", e))?
                .rows_affected();
        if graph_changed > 0 {
            Self::bump_graph_revision(&mut tx, uri.tenant(), "delete", Some(uri.as_str()), None)
                .await?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit delete", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(uri).await;
        }
        Ok(())
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        if from.tenant() != to.tenant() {
            return Err(ContextError::PermissionDenied(format!(
                "cross-tenant rename is not allowed: {} -> {}",
                from.tenant(),
                to.tenant()
            )));
        }
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin rename", e))?;
        let json: Option<String> =
            sqlx::query_scalar("SELECT entry_json FROM context_entries WHERE uri = ?")
                .bind(from.to_string())
                .fetch_optional(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("read rename source", e))?;
        let mut entry: ContextEntry =
            serde_json::from_str(&json.ok_or_else(|| ContextError::NotFound(from.to_string()))?)?;
        entry.uri = to.clone();
        entry.updated_at = chrono::Utc::now();
        let entry_json = serde_json::to_string(&entry)?;
        sqlx::query(
            "UPDATE context_entries SET uri = ?, entry_json = ?, updated_at = ? WHERE uri = ?",
        )
        .bind(to.to_string())
        .bind(entry_json)
        .bind(entry.updated_at.to_rfc3339())
        .bind(from.to_string())
        .execute(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("rename", e))?;
        let versions: Vec<(i64, String)> =
            sqlx::query_as("SELECT mvcc_version, entry_json FROM context_versions WHERE uri = ?")
                .bind(from.to_string())
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("read rename versions", e))?;
        for (version, json) in versions {
            let mut historical: ContextEntry = serde_json::from_str(&json)?;
            historical.uri = to.clone();
            sqlx::query(
                "UPDATE context_versions SET uri = ?, entry_json = ? WHERE uri = ? AND mvcc_version = ?",
            )
            .bind(to.to_string())
            .bind(serde_json::to_string(&historical)?)
            .bind(from.to_string())
            .bind(version)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("rename version", e))?;
        }
        let relations: Vec<(String, String, String, String)> = sqlx::query_as(
            "SELECT from_uri, to_uri, relation_kind, created_at FROM context_relations WHERE from_uri = ? OR to_uri = ?",
        )
        .bind(from.to_string())
        .bind(from.to_string())
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("read rename relations", e))?;
        sqlx::query("DELETE FROM context_relations WHERE from_uri = ? OR to_uri = ?")
            .bind(from.to_string())
            .bind(from.to_string())
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("remove renamed relations", e))?;
        let graph_changed = !relations.is_empty();
        for (edge_from, edge_to, kind, created_at) in relations {
            sqlx::query("INSERT OR IGNORE INTO context_relations (from_uri, to_uri, relation_kind, created_at) VALUES (?, ?, ?, ?)")
                .bind(if edge_from == from.as_str() { to.as_str() } else { &edge_from })
                .bind(if edge_to == from.as_str() { to.as_str() } else { &edge_to })
                .bind(kind)
                .bind(created_at)
                .execute(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("restore renamed relation", e))?;
        }
        if graph_changed {
            Self::bump_graph_revision(
                &mut tx,
                from.tenant(),
                "rename",
                Some(from.as_str()),
                Some(to.as_str()),
            )
            .await?;
        }
        enqueue_sqlite(
            &mut tx,
            &IndexMutation::Rename {
                collection: collection_from_entry(&entry),
                from: from.clone(),
                point: point_from_entry(&entry)?,
            },
        )
        .await?;
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit rename", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(from).await;
            cache.invalidate(to).await;
        }
        Ok(())
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin batch write", e))?;
        let mut versions = Vec::with_capacity(entries.len());
        for chunk in self.batch_config.chunks(entries)? {
            let uris = chunk
                .iter()
                .map(|entry| entry.uri.to_string())
                .collect::<Vec<_>>();
            let placeholders = std::iter::repeat_n("?", uris.len())
                .collect::<Vec<_>>()
                .join(",");
            let version_sql = format!(
                "SELECT uri, mvcc_version FROM context_entries WHERE uri IN ({placeholders})"
            );
            let mut query = sqlx::query(&version_sql);
            for uri in &uris {
                query = query.bind(uri);
            }
            let base_versions = query
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("read batch versions", e))?
                .into_iter()
                .map(|row| Ok((row.try_get(0)?, row.try_get(1)?)))
                .collect::<std::result::Result<HashMap<String, i64>, sqlx::Error>>()
                .map_err(|e| Self::storage_err("decode batch versions", e))?;
            let (rows, chunk_versions) = crate::graph::prepare_batch_chunk(chunk, &base_versions)?;
            let json = serde_json::to_string(&rows)?;
            sqlx::query(r#"
                WITH input AS (
                    SELECT * FROM json_each(?) AS j
                ), rows AS (
                    SELECT json_extract(value,'$.uri') uri, json_extract(value,'$.tenant_id') tenant_id,
                           json_extract(value,'$.l0') l0, json_extract(value,'$.l1') l1,
                           json_extract(value,'$.l2') l2, json_extract(value,'$.content_type') content_type,
                           json_extract(value,'$.state_scope') state_scope, json_extract(value,'$.tags') tags,
                           json_extract(value,'$.custom') custom, json_extract(value,'$.entry') entry,
                           json_extract(value,'$.version') version, json_extract(value,'$.created_at') created_at,
                           json_extract(value,'$.updated_at') updated_at
                    FROM input WHERE json_extract(value,'$.is_current')
                )
                INSERT INTO context_entries (uri,tenant_id,l0_abstract,l1_overview,l2_full_text,content_type,state_scope,tags_json,custom_json,entry_json,mvcc_version,created_at,updated_at)
                SELECT uri,tenant_id,l0,l1,l2,content_type,state_scope,json(tags),json(custom),json(entry),version,created_at,updated_at FROM rows WHERE true
                ON CONFLICT(uri) DO UPDATE SET tenant_id=excluded.tenant_id,l0_abstract=excluded.l0_abstract,l1_overview=excluded.l1_overview,l2_full_text=excluded.l2_full_text,content_type=excluded.content_type,state_scope=excluded.state_scope,tags_json=excluded.tags_json,custom_json=excluded.custom_json,entry_json=excluded.entry_json,mvcc_version=excluded.mvcc_version,updated_at=excluded.updated_at
            "#).bind(&json).execute(&mut *tx).await.map_err(|e| Self::storage_err("write batch current", e))?;
            sqlx::query(r#"INSERT INTO context_versions(uri,mvcc_version,l0_abstract,entry_json,created_at)
                SELECT json_extract(value,'$.uri'),json_extract(value,'$.version'),json_extract(value,'$.l0'),json(json_extract(value,'$.entry')),json_extract(value,'$.updated_at') FROM json_each(?)"#)
                .bind(&json).execute(&mut *tx).await.map_err(|e| Self::storage_err("write batch history", e))?;
            sqlx::query(r#"INSERT INTO context_index_outbox(id,mutation_json,uri,mvcc_version,status,attempts,available_at,created_at,updated_at)
                SELECT json_extract(value,'$.outbox_id'),json(json_extract(value,'$.mutation')),json_extract(value,'$.uri'),json_extract(value,'$.version'),'pending',0,json_extract(value,'$.updated_at'),json_extract(value,'$.updated_at'),json_extract(value,'$.updated_at') FROM json_each(?) WHERE json_extract(value,'$.mutation') IS NOT NULL ORDER BY json_extract(value,'$.ordinal')"#)
                .bind(&json).execute(&mut *tx).await.map_err(|e| Self::storage_err("write batch outbox", e))?;
            versions.extend(chunk_versions);
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit batch write", e))?;
        if let Some(cache) = &self.read_cache {
            for entry in entries {
                cache.invalidate(&entry.uri).await;
            }
        }
        Ok(versions)
    }
}

#[async_trait]
impl FsOps for SqliteContextStore {
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        let prefix = Self::dir_prefix(dir);
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds SQLite INTEGER".into()))?;
        let rows = sqlx::query(
            "WITH scoped AS (SELECT uri, l0_abstract, content_type, substr(uri, length(?) + 1) AS rest FROM context_entries WHERE uri LIKE ? ESCAPE '\\'), children AS (SELECT CASE WHEN instr(rest, '/') > 0 THEN substr(rest, 1, instr(rest, '/') - 1) ELSE rest END AS name, max(instr(rest, '/') > 0) AS is_dir, max(CASE WHEN instr(rest, '/') = 0 THEN l0_abstract END) AS abstract_, max(CASE WHEN instr(rest, '/') = 0 THEN content_type END) AS content_type FROM scoped GROUP BY name) SELECT name, is_dir, abstract_, content_type FROM children WHERE (? IS NULL OR name > ?) ORDER BY name LIMIT ?",
        )
        .bind(&prefix).bind(Self::prefix_pattern(&prefix))
        .bind(page.after.as_deref()).bind(page.after.as_deref()).bind(limit)
        .fetch_all(self.sqlite_pool()).await.map_err(|e| Self::storage_err("ls", e))?;
        let has_more = rows.len() > page.effective_limit();
        let mut items = Vec::with_capacity(rows.len().min(page.effective_limit()));
        for row in rows.into_iter().take(page.effective_limit()) {
            let name: String = row.try_get(0).map_err(|e| Self::storage_err("ls row", e))?;
            let is_dir: bool = row.try_get(1).map_err(|e| Self::storage_err("ls row", e))?;
            items.push(DirEntry {
                uri: ContextUri::parse(format!("{prefix}{name}"))?,
                is_dir,
                abstract_: row
                    .try_get::<Option<String>, _>(2)
                    .map_err(|e| Self::storage_err("ls row", e))?
                    .unwrap_or_default(),
                content_type: if is_dir {
                    None
                } else {
                    row.try_get::<Option<String>, _>(3)
                        .map_err(|e| Self::storage_err("ls row", e))?
                        .and_then(|v| ContentType::from_path_segment(&v))
                },
            });
        }
        let next_cursor = has_more
            .then(|| {
                items
                    .last()
                    .and_then(|item| item.uri.as_str().strip_prefix(&prefix).map(str::to_owned))
            })
            .flatten();
        Ok(Page::new(items, next_cursor))
    }

    async fn find(&self, pattern: &FindPattern, page: PageRequest) -> Result<Page<ContextUri>> {
        let scope = pattern
            .scope
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds SQLite INTEGER".into()))?;
        let name_pattern = pattern
            .name_glob
            .as_deref()
            .map(|glob| format!("%/{}", Self::glob_pattern(glob)));
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE uri LIKE ? ESCAPE '\\' AND (? IS NULL OR content_type = ?) AND (? IS NULL OR uri LIKE ? ESCAPE '\\') AND (? IS NULL OR uri > ?) ORDER BY uri LIMIT ?",
        )
        .bind(Self::prefix_pattern(&scope))
        .bind(pattern.content_type.map(|kind| kind.as_path_segment()))
        .bind(pattern.content_type.map(|kind| kind.as_path_segment()))
        .bind(name_pattern.as_deref())
        .bind(name_pattern.as_deref())
        .bind(page.after.as_deref())
        .bind(page.after.as_deref())
        .bind(limit)
        .fetch_all(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("find", e))?;
        let has_more = rows.len() > page.effective_limit();
        let items = rows
            .into_iter()
            .take(page.effective_limit())
            .map(ContextUri::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let next_cursor = has_more
            .then(|| items.last().map(ToString::to_string))
            .flatten();
        Ok(Page::new(items, next_cursor))
    }

    async fn grep(&self, pattern: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        let folded_pattern = pattern.to_lowercase();
        // SQLite's built-in lower() is ASCII-only. Fetch the URI-scoped projection and apply the
        // same Unicode normalization to both operands in Rust to avoid SQL-side false negatives.
        let rows = sqlx::query(
            "SELECT uri, l0_abstract, l1_overview FROM context_entries WHERE uri LIKE ? ESCAPE '\\' ORDER BY uri",
        )
        .bind(Self::prefix_pattern(scope.as_str()))
        .fetch_all(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("grep", e))?;
        rows.into_iter()
            .map(|row| {
                let uri: String = row
                    .try_get(0)
                    .map_err(|e| Self::storage_err("grep row", e))?;
                let l0: String = row
                    .try_get(1)
                    .map_err(|e| Self::storage_err("grep row", e))?;
                let l1: Option<String> = row
                    .try_get(2)
                    .map_err(|e| Self::storage_err("grep row", e))?;
                let line = if l0.to_lowercase().contains(&folded_pattern) {
                    Some(l0)
                } else {
                    l1.filter(|overview| overview.to_lowercase().contains(&folded_pattern))
                };
                let uri = ContextUri::parse(uri)?;
                Ok(line.map(|line| GrepHit {
                    uri,
                    line,
                    level: ContentLevel::L0,
                }))
            })
            .collect::<Result<Vec<_>>>()
            .map(|hits| hits.into_iter().flatten().collect())
    }

    async fn tree(
        &self,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        let prefix = Self::dir_prefix(root);
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds SQLite INTEGER".into()))?;
        let mut rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE uri LIKE ? ESCAPE '\\' AND (? IS NULL OR uri > ?) ORDER BY uri LIMIT ?",
        ).bind(Self::prefix_pattern(&prefix)).bind(page.after.as_deref()).bind(page.after.as_deref()).bind(limit)
        .fetch_all(self.sqlite_pool()).await.map_err(|e| Self::storage_err("tree", e))?;
        let has_more = rows.len() > page.effective_limit();
        rows.truncate(page.effective_limit());
        let next_cursor = has_more.then(|| rows.last().cloned()).flatten();
        Ok(Page::new(
            vec![TreeNode {
                uri: root.clone(),
                is_dir: true,
                children: build_tree(&prefix, &rows, 0, depth),
            }],
            next_cursor,
        ))
    }

    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        if level != ContentLevel::L2
            && let Some(cache) = &self.read_cache
            && let Some(hit) = cache.get(uri, level).await
        {
            return hit.ok_or_else(|| ContextError::NotFound(uri.to_string()));
        }
        let entry = self.load_entry(uri).await;
        if let Err(ContextError::NotFound(_)) = &entry
            && let Some(cache) = &self.read_cache
        {
            cache.put_negative(uri, level).await;
        }
        let entry = entry?;
        let payload = match (&entry.payload, level) {
            (ContentPayload::Text { sparse, dense, .. }, ContentLevel::L0) => {
                ContentPayload::Text {
                    sparse: sparse.clone(),
                    dense: dense.clone(),
                    full: sparse.clone(),
                }
            }
            (ContentPayload::Text { sparse, dense, .. }, ContentLevel::L1) => {
                ContentPayload::Text {
                    sparse: sparse.clone(),
                    dense: dense.clone(),
                    full: dense.clone(),
                }
            }
            (_, ContentLevel::L2) => entry.payload.clone(),
            _ => entry.payload.clone(),
        };
        if level != ContentLevel::L2
            && let Some(cache) = &self.read_cache
        {
            cache
                .put(
                    uri,
                    level,
                    payload.clone(),
                    std::time::Duration::from_secs(300),
                )
                .await;
        }
        Ok(payload)
    }
}

#[async_trait]
impl VersionOps for SqliteContextStore {
    async fn version_history(
        &self,
        uri: &ContextUri,
        page: PageRequest,
    ) -> Result<Page<VersionEntry>> {
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds SQLite INTEGER".into()))?;
        let after = page
            .after
            .as_deref()
            .map(str::parse::<i64>)
            .transpose()
            .map_err(|e| ContextError::Storage(format!("invalid version cursor: {e}")))?;
        let rows = sqlx::query(
            "SELECT mvcc_version, l0_abstract, created_at FROM context_versions WHERE uri = ? AND (? IS NULL OR mvcc_version > ?) ORDER BY mvcc_version LIMIT ?",
        ).bind(uri.to_string()).bind(after).bind(after).bind(limit)
        .fetch_all(self.sqlite_pool()).await.map_err(|e| Self::storage_err("version history", e))?;
        let has_more = rows.len() > page.effective_limit();
        let items = rows
            .into_iter()
            .take(page.effective_limit())
            .map(|row| {
                let version: i64 = row
                    .try_get(0)
                    .map_err(|e| Self::storage_err("version row", e))?;
                let message: String = row
                    .try_get(1)
                    .map_err(|e| Self::storage_err("version row", e))?;
                let timestamp: String = row
                    .try_get(2)
                    .map_err(|e| Self::storage_err("version row", e))?;
                let ts = chrono::DateTime::parse_from_rfc3339(&timestamp)
                    .map_err(|e| Self::storage_err("version timestamp", e))?
                    .with_timezone(&chrono::Utc);
                Ok(VersionEntry {
                    version: Self::version_from_sql(version)?,
                    message,
                    ts,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let next_cursor = has_more
            .then(|| items.last().map(|item| item.version.0.to_string()))
            .flatten();
        Ok(Page::new(items, next_cursor))
    }

    async fn rollback(&self, uri: &ContextUri, to: MvccVersion) -> Result<()> {
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin rollback", e))?;
        let json: Option<String> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = ? AND mvcc_version = ?",
        )
        .bind(uri.to_string())
        .bind(Self::version_to_sql(to)?)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("rollback read", e))?;
        let mut entry: ContextEntry = serde_json::from_str(&json.ok_or_else(|| {
            ContextError::VersionConflict(format!("no version {} for {uri}", to.0))
        })?)?;
        // Historical JSON predating a rename still contains the old URI. Rollback always writes
        // the selected snapshot to the URI explicitly requested by the caller.
        entry.uri = uri.clone();
        let exists: i64 =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM context_entries WHERE uri = ?)")
                .bind(uri.to_string())
                .fetch_one(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("rollback current entry", e))?;
        if exists == 0 {
            return Err(ContextError::NotFound(uri.to_string()));
        }
        let version = Self::write_in_tx(&mut tx, &entry).await?;
        if let Some(mutation) = upsert_mutation(&entry, version)? {
            enqueue_sqlite(&mut tx, &mutation).await?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit rollback", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(uri).await;
        }
        Ok(())
    }

    async fn diff(&self, uri: &ContextUri, a: MvccVersion, b: MvccVersion) -> Result<ContextDiff> {
        async fn snapshot(
            pool: &SqlitePool,
            uri: &ContextUri,
            version: i64,
        ) -> std::result::Result<Option<String>, sqlx::Error> {
            sqlx::query_scalar(
                "SELECT entry_json FROM context_versions WHERE uri = ? AND mvcc_version = ?",
            )
            .bind(uri.to_string())
            .bind(version)
            .fetch_optional(pool)
            .await
        }
        let a_version = Self::version_to_sql(a)?;
        let b_version = Self::version_to_sql(b)?;
        let left = snapshot(self.sqlite_pool(), uri, a_version)
            .await
            .map_err(|e| Self::storage_err("diff a", e))?
            .ok_or_else(|| {
                ContextError::VersionConflict(format!("no version {} for {uri}", a.0))
            })?;
        let right = snapshot(self.sqlite_pool(), uri, b_version)
            .await
            .map_err(|e| Self::storage_err("diff b", e))?
            .ok_or_else(|| {
                ContextError::VersionConflict(format!("no version {} for {uri}", b.0))
            })?;
        let left: serde_json::Value = serde_json::from_str(&left)?;
        let right: serde_json::Value = serde_json::from_str(&right)?;
        let mut changes = Vec::new();
        collect_json_changes("$", &left, &right, &mut changes);
        Ok(ContextDiff {
            summary: if changes.is_empty() {
                format!("{uri}: v{} and v{} are identical", a.0, b.0)
            } else {
                format!("{uri}: v{} -> v{}\n{}", a.0, b.0, changes.join("\n"))
            },
        })
    }
}

#[async_trait]
impl TenantOps for SqliteContextStore {
    async fn list_tenants(&self, page: PageRequest) -> Result<Page<TenantId>> {
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds SQLite INTEGER".into()))?;
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT tenant_id FROM context_entries WHERE (? IS NULL OR tenant_id > ?) ORDER BY tenant_id LIMIT ?")
            .bind(page.after.as_deref()).bind(page.after.as_deref()).bind(limit)
            .fetch_all(self.sqlite_pool()).await.map_err(|e| Self::storage_err("list tenants", e))?;
        let has_more = rows.len() > page.effective_limit();
        let items = rows
            .into_iter()
            .take(page.effective_limit())
            .map(|id| {
                Uuid::parse_str(&id)
                    .map(TenantId)
                    .map_err(|e| Self::storage_err("tenant id", e))
            })
            .collect::<Result<Vec<_>>>()?;
        let next_cursor = has_more
            .then(|| items.last().map(|item| item.0.to_string()))
            .flatten();
        Ok(Page::new(items, next_cursor))
    }
}

#[async_trait]
impl ContentStore for SqliteContextStore {
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        <Self as FsOps>::read(self, uri, level).await
    }
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        <Self as ContentRepo>::write(self, entry).await
    }
    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        <Self as ContentRepo>::delete(self, uri).await
    }
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        <Self as ContentRepo>::rename(self, from, to).await
    }
    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        <Self as ContentRepo>::batch_write(self, entries).await
    }
    async fn scan_by_prefix(&self, prefix: &str, page: PageRequest) -> Result<Page<ContextEntry>> {
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("scan limit exceeds SQLite INTEGER".into()))?;
        let scope = ContextUri::parse(prefix.trim_end_matches('/'))?;
        let exact = scope.to_string();
        let descendants = Self::prefix_pattern(&Self::dir_prefix(&scope));
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT entry_json FROM context_entries WHERE (uri = ? OR uri LIKE ? ESCAPE '\\') AND (? IS NULL OR uri > ?) ORDER BY uri LIMIT ?",
        )
        .bind(exact)
        .bind(descendants)
        .bind(page.after.as_deref()).bind(page.after.as_deref())
        .bind(limit)
        .fetch_all(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("scan by prefix", e))?;
        let has_more = rows.len() > page.effective_limit();
        let items = rows
            .into_iter()
            .take(page.effective_limit())
            .map(|json| serde_json::from_str::<ContextEntry>(&json).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        let next_cursor = has_more
            .then(|| items.last().map(|item| item.uri.to_string()))
            .flatten();
        Ok(Page::new(items, next_cursor))
    }

    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("scan limit exceeds SQLite INTEGER".into()))?;
        let scope = ContextUri::parse(prefix.trim_end_matches('/'))?;
        let exact = scope.to_string();
        let descendants = Self::prefix_pattern(&Self::dir_prefix(&scope));
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT entry_json FROM context_entries WHERE (uri = ? OR uri LIKE ? ESCAPE '\\') AND content_type = ? AND (? IS NULL OR uri > ?) ORDER BY uri LIMIT ?",
        )
        .bind(exact)
        .bind(descendants)
        .bind(content_type.as_path_segment())
        .bind(page.after.as_deref()).bind(page.after.as_deref())
        .bind(limit)
        .fetch_all(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("scan by type", e))?;
        let has_more = rows.len() > page.effective_limit();
        let items = rows
            .into_iter()
            .take(page.effective_limit())
            .map(|json| serde_json::from_str::<ContextEntry>(&json).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        let next_cursor = has_more
            .then(|| items.last().map(|item| item.uri.to_string()))
            .flatten();
        Ok(Page::new(items, next_cursor))
    }
}

#[async_trait]
impl BrowsingOps for SqliteContextStore {
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        <Self as FsOps>::ls(self, dir, page).await
    }
    async fn tree(
        &self,
        dir: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        <Self as FsOps>::tree(self, dir, depth, page).await
    }
    async fn find(
        &self,
        scope: &ContextUri,
        pattern: &str,
        page: PageRequest,
    ) -> Result<Page<ContextUri>> {
        let limit = i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds SQLite INTEGER".into()))?;
        let exact = scope.to_string();
        let descendants = Self::prefix_pattern(&Self::dir_prefix(scope));
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE (uri = ? OR uri LIKE ? ESCAPE '\\') AND instr(uri, ?) > 0 AND (? IS NULL OR uri > ?) ORDER BY uri LIMIT ?",
        ).bind(exact).bind(descendants).bind(pattern).bind(page.after.as_deref()).bind(page.after.as_deref()).bind(limit)
        .fetch_all(self.sqlite_pool()).await.map_err(|e| Self::storage_err("browse find", e))?;
        let has_more = rows.len() > page.effective_limit();
        let items = rows
            .into_iter()
            .take(page.effective_limit())
            .map(ContextUri::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let next_cursor = has_more
            .then(|| items.last().map(ToString::to_string))
            .flatten();
        Ok(Page::new(items, next_cursor))
    }
    async fn grep(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<GrepHit>> {
        <Self as FsOps>::grep(self, pattern, scope).await
    }
}

#[async_trait]
impl GraphStore for SqliteContextStore {
    async fn add_edge(
        &self,
        from: &ContextUri,
        to: &ContextUri,
        kind: GraphRelation,
    ) -> Result<()> {
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin add edge", e))?;
        let changed = sqlx::query("INSERT OR IGNORE INTO context_relations (from_uri, to_uri, relation_kind, created_at) VALUES (?, ?, ?, ?)")
            .bind(from.to_string()).bind(to.to_string()).bind(format!("{kind:?}"))
            .bind(chrono::Utc::now().to_rfc3339()).execute(&mut *tx).await
            .map_err(|e| Self::storage_err("add edge", e))?.rows_affected();
        if changed > 0 {
            Self::bump_graph_revision(
                &mut tx,
                from.tenant(),
                "add",
                Some(from.as_str()),
                Some(to.as_str()),
            )
            .await?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit add edge", e))?;
        Ok(())
    }
    async fn remove_edge(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin remove edge", e))?;
        let changed =
            sqlx::query("DELETE FROM context_relations WHERE from_uri = ? AND to_uri = ?")
                .bind(from.to_string())
                .bind(to.to_string())
                .execute(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("remove edge", e))?
                .rows_affected();
        if changed > 0 {
            Self::bump_graph_revision(
                &mut tx,
                from.tenant(),
                "remove",
                Some(from.as_str()),
                Some(to.as_str()),
            )
            .await?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit remove edge", e))?;
        Ok(())
    }
    async fn outgoing_neighbors(
        &self,
        uri: &ContextUri,
        kind: Option<GraphRelation>,
    ) -> Result<Vec<ContextUri>> {
        let rows: Vec<String> = if let Some(kind) = kind {
            sqlx::query_scalar("SELECT DISTINCT to_uri FROM context_relations WHERE from_uri = ? AND relation_kind = ? ORDER BY to_uri")
                .bind(uri.to_string()).bind(format!("{kind:?}")).fetch_all(self.sqlite_pool()).await
        } else {
            sqlx::query_scalar("SELECT DISTINCT to_uri FROM context_relations WHERE from_uri = ? ORDER BY to_uri")
                .bind(uri.to_string()).fetch_all(self.sqlite_pool()).await
        }.map_err(|e| Self::storage_err("outgoing neighbors", e))?;
        rows.into_iter()
            .map(ContextUri::parse)
            .collect::<std::result::Result<_, _>>()
    }
    async fn incoming_neighbors(
        &self,
        uri: &ContextUri,
        kind: Option<GraphRelation>,
    ) -> Result<Vec<ContextUri>> {
        let rows: Vec<String> = if let Some(kind) = kind {
            sqlx::query_scalar("SELECT DISTINCT from_uri FROM context_relations WHERE to_uri = ? AND relation_kind = ? ORDER BY from_uri")
                .bind(uri.to_string()).bind(format!("{kind:?}")).fetch_all(self.sqlite_pool()).await
        } else {
            sqlx::query_scalar("SELECT DISTINCT from_uri FROM context_relations WHERE to_uri = ? ORDER BY from_uri")
                .bind(uri.to_string()).fetch_all(self.sqlite_pool()).await
        }.map_err(|e| Self::storage_err("incoming neighbors", e))?;
        rows.into_iter()
            .map(ContextUri::parse)
            .collect::<std::result::Result<_, _>>()
    }
    async fn batch_traverse(
        &self,
        seeds: &[ContextUri],
        kinds: &[GraphRelation],
        max_hops: usize,
    ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
        let mut result = Vec::new();
        let mut frontier = seeds.to_vec();
        let mut visited = HashSet::new();
        for _ in 0..max_hops {
            let mut next = Vec::new();
            for uri in frontier {
                if !visited.insert(uri.clone()) {
                    continue;
                }
                for kind in kinds {
                    for neighbor in self.outgoing_neighbors(&uri, Some(*kind)).await? {
                        result.push((uri.clone(), neighbor.clone(), *kind));
                        next.push(neighbor);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        Ok(result)
    }
    async fn centrality(&self, uri: &ContextUri) -> Result<f32> {
        let config = self.centrality_config;
        let scope = uri.tenant();
        let rows = sqlx::query(
            "SELECT from_uri, to_uri FROM context_relations WHERE from_uri LIKE ? OR to_uri LIKE ? ORDER BY from_uri, to_uri",
        )
        .bind(format!("uwu://{scope}/%"))
        .bind(format!("uwu://{scope}/%"))
        .fetch_all(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("centrality edges", e))?;
        let edges = rows
            .into_iter()
            .map(|row| {
                Ok((
                    row.try_get::<String, _>(0)
                        .map_err(|e| Self::storage_err("centrality edge from_uri", e))?,
                    row.try_get::<String, _>(1)
                        .map_err(|e| Self::storage_err("centrality edge to_uri", e))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let node_list = edges
            .iter()
            .flat_map(|(from, to)| [from.clone(), to.clone()])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if !node_list.iter().any(|node| node == uri.as_str()) {
            return Ok(0.0);
        }
        let revision: i64 = sqlx::query_scalar(
            "SELECT COALESCE((SELECT revision FROM context_graph_revisions WHERE scope = ?), 0)",
        )
        .bind(scope)
        .fetch_one(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("centrality revision", e))?;
        let cache_key = config.cache_key();
        if let Some(score) = sqlx::query_scalar::<_, f32>(
            "SELECT score FROM context_graph_centrality WHERE scope = ? AND revision = ? AND algorithm_config = ? AND uri = ?",
        )
        .bind(scope)
        .bind(revision)
        .bind(&cache_key)
        .bind(uri.as_str())
        .fetch_optional(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("read centrality cache", e))?
        {
            return Ok(score);
        }
        let previous_revision: Option<i64> = sqlx::query_scalar(
            "SELECT max(revision) FROM context_graph_centrality WHERE scope = ? AND algorithm_config = ? AND revision < ?",
        )
        .bind(scope)
        .bind(&cache_key)
        .bind(revision)
        .fetch_one(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("read previous centrality revision", e))?;
        let scores = if let Some(previous_revision) = previous_revision {
            let previous = sqlx::query_as::<_, (String, f32)>(
                "SELECT uri, score FROM context_graph_centrality WHERE scope = ? AND revision = ? AND algorithm_config = ?",
            )
            .bind(scope)
            .bind(previous_revision)
            .bind(&cache_key)
            .fetch_all(self.sqlite_pool())
            .await
            .map_err(|e| Self::storage_err("read previous centrality snapshot", e))?
            .into_iter()
            .collect::<HashMap<_, _>>();
            let dirty_seeds = sqlx::query_as::<_, (Option<String>, Option<String>)>(
                "SELECT from_uri, to_uri FROM context_graph_mutations WHERE scope = ? AND revision > ? AND revision <= ?",
            )
            .bind(scope)
            .bind(previous_revision)
            .bind(revision)
            .fetch_all(self.sqlite_pool())
            .await
            .map_err(|e| Self::storage_err("read dirty graph nodes", e))?
            .into_iter()
            .flat_map(|(from, to)| from.into_iter().chain(to))
            .collect::<HashSet<_>>();
            incremental_pagerank_scores(&node_list, &edges, &previous, &dirty_seeds, config)
                .unwrap_or_else(|| pagerank_scores(&node_list, &edges, config))
        } else {
            pagerank_scores(&node_list, &edges, config)
        };
        let mut tx = self
            .sqlite_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin centrality cache", e))?;
        for (node, score) in &scores {
            sqlx::query("INSERT OR REPLACE INTO context_graph_centrality(scope, revision, algorithm_config, uri, score) VALUES (?,?,?,?,?)")
                .bind(scope).bind(revision).bind(&cache_key).bind(node).bind(score)
                .execute(&mut *tx).await.map_err(|e| Self::storage_err("write centrality cache", e))?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit centrality cache", e))?;
        Ok(scores.get(uri.as_str()).copied().unwrap_or(0.0))
    }

    async fn graph_revision(&self, scope: &ContextUri) -> Result<u64> {
        let revision: Option<i64> =
            sqlx::query_scalar("SELECT revision FROM context_graph_revisions WHERE scope = ?")
                .bind(scope.tenant())
                .fetch_optional(self.sqlite_pool())
                .await
                .map_err(|e| Self::storage_err("read graph revision", e))?;
        u64::try_from(revision.unwrap_or(0))
            .map_err(|_| ContextError::Storage("negative graph revision".into()))
    }
}

#[async_trait]
impl BlobStore for SqliteContextStore {
    async fn put(&self, data: &[u8], mime_type: &str) -> Result<BlobRef> {
        let hash = blake3::hash(data).to_hex().to_string();
        sqlx::query("INSERT OR IGNORE INTO context_blobs (content_hash, data, mime_type, size, created_at) VALUES (?, ?, ?, ?, ?)")
            .bind(&hash).bind(data).bind(mime_type).bind(data.len() as i64)
            .bind(chrono::Utc::now().to_rfc3339()).execute(self.sqlite_pool()).await
            .map_err(|e| Self::storage_err("blob put", e))?;
        Ok(BlobRef {
            hash: ContentHash(hash),
            size: data.len(),
            mime_type: mime_type.to_string(),
        })
    }
    async fn get(&self, blob_ref: &BlobRef) -> Result<Vec<u8>> {
        let row: Option<(Vec<u8>, String, i64)> = sqlx::query_as(
            "SELECT data, mime_type, size FROM context_blobs WHERE content_hash = ?",
        )
        .bind(&blob_ref.hash.0)
        .fetch_optional(self.sqlite_pool())
        .await
        .map_err(|e| Self::storage_err("blob get", e))?;
        let (data, mime_type, stored_size) = row.ok_or_else(|| {
            ContextError::NotFound(format!("blob not found: {}", blob_ref.hash.0))
        })?;
        validate_blob(blob_ref, &data, &mime_type, stored_size)?;
        Ok(data)
    }
    async fn dedup_check(&self, hash: &ContentHash) -> Result<bool> {
        let exists: i64 =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM context_blobs WHERE content_hash = ?)")
                .bind(&hash.0)
                .fetch_one(self.sqlite_pool())
                .await
                .map_err(|e| Self::storage_err("blob dedup", e))?;
        Ok(exists != 0)
    }
}

fn validate_blob(blob_ref: &BlobRef, data: &[u8], mime_type: &str, stored_size: i64) -> Result<()> {
    let actual_hash = blake3::hash(data).to_hex().to_string();
    let actual_size = data.len();
    if actual_hash != blob_ref.hash.0
        || usize::try_from(stored_size).ok() != Some(actual_size)
        || blob_ref.size != actual_size
        || blob_ref.mime_type != mime_type
    {
        return Err(ContextError::Storage(format!(
            "blob integrity mismatch for {}",
            blob_ref.hash.0
        )));
    }
    Ok(())
}

fn collect_json_changes(
    path: &str,
    left: &serde_json::Value,
    right: &serde_json::Value,
    output: &mut Vec<String>,
) {
    match (left, right) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            let keys = left.keys().chain(right.keys()).collect::<BTreeSet<_>>();
            for key in keys {
                let child = format!("{path}.{key}");
                match (left.get(key), right.get(key)) {
                    (Some(a), Some(b)) => collect_json_changes(&child, a, b, output),
                    (Some(a), None) => output.push(format!("- {child}: {a}")),
                    (None, Some(b)) => output.push(format!("+ {child}: {b}")),
                    (None, None) => {}
                }
            }
        }
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            for index in 0..left.len().max(right.len()) {
                let child = format!("{path}[{index}]");
                match (left.get(index), right.get(index)) {
                    (Some(a), Some(b)) => collect_json_changes(&child, a, b, output),
                    (Some(a), None) => output.push(format!("- {child}: {a}")),
                    (None, Some(b)) => output.push(format!("+ {child}: {b}")),
                    (None, None) => {}
                }
            }
        }
        _ if left != right => output.push(format!("~ {path}: {left} -> {right}")),
        _ => {}
    }
}

fn payload_levels(payload: &ContentPayload) -> (String, Option<String>, String) {
    let projection = payload.index_projection();
    (projection.l0, projection.l1, projection.l2)
}

fn state_scope_name(scope: StateScope) -> &'static str {
    match scope {
        StateScope::Short => "short",
        StateScope::Mid => "mid",
        StateScope::Long => "long",
    }
}

fn build_tree(prefix: &str, uris: &[String], depth: usize, max_depth: usize) -> Vec<TreeNode> {
    if depth >= max_depth {
        return Vec::new();
    }
    let mut names = BTreeSet::new();
    for uri in uris {
        if let Some(rest) = uri.strip_prefix(prefix)
            && let Some(name) = rest.split('/').next().filter(|name| !name.is_empty())
        {
            names.insert(name.to_string());
        }
    }
    names
        .into_iter()
        .filter_map(|name| {
            let child_uri = format!("{prefix}{name}");
            let child_prefix = format!("{child_uri}/");
            let is_dir = uris.iter().any(|uri| uri.starts_with(&child_prefix));
            ContextUri::parse(&child_uri).ok().map(|uri| TreeNode {
                uri,
                is_dir,
                children: if is_dir {
                    build_tree(&child_prefix, uris, depth + 1, max_depth)
                } else {
                    Vec::new()
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContextMeta, MediaType};
    use uwu_database::config::{
        CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend,
        VectorBackend, VectorConfig,
    };

    async fn store() -> SqliteContextStore {
        let cfg = RuntimeConfig {
            deploy: DeployConfig::default(),
            database: DbConfig {
                backend: SqlBackend::Sqlite,
                url: "sqlite::memory:".into(),
                max_connections: 1,
                min_connections: 0,
                acquire_timeout_secs: 5,
                idle_timeout_secs: 60,
                max_lifetime_secs: 300,
                test_before_acquire: false,
                statement_cache_capacity: 100,
                application_name: None,
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
        };
        let db = uwu_database::Database::connect(&cfg).await.unwrap();
        migrate_sqlite(&db.pool).await.unwrap();
        SqliteContextStore::try_new(Arc::new(db.pool), GraphCentralityConfig::default()).unwrap()
    }

    fn entry(uri: &str, text: &str) -> ContextEntry {
        let mut entry = ContextEntry::new_text(
            ContextUri::parse(uri).unwrap(),
            TenantId(Uuid::new_v4()),
            text,
        );
        entry.metadata = ContextMeta {
            content_type: Some(ContentType::Fact),
            tags: vec!["sqlite".into()],
            custom: serde_json::json!({"source": "test"}),
            ..ContextMeta::default()
        };
        entry.media_type = MediaType::Text;
        entry
    }

    #[tokio::test]
    async fn every_pooled_connection_enforces_foreign_keys() {
        let cfg = RuntimeConfig {
            deploy: DeployConfig::default(),
            database: DbConfig {
                backend: SqlBackend::Sqlite,
                url: "sqlite::memory:".into(),
                max_connections: 4,
                min_connections: 4,
                acquire_timeout_secs: 5,
                idle_timeout_secs: 60,
                max_lifetime_secs: 300,
                test_before_acquire: false,
                statement_cache_capacity: 100,
                application_name: None,
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
        };
        let db = uwu_database::Database::connect(&cfg).await.unwrap();
        let pool = db.pool.as_sqlite().unwrap();
        let mut connections = Vec::new();
        for _ in 0..4 {
            connections.push(pool.acquire().await.unwrap());
        }
        for connection in &mut connections {
            let enabled: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
                .fetch_one(&mut **connection)
                .await
                .unwrap();
            assert_eq!(enabled, 1);
        }
    }

    #[tokio::test]
    async fn write_read_and_mvcc_are_atomic() {
        let store = store().await;
        let mut value = entry("uwu://tenant/agent/a/memory/fact/sqlite/one", "first");
        assert_eq!(
            ContentRepo::write(&store, value.clone()).await.unwrap(),
            MvccVersion(1)
        );
        value.payload = ContentPayload::Text {
            sparse: "second".into(),
            dense: "second dense".into(),
            full: "second full".into(),
        };
        // A stale caller version cannot reset or reuse the database-owned sequence.
        value.mvcc_version = MvccVersion(0);
        assert_eq!(
            ContentRepo::write(&store, value).await.unwrap(),
            MvccVersion(2)
        );
        let payload = FsOps::read(
            &store,
            &ContextUri::parse("uwu://tenant/agent/a/memory/fact/sqlite/one").unwrap(),
            ContentLevel::L2,
        )
        .await
        .unwrap();
        assert_eq!(payload.sparse_text(), "second");
        let history = store
            .version_history(
                &ContextUri::parse("uwu://tenant/agent/a/memory/fact/sqlite/one").unwrap(),
                PageRequest::default(),
            )
            .await
            .unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn batch_browsing_graph_and_blob_paths_work() {
        let store = store().await;
        let entries = vec![
            entry("uwu://tenant/agent/a/memory/fact/sqlite/a", "alpha needle"),
            entry("uwu://tenant/agent/a/memory/fact/sqlite/b", "beta"),
        ];
        assert_eq!(
            ContentRepo::batch_write(&store, &entries).await.unwrap(),
            vec![MvccVersion(1), MvccVersion(1)]
        );
        let root = ContextUri::parse("uwu://tenant/agent/a/memory/fact/sqlite").unwrap();
        assert_eq!(
            FsOps::ls(&store, &root, PageRequest::default())
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(FsOps::grep(&store, "needle", &root).await.unwrap().len(), 1);
        store
            .add_edge(
                &entries[0].uri,
                &entries[1].uri,
                GraphRelation::Corroborates,
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .outgoing_neighbors(&entries[0].uri, Some(GraphRelation::Corroborates))
                .await
                .unwrap(),
            vec![entries[1].uri.clone()]
        );
        let blob = store.put(b"sqlite blob", "text/plain").await.unwrap();
        assert!(store.dedup_check(&blob.hash).await.unwrap());
        assert_eq!(store.get(&blob).await.unwrap(), b"sqlite blob");
    }

    #[tokio::test]
    async fn tenant_boundaries_delete_edges_and_literal_prefixes_are_enforced() {
        let store = store().await;
        let tenant = Uuid::new_v4();
        let other = Uuid::new_v4();
        let mismatched = ContextEntry::new_text(
            ContextUri::parse(format!("uwu://{tenant}/agent/a/memory/fact/mismatch")).unwrap(),
            TenantId(other),
            "invalid",
        );
        assert!(matches!(
            ContentRepo::write(&store, mismatched).await,
            Err(ContextError::InvalidUri(_))
        ));

        let literal = entry(
            "uwu://tenant/agent/a/memory/fact/literal_100%/one",
            "literal",
        );
        let wildcard_neighbor = entry(
            "uwu://tenant/agent/a/memory/fact/literalX100Y/two",
            "wildcard",
        );
        ContentRepo::batch_write(&store, &[literal.clone(), wildcard_neighbor.clone()])
            .await
            .unwrap();
        assert_eq!(
            ContentStore::scan_by_prefix(
                &store,
                "uwu://tenant/agent/a/memory/fact/literal_100%",
                PageRequest::new(10),
            )
            .await
            .unwrap()
            .len(),
            1
        );

        store
            .add_edge(
                &literal.uri,
                &wildcard_neighbor.uri,
                GraphRelation::DerivedFrom,
            )
            .await
            .unwrap();
        ContentRepo::delete(&store, &literal.uri).await.unwrap();
        assert!(
            store
                .outgoing_neighbors(&wildcard_neighbor.uri, None)
                .await
                .unwrap()
                .is_empty()
        );

        let cross_tenant =
            ContextUri::parse("uwu://other/agent/a/memory/fact/literal_100%/one").unwrap();
        assert!(matches!(
            ContentRepo::rename(&store, &wildcard_neighbor.uri, &cross_tenant).await,
            Err(ContextError::PermissionDenied(_))
        ));
    }

    #[tokio::test]
    async fn graph_revision_cache_and_edge_mutations_follow_contract() {
        let store = store().await;
        let a = entry("uwu://tenant/agent/a/memory/fact/graph/a", "a");
        let b = entry("uwu://tenant/agent/a/memory/fact/graph/b", "b");
        let c = entry("uwu://tenant/agent/a/memory/fact/graph/c", "c");
        ContentRepo::batch_write(&store, &[a.clone(), b.clone(), c.clone()])
            .await
            .unwrap();
        store
            .add_edge(&a.uri, &b.uri, GraphRelation::DerivedFrom)
            .await
            .unwrap();
        store
            .add_edge(&b.uri, &c.uri, GraphRelation::DerivedFrom)
            .await
            .unwrap();
        assert_eq!(store.graph_revision(&a.uri).await.unwrap(), 2);
        let first = store.centrality(&b.uri).await.unwrap();
        assert_eq!(first, store.centrality(&b.uri).await.unwrap());
        let snapshots: i64 =
            sqlx::query_scalar("SELECT count(DISTINCT revision) FROM context_graph_centrality")
                .fetch_one(store.sqlite_pool())
                .await
                .unwrap();
        assert_eq!(snapshots, 1);

        let renamed = ContextUri::parse("uwu://tenant/agent/a/memory/fact/graph/b2").unwrap();
        ContentRepo::rename(&store, &b.uri, &renamed).await.unwrap();
        assert_eq!(store.graph_revision(&a.uri).await.unwrap(), 3);
        assert!(store.centrality(&renamed).await.unwrap() > 0.0);
        ContentRepo::delete(&store, &renamed).await.unwrap();
        assert_eq!(store.graph_revision(&a.uri).await.unwrap(), 4);
        let mutations: Vec<String> =
            sqlx::query_scalar("SELECT operation FROM context_graph_mutations ORDER BY revision")
                .fetch_all(store.sqlite_pool())
                .await
                .unwrap();
        assert_eq!(mutations, ["add", "add", "rename", "delete"]);
    }

    #[tokio::test]
    async fn rename_and_rollback_preserve_history() {
        let store = store().await;
        let from = ContextUri::parse("uwu://tenant/agent/a/memory/fact/sqlite/from").unwrap();
        let to = ContextUri::parse("uwu://tenant/agent/a/memory/fact/sqlite/to").unwrap();
        let mut value = entry(from.as_str(), "v1");
        ContentRepo::write(&store, value.clone()).await.unwrap();
        value.payload = ContentPayload::Text {
            sparse: "v2".into(),
            dense: "v2".into(),
            full: "v2".into(),
        };
        ContentRepo::write(&store, value).await.unwrap();
        ContentRepo::rename(&store, &from, &to).await.unwrap();
        store.rollback(&to, MvccVersion(1)).await.unwrap();
        assert_eq!(
            FsOps::read(&store, &to, ContentLevel::L0)
                .await
                .unwrap()
                .sparse_text(),
            "v1"
        );
        assert_eq!(
            store
                .version_history(&to, PageRequest::default())
                .await
                .unwrap()
                .len(),
            3
        );
        assert!(matches!(
            FsOps::read(&store, &from, ContentLevel::L0).await,
            Err(ContextError::NotFound(_))
        ));
    }
}
