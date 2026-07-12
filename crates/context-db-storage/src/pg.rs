//! PG 适配器：用 uwu_database::DbPool 实现 context-db 的四个窄端口。
//!
//! - [`PgContextStore`] 同时实现 `FsOps` + `ContentRepo` + `VersionOps` + `TenantOps`，
//!   自动满足 `ContextStore` supertrait。
//! - URI 寻址通过 `context_entries` 表的 TEXT 列 + LIKE 前缀查询实现。
//! - 版本历史通过 `context_versions` 表存储完整快照。
use agent_context_db_core::{Page, PageRequest};

use agent_context_db_core::{
    AclProtectedStore, BlobRef, BlobStore, BrowsingOps, ContentHash, ContentLevel, ContentPayload,
    ContentRepo, ContentStore, ContentType, ContextDiff, ContextEntry, ContextError, ContextUri,
    DirEntry, FindPattern, FsOps, GraphRelation, GraphStore, GrepHit, MvccVersion, PathAcl,
    Principal, Result, StateScope, StorageEngine, TenantId, TenantOps, TreeNode, VersionEntry,
    VersionOps, sanitize_entry_for_write,
};
use async_trait::async_trait;
use sqlx::{Postgres, Row, Transaction};
use uuid::Uuid;

use std::sync::Arc;
use uwu_database::DbPool;

use crate::graph::{
    BatchWriteConfig, GraphCentralityConfig, incremental_pagerank_scores, pagerank_scores,
};
use crate::outbox::{
    IndexMutation, collection_from_entry, enqueue_pg, point_from_entry, upsert_mutation,
};

// ===========================================================================
// PgContextStore
// ===========================================================================

/// PG 适配器：持有 `DbPool` + 可选内容缓存。
#[derive(Clone)]
pub struct PgContextStore {
    pool: sqlx::PgPool,
    /// 可选的 L0/L1 读取缓存（通过 UwuCacheAdapter 桥接 uwu_database::Cache）。
    read_cache: Option<Arc<dyn agent_context_db_core::ReadCache>>,
    centrality_config: GraphCentralityConfig,
    batch_config: BatchWriteConfig,
}

impl PgContextStore {
    /// C.3: 构造时验证后端类型，避免运行时 expect panic。
    pub fn new(pool: Arc<DbPool>, centrality_config: GraphCentralityConfig) -> Result<Self> {
        let pool = pool
            .as_postgres()
            .map_err(|error| {
                ContextError::Storage(format!("PgContextStore requires postgres backend: {error}"))
            })?
            .clone();
        Ok(Self {
            pool,
            read_cache: None,
            centrality_config,
            batch_config: BatchWriteConfig::default(),
        })
    }

    /// 注入读取缓存（使用 uwu_database::Cache 或任意 ReadCache 实现）。
    pub fn with_cache(mut self, cache: Arc<dyn agent_context_db_core::ReadCache>) -> Self {
        self.read_cache = Some(cache);
        self
    }

    fn pg_pool(&self) -> &sqlx::PgPool {
        &self.pool
    }

    /// I.3: 统一 map_err 辅助 —— 记录 tracing::error! 并返回 `ContextError::Storage`。
    fn storage_err(op: &str, e: impl std::error::Error + 'static) -> ContextError {
        tracing::error!(
            op,
            error = ?agent_context_db_core::ErrorReport::from_error(&e),
            "storage operation failed"
        );
        ContextError::Storage(format!("{op} failed: {e}"))
    }

    /// 目录前缀：`{dir_uri}/` 用于 LIKE 查询。
    fn dir_prefix(dir: &ContextUri) -> String {
        let s = dir.to_string().trim_end_matches('/').to_string();
        format!("{}/", s)
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

    fn version_from_sql(version: i64) -> Result<MvccVersion> {
        u64::try_from(version).map(MvccVersion).map_err(|_| {
            ContextError::Storage(format!("invalid negative PostgreSQL version {version}"))
        })
    }

    fn page_limit(page: &PageRequest) -> Result<i64> {
        i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds PostgreSQL BIGINT".into()))
    }

    fn finish_page<T>(mut items: Vec<T>, limit: usize, cursor: impl Fn(&T) -> String) -> Page<T> {
        let has_more = items.len() > limit;
        if has_more {
            items.truncate(limit);
        }
        let next_cursor = has_more.then(|| items.last().map(&cursor)).flatten();
        Page::new(items, next_cursor)
    }

    async fn bump_graph_revision(
        tx: &mut Transaction<'_, Postgres>,
        scope: &str,
        operation: &str,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<u64> {
        let revision: i64 = sqlx::query_scalar("INSERT INTO context_graph_revisions(scope, revision) VALUES ($1,1) ON CONFLICT(scope) DO UPDATE SET revision=context_graph_revisions.revision+1 RETURNING revision")
            .bind(scope).fetch_one(&mut **tx).await.map_err(|e| Self::storage_err("bump graph revision", e))?;
        sqlx::query("INSERT INTO context_graph_mutations(scope, revision, operation, from_uri, to_uri) VALUES ($1,$2,$3,$4,$5)")
            .bind(scope).bind(revision).bind(operation).bind(from).bind(to).execute(&mut **tx).await
            .map_err(|e| Self::storage_err("log graph mutation", e))?;
        u64::try_from(revision).map_err(|_| ContextError::Storage("negative graph revision".into()))
    }

    async fn write_in_tx(
        tx: &mut Transaction<'_, Postgres>,
        input: &ContextEntry,
    ) -> Result<MvccVersion> {
        input.validate_tenant_binding()?;
        let mut entry = sanitize_entry_for_write(input)?;
        let uri = entry.uri.to_string();
        let (l0, l1, _) = extract_payload_levels(&entry.payload);
        let content_type = entry
            .metadata
            .content_type
            .unwrap_or(ContentType::Evidence)
            .as_path_segment();
        let state_scope = entry.metadata.state_scope.map(|scope| match scope {
            StateScope::Short => "short",
            StateScope::Mid => "mid",
            StateScope::Long => "long",
        });
        let tags = serde_json::to_value(&entry.metadata.tags)?;
        let custom = entry.metadata.custom.clone();
        let now = chrono::Utc::now();
        entry.updated_at = now;

        let mvcc: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO context_entries
                (uri, tenant_id, l0_abstract, l1_overview, l2_detail_ref,
                 content_type, state_scope, tags, custom, entry_json,
                 mvcc_version, created_at, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'{}'::jsonb,1,$10,$11)
            ON CONFLICT (uri) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                l0_abstract = EXCLUDED.l0_abstract,
                l1_overview = EXCLUDED.l1_overview,
                l2_detail_ref = EXCLUDED.l2_detail_ref,
                content_type = EXCLUDED.content_type,
                state_scope = EXCLUDED.state_scope,
                tags = EXCLUDED.tags,
                custom = EXCLUDED.custom,
                mvcc_version = context_entries.mvcc_version + 1,
                updated_at = EXCLUDED.updated_at
            RETURNING mvcc_version
            "#,
        )
        .bind(&uri)
        .bind(entry.tenant.0)
        .bind(&l0)
        .bind(&l1)
        .bind::<Option<Uuid>>(None)
        .bind(content_type)
        .bind(state_scope)
        .bind(&tags)
        .bind(&custom)
        .bind(entry.created_at)
        .bind(now)
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| Self::storage_err("write entry", e))?;
        let version = u64::try_from(mvcc)
            .map_err(|_| ContextError::Storage(format!("invalid MVCC version {mvcc}")))?;
        entry.mvcc_version = MvccVersion(version);
        let entry_json = serde_json::to_value(&entry)?;

        sqlx::query("UPDATE context_entries SET entry_json = $2 WHERE uri = $1")
            .bind(&uri)
            .bind(&entry_json)
            .execute(&mut **tx)
            .await
            .map_err(|e| Self::storage_err("store current entry JSON", e))?;
        sqlx::query(
            r#"
            INSERT INTO context_versions
                (uri, mvcc_version, tenant_id, l0_abstract, l1_overview,
                 l2_detail_ref, content_type, state_scope, tags, custom,
                 entry_json, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            "#,
        )
        .bind(&uri)
        .bind(mvcc)
        .bind(entry.tenant.0)
        .bind(&l0)
        .bind(&l1)
        .bind::<Option<Uuid>>(None)
        .bind(content_type)
        .bind(state_scope)
        .bind(&tags)
        .bind(&custom)
        .bind(&entry_json)
        .bind(now)
        .execute(&mut **tx)
        .await
        .map_err(|e| Self::storage_err("write version", e))?;
        Ok(MvccVersion(version))
    }
}

/// 从 ContentPayload 提取各层文本（L0/L1/L2）。
fn extract_payload_levels(payload: &ContentPayload) -> (String, Option<String>, String) {
    let projection = payload.index_projection();
    (projection.l0, projection.l1, projection.l2)
}

fn validate_blob_ref(
    blob_ref: &BlobRef,
    data: &[u8],
    mime_type: &str,
    stored_size: i64,
) -> Result<()> {
    let actual_size = data.len();
    if blake3::hash(data).to_hex().to_string() != blob_ref.hash.0
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
            let keys = left
                .keys()
                .chain(right.keys())
                .collect::<std::collections::BTreeSet<_>>();
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
        _ if left != right => output.push(format!("~ {path}: {left} -> {right}")),
        _ => {}
    }
}

/// 从 L0 摘要和 L1 概览重建 ContentPayload。
fn payload_from_levels(l0: &str, l1: Option<&str>, _l2_ref: Option<Uuid>) -> ContentPayload {
    ContentPayload::Text {
        sparse: l0.to_string(),
        dense: l1.unwrap_or("").to_string(),
        full: l1.unwrap_or(l0).to_string(),
    }
}

// ===========================================================================
// ContentRepo 实现
// ===========================================================================

#[async_trait]
impl ContentRepo for PgContextStore {
    #[tracing::instrument(skip(self, entry), fields(uri = tracing::field::Empty))]
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let uri = entry.uri.clone();
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin write", e))?;
        let version = Self::write_in_tx(&mut tx, &entry).await?;
        if let Some(mutation) = upsert_mutation(&entry, version)? {
            enqueue_pg(&mut tx, &mutation).await?;
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit write", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(&uri).await;
        }
        Ok(version)
    }

    #[tracing::instrument(skip(self), fields(uri = tracing::field::Empty))]
    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin transaction", e))?;
        let uri_str = uri.to_string();
        let affected = sqlx::query("DELETE FROM context_entries WHERE uri = $1")
            .bind(&uri_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("delete", e))?;

        if affected.rows_affected() == 0 {
            return Err(ContextError::NotFound(uri_str));
        }
        let collection = sqlx::query_scalar::<_, serde_json::Value>(
            "SELECT entry_json FROM context_versions WHERE uri = $1 ORDER BY mvcc_version DESC LIMIT 1",
        )
        .bind(&uri_str)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("read deleted collection", e))?
        .and_then(|value| serde_json::from_value::<ContextEntry>(value).ok())
        .map(|entry| collection_from_entry(&entry))
        .unwrap_or_else(|| crate::outbox::DEFAULT_COLLECTION.to_owned());
        enqueue_pg(
            &mut tx,
            &IndexMutation::Delete {
                collection,
                uri: uri.clone(),
            },
        )
        .await?;
        // Also delete versions — error propagates with transaction
        sqlx::query("DELETE FROM context_versions WHERE uri = $1")
            .bind(&uri_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("delete versions", e))?;
        let graph_changed =
            sqlx::query("DELETE FROM context_relations WHERE from_uri = $1 OR to_uri = $1")
                .bind(&uri_str)
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
            .map_err(|e| Self::storage_err("commit", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(uri).await;
        }
        Ok(())
    }

    #[tracing::instrument(skip(self, from, to), fields(from = tracing::field::Empty, to = tracing::field::Empty))]
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        if from.tenant() != to.tenant() {
            return Err(ContextError::PermissionDenied(format!(
                "cross-tenant rename is not allowed: {} -> {}",
                from.tenant(),
                to.tenant()
            )));
        }
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin transaction", e))?;
        let from_str = from.to_string();
        let to_str = to.to_string();

        let current_json: Option<serde_json::Value> =
            sqlx::query_scalar("SELECT entry_json FROM context_entries WHERE uri = $1")
                .bind(&from_str)
                .fetch_optional(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("read rename source", e))?;
        let mut entry: ContextEntry = serde_json::from_value(
            current_json.ok_or_else(|| ContextError::NotFound(from_str.clone()))?,
        )?;
        entry.uri = to.clone();
        entry.updated_at = chrono::Utc::now();
        let entry_json = serde_json::to_value(&entry)?;
        let affected = sqlx::query(
            "UPDATE context_entries SET uri = $1, entry_json = $2, updated_at = $3 WHERE uri = $4",
        )
        .bind(&to_str)
        .bind(&entry_json)
        .bind(entry.updated_at)
        .bind(&from_str)
        .execute(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("rename", e))?;

        if affected.rows_affected() == 0 {
            return Err(ContextError::NotFound(from_str));
        }

        let versions: Vec<(i64, serde_json::Value)> =
            sqlx::query_as("SELECT mvcc_version, entry_json FROM context_versions WHERE uri = $1")
                .bind(&from_str)
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("read rename versions", e))?;
        for (version, json) in versions {
            let mut historical: ContextEntry = serde_json::from_value(json)?;
            historical.uri = to.clone();
            sqlx::query(
                "UPDATE context_versions SET uri = $1, entry_json = $2 WHERE uri = $3 AND mvcc_version = $4",
            )
            .bind(&to_str)
            .bind(serde_json::to_value(&historical)?)
            .bind(&from_str)
            .bind(version)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("rename version", e))?;
        }
        let relations: Vec<(String, String, String, chrono::DateTime<chrono::Utc>)> =
            sqlx::query_as("SELECT from_uri, to_uri, relation_kind, created_at FROM context_relations WHERE from_uri = $1 OR to_uri = $1")
                .bind(&from_str)
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("read rename relations", e))?;
        sqlx::query("DELETE FROM context_relations WHERE from_uri = $1 OR to_uri = $1")
            .bind(&from_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("remove renamed relations", e))?;
        let graph_changed = !relations.is_empty();
        for (edge_from, edge_to, kind, created_at) in relations {
            sqlx::query("INSERT INTO context_relations (from_uri, to_uri, relation_kind, created_at) VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING")
                .bind(if edge_from == from_str { &to_str } else { &edge_from })
                .bind(if edge_to == from_str { &to_str } else { &edge_to })
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

        enqueue_pg(
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
            .map_err(|e| Self::storage_err("commit", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(from).await;
            cache.invalidate(to).await;
        }
        Ok(())
    }

    /// 批量写入：所有当前行和版本快照共享一个事务。
    #[tracing::instrument(skip(self, entries), fields(count = entries.len()))]
    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        if entries.is_empty() {
            return Ok(vec![]);
        }
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("batch begin", e))?;

        let mut versions = Vec::with_capacity(entries.len());
        for chunk in self.batch_config.chunks(entries)? {
            let uris = chunk
                .iter()
                .map(|entry| entry.uri.to_string())
                .collect::<Vec<_>>();
            let base_versions = sqlx::query_as::<_, (String, i64)>(
                "SELECT uri, mvcc_version FROM context_entries WHERE uri = ANY($1) FOR UPDATE",
            )
            .bind(&uris)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("lock batch versions", e))?
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
            let (rows, chunk_versions) = crate::graph::prepare_batch_chunk(chunk, &base_versions)?;
            let json = serde_json::to_value(&rows)?;
            sqlx::query(r#"
                WITH input AS (SELECT * FROM jsonb_to_recordset($1::jsonb) AS x(uri text,tenant_id text,l0 text,l1 text,l2 text,content_type text,state_scope text,tags jsonb,custom jsonb,entry jsonb,version bigint,created_at text,updated_at text,is_current boolean))
                INSERT INTO context_entries(uri,tenant_id,l0_abstract,l1_overview,l2_detail_ref,content_type,state_scope,tags,custom,entry_json,mvcc_version,created_at,updated_at)
                SELECT uri,tenant_id::uuid,l0,l1,NULL,content_type,state_scope,tags,custom,entry,version,created_at::timestamptz,updated_at::timestamptz FROM input WHERE is_current
                ON CONFLICT(uri) DO UPDATE SET tenant_id=excluded.tenant_id,l0_abstract=excluded.l0_abstract,l1_overview=excluded.l1_overview,l2_detail_ref=excluded.l2_detail_ref,content_type=excluded.content_type,state_scope=excluded.state_scope,tags=excluded.tags,custom=excluded.custom,entry_json=excluded.entry_json,mvcc_version=excluded.mvcc_version,updated_at=excluded.updated_at
            "#).bind(&json).execute(&mut *tx).await.map_err(|e| Self::storage_err("write batch current", e))?;
            sqlx::query(r#"INSERT INTO context_versions(uri,mvcc_version,tenant_id,l0_abstract,l1_overview,l2_detail_ref,content_type,state_scope,tags,custom,entry_json,created_at)
                SELECT uri,version,tenant_id::uuid,l0,l1,NULL,content_type,state_scope,tags,custom,entry,updated_at::timestamptz FROM jsonb_to_recordset($1::jsonb) AS x(uri text,tenant_id text,l0 text,l1 text,content_type text,state_scope text,tags jsonb,custom jsonb,entry jsonb,version bigint,updated_at text)"#)
                .bind(&json).execute(&mut *tx).await.map_err(|e| Self::storage_err("write batch history", e))?;
            sqlx::query(r#"INSERT INTO context_index_outbox(id,mutation_json,uri,mvcc_version,status,attempts,available_at,created_at,updated_at)
                SELECT outbox_id::uuid,mutation,uri,version,'pending',0,updated_at::timestamptz,updated_at::timestamptz,updated_at::timestamptz FROM jsonb_to_recordset($1::jsonb) AS x(ordinal bigint,uri text,version bigint,outbox_id text,mutation jsonb,updated_at text) WHERE mutation IS NOT NULL ORDER BY ordinal"#)
                .bind(&json).execute(&mut *tx).await.map_err(|e| Self::storage_err("write batch outbox", e))?;
            versions.extend(chunk_versions);
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("batch commit", e))?;
        if let Some(cache) = &self.read_cache {
            for entry in entries {
                cache.invalidate(&entry.uri).await;
            }
        }
        Ok(versions)
    }
}

/// D.7: 连接池配置 — 从 UwuConfig 构建。
#[derive(Debug, Clone)]
pub struct PgStoreConfig {
    pub max_connections: u32,
    pub min_connections: u32,
    pub statement_cache_capacity: usize,
}

impl Default for PgStoreConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            statement_cache_capacity: 100,
        }
    }
}

impl PgStoreConfig {
    pub fn from_uwu_config(cfg: &agent_context_db_core::config::StorageConfig) -> Result<Self> {
        Ok(Self {
            max_connections: crate::max_connections(cfg)?,
            ..Default::default()
        })
    }
}

// ===========================================================================
// FsOps 实现
// ===========================================================================

#[async_trait]
impl FsOps for PgContextStore {
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        let prefix = Self::dir_prefix(dir);
        let after = page.after.as_deref().unwrap_or("");
        let limit = Self::page_limit(&page)?;
        let rows = sqlx::query_as::<_, (String, String, String, Option<String>)>(
            r#"
            WITH children AS (
                SELECT DISTINCT ON (child_uri)
                       child_uri, uri, l0_abstract, content_type
                FROM (
                    SELECT concat($1, split_part(substring(uri FROM char_length($1) + 1), '/', 1)) AS child_uri,
                           uri, l0_abstract, content_type
                    FROM context_entries
                    WHERE uri LIKE $2 ESCAPE '\\'
                ) candidates
                ORDER BY child_uri, uri
            )
            SELECT child_uri, uri, l0_abstract, content_type
            FROM children WHERE child_uri > $3
            ORDER BY child_uri LIMIT $4
            "#,
        )
        .bind(&prefix)
        .bind(Self::prefix_pattern(&prefix))
        .bind(after)
        .bind(limit)
        .fetch_all(self.pg_pool())
        .await
        .map_err(|e| Self::storage_err("ls", e))?;
        let items = rows
            .into_iter()
            .map(|(child, uri, abstract_, content_type)| {
                let is_dir = child != uri;
                Ok(DirEntry {
                    uri: ContextUri::parse(child)?,
                    is_dir,
                    abstract_: if is_dir { String::new() } else { abstract_ },
                    content_type: if is_dir {
                        None
                    } else {
                        content_type
                            .as_deref()
                            .and_then(ContentType::from_path_segment)
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::finish_page(items, page.effective_limit(), |item| {
            item.uri.to_string()
        }))
    }

    async fn find(&self, pattern: &FindPattern, page: PageRequest) -> Result<Page<ContextUri>> {
        let pg = self.pg_pool();
        let scope = pattern
            .scope
            .as_ref()
            .map(|u| u.to_string())
            .unwrap_or_default();

        let after = page.after.as_deref().unwrap_or("");
        let limit = Self::page_limit(&page)?;
        let content_type = pattern.content_type.map(|kind| kind.as_path_segment());
        let name_pattern = pattern.name_glob.as_deref().map(Self::glob_pattern);
        let results: Vec<String> = sqlx::query_scalar(
            r#"SELECT uri FROM context_entries
               WHERE uri LIKE $1 ESCAPE '\\' AND uri > $2
                 AND ($3::text IS NULL OR content_type = $3)
                 AND ($4::text IS NULL OR split_part(uri, '/', -1) LIKE $4 ESCAPE '\\')
               ORDER BY uri LIMIT $5"#,
        )
        .bind(Self::prefix_pattern(&scope))
        .bind(after)
        .bind(content_type)
        .bind(name_pattern)
        .bind(limit)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("find", e))?;
        let items = results
            .into_iter()
            .map(ContextUri::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self::finish_page(
            items,
            page.effective_limit(),
            ToString::to_string,
        ))
    }

    async fn grep(&self, regex: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        let pg = self.pg_pool();
        // 该接口当前定义为不区分大小写的字面量包含匹配。
        let pattern = format!("%{}%", Self::escape_like(regex));

        let rows = sqlx::query_as::<_, (String, String, Option<String>)>(
            r#"
            SELECT uri, l0_abstract, l1_overview FROM context_entries
            WHERE uri LIKE $1 ESCAPE '\\'
              AND (l0_abstract ILIKE $2 ESCAPE '\\' OR l1_overview ILIKE $2 ESCAPE '\\')
            ORDER BY uri
            "#,
        )
        .bind(Self::prefix_pattern(&Self::dir_prefix(scope)))
        .bind(&pattern)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("grep", e))?;

        let mut hits = Vec::new();
        for (uri_str, l0, l1) in rows {
            // 返回匹配行（截取包含匹配词的部分）
            let matched_line = if l0.to_lowercase().contains(&regex.to_lowercase()) {
                l0
            } else if let Some(ov) = &l1 {
                ov.clone()
            } else {
                continue;
            };
            let uri_parsed = match ContextUri::parse(uri_str) {
                Ok(u) => u,
                Err(_) => continue,
            };
            hits.push(GrepHit {
                uri: uri_parsed,
                line: matched_line,
                level: ContentLevel::L0,
            });
        }
        Ok(hits)
    }

    async fn tree(
        &self,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        let pg = self.pg_pool();
        let prefix = Self::dir_prefix(root);

        let after = page.after.as_deref().unwrap_or("");
        let limit = Self::page_limit(&page)?;
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE uri LIKE $1 ESCAPE '\\' AND uri > $2 ORDER BY uri LIMIT $3",
        )
        .bind(Self::prefix_pattern(&prefix))
        .bind(after)
        .bind(limit)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("tree", e))?;
        let has_more = rows.len() > page.effective_limit();
        let selected = &rows[..rows.len().min(page.effective_limit())];
        let next_cursor = has_more.then(|| selected.last().cloned()).flatten();
        Ok(Page::new(
            vec![TreeNode {
                uri: root.clone(),
                is_dir: true,
                children: build_tree_level(&prefix, selected, 0, depth),
            }],
            next_cursor,
        ))
    }

    #[tracing::instrument(skip(self), fields(uri = tracing::field::Empty, level = ?level))]
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        // 0. 缓存命中（L0/L1 可缓存，L2 不走缓存）
        if level != ContentLevel::L2
            && let Some(ref cache) = self.read_cache
        {
            match cache.get(uri, level).await {
                Some(Some(cached)) => return Ok(cached),
                // 负缓存命中：已知该 URI 缺失，直接 NotFound（穿透防护）
                Some(None) => return Err(ContextError::NotFound(uri.to_string())),
                None => { /* miss，继续回源 */ }
            }
        }

        let pg = self.pg_pool();
        let uri_str = uri.to_string();

        let row = sqlx::query_as::<
            _,
            (String, Option<String>, Option<uuid::Uuid>, Option<serde_json::Value>),
        >(
            "SELECT l0_abstract, l1_overview, l2_detail_ref, entry_json FROM context_entries WHERE uri = $1",
        )
        .bind(&uri_str)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("read", e))?;

        let result = match row {
            None => return Err(ContextError::NotFound(uri_str)),
            Some((l0, l1, _l2_ref, entry_json)) => match level {
                ContentLevel::L0 => Ok(payload_from_levels(&l0, l1.as_deref(), None)),
                ContentLevel::L1 => Ok(ContentPayload::Text {
                    sparse: l0.clone(),
                    dense: l1.unwrap_or_default(),
                    full: l0,
                }),
                ContentLevel::L2 => {
                    let json = entry_json.ok_or_else(|| {
                        ContextError::Storage(format!(
                            "entry {uri_str} predates complete JSON migration and has no recoverable snapshot"
                        ))
                    })?;
                    let entry: ContextEntry = serde_json::from_value(json)?;
                    Ok(entry.payload)
                }
            },
        };

        // 写回缓存（L0/L1 结果）；NotFound 写入负缓存（穿透防护）
        if let Some(cache) = self.read_cache.as_ref()
            && level != ContentLevel::L2
        {
            match &result {
                Ok(payload) => {
                    cache
                        .put(
                            uri,
                            level,
                            payload.clone(),
                            std::time::Duration::from_secs(300),
                        )
                        .await
                }
                Err(ContextError::NotFound(_)) => cache.put_negative(uri, level).await,
                Err(_) => {} // 其他错误不缓存，避免污染
            }
        }
        result
    }
}

// ===========================================================================
// VersionOps 实现
// ===========================================================================

#[async_trait]
impl VersionOps for PgContextStore {
    async fn version_history(
        &self,
        uri: &ContextUri,
        page: PageRequest,
    ) -> Result<Page<VersionEntry>> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();

        let after = page
            .after
            .as_deref()
            .map(str::parse::<i64>)
            .transpose()
            .map_err(|e| ContextError::Storage(format!("invalid version cursor: {e}")))?
            .unwrap_or(0);
        let rows = sqlx::query_as::<_, (i64, String, chrono::DateTime<chrono::Utc>)>(
            "SELECT mvcc_version, l0_abstract, created_at FROM context_versions WHERE uri = $1 AND mvcc_version > $2 ORDER BY mvcc_version LIMIT $3",
        )
        .bind(&uri_str).bind(after).bind(Self::page_limit(&page)?)
        .fetch_all(pg).await.map_err(|e| Self::storage_err("version_history", e))?;
        let items = rows
            .into_iter()
            .map(|(v, message, ts)| {
                Ok(VersionEntry {
                    version: Self::version_from_sql(v)?,
                    message,
                    ts,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::finish_page(items, page.effective_limit(), |item| {
            item.version.0.to_string()
        }))
    }

    async fn rollback(&self, uri: &ContextUri, to: MvccVersion) -> Result<()> {
        let uri_str = uri.to_string();
        let target_v = i64::try_from(to.0)
            .map_err(|_| ContextError::VersionConflict(format!("version {} is too large", to.0)))?;
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin rollback", e))?;
        let entry_json: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(target_v)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("rollback read", e))?;
        let current: Option<i64> = sqlx::query_scalar(
            "SELECT mvcc_version FROM context_entries WHERE uri = $1 FOR UPDATE",
        )
        .bind(&uri_str)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("lock rollback target", e))?;
        if current.is_none() {
            return Err(ContextError::NotFound(uri_str));
        }
        let mut entry: ContextEntry = serde_json::from_value(entry_json.ok_or_else(|| {
            ContextError::VersionConflict(format!("no version {} for {uri}", to.0))
        })?)?;
        entry.uri = uri.clone();
        let version = Self::write_in_tx(&mut tx, &entry).await?;
        if let Some(mutation) = upsert_mutation(&entry, version)? {
            enqueue_pg(&mut tx, &mutation).await?;
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
        let pg = self.pg_pool();
        let uri_str = uri.to_string();
        let a_version = a.0;
        let b_version = b.0;
        let a = i64::try_from(a_version).map_err(|_| {
            ContextError::VersionConflict(format!("version {a_version} is too large"))
        })?;
        let b = i64::try_from(b_version).map_err(|_| {
            ContextError::VersionConflict(format!("version {b_version} is too large"))
        })?;

        let v_a: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(a)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("diff read a", e))?;

        let v_b: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(b)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("diff read b", e))?;

        let (a_json, b_json) = match (v_a, v_b) {
            (Some(a_json), Some(b_json)) => (a_json, b_json),
            _ => {
                return Err(ContextError::VersionConflict(format!(
                    "one or both versions not found for {uri_str}"
                )));
            }
        };
        let mut changes = Vec::new();
        collect_json_changes("$", &a_json, &b_json, &mut changes);
        let summary = if changes.is_empty() {
            format!("{uri_str}: v{a_version} and v{b_version} are identical")
        } else {
            format!(
                "{uri_str}: v{a_version} -> v{b_version}\n{}",
                changes.join("\n")
            )
        };

        Ok(ContextDiff { summary })
    }
}

// ===========================================================================
// TenantOps 实现
// ===========================================================================

#[async_trait]
impl TenantOps for PgContextStore {
    async fn list_tenants(&self, page: PageRequest) -> Result<Page<TenantId>> {
        let pg = self.pg_pool();

        let after = page.after.as_deref().unwrap_or("");
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT tenant_id::text FROM context_entries WHERE tenant_id::text > $1 ORDER BY tenant_id::text LIMIT $2",
        )
        .bind(after).bind(Self::page_limit(&page)?)
        .fetch_all(pg).await.map_err(|e| Self::storage_err("list_tenants", e))?;
        let items = rows
            .into_iter()
            .map(|s| {
                Uuid::parse_str(&s)
                    .map(TenantId)
                    .map_err(|e| Self::storage_err("tenant id", e))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::finish_page(items, page.effective_limit(), |item| {
            item.0.to_string()
        }))
    }
}

// ===========================================================================
// 6 域存储端口实现（ContentStore / BrowsingOps / GraphStore / BlobStore）
// ===========================================================================

#[async_trait]
impl ContentStore for PgContextStore {
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
        let pg = self.pg_pool();
        let limit = Self::page_limit(&page)?;
        let scope = ContextUri::parse(prefix.trim_end_matches('/'))?;
        let exact = scope.to_string();
        let descendants = Self::prefix_pattern(&Self::dir_prefix(&scope));
        let rows = sqlx::query_as::<_, (serde_json::Value,)>(
            "SELECT entry_json FROM context_entries WHERE (uri = $1 OR uri LIKE $2 ESCAPE '\\') AND uri > $3 AND entry_json IS NOT NULL ORDER BY uri LIMIT $4",
        )
        .bind(&exact)
        .bind(&descendants)
        .bind(page.after.as_deref().unwrap_or(""))
        .bind(limit)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("scan_by_prefix", e))?;

        let items = rows
            .into_iter()
            .map(|(json,)| serde_json::from_value(json).map_err(Into::into))
            .collect::<Result<Vec<ContextEntry>>>()?;
        Ok(Self::finish_page(items, page.effective_limit(), |item| {
            item.uri.to_string()
        }))
    }

    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        let limit = Self::page_limit(&page)?;
        let scope = ContextUri::parse(prefix.trim_end_matches('/'))?;
        let exact = scope.to_string();
        let descendants = Self::prefix_pattern(&Self::dir_prefix(&scope));
        let rows = sqlx::query_as::<_, (serde_json::Value,)>(
            "SELECT entry_json FROM context_entries WHERE (uri = $1 OR uri LIKE $2 ESCAPE '\\') AND content_type = $3 AND uri > $4 AND entry_json IS NOT NULL ORDER BY uri LIMIT $5",
        )
        .bind(&exact)
        .bind(&descendants)
        .bind(content_type.as_path_segment())
        .bind(page.after.as_deref().unwrap_or(""))
        .bind(limit)
        .fetch_all(self.pg_pool())
        .await
        .map_err(|e| Self::storage_err("scan_by_type", e))?;
        let items = rows
            .into_iter()
            .map(|(json,)| serde_json::from_value(json).map_err(Into::into))
            .collect::<Result<Vec<ContextEntry>>>()?;
        Ok(Self::finish_page(items, page.effective_limit(), |item| {
            item.uri.to_string()
        }))
    }
}

#[async_trait]
impl BrowsingOps for PgContextStore {
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
        let pg = self.pg_pool();
        let descendants = Self::prefix_pattern(&Self::dir_prefix(scope));
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE uri LIKE $1 ESCAPE '\\' AND strpos(uri, $2) > 0 AND uri > $3 ORDER BY uri LIMIT $4",
        )
        .bind(&descendants)
        .bind(pattern)
        .bind(page.after.as_deref().unwrap_or(""))
        .bind(Self::page_limit(&page)?)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("find", e))?;
        let items = rows
            .into_iter()
            .map(ContextUri::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self::finish_page(
            items,
            page.effective_limit(),
            ToString::to_string,
        ))
    }

    async fn grep(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<GrepHit>> {
        <Self as FsOps>::grep(self, pattern, scope).await
    }
}

#[async_trait]
impl GraphStore for PgContextStore {
    async fn add_edge(
        &self,
        from: &ContextUri,
        to: &ContextUri,
        kind: GraphRelation,
    ) -> Result<()> {
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin add edge", e))?;
        let changed = sqlx::query(
            "INSERT INTO context_relations (from_uri, to_uri, relation_kind, created_at) VALUES ($1,$2,$3,now()) ON CONFLICT DO NOTHING",
        ).bind(from.to_string()).bind(to.to_string()).bind(format!("{:?}", kind))
        .execute(&mut *tx).await.map_err(|e| Self::storage_err("add_edge", e))?.rows_affected();
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
        let pg = self.pg_pool();
        sqlx::query("DELETE FROM context_relations WHERE from_uri = $1 AND to_uri = $2")
            .bind(from.to_string())
            .bind(to.to_string())
            .execute(pg)
            .await
            .map_err(|e| Self::storage_err("remove_edge", e))?;
        Ok(())
    }

    async fn outgoing_neighbors(
        &self,
        uri: &ContextUri,
        kind: Option<GraphRelation>,
    ) -> Result<Vec<ContextUri>> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();
        let rows: Vec<String> = if let Some(k) = kind {
            sqlx::query_scalar("SELECT DISTINCT to_uri FROM context_relations WHERE from_uri = $1 AND relation_kind = $2 ORDER BY to_uri")
                .bind(&uri_str).bind(format!("{:?}", k)).fetch_all(pg).await
        } else {
            sqlx::query_scalar("SELECT DISTINCT to_uri FROM context_relations WHERE from_uri = $1 ORDER BY to_uri")
                .bind(&uri_str).fetch_all(pg).await
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
        let pg = self.pg_pool();
        let rows: Vec<String> = if let Some(k) = kind {
            sqlx::query_scalar("SELECT DISTINCT from_uri FROM context_relations WHERE to_uri = $1 AND relation_kind = $2 ORDER BY from_uri")
                .bind(uri.to_string()).bind(format!("{:?}", k)).fetch_all(pg).await
        } else {
            sqlx::query_scalar("SELECT DISTINCT from_uri FROM context_relations WHERE to_uri = $1 ORDER BY from_uri")
                .bind(uri.to_string()).fetch_all(pg).await
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
        let mut results = Vec::new();
        let mut frontier: Vec<ContextUri> = seeds.to_vec();
        let mut visited = std::collections::HashSet::new();

        for _hop in 0..max_hops {
            let mut next_frontier = Vec::new();
            for uri in &frontier {
                if !visited.insert(uri.clone()) {
                    continue;
                }
                for kind in kinds {
                    let edges = self.outgoing_neighbors(uri, Some(*kind)).await?;
                    for edge in &edges {
                        results.push((uri.clone(), edge.clone(), *kind));
                        next_frontier.push(edge.clone());
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        Ok(results)
    }

    async fn centrality(&self, uri: &ContextUri) -> Result<f32> {
        let config = self.centrality_config;
        let pg = self.pg_pool();
        let mut nodes = std::collections::BTreeSet::new();
        let mut frontier = vec![uri.to_string()];
        nodes.insert(uri.to_string());

        for _ in 0..config.max_hops() {
            if frontier.is_empty() || nodes.len() >= config.max_nodes() {
                break;
            }
            let rows: Vec<String> = sqlx::query_scalar(
                "SELECT DISTINCT to_uri FROM context_relations WHERE from_uri = ANY($1)
                 UNION
                 SELECT DISTINCT from_uri FROM context_relations WHERE to_uri = ANY($1)",
            )
            .bind(&frontier)
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("centrality.frontier", e))?;

            frontier.clear();
            for node in rows {
                if nodes.len() >= config.max_nodes() {
                    break;
                }
                if nodes.insert(node.clone()) {
                    frontier.push(node);
                }
            }
        }

        let node_list = nodes.into_iter().collect::<Vec<_>>();
        if node_list.len() <= 1 {
            return Ok(0.0);
        }
        let node_set = node_list
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let rows = sqlx::query(
            "SELECT from_uri, to_uri FROM context_relations
             WHERE from_uri = ANY($1) OR to_uri = ANY($1)",
        )
        .bind(&node_list)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("centrality.edges", e))?;

        let edges = rows
            .into_iter()
            .map(|row| {
                let from = row
                    .try_get::<String, _>("from_uri")
                    .map_err(|e| Self::storage_err("centrality.edges.from_uri", e))?;
                let to = row
                    .try_get::<String, _>("to_uri")
                    .map_err(|e| Self::storage_err("centrality.edges.to_uri", e))?;
                Ok((from, to))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(from, to)| node_set.contains(from) && node_set.contains(to))
            .collect::<Vec<_>>();
        let scope = uri.tenant();
        let revision: i64 = sqlx::query_scalar(
            "SELECT COALESCE((SELECT revision FROM context_graph_revisions WHERE scope=$1), 0)",
        )
        .bind(scope)
        .fetch_one(pg)
        .await
        .map_err(|e| Self::storage_err("centrality revision", e))?;
        let cache_key = config.cache_key();
        if let Some(score) = sqlx::query_scalar::<_, f32>(
            "SELECT score FROM context_graph_centrality WHERE scope=$1 AND revision=$2 AND algorithm_config=$3 AND uri=$4",
        ).bind(scope).bind(revision).bind(&cache_key).bind(uri.as_str())
            .fetch_optional(pg).await.map_err(|e| Self::storage_err("read centrality cache", e))?
        { return Ok(score); }
        let previous_revision: Option<i64> = sqlx::query_scalar(
            "SELECT max(revision) FROM context_graph_centrality WHERE scope=$1 AND algorithm_config=$2 AND revision < $3",
        )
        .bind(scope)
        .bind(&cache_key)
        .bind(revision)
        .fetch_one(pg)
        .await
        .map_err(|e| Self::storage_err("read previous centrality revision", e))?;
        let scores = if let Some(previous_revision) = previous_revision {
            let previous = sqlx::query_as::<_, (String, f32)>(
                "SELECT uri, score FROM context_graph_centrality WHERE scope=$1 AND revision=$2 AND algorithm_config=$3",
            )
            .bind(scope)
            .bind(previous_revision)
            .bind(&cache_key)
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("read previous centrality snapshot", e))?
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
            let dirty_seeds = sqlx::query_as::<_, (Option<String>, Option<String>)>(
                "SELECT from_uri, to_uri FROM context_graph_mutations WHERE scope=$1 AND revision > $2 AND revision <= $3",
            )
            .bind(scope)
            .bind(previous_revision)
            .bind(revision)
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("read dirty graph nodes", e))?
            .into_iter()
            .flat_map(|(from, to)| from.into_iter().chain(to))
            .collect::<std::collections::HashSet<_>>();
            incremental_pagerank_scores(&node_list, &edges, &previous, &dirty_seeds, config)
                .unwrap_or_else(|| pagerank_scores(&node_list, &edges, config))
        } else {
            pagerank_scores(&node_list, &edges, config)
        };
        let mut tx = pg
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin centrality cache", e))?;
        for (node, score) in &scores {
            sqlx::query("INSERT INTO context_graph_centrality(scope,revision,algorithm_config,uri,score) VALUES($1,$2,$3,$4,$5) ON CONFLICT(scope,revision,algorithm_config,uri) DO UPDATE SET score=EXCLUDED.score")
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
            sqlx::query_scalar("SELECT revision FROM context_graph_revisions WHERE scope=$1")
                .bind(scope.tenant())
                .fetch_optional(self.pg_pool())
                .await
                .map_err(|e| Self::storage_err("read graph revision", e))?;
        u64::try_from(revision.unwrap_or(0))
            .map_err(|_| ContextError::Storage("negative graph revision".into()))
    }
}

#[async_trait]
impl BlobStore for PgContextStore {
    async fn put(&self, data: &[u8], mime_type: &str) -> Result<BlobRef> {
        let hash = blake3::hash(data);
        let hash_str = hash.to_hex().to_string();
        let size = data.len();
        let pg = self.pg_pool();

        sqlx::query(
            "INSERT INTO context_blobs (content_hash, data, mime_type, size, created_at)
             VALUES ($1, $2, $3, $4, now())
             ON CONFLICT (content_hash) DO NOTHING",
        )
        .bind(&hash_str)
        .bind(data)
        .bind(mime_type)
        .bind(size as i64)
        .execute(pg)
        .await
        .map_err(|e| Self::storage_err("blob put", e))?;

        Ok(BlobRef {
            hash: ContentHash(hash_str),
            size,
            mime_type: mime_type.to_string(),
        })
    }

    async fn get(&self, blob_ref: &BlobRef) -> Result<Vec<u8>> {
        let row: Option<(Vec<u8>, String, i64)> = sqlx::query_as(
            "SELECT data, mime_type, size FROM context_blobs WHERE content_hash = $1",
        )
        .bind(&blob_ref.hash.0)
        .fetch_optional(self.pg_pool())
        .await
        .map_err(|e| Self::storage_err("blob get", e))?;
        let (data, mime_type, stored_size) = row.ok_or_else(|| {
            ContextError::NotFound(format!("blob not found: {}", blob_ref.hash.0))
        })?;
        validate_blob_ref(blob_ref, &data, &mime_type, stored_size)?;
        Ok(data)
    }

    async fn dedup_check(&self, hash: &ContentHash) -> Result<bool> {
        let pg = self.pg_pool();
        let exists: Option<String> =
            sqlx::query_scalar("SELECT content_hash FROM context_blobs WHERE content_hash = $1")
                .bind(&hash.0)
                .fetch_optional(pg)
                .await
                .map_err(|e| Self::storage_err("dedup_check", e))?;
        Ok(exists.is_some())
    }
}

// ===========================================================================
// PgEngine — 6域存储引擎装配
// ===========================================================================

pub struct AclPgEngine {
    store: Arc<AclProtectedStore<PgContextStore>>,
    raw_store: Arc<PgContextStore>,
}

impl AclPgEngine {
    pub fn from_pool(pool: Arc<DbPool>, acl: Arc<PathAcl>, principal: Principal) -> Result<Self> {
        let raw_store = Arc::new(PgContextStore::new(pool, GraphCentralityConfig::default())?);
        let store = Arc::new(AclProtectedStore::new(
            raw_store.as_ref().clone(),
            acl,
            principal,
        ));
        Ok(Self { store, raw_store })
    }
}

impl StorageEngine for AclPgEngine {
    fn content(&self) -> &dyn ContentStore {
        self.store.as_ref()
    }

    fn browsing(&self) -> &dyn BrowsingOps {
        self.store.as_ref()
    }

    fn version(&self) -> &dyn VersionOps {
        self.raw_store.as_ref()
    }

    fn tenant(&self) -> &dyn TenantOps {
        self.raw_store.as_ref()
    }

    fn graph(&self) -> Option<&dyn GraphStore> {
        Some(self.raw_store.as_ref())
    }

    fn blob(&self) -> &dyn BlobStore {
        self.raw_store.as_ref()
    }
}

// ===========================================================================
// 辅助函数
// ===========================================================================

#[allow(dead_code)]
fn parse_state_scope(s: &str) -> Option<StateScope> {
    match s {
        "short" => Some(StateScope::Short),
        "mid" => Some(StateScope::Mid),
        "long" => Some(StateScope::Long),
        _ => None,
    }
}

/// 递归构建树节点。
fn build_tree_level(
    prefix: &str,
    all_uris: &[String],
    current_depth: usize,
    max_depth: usize,
) -> Vec<TreeNode> {
    if current_depth >= max_depth {
        return vec![];
    }

    let mut children: Vec<TreeNode> = Vec::new();
    let mut seen = std::collections::BTreeMap::new();

    for uri_str in all_uris {
        let rest = match uri_str.strip_prefix(prefix) {
            Some(r) => r,
            None => continue,
        };
        if rest.is_empty() {
            continue;
        }
        let (name, is_dir) = match rest.split_once('/') {
            Some((name, _)) => (name, true),
            None => (rest, false),
        };
        if name.is_empty() {
            continue;
        }
        if let Ok(child_uri) = ContextUri::parse(format!("{}{}", prefix, name)) {
            seen.entry(name.to_string()).or_insert((child_uri, is_dir));
        }
    }

    for (_name, (child_uri, is_dir)) in seen {
        if is_dir {
            let child_prefix = format!("{}{}/", prefix, _name);
            let sub_children =
                build_tree_level(&child_prefix, all_uris, current_depth + 1, max_depth);
            children.push(TreeNode {
                uri: child_uri,
                is_dir: true,
                children: sub_children,
            });
        } else {
            children.push(TreeNode {
                uri: child_uri,
                is_dir: false,
                children: vec![],
            });
        }
    }
    children
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_prefix_ends_with_slash() {
        let dir = ContextUri::parse("uwu://t/agent/a").unwrap();
        assert_eq!(PgContextStore::dir_prefix(&dir), "uwu://t/agent/a/");
    }
}

// ===========================================================================
// PG 集成测试
// ===========================================================================

#[cfg(test)]
mod pg_tests {
    use super::*;
    use agent_context_db_core::{ContentType, ContextMeta, MediaType, StateScope};
    use std::sync::Arc;
    use uuid::Uuid;
    use uwu_database::config::{
        CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend,
        VectorBackend, VectorConfig,
    };
    use uwu_database::sql;

    fn pg_url() -> Option<String> {
        std::env::var("DATABASE_URL").ok()
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
                application_name: Some("ctx_db_test".into()),
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

    async fn setup_store(url: String) -> PgContextStore {
        let cfg = test_cfg(url);
        let pool = sql::build_pool(&cfg.database).await.unwrap();
        let arc_pool = Arc::new(pool);

        // 运行 context-db 迁移（手动创建表结构以确保存在）
        arc_pool.as_postgres().unwrap();
        arc_pool
            .exec(
                "CREATE TABLE IF NOT EXISTS context_entries (\
                uri             TEXT PRIMARY KEY,\
                tenant_id       UUID NOT NULL,\
                l0_abstract     TEXT NOT NULL,\
                l1_overview     TEXT,\
                l2_detail_ref   UUID,\
                content_type    TEXT NOT NULL DEFAULT 'evidence',\
                state_scope     TEXT,\
                tags            JSONB NOT NULL DEFAULT '[]',\
                custom          JSONB NOT NULL DEFAULT '{}',\
                entry_json      JSONB,\
                mvcc_version    BIGINT NOT NULL DEFAULT 0,\
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),\
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()\
            )",
            )
            .await
            .unwrap();
        arc_pool
            .exec(
                "CREATE TABLE IF NOT EXISTS context_versions (\
                uri             TEXT NOT NULL,\
                mvcc_version    BIGINT NOT NULL,\
                tenant_id       UUID NOT NULL,\
                l0_abstract     TEXT NOT NULL,\
                l1_overview     TEXT,\
                l2_detail_ref   UUID,\
                content_type    TEXT NOT NULL DEFAULT 'evidence',\
                state_scope     TEXT,\
                tags            JSONB NOT NULL DEFAULT '[]',\
                custom          JSONB NOT NULL DEFAULT '{}',\
                entry_json      JSONB NOT NULL,\
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),\
                PRIMARY KEY (uri, mvcc_version)\
            )",
            )
            .await
            .unwrap();

        PgContextStore::new(arc_pool, GraphCentralityConfig::default())
            .unwrap_or_else(|error| panic!("test postgres store construction failed: {error}"))
    }

    fn ctx_uri(path: &str) -> ContextUri {
        ContextUri::parse(format!("uwu://t/agent/{path}")).unwrap()
    }

    fn tenant() -> TenantId {
        TenantId(Uuid::new_v4())
    }

    fn entry(uri: &ContextUri, tenant: TenantId, text: &str) -> ContextEntry {
        ContextEntry::new_text(uri.clone(), tenant, text)
    }

    // ── ContentRepo: write ──────────────────────────────

    #[tokio::test]
    async fn test_write_and_read_l0() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let t = tenant();
        let uri = ctx_uri("mem/cases/c1");
        let e = entry(&uri, t, "hello world");

        let v = ContentRepo::write(&store, e).await.unwrap();
        assert_eq!(v.0, 1);

        let payload = ContentStore::read(&store, &uri, ContentLevel::L0)
            .await
            .unwrap();
        assert!(matches!(&payload, ContentPayload::Text { sparse, .. } if sparse == "hello world"));
    }

    #[tokio::test]
    async fn test_write_updates_existing() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let t = tenant();
        let uri = ctx_uri("mem/skills/s1");

        ContentRepo::write(&store, entry(&uri, t, "v1"))
            .await
            .unwrap();
        let v2 = ContentRepo::write(&store, entry(&uri, t, "v2 updated"))
            .await
            .unwrap();

        assert_eq!(v2.0, 2, "version should increment on update");
        let payload = ContentStore::read(&store, &uri, ContentLevel::L0)
            .await
            .unwrap();
        assert!(matches!(&payload, ContentPayload::Text { sparse, .. } if sparse == "v2 updated"));
    }

    #[tokio::test]
    async fn test_write_with_all_fields() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let t = tenant();
        let uri = ctx_uri("full/entry");
        let now = chrono::Utc::now();

        let entry = ContextEntry {
            uri: uri.clone(),
            tenant: t,
            payload: ContentPayload::Text {
                sparse: "L0 summary".into(),
                dense: "L1 overview".into(),
                full: "L0 summary\nL1 overview".into(),
            },
            media_type: MediaType::Text,
            metadata: ContextMeta {
                content_type: Some(ContentType::Error),
                state_scope: Some(StateScope::Long),
                tags: vec!["important".into(), "bug".into()],
                custom: serde_json::json!({"priority": "high"}),
                ..Default::default()
            },
            mvcc_version: MvccVersion(0),
            created_at: now,
            updated_at: now,
            derivation: None,
        };

        ContentRepo::write(&store, entry).await.unwrap();

        // 通过数据库直接验证
        let pool = store.pg_pool();
        let row: (String, String, Option<String>) = sqlx::query_as(
            "SELECT l0_abstract, content_type, state_scope FROM context_entries WHERE uri = $1",
        )
        .bind(uri.to_string())
        .fetch_one(pool)
        .await
        .unwrap();
        assert_eq!(row.0, "L0 summary");
        assert_eq!(row.1, "error");
        assert_eq!(row.2, Some("long".into()));
    }

    // ── ContentRepo: delete ─────────────────────────────

    #[tokio::test]
    async fn test_delete_entry() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let t = tenant();
        let uri = ctx_uri("tmp/to_delete");

        ContentRepo::write(&store, entry(&uri, t, "bye"))
            .await
            .unwrap();
        ContentRepo::delete(&store, &uri).await.unwrap();

        let r = FsOps::read(&store, &uri, ContentLevel::L0).await;
        assert!(r.is_err(), "should not find deleted entry");
    }

    #[tokio::test]
    async fn test_delete_nonexistent_returns_not_found() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let uri = ctx_uri("nonexistent/xyz");
        let r = ContentRepo::delete(&store, &uri).await;
        assert!(r.is_err());
    }

    // ── ContentRepo: rename ─────────────────────────────

    #[tokio::test]
    async fn test_rename_entry() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let t = tenant();
        let from = ctx_uri("old/name");
        let to = ctx_uri("new/name");

        agent_context_db_core::ContentRepo::write(&store, entry(&from, t, "move me"))
            .await
            .unwrap();
        agent_context_db_core::ContentRepo::rename(&store, &from, &to)
            .await
            .unwrap();

        // 旧路径不可读
        assert!(
            agent_context_db_core::ContentStore::read(&store, &from, ContentLevel::L0)
                .await
                .is_err()
        );
        // 新路径可读
        let p = agent_context_db_core::ContentStore::read(&store, &to, ContentLevel::L0)
            .await
            .unwrap();
        assert!(matches!(p, ContentPayload::Text { sparse, .. } if sparse == "move me"));
    }

    #[tokio::test]
    async fn test_rename_nonexistent_returns_not_found() {
        let Some(url) = pg_url() else { return };
        let store = setup_store(url).await;
        let r = agent_context_db_core::ContentRepo::rename(
            &store,
            &ctx_uri("ghost/src"),
            &ctx_uri("ghost/dst"),
        )
        .await;
        assert!(r.is_err());
    }
}
