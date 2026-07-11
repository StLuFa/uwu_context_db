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

use crate::outbox::{
    IndexMutation, collection_from_entry, enqueue_pg, point_from_entry, upsert_mutation,
};

// ===========================================================================
// PgContextStore
// ===========================================================================

/// PG 适配器：持有 `DbPool` + 可选内容缓存。
#[derive(Clone)]
pub struct PgContextStore {
    pool: Arc<DbPool>,
    /// 可选的 L0/L1 读取缓存（通过 UwuCacheAdapter 桥接 uwu_database::Cache）。
    read_cache: Option<Arc<dyn agent_context_db_core::ReadCache>>,
}

impl PgContextStore {
    /// C.3: 构造时验证后端类型，避免运行时 expect panic。
    pub fn new(pool: Arc<DbPool>) -> Self {
        let _ = pool
            .as_postgres()
            .expect("PgContextStore requires postgres backend");
        Self {
            pool,
            read_cache: None,
        }
    }

    /// 注入读取缓存（使用 uwu_database::Cache 或任意 ReadCache 实现）。
    pub fn with_cache(mut self, cache: Arc<dyn agent_context_db_core::ReadCache>) -> Self {
        self.read_cache = Some(cache);
        self
    }

    fn pg_pool(&self) -> &sqlx::PgPool {
        self.pool
            .as_postgres()
            .expect("PgContextStore: backend validated at construction")
    }

    /// I.3: 统一 map_err 辅助 —— 记录 tracing::error! 并返回 `ContextError::Storage`。
    fn storage_err(op: &str, e: impl std::fmt::Display) -> ContextError {
        tracing::error!(op, error = %e, "storage operation failed");
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

    fn page_limit(page: &PageRequest) -> Result<i64> {
        i64::try_from(page.effective_limit() + 1)
            .map_err(|_| ContextError::Storage("page limit exceeds PostgreSQL BIGINT".into()))
    }

    fn finish_page<T>(mut items: Vec<T>, limit: usize, cursor: impl Fn(&T) -> String) -> Page<T> {
        let has_more = items.len() > limit;
        if has_more {
            items.truncate(limit);
        }
        let next_cursor = has_more.then(|| cursor(items.last().expect("non-empty limited page")));
        Page::new(items, next_cursor)
    }

    async fn write_in_tx(
        tx: &mut Transaction<'_, Postgres>,
        input: &ContextEntry,
    ) -> Result<MvccVersion> {
        input.validate_tenant_binding()?;
        let mut entry = sanitize_entry_for_write(input);
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
    match payload {
        ContentPayload::Text {
            sparse,
            dense,
            full,
        } => (sparse.clone(), Some(dense.clone()), full.clone()),
        ContentPayload::Image { .. } => ("[image]".to_string(), None, String::new()),
        ContentPayload::Audio { transcript, .. } => (transcript.clone(), None, transcript.clone()),
        ContentPayload::Structured { summary, data, .. } => {
            (summary.clone(), Some(data.to_string()), String::new())
        }
        ContentPayload::Composite { summary, .. } => (summary.clone(), None, String::new()),
    }
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
    #[tracing::instrument(skip(self, entry), fields(uri = %entry.uri))]
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

    #[tracing::instrument(skip(self), fields(uri = %uri))]
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
        sqlx::query("DELETE FROM context_relations WHERE from_uri = $1 OR to_uri = $1")
            .bind(&uri_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("delete relations", e))?;

        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit", e))?;
        if let Some(cache) = &self.read_cache {
            cache.invalidate(uri).await;
        }
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(from = %from, to = %to))]
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
        for entry in entries {
            let version = Self::write_in_tx(&mut tx, entry).await?;
            if let Some(mutation) = upsert_mutation(entry, version)? {
                enqueue_pg(&mut tx, &mutation).await?;
            }
            versions.push(version);
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
    pub fn from_uwu_config(cfg: &agent_context_db_core::config::StorageConfig) -> Self {
        Self {
            max_connections: cfg.max_connections as u32,
            ..Default::default()
        }
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
        let name_pattern = pattern.name_glob.as_deref().map(Self::prefix_pattern);
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
        let next_cursor =
            has_more.then(|| selected.last().expect("non-empty limited page").clone());
        Ok(Page::new(
            vec![TreeNode {
                uri: root.clone(),
                is_dir: true,
                children: build_tree_level(&prefix, selected, 0, depth),
            }],
            next_cursor,
        ))
    }

    #[tracing::instrument(skip(self), fields(uri = %uri, level = ?level))]
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
            .map(|(v, message, ts)| VersionEntry {
                version: MvccVersion(v as u64),
                message,
                ts,
            })
            .collect();
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
        Self::write_in_tx(&mut tx, &entry).await?;
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

        let v_a: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(a.0 as i64)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("diff read a", e))?;

        let v_b: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(b.0 as i64)
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
            format!("{uri_str}: v{} and v{} are identical", a.0, b.0)
        } else {
            format!("{uri_str}: v{} -> v{}\n{}", a.0, b.0, changes.join("\n"))
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
        let pg = self.pg_pool();
        sqlx::query(
            "INSERT INTO context_relations (from_uri, to_uri, relation_kind, created_at)
             VALUES ($1, $2, $3, now())
             ON CONFLICT (from_uri, to_uri, relation_kind) DO NOTHING",
        )
        .bind(from.to_string())
        .bind(to.to_string())
        .bind(format!("{:?}", kind))
        .execute(pg)
        .await
        .map_err(|e| Self::storage_err("add_edge", e))?;
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

    async fn neighbors(
        &self,
        uri: &ContextUri,
        kind: Option<GraphRelation>,
    ) -> Result<Vec<ContextUri>> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();
        let rows: Vec<String> = if let Some(k) = kind {
            sqlx::query_scalar(
                "SELECT DISTINCT to_uri FROM context_relations
                 WHERE from_uri = $1 AND relation_kind = $2
                 UNION
                 SELECT DISTINCT from_uri FROM context_relations
                 WHERE to_uri = $1 AND relation_kind = $2",
            )
            .bind(&uri_str)
            .bind(format!("{:?}", k))
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("neighbors", e))?
        } else {
            sqlx::query_scalar(
                "SELECT DISTINCT to_uri FROM context_relations WHERE from_uri = $1
                 UNION
                 SELECT DISTINCT from_uri FROM context_relations WHERE to_uri = $1",
            )
            .bind(&uri_str)
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("neighbors", e))?
        };
        Ok(rows
            .into_iter()
            .map(ContextUri::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?)
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
                    let edges = self.neighbors(uri, Some(*kind)).await?;
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
        const MAX_NODES: usize = 256;
        const MAX_HOPS: usize = 3;
        const MAX_ITERS: usize = 32;
        const EPSILON: f32 = 1e-5;
        const DAMPING: f32 = 0.85;

        let pg = self.pg_pool();
        let mut nodes = std::collections::BTreeSet::new();
        let mut frontier = vec![uri.to_string()];
        nodes.insert(uri.to_string());

        for _ in 0..MAX_HOPS {
            if frontier.is_empty() || nodes.len() >= MAX_NODES {
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
                if nodes.len() >= MAX_NODES {
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

        let mut outgoing: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for row in rows {
            let from = row.try_get::<String, _>("from_uri").unwrap_or_default();
            let to = row.try_get::<String, _>("to_uri").unwrap_or_default();
            if node_set.contains(&from) && node_set.contains(&to) {
                outgoing.entry(from).or_default().push(to);
            }
        }

        let n = node_list.len() as f32;
        let base = (1.0 - DAMPING) / n;
        let mut ranks = node_list
            .iter()
            .map(|node| (node.clone(), 1.0 / n))
            .collect::<std::collections::HashMap<_, _>>();

        for _ in 0..MAX_ITERS {
            let dangling_mass = node_list
                .iter()
                .filter(|node| outgoing.get(*node).is_none_or(Vec::is_empty))
                .map(|node| ranks.get(node).copied().unwrap_or(0.0))
                .sum::<f32>()
                / n;
            let mut next = node_list
                .iter()
                .map(|node| (node.clone(), base + DAMPING * dangling_mass))
                .collect::<std::collections::HashMap<_, _>>();

            for (from, targets) in &outgoing {
                if targets.is_empty() {
                    continue;
                }
                let contribution =
                    DAMPING * ranks.get(from).copied().unwrap_or(0.0) / targets.len() as f32;
                for target in targets {
                    if let Some(value) = next.get_mut(target) {
                        *value += contribution;
                    }
                }
            }

            let delta = node_list
                .iter()
                .map(|node| {
                    (next.get(node).copied().unwrap_or(0.0)
                        - ranks.get(node).copied().unwrap_or(0.0))
                    .abs()
                })
                .sum::<f32>();
            ranks = next;
            if delta < EPSILON {
                break;
            }
        }

        let raw = ranks.get(uri.as_str()).copied().unwrap_or(0.0);
        Ok((raw * n).clamp(0.0, 1.0))
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
    pub fn from_pool(pool: Arc<DbPool>, acl: Arc<PathAcl>, principal: Principal) -> Self {
        let raw_store = Arc::new(PgContextStore::new(pool));
        let store = Arc::new(AclProtectedStore::new(
            raw_store.as_ref().clone(),
            acl,
            principal,
        ));
        Self { store, raw_store }
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
        let slash_pos = rest.find('/');
        let (name, is_dir) = if let Some(pos) = slash_pos {
            (&rest[..pos], true)
        } else {
            (rest, false)
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

    fn require_pg() -> String {
        pg_url().expect("SKIP: DATABASE_URL not set")
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

    async fn setup_store() -> PgContextStore {
        let _url = require_pg();
        let cfg = test_cfg();
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

        PgContextStore::new(arc_pool)
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
        let store = setup_store().await;
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
        let store = setup_store().await;
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
        let store = setup_store().await;
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
        let store = setup_store().await;
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
        let store = setup_store().await;
        let uri = ctx_uri("nonexistent/xyz");
        let r = ContentRepo::delete(&store, &uri).await;
        assert!(r.is_err());
    }

    // ── ContentRepo: rename ─────────────────────────────

    #[tokio::test]
    async fn test_rename_entry() {
        let store = setup_store().await;
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
        let store = setup_store().await;
        let r = agent_context_db_core::ContentRepo::rename(
            &store,
            &ctx_uri("ghost/src"),
            &ctx_uri("ghost/dst"),
        )
        .await;
        assert!(r.is_err());
    }
}
