//! PG 适配器：用 uwu_database::DbPool 实现 context-db 的四个窄端口。
//!
//! - [`PgContextStore`] 同时实现 `FsOps` + `ContentRepo` + `VersionOps` + `TenantOps`，
//!   自动满足 `ContextStore` supertrait。
//! - URI 寻址通过 `context_entries` 表的 TEXT 列 + LIKE 前缀查询实现。
//! - 版本历史通过 `context_versions` 表存储完整快照。

use agent_context_db_core::{
    AclProtectedStore, BlobRef, BlobStore, BrowsingOps, ContentHash, ContentLevel, ContentPayload,
    ContentRepo, ContentStore, ContentType, ContextDiff, ContextEntry, ContextError, ContextUri,
    DirEntry, FindPattern, FsOps, GraphRelation, GraphStore, GrepHit, MvccVersion, PathAcl,
    Principal, Result, StateScope, StorageEngine, TenantId, TenantOps, TreeNode, VersionEntry,
    VersionOps, sanitize_entry_for_write,
};
use async_trait::async_trait;
use uuid::Uuid;

use std::sync::Arc;
use uwu_database::DbPool;

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
        let entry = sanitize_entry_for_write(&entry);
        let mut tx = self.pg_pool().begin().await.map_err(|e| {
            tracing::error!(error = %e, "begin transaction failed");
            ContextError::Storage(format!("begin transaction failed: {e}"))
        })?;
        let uri_str = entry.uri.to_string();
        let tenant_str = entry.tenant.0.to_string();
        let (l0_text, l1_text, _l2_text) = extract_payload_levels(&entry.payload);
        let l2_ref: Option<Uuid> = None; // L2 stored in l2_full_text, blob ref not used
        let content_type = entry
            .metadata
            .content_type
            .unwrap_or(ContentType::Evidence)
            .as_path_segment();
        let state_scope: Option<String> = entry.metadata.state_scope.map(|s| match s {
            StateScope::Short => "short".to_string(),
            StateScope::Mid => "mid".to_string(),
            StateScope::Long => "long".to_string(),
        });
        let tags = serde_json::to_value(&entry.metadata.tags).unwrap_or(serde_json::json!([]));
        let custom = &entry.metadata.custom;
        let mvcc = entry.mvcc_version.0 as i64 + 1;

        // Upsert into context_entries
        sqlx::query(
            r#"
            INSERT INTO context_entries
                (uri, tenant_id, l0_abstract, l1_overview, l2_detail_ref,
                 content_type, state_scope, tags, custom,
                 mvcc_version, created_at, updated_at)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (uri) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                l0_abstract = EXCLUDED.l0_abstract,
                l1_overview = EXCLUDED.l1_overview,
                l2_detail_ref = EXCLUDED.l2_detail_ref,
                content_type = EXCLUDED.content_type,
                state_scope = EXCLUDED.state_scope,
                tags = EXCLUDED.tags,
                custom = EXCLUDED.custom,
                mvcc_version = EXCLUDED.mvcc_version,
                updated_at = EXCLUDED.updated_at
            "#,
        )
        .bind(&uri_str)
        .bind(&tenant_str)
        .bind(&l0_text)
        .bind(&l1_text)
        .bind(&l2_ref)
        .bind(content_type)
        .bind(&state_scope)
        .bind(&tags)
        .bind(custom)
        .bind(mvcc)
        .bind(&entry.created_at)
        .bind(&entry.updated_at)
        .execute(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("write", e))?;

        // Insert version record
        let entry_json = serde_json::to_value(&entry).unwrap_or(serde_json::Value::Null);
        sqlx::query(
            r#"
            INSERT INTO context_versions
                (uri, mvcc_version, tenant_id, l0_abstract, l1_overview,
                 l2_detail_ref, content_type, state_scope,
                 tags, custom, entry_json)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(&uri_str)
        .bind(mvcc)
        .bind(&tenant_str)
        .bind(&l0_text)
        .bind(&l1_text)
        .bind(&l2_ref)
        .bind(content_type)
        .bind(&state_scope)
        .bind(&tags)
        .bind(custom)
        .bind(&entry_json)
        .execute(&mut *tx)
        .await
        .map_err(|e| Self::storage_err("write version", e))?;

        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit", e))?;

        Ok(MvccVersion(mvcc as u64))
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
        // Also delete versions — error propagates with transaction
        sqlx::query("DELETE FROM context_versions WHERE uri = $1")
            .bind(&uri_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("delete versions", e))?;

        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit", e))?;
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(from = %from, to = %to))]
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        let mut tx = self
            .pg_pool()
            .begin()
            .await
            .map_err(|e| Self::storage_err("begin transaction", e))?;
        let from_str = from.to_string();
        let to_str = to.to_string();

        let affected =
            sqlx::query("UPDATE context_entries SET uri = $1, updated_at = now() WHERE uri = $2")
                .bind(&to_str)
                .bind(&from_str)
                .execute(&mut *tx)
                .await
                .map_err(|e| Self::storage_err("rename", e))?;

        if affected.rows_affected() == 0 {
            return Err(ContextError::NotFound(from_str));
        }

        // Also update versions — error propagates with transaction
        sqlx::query("UPDATE context_versions SET uri = $1 WHERE uri = $2")
            .bind(&to_str)
            .bind(&from_str)
            .execute(&mut *tx)
            .await
            .map_err(|e| Self::storage_err("rename versions", e))?;

        tx.commit()
            .await
            .map_err(|e| Self::storage_err("commit", e))?;
        Ok(())
    }

    /// D.1: 批量写入 — 单事务 UNNEST 批量插入。
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
            let (l0, l1, _l2) = extract_payload_levels(&entry.payload);
            let mvcc = entry.mvcc_version.0 as i64 + 1;
            sqlx::query(
                "INSERT INTO context_entries (uri, tenant_id, l0_abstract, l1_overview, l2_detail_ref, content_type, state_scope, tags, custom, mvcc_version, created_at, updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) ON CONFLICT (uri) DO UPDATE SET l0_abstract=EXCLUDED.l0_abstract, l1_overview=EXCLUDED.l1_overview, content_type=EXCLUDED.content_type, state_scope=EXCLUDED.state_scope, tags=EXCLUDED.tags, custom=EXCLUDED.custom, mvcc_version=EXCLUDED.mvcc_version, updated_at=EXCLUDED.updated_at"
            )
            .bind(&entry.uri.to_string())
            .bind(&entry.tenant.0.to_string())
            .bind(&l0).bind(&l1).bind::<Option<Uuid>>(None)
            .bind(entry.metadata.content_type.unwrap_or(ContentType::Evidence).as_path_segment())
            .bind(entry.metadata.state_scope.map(|s| match s { StateScope::Short => "short".to_string(), StateScope::Mid => "mid".to_string(), StateScope::Long => "long".to_string() }))
            .bind(&serde_json::to_value(&entry.metadata.tags).unwrap_or(serde_json::json!([]))).bind(&entry.metadata.custom)
            .bind(mvcc).bind(&entry.created_at).bind(&entry.updated_at)
            .execute(&mut *tx).await
            .map_err(|e| Self::storage_err("batch write entry", e))?;
            versions.push(MvccVersion(mvcc as u64));
        }
        tx.commit()
            .await
            .map_err(|e| Self::storage_err("batch commit", e))?;
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
    async fn ls(&self, dir: &ContextUri) -> Result<Vec<DirEntry>> {
        let pg = self.pg_pool();
        let prefix = Self::dir_prefix(dir);

        // 获取所有以 dir/ 开头的条目
        let rows = sqlx::query_as::<_, (String, String, Option<String>)>(
            r#"
            SELECT uri, l0_abstract, content_type FROM context_entries
            WHERE uri LIKE $1
            ORDER BY uri
            "#,
        )
        .bind(format!("{}%", &prefix))
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("ls", e))?;

        let mut seen = std::collections::BTreeMap::new();
        for (uri_str, abstract_, content_type_str) in rows {
            let rest = uri_str.strip_prefix(&prefix).unwrap_or(&uri_str);
            let ct = content_type_str
                .as_deref()
                .and_then(ContentType::from_path_segment);
            let slash_pos = rest.find('/');
            if let Some(pos) = slash_pos {
                let dir_name = &rest[..pos];
                if !seen.contains_key(dir_name) {
                    let dir_uri = ContextUri::parse(format!("{}{}", prefix, dir_name))
                        .map_err(|e| ContextError::Storage(format!("bad uri: {e}")))?;
                    seen.insert(
                        dir_name.to_string(),
                        DirEntry {
                            uri: dir_uri,
                            is_dir: true,
                            abstract_: String::new(),
                            content_type: ct,
                        },
                    );
                }
            } else {
                let context_uri = ContextUri::parse(uri_str.clone())
                    .map_err(|e| ContextError::Storage(format!("bad uri: {e}")))?;
                seen.entry(rest.to_string()).or_insert(DirEntry {
                    uri: context_uri,
                    is_dir: false,
                    abstract_,
                    content_type: ct,
                });
            }
        }

        Ok(seen.into_values().collect())
    }

    async fn find(&self, pattern: &FindPattern) -> Result<Vec<ContextUri>> {
        let pg = self.pg_pool();
        let scope = pattern
            .scope
            .as_ref()
            .map(|u| u.to_string())
            .unwrap_or_default();

        // 构建查询：有 content_type 过滤时加 WHERE 条件
        let results: Vec<String> = if let Some(content_type) = pattern.content_type {
            sqlx::query_scalar::<_, String>(
                r#"
                SELECT uri FROM context_entries
                WHERE uri LIKE $1 AND content_type = $2
                ORDER BY uri
                "#,
            )
            .bind(format!("{}%", &scope))
            .bind(content_type.as_path_segment())
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("find", e))?
        } else {
            sqlx::query_scalar::<_, String>(
                "SELECT uri FROM context_entries WHERE uri LIKE $1 ORDER BY uri",
            )
            .bind(format!("{}%", &scope))
            .fetch_all(pg)
            .await
            .map_err(|e| Self::storage_err("find", e))?
        };

        Ok(results
            .into_iter()
            .filter_map(|s: String| ContextUri::parse(s).ok())
            .collect())
    }

    async fn grep(&self, regex: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        let pg = self.pg_pool();
        let scope_str = scope.to_string();
        // PG 的 regex 用 ~ 操作符；fallback 到 ILIKE（更简单可靠）
        let pattern = format!("%{}%", regex);

        let rows = sqlx::query_as::<_, (String, String, Option<String>)>(
            r#"
            SELECT uri, l0_abstract, l1_overview FROM context_entries
            WHERE uri LIKE $1
              AND (l0_abstract ILIKE $2 OR l1_overview ILIKE $2)
            ORDER BY uri
            "#,
        )
        .bind(format!("{}%", &scope_str))
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

    async fn tree(&self, root: &ContextUri, depth: usize) -> Result<TreeNode> {
        let pg = self.pg_pool();
        let prefix = Self::dir_prefix(root);

        let rows: Vec<String> =
            sqlx::query_scalar("SELECT uri FROM context_entries WHERE uri LIKE $1 ORDER BY uri")
                .bind(format!("{}%", &prefix))
                .fetch_all(pg)
                .await
                .map_err(|e| Self::storage_err("tree", e))?;

        let root_node = TreeNode {
            uri: root.clone(),
            is_dir: true,
            children: build_tree_level(&prefix, &rows, 0, depth),
        };
        Ok(root_node)
    }

    #[tracing::instrument(skip(self), fields(uri = %uri, level = ?level))]
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        // 0. 缓存命中（L0/L1 可缓存，L2 不走缓存）
        if level != ContentLevel::L2 {
            if let Some(ref cache) = self.read_cache {
                match cache.get(uri, level).await {
                    Some(Some(cached)) => return Ok(cached),
                    // 负缓存命中：已知该 URI 缺失，直接 NotFound（穿透防护）
                    Some(None) => return Err(ContextError::NotFound(uri.to_string())),
                    None => { /* miss，继续回源 */ }
                }
            }
        }

        let pg = self.pg_pool();
        let uri_str = uri.to_string();

        let row = sqlx::query_as::<_, (String, Option<String>, Option<uuid::Uuid>)>(
            "SELECT l0_abstract, l1_overview, l2_detail_ref FROM context_entries WHERE uri = $1",
        )
        .bind(&uri_str)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("read", e))?;

        let result = match row {
            None => return Err(ContextError::NotFound(uri_str)),
            Some((l0, l1, _l2_ref)) => match level {
                ContentLevel::L0 => Ok(payload_from_levels(&l0, l1.as_deref(), None)),
                ContentLevel::L1 => Ok(ContentPayload::Text {
                    sparse: l0.clone(),
                    dense: l1.unwrap_or_default(),
                    full: l0,
                }),
                ContentLevel::L2 => {
                    // L2: 返回完整条目 JSON
                    let entry_row = sqlx::query_as::<_, (serde_json::Value,)>(
                        r#"
                        SELECT row_to_json(t) FROM (
                            SELECT uri, tenant_id, l0_abstract, l1_overview, l2_detail_ref,
                                   content_type, state_scope, tags, custom,
                                   mvcc_version, created_at, updated_at
                            FROM context_entries WHERE uri = $1
                        ) t
                        "#,
                    )
                    .bind(&uri_str)
                    .fetch_one(pg)
                    .await
                    .map_err(|e| Self::storage_err("read L2", e))?;

                    let bytes = serde_json::to_vec(&entry_row.0)?;
                    let full = String::from_utf8(bytes).map_err(|e| {
                        ContextError::Storage(format!("L2 row is not valid UTF-8: {e}"))
                    })?;
                    Ok(ContentPayload::Text {
                        sparse: l0,
                        dense: l1.unwrap_or_default(),
                        full,
                    })
                }
            },
        };

        // 写回缓存（L0/L1 结果）；NotFound 写入负缓存（穿透防护）
        if let Some(cache) = self.read_cache.as_ref() {
            if level != ContentLevel::L2 {
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
        }
        result
    }
}

// ===========================================================================
// VersionOps 实现
// ===========================================================================

#[async_trait]
impl VersionOps for PgContextStore {
    async fn version_history(&self, uri: &ContextUri) -> Result<Vec<VersionEntry>> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();

        let rows = sqlx::query_as::<_, (i64, String, chrono::DateTime<chrono::Utc>)>(
            r#"
            SELECT mvcc_version, l0_abstract, created_at
            FROM context_versions
            WHERE uri = $1
            ORDER BY mvcc_version
            "#,
        )
        .bind(&uri_str)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("version_history", e))?;

        Ok(rows
            .into_iter()
            .map(|(v, msg, ts)| VersionEntry {
                version: MvccVersion(v as u64),
                message: msg,
                ts,
            })
            .collect())
    }

    async fn rollback(&self, uri: &ContextUri, to: MvccVersion) -> Result<()> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();
        let target_v = to.0 as i64;

        // 读出目标版本的完整条目 JSON
        let entry_json: Option<serde_json::Value> = sqlx::query_scalar(
            "SELECT entry_json FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(target_v)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("rollback read", e))?;

        let json = entry_json
            .ok_or_else(|| ContextError::VersionConflict(format!("no version {:?}", to)))?;

        let mut entry: ContextEntry = serde_json::from_value(json)?;

        // 把 entry 写回当前表（version++ 作为 rollback 操作的新版本）
        entry.mvcc_version = MvccVersion(0); // write() 会 +1
        entry.updated_at = chrono::Utc::now();
        <Self as ContentRepo>::write(self, entry).await?;
        Ok(())
    }

    async fn diff(&self, uri: &ContextUri, a: MvccVersion, b: MvccVersion) -> Result<ContextDiff> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();

        // 简单 diff：返回两个版本的 l0_abstract 对比
        let v_a: Option<String> = sqlx::query_scalar(
            "SELECT l0_abstract FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(a.0 as i64)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("diff read a", e))?;

        let v_b: Option<String> = sqlx::query_scalar(
            "SELECT l0_abstract FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(b.0 as i64)
        .fetch_optional(pg)
        .await
        .map_err(|e| Self::storage_err("diff read b", e))?;

        let summary = match (v_a, v_b) {
            (Some(a_str), Some(b_str)) => {
                format!(
                    "{}: v{:?} → v{:?}\n---\n{}\n+++\n{}",
                    uri_str, a.0, b.0, a_str, b_str
                )
            }
            _ => format!("{}: one or both versions not found", uri_str),
        };

        Ok(ContextDiff { summary })
    }
}

// ===========================================================================
// TenantOps 实现
// ===========================================================================

#[async_trait]
impl TenantOps for PgContextStore {
    async fn list_tenants(&self) -> Result<Vec<TenantId>> {
        let pg = self.pg_pool();

        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT DISTINCT tenant_id::text FROM context_entries ORDER BY tenant_id",
        )
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("list_tenants", e))?;

        Ok(rows
            .into_iter()
            .filter_map(|s| uuid::Uuid::parse_str(&s).ok())
            .map(TenantId)
            .collect())
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

    async fn scan_by_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<ContextEntry>> {
        let pg = self.pg_pool();
        let rows = sqlx::query_as::<_, (String,)>(
            "SELECT uri FROM context_entries WHERE uri LIKE $1 ORDER BY uri LIMIT $2",
        )
        .bind(format!("{}%", prefix))
        .bind(limit as i64)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("scan_by_prefix", e))?;

        let mut entries = Vec::new();
        for (uri_str,) in rows {
            let uri = ContextUri::parse(&uri_str)
                .map_err(|e| ContextError::InvalidUri(format!("scan_by_prefix parse: {e}")))?;
            let payload = <Self as ContentStore>::read(self, &uri, ContentLevel::L0).await?;
            entries.push(ContextEntry::new_text(
                uri,
                TenantId(uuid::Uuid::nil()),
                payload.sparse_text(),
            ));
        }
        Ok(entries)
    }
}

#[async_trait]
impl BrowsingOps for PgContextStore {
    async fn ls(&self, dir: &ContextUri) -> Result<Vec<DirEntry>> {
        <Self as FsOps>::ls(self, dir).await
    }

    async fn tree(&self, dir: &ContextUri, depth: usize) -> Result<TreeNode> {
        <Self as FsOps>::tree(self, dir, depth).await
    }

    async fn find(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<ContextUri>> {
        let pg = self.pg_pool();
        let scope_str = scope.to_string();
        let like_pattern = format!("{}%{}%", scope_str, pattern);
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE uri LIKE $1 ORDER BY uri LIMIT 100",
        )
        .bind(&like_pattern)
        .fetch_all(pg)
        .await
        .map_err(|e| Self::storage_err("find", e))?;
        Ok(rows
            .into_iter()
            .map(ContextUri::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?)
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
        .bind(&from.to_string())
        .bind(&to.to_string())
        .bind(format!("{:?}", kind))
        .execute(pg)
        .await
        .map_err(|e| Self::storage_err("add_edge", e))?;
        Ok(())
    }

    async fn remove_edge(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        let pg = self.pg_pool();
        sqlx::query("DELETE FROM context_relations WHERE from_uri = $1 AND to_uri = $2")
            .bind(&from.to_string())
            .bind(&to.to_string())
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
        let in_degree = self.neighbors(uri, None).await?.len();
        // 简化：PageRank 难度过高，用归一化入度近似中心性
        Ok((in_degree as f32 / (in_degree + 10) as f32).clamp(0.0, 1.0))
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
        let pg = self.pg_pool();
        let data: Vec<u8> =
            sqlx::query_scalar("SELECT data FROM context_blobs WHERE content_hash = $1")
                .bind(&blob_ref.hash.0)
                .fetch_optional(pg)
                .await
                .map_err(|e| Self::storage_err("blob get", e))?
                .ok_or_else(|| {
                    ContextError::NotFound(format!("blob not found: {}", blob_ref.hash.0))
                })?;
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
            .exec(&format!(
                "CREATE TABLE IF NOT EXISTS context_entries (\
                uri             TEXT PRIMARY KEY,\
                tenant_id       UUID NOT NULL,\
                l0_abstract     TEXT NOT NULL,\
                l1_overview     TEXT,\
                l2_detail_ref   UUID,\
                content_type    TEXT NOT NULL DEFAULT 'evidence',\
                state_scope     TEXT,\
                tags            JSONB NOT NULL DEFAULT '[]',\
                custom          JSONB NOT NULL DEFAULT '{{}}',\
                mvcc_version    BIGINT NOT NULL DEFAULT 0,\
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),\
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()\
            )"
            ))
            .await
            .unwrap();
        arc_pool
            .exec(&format!(
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
                custom          JSONB NOT NULL DEFAULT '{{}}',\
                entry_json      JSONB NOT NULL,\
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),\
                PRIMARY KEY (uri, mvcc_version)\
            )"
            ))
            .await
            .unwrap();

        PgContextStore::new(arc_pool)
    }

    fn ctx_uri(path: &str) -> ContextUri {
        ContextUri::parse(&format!("uwu://t/agent/{path}")).unwrap()
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
        .bind(&uri.to_string())
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
