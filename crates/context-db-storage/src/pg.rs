//! PG 适配器：用 uwu_database::DbPool 实现 context-db 的四个窄端口。
//!
//! - [`PgContextStore`] 同时实现 `FsOps` + `ContentRepo` + `VersionOps` + `TenantOps`，
//!   自动满足 `ContextStore` supertrait。
//! - URI 寻址通过 `context_entries` 表的 TEXT 列 + LIKE 前缀查询实现。
//! - 版本历史通过 `context_versions` 表存储完整快照。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentRepo, ContentType, ContextDiff, ContextEntry,
    ContextError, ContextUri, DirEntry, FindPattern, FsOps, GrepHit, MemoryClass,
    MvccVersion, Result, StateScope, TenantId, TenantOps, TreeNode, VersionEntry, VersionOps,
};
use async_trait::async_trait;

use std::sync::Arc;
use uwu_database::DbPool;

// ===========================================================================
// PgContextStore
// ===========================================================================

/// PG 适配器：持有 `DbPool`，实现全部四个窄端口。
#[derive(Clone)]
pub struct PgContextStore {
    pool: Arc<DbPool>,
}

impl PgContextStore {
    pub fn new(pool: Arc<DbPool>) -> Self {
        Self { pool }
    }

    fn pg_pool(&self) -> &sqlx::PgPool {
        // Safety: uwu_database 的 RuntimeConfig 保证 backend 匹配 feature
        self.pool.as_postgres().expect("PgContextStore requires postgres backend")
    }

    /// 目录前缀：`{dir_uri}/` 用于 LIKE 查询。
    fn dir_prefix(dir: &ContextUri) -> String {
        let s = dir.to_string().trim_end_matches('/').to_string();
        format!("{}/", s)
    }
}

// ===========================================================================
// ContentRepo 实现
// ===========================================================================

#[async_trait]
impl ContentRepo for PgContextStore {
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let pg = self.pg_pool();
        let uri_str = entry.uri.to_string();
        let tenant_str = entry.tenant.0.to_string();
        let l2_ref = entry.l2_detail_uri.map(|r| r.0);
        let content_type = match entry.content_type {
            ContentType::Text => "text",
            ContentType::Image => "image",
            ContentType::Audio => "audio",
            ContentType::Video => "video",
            ContentType::Binary => "binary",
        };
        let memory_class: Option<String> = entry.metadata.memory_class.map(|c| {
            memory_class_name(c)
        });
        let state_scope: Option<String> = entry.metadata.state_scope.map(|s| {
            match s {
                StateScope::Short => "short".to_string(),
                StateScope::Mid => "mid".to_string(),
                StateScope::Long => "long".to_string(),
            }
        });
        let tags = serde_json::to_value(&entry.metadata.tags)
            .unwrap_or(serde_json::json!([]));
        let custom = &entry.metadata.custom;
        let mvcc = entry.mvcc_version.0 as i64 + 1;

        // Upsert into context_entries
        sqlx::query(
            r#"
            INSERT INTO context_entries
                (uri, tenant_id, l0_abstract, l1_overview, l2_detail_ref,
                 content_type, memory_class, state_scope, tags, custom,
                 mvcc_version, created_at, updated_at)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (uri) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                l0_abstract = EXCLUDED.l0_abstract,
                l1_overview = EXCLUDED.l1_overview,
                l2_detail_ref = EXCLUDED.l2_detail_ref,
                content_type = EXCLUDED.content_type,
                memory_class = EXCLUDED.memory_class,
                state_scope = EXCLUDED.state_scope,
                tags = EXCLUDED.tags,
                custom = EXCLUDED.custom,
                mvcc_version = EXCLUDED.mvcc_version,
                updated_at = EXCLUDED.updated_at
            "#,
        )
        .bind(&uri_str)
        .bind(&tenant_str)
        .bind(&entry.l0_abstract)
        .bind(&entry.l1_overview)
        .bind(&l2_ref)
        .bind(content_type)
        .bind(&memory_class)
        .bind(&state_scope)
        .bind(&tags)
        .bind(custom)
        .bind(mvcc)
        .bind(&entry.created_at)
        .bind(&entry.updated_at)
        .execute(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("write failed: {e}")))?;

        // Insert version record
        let entry_json = serde_json::to_value(&entry)
            .unwrap_or(serde_json::Value::Null);
        sqlx::query(
            r#"
            INSERT INTO context_versions
                (uri, mvcc_version, tenant_id, l0_abstract, l1_overview,
                 l2_detail_ref, content_type, memory_class, state_scope,
                 tags, custom, entry_json)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            "#,
        )
        .bind(&uri_str)
        .bind(mvcc)
        .bind(&tenant_str)
        .bind(&entry.l0_abstract)
        .bind(&entry.l1_overview)
        .bind(&l2_ref)
        .bind(content_type)
        .bind(&memory_class)
        .bind(&state_scope)
        .bind(&tags)
        .bind(custom)
        .bind(&entry_json)
        .execute(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("write version failed: {e}")))?;

        Ok(MvccVersion(mvcc as u64))
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();
        let affected = sqlx::query("DELETE FROM context_entries WHERE uri = $1")
            .bind(&uri_str)
            .execute(pg)
            .await
            .map_err(|e| ContextError::Storage(format!("delete failed: {e}")))?;

        if affected.rows_affected() == 0 {
            return Err(ContextError::NotFound(uri_str));
        }
        // Also delete versions
        let _ = sqlx::query("DELETE FROM context_versions WHERE uri = $1")
            .bind(&uri_str)
            .execute(pg)
            .await;
        Ok(())
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        let pg = self.pg_pool();
        let from_str = from.to_string();
        let to_str = to.to_string();

        let affected = sqlx::query(
            "UPDATE context_entries SET uri = $1, updated_at = now() WHERE uri = $2",
        )
        .bind(&to_str)
        .bind(&from_str)
        .execute(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("rename failed: {e}")))?;

        if affected.rows_affected() == 0 {
            return Err(ContextError::NotFound(from_str));
        }

        // Also update versions
        let _ = sqlx::query(
            "UPDATE context_versions SET uri = $1 WHERE uri = $2",
        )
        .bind(&to_str)
        .bind(&from_str)
        .execute(pg)
        .await;
        Ok(())
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
            SELECT uri, l0_abstract, memory_class FROM context_entries
            WHERE uri LIKE $1
            ORDER BY uri
            "#,
        )
        .bind(format!("{}%", &prefix))
        .fetch_all(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("ls failed: {e}")))?;

        let mut seen = std::collections::BTreeMap::new();
        for (uri_str, abstract_, memory_class_str) in rows {
            let rest = uri_str.strip_prefix(&prefix).unwrap_or(&uri_str);
            let mc = memory_class_str.as_deref().and_then(|s| parse_memory_class(s));
            // 直接子项：rest 不含 '/'
            let slash_pos = rest.find('/');
            if let Some(pos) = slash_pos {
                // 这是一个目录：rest[..pos] 是目录名
                let dir_name = &rest[..pos];
                seen.entry(dir_name.to_string())
                    .or_insert_with(|| DirEntry {
                        uri: ContextUri(format!("{}{}", prefix, dir_name)),
                        is_dir: true,
                        abstract_: String::new(),
                        memory_class: mc,
                    });
            } else {
                // 这是一个文件
                let context_uri = ContextUri(uri_str.clone());
                seen.entry(rest.to_string())
                    .or_insert(DirEntry {
                        uri: context_uri,
                        is_dir: false,
                        abstract_,
                        memory_class: mc,
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

        // 构建查询：有 class 过滤时加 WHERE 条件
        let results: Vec<String> = if let Some(mc) = pattern.class {
            let mc_name = memory_class_name(mc);
            sqlx::query_scalar::<_, String>(
                r#"
                SELECT uri FROM context_entries
                WHERE uri LIKE $1 AND memory_class = $2
                ORDER BY uri
                "#,
            )
            .bind(format!("{}%", &scope))
            .bind(mc_name)
            .fetch_all(pg)
            .await
            .map_err(|e| ContextError::Storage(format!("find failed: {e}")))?
        } else {
            sqlx::query_scalar::<_, String>(
                "SELECT uri FROM context_entries WHERE uri LIKE $1 ORDER BY uri",
            )
            .bind(format!("{}%", &scope))
            .fetch_all(pg)
            .await
            .map_err(|e| ContextError::Storage(format!("find failed: {e}")))?
        };

        Ok(results.into_iter().map(ContextUri).collect())
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
        .map_err(|e| ContextError::Storage(format!("grep failed: {e}")))?;

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
            hits.push(GrepHit {
                uri: ContextUri(uri_str),
                line: matched_line,
                level: ContentLevel::L0,
            });
        }
        Ok(hits)
    }

    async fn tree(&self, root: &ContextUri, depth: usize) -> Result<TreeNode> {
        let pg = self.pg_pool();
        let prefix = Self::dir_prefix(root);

        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT uri FROM context_entries WHERE uri LIKE $1 ORDER BY uri",
        )
        .bind(format!("{}%", &prefix))
        .fetch_all(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("tree failed: {e}")))?;

        let root_node = TreeNode {
            uri: root.clone(),
            is_dir: true,
            children: build_tree_level(&prefix, &rows, 0, depth),
        };
        Ok(root_node)
    }

    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        let pg = self.pg_pool();
        let uri_str = uri.to_string();

        let row = sqlx::query_as::<_, (String, Option<String>, Option<uuid::Uuid>)>(
            "SELECT l0_abstract, l1_overview, l2_detail_ref FROM context_entries WHERE uri = $1",
        )
        .bind(&uri_str)
        .fetch_optional(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("read failed: {e}")))?;

        match row {
            None => Err(ContextError::NotFound(uri_str)),
            Some((l0, l1, _l2_ref)) => match level {
                ContentLevel::L0 => Ok(ContentPayload::Abstract(l0)),
                ContentLevel::L1 => Ok(ContentPayload::Overview(
                    l1.unwrap_or_default(),
                )),
                ContentLevel::L2 => {
                    // L2: 返回完整条目 JSON（包含所有元数据）
                    // 实际 blob 内容由 AGFS 存储；此处返回 entry 的完整序列化，
                    // 包含 l0_abstract + l1_overview + metadata，供上层重建完整 ContextEntry
                    let entry_row = sqlx::query_as::<_, (serde_json::Value,)>(
                        r#"
                        SELECT row_to_json(t) FROM (
                            SELECT uri, tenant_id, l0_abstract, l1_overview, l2_detail_ref,
                                   content_type, memory_class, state_scope, tags, custom,
                                   mvcc_version, created_at, updated_at
                            FROM context_entries WHERE uri = $1
                        ) t
                        "#,
                    )
                    .bind(&uri_str)
                    .fetch_one(pg)
                    .await
                    .map_err(|e| ContextError::Storage(format!("read L2 failed: {e}")))?;

                    let bytes = serde_json::to_vec(&entry_row.0)
                        .unwrap_or_else(|_| l0.into_bytes());
                    Ok(ContentPayload::Detail(bytes))
                }
            },
        }
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
        .map_err(|e| ContextError::Storage(format!("version_history failed: {e}")))?;

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
        .map_err(|e| ContextError::Storage(format!("rollback read failed: {e}")))?;

        let json = entry_json.ok_or_else(|| {
            ContextError::VersionConflict(format!("no version {:?}", to))
        })?;

        let mut entry: ContextEntry = serde_json::from_value(json)
            .map_err(|e| ContextError::Serialization(format!("rollback parse: {e}")))?;

        // 把 entry 写回当前表（version++ 作为 rollback 操作的新版本）
        entry.mvcc_version = MvccVersion(0); // write() 会 +1
        entry.updated_at = chrono::Utc::now();
        self.write(entry).await?;
        Ok(())
    }

    async fn diff(
        &self,
        uri: &ContextUri,
        a: MvccVersion,
        b: MvccVersion,
    ) -> Result<ContextDiff> {
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
        .map_err(|e| ContextError::Storage(format!("diff read a failed: {e}")))?;

        let v_b: Option<String> = sqlx::query_scalar(
            "SELECT l0_abstract FROM context_versions WHERE uri = $1 AND mvcc_version = $2",
        )
        .bind(&uri_str)
        .bind(b.0 as i64)
        .fetch_optional(pg)
        .await
        .map_err(|e| ContextError::Storage(format!("diff read b failed: {e}")))?;

        let summary = match (v_a, v_b) {
            (Some(a_str), Some(b_str)) => {
                format!("{}: v{:?} → v{:?}\n---\n{}\n+++\n{}", uri_str, a.0, b.0, a_str, b_str)
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
        .map_err(|e| ContextError::Storage(format!("list_tenants failed: {e}")))?;

        Ok(rows
            .into_iter()
            .filter_map(|s| uuid::Uuid::parse_str(&s).ok())
            .map(TenantId)
            .collect())
    }
}

// ===========================================================================
// 辅助函数
// ===========================================================================

fn memory_class_name(c: MemoryClass) -> String {
    match c {
        MemoryClass::Profile => "profile",
        MemoryClass::Preferences => "preferences",
        MemoryClass::Entities => "entities",
        MemoryClass::Events => "events",
        MemoryClass::Cases => "cases",
        MemoryClass::Patterns => "patterns",
        MemoryClass::Tools => "tools",
        MemoryClass::Skills => "skills",
    }
    .to_string()
}

#[allow(dead_code)]
fn parse_memory_class(s: &str) -> Option<MemoryClass> {
    match s {
        "profile" => Some(MemoryClass::Profile),
        "preferences" => Some(MemoryClass::Preferences),
        "entities" => Some(MemoryClass::Entities),
        "events" => Some(MemoryClass::Events),
        "cases" => Some(MemoryClass::Cases),
        "patterns" => Some(MemoryClass::Patterns),
        "tools" => Some(MemoryClass::Tools),
        "skills" => Some(MemoryClass::Skills),
        _ => None,
    }
}

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
        seen.entry(name.to_string())
            .or_insert((ContextUri(format!("{}{}", prefix, name)), is_dir));
    }

    for (_name, (child_uri, is_dir)) in seen {
        if is_dir {
            let child_prefix = format!("{}{}/", prefix, _name);
            let sub_children = build_tree_level(&child_prefix, all_uris, current_depth + 1, max_depth);
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
    fn memory_class_roundtrips() {
        for c in &[
            MemoryClass::Profile,
            MemoryClass::Preferences,
            MemoryClass::Entities,
            MemoryClass::Events,
            MemoryClass::Cases,
            MemoryClass::Patterns,
            MemoryClass::Tools,
            MemoryClass::Skills,
        ] {
            let name = memory_class_name(*c);
            assert_eq!(parse_memory_class(&name), Some(*c));
        }
    }

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
    use agent_context_db_core::{ContentRef, ContextMeta, ContextStore, StateScope};
    use std::sync::Arc;
    use uwu_database::config::{
        CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend,
        VectorBackend, VectorConfig,
    };
    use uwu_database::sql;
    use uuid::Uuid;

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
            cache: CacheConfig { backend: CacheBackend::None, capacity: 0, url: None },
            vector: VectorConfig { backend: VectorBackend::Memory, url: None, api_key: None },
        }
    }

    async fn setup_store() -> PgContextStore {
        let _url = require_pg();
        let cfg = test_cfg();
        let pool = sql::build_pool(&cfg.database).await.unwrap();
        let arc_pool = Arc::new(pool);

        // 运行 context-db 迁移（手动创建表结构以确保存在）
        arc_pool.as_postgres().unwrap();
        arc_pool.exec(&format!(
            "CREATE TABLE IF NOT EXISTS context_entries (\
                uri             TEXT PRIMARY KEY,\
                tenant_id       UUID NOT NULL,\
                l0_abstract     TEXT NOT NULL,\
                l1_overview     TEXT,\
                l2_detail_ref   UUID,\
                content_type    TEXT NOT NULL DEFAULT 'text',\
                memory_class    TEXT,\
                state_scope     TEXT,\
                tags            JSONB NOT NULL DEFAULT '[]',\
                custom          JSONB NOT NULL DEFAULT '{{}}',\
                mvcc_version    BIGINT NOT NULL DEFAULT 0,\
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),\
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()\
            )"
        )).await.unwrap();
        arc_pool.exec(&format!(
            "CREATE TABLE IF NOT EXISTS context_versions (\
                uri             TEXT NOT NULL,\
                mvcc_version    BIGINT NOT NULL,\
                tenant_id       UUID NOT NULL,\
                l0_abstract     TEXT NOT NULL,\
                l1_overview     TEXT,\
                l2_detail_ref   UUID,\
                content_type    TEXT NOT NULL DEFAULT 'text',\
                memory_class    TEXT,\
                state_scope     TEXT,\
                tags            JSONB NOT NULL DEFAULT '[]',\
                custom          JSONB NOT NULL DEFAULT '{{}}',\
                entry_json      JSONB NOT NULL,\
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),\
                PRIMARY KEY (uri, mvcc_version)\
            )"
        )).await.unwrap();

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

        let v = store.write(e).await.unwrap();
        assert_eq!(v.0, 1);

        let payload = store.read(&uri, ContentLevel::L0).await.unwrap();
        assert!(matches!(payload, ContentPayload::Abstract(s) if s == "hello world"));
    }

    #[tokio::test]
    async fn test_write_updates_existing() {
        let store = setup_store().await;
        let t = tenant();
        let uri = ctx_uri("mem/skills/s1");

        store.write(entry(&uri, t, "v1")).await.unwrap();
        let v2 = store.write(entry(&uri, t, "v2 updated")).await.unwrap();

        assert_eq!(v2.0, 2, "version should increment on update");
        let payload = store.read(&uri, ContentLevel::L0).await.unwrap();
        assert!(matches!(payload, ContentPayload::Abstract(s) if s == "v2 updated"));
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
            l0_abstract: "L0 summary".into(),
            l1_overview: Some("L1 overview".into()),
            l2_detail_uri: Some(ContentRef(Uuid::new_v4())),
            content_type: ContentType::Text,
            metadata: ContextMeta {
                memory_class: Some(MemoryClass::Cases),
                state_scope: Some(StateScope::Long),
                tags: vec!["important".into(), "bug".into()],
                custom: serde_json::json!({"priority": "high"}),
            },
            mvcc_version: MvccVersion(0),
            created_at: now,
            updated_at: now,
        };

        store.write(entry).await.unwrap();

        // 通过数据库直接验证
        let pool = store.pg_pool();
        let row: (String, Option<String>, Option<String>) = sqlx::query_as(
            "SELECT l0_abstract, memory_class, state_scope FROM context_entries WHERE uri = $1"
        )
        .bind(&uri.0)
        .fetch_one(pool)
        .await
        .unwrap();
        assert_eq!(row.0, "L0 summary");
        assert_eq!(row.1, Some("cases".into()));
        assert_eq!(row.2, Some("long".into()));
    }

    // ── ContentRepo: delete ─────────────────────────────

    #[tokio::test]
    async fn test_delete_entry() {
        let store = setup_store().await;
        let t = tenant();
        let uri = ctx_uri("tmp/to_delete");

        store.write(entry(&uri, t, "bye")).await.unwrap();
        store.delete(&uri).await.unwrap();

        let r = store.read(&uri, ContentLevel::L0).await;
        assert!(r.is_err(), "should not find deleted entry");
    }

    #[tokio::test]
    async fn test_delete_nonexistent_returns_not_found() {
        let store = setup_store().await;
        let uri = ctx_uri("nonexistent/xyz");
        let r = store.delete(&uri).await;
        assert!(r.is_err());
    }

    // ── ContentRepo: rename ─────────────────────────────

    #[tokio::test]
    async fn test_rename_entry() {
        let store = setup_store().await;
        let t = tenant();
        let from = ctx_uri("old/name");
        let to = ctx_uri("new/name");

        store.write(entry(&from, t, "move me")).await.unwrap();
        store.rename(&from, &to).await.unwrap();

        // 旧路径不可读
        assert!(store.read(&from, ContentLevel::L0).await.is_err());
        // 新路径可读
        let p = store.read(&to, ContentLevel::L0).await.unwrap();
        assert!(matches!(p, ContentPayload::Abstract(s) if s == "move me"));
    }

    #[tokio::test]
    async fn test_rename_nonexistent_returns_not_found() {
        let store = setup_store().await;
        let r = store.rename(
            &ctx_uri("ghost/src"),
            &ctx_uri("ghost/dst"),
        ).await;
        assert!(r.is_err());
    }

    // ── FsOps: ls ───────────────────────────────────────

    #[tokio::test]
    async fn test_ls_directory() {
        let store = setup_store().await;
        let t = tenant();

        store.write(entry(&ctx_uri("dir/file1"), t, "a")).await.unwrap();
        store.write(entry(&ctx_uri("dir/file2"), t, "b")).await.unwrap();
        store.write(entry(&ctx_uri("dir/sub/file3"), t, "c")).await.unwrap();

        let dir = ctx_uri("dir");
        let entries = store.ls(&dir).await.unwrap();

        // file1 和 file2 是直接子项，sub 是子目录
        let files: Vec<_> = entries.iter().filter(|e| !e.is_dir).collect();
        let dirs: Vec<_> = entries.iter().filter(|e| e.is_dir).collect();

        assert_eq!(files.len(), 2, "should have 2 direct child files");
        assert_eq!(dirs.len(), 1, "should have 1 subdirectory");
        assert!(dirs.iter().any(|d| d.uri.0.contains("sub")));
    }

    // ── FsOps: find ─────────────────────────────────────

    #[tokio::test]
    async fn test_find_by_scope() {
        let store = setup_store().await;
        let t = tenant();

        store.write(entry(&ctx_uri("scope/a/x1"), t, "x")).await.unwrap();
        store.write(entry(&ctx_uri("scope/a/x2"), t, "y")).await.unwrap();
        store.write(entry(&ctx_uri("scope/b/y1"), t, "z")).await.unwrap();

        let results = store.find(&FindPattern {
            scope: Some(ctx_uri("scope/a")),
            ..Default::default()
        }).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|u| u.0.starts_with("uwu://t/agent/scope/a")));
    }

    #[tokio::test]
    async fn test_find_by_memory_class() {
        let store = setup_store().await;
        let t = tenant();

        let mut e1 = entry(&ctx_uri("class/c"), t, "case");
        e1.metadata.memory_class = Some(MemoryClass::Cases);
        store.write(e1).await.unwrap();

        let mut e2 = entry(&ctx_uri("class/s"), t, "skill");
        e2.metadata.memory_class = Some(MemoryClass::Skills);
        store.write(e2).await.unwrap();

        let results = store.find(&FindPattern {
            scope: Some(ctx_uri("class")),
            class: Some(MemoryClass::Cases),
            ..Default::default()
        }).await.unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].0.contains("class/c"));
    }

    // ── FsOps: grep ─────────────────────────────────────

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let store = setup_store().await;
        let t = tenant();

        store.write(entry(&ctx_uri("grep/hit1"), t, "Rust is great")).await.unwrap();
        store.write(entry(&ctx_uri("grep/hit2"), t, "Python is nice")).await.unwrap();
        store.write(entry(&ctx_uri("grep/miss"), t, "nothing here")).await.unwrap();

        let hits = store.grep("rust", &ctx_uri("grep")).await.unwrap();
        assert_eq!(hits.len(), 1, "case-insensitive grep should find RUST");
        assert!(hits[0].line.to_lowercase().contains("rust"));
    }

    // ── FsOps: tree ─────────────────────────────────────

    #[tokio::test]
    async fn test_tree_respects_depth() {
        let store = setup_store().await;
        let t = tenant();

        store.write(entry(&ctx_uri("tree/a/b/c/d"), t, "deep")).await.unwrap();
        store.write(entry(&ctx_uri("tree/a/shallow"), t, "shallow")).await.unwrap();

        // depth=1: 只到 tree/a/
        let node = store.tree(&ctx_uri("tree"), 1).await.unwrap();
        assert!(node.is_dir);
        assert!(!node.children.is_empty());

        // depth=0: 只看根
        let node = store.tree(&ctx_uri("tree/a"), 0).await.unwrap();
        assert!(node.children.is_empty());
    }

    // ── FsOps: read multi-level ──────────────────────────

    #[tokio::test]
    async fn test_read_l1_overview() {
        let store = setup_store().await;
        let t = tenant();
        let uri = ctx_uri("l1/test");

        let mut e = entry(&uri, t, "L0 summary");
        e.l1_overview = Some("L1 detailed overview".into());
        store.write(e).await.unwrap();

        let l1 = store.read(&uri, ContentLevel::L1).await.unwrap();
        assert!(matches!(l1, ContentPayload::Overview(s) if s == "L1 detailed overview"));
    }

    // ── VersionOps ──────────────────────────────────────

    #[tokio::test]
    async fn test_version_history() {
        let store = setup_store().await;
        let t = tenant();
        let uri = ctx_uri("ver/hist");

        store.write(entry(&uri, t, "first")).await.unwrap();
        store.write(entry(&uri, t, "second")).await.unwrap();
        store.write(entry(&uri, t, "third")).await.unwrap();

        let history = store.version_history(&uri).await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].version.0, 1);
        assert_eq!(history[2].version.0, 3);
    }

    #[tokio::test]
    async fn test_rollback() {
        let store = setup_store().await;
        let t = tenant();
        let uri = ctx_uri("ver/rb");

        let v1 = store.write(entry(&uri, t, "v1")).await.unwrap();
        store.write(entry(&uri, t, "v2")).await.unwrap();

        store.rollback(&uri, v1).await.unwrap();

        // rollback 创建了 v3，内容是 v1
        let payload = store.read(&uri, ContentLevel::L0).await.unwrap();
        assert!(matches!(payload, ContentPayload::Abstract(s) if s == "v1"));

        let history = store.version_history(&uri).await.unwrap();
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn test_diff_between_versions() {
        let store = setup_store().await;
        let t = tenant();
        let uri = ctx_uri("ver/diff");

        let v1 = store.write(entry(&uri, t, "original")).await.unwrap();
        let v2 = store.write(entry(&uri, t, "modified")).await.unwrap();

        let diff = store.diff(&uri, v1, v2).await.unwrap();
        assert!(diff.summary.contains("original"));
        assert!(diff.summary.contains("modified"));
    }

    // ── TenantOps ───────────────────────────────────────

    #[tokio::test]
    async fn test_list_tenants() {
        let store = setup_store().await;
        let t1 = tenant();
        let t2 = tenant();

        store.write(entry(&ctx_uri("tenants/e1"), t1, "a")).await.unwrap();
        store.write(entry(&ctx_uri("tenants/e2"), t2, "b")).await.unwrap();

        let tenants = store.list_tenants().await.unwrap();
        assert!(tenants.len() >= 2);
        assert!(tenants.iter().any(|x| x.0 == t1.0));
        assert!(tenants.iter().any(|x| x.0 == t2.0));
    }

    // ── ContextStore supertrait ─────────────────────────

    #[tokio::test]
    async fn test_context_store_supertrait_satisfied() {
        // 编译时验证：PgContextStore 实现了 ContextStore
        fn assert_store<T: ContextStore>() {}
        assert_store::<PgContextStore>();
    }
}
