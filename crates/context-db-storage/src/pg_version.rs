//! `PgVersionStore` —— Postgres 后端的版本 DAG 存储。
//!
//! # 设计要点
//!
//! - **Git 风格差量存储**：每个 commit 只存储相对 `parent[0]` 的变更（`version_entry_deltas`）。
//!   完整快照按需从最近祖先重建（沿 parent[0] 链遍历应用 delta）。
//! - **合并 commit** 存两个 parent（顺序：`into`, `from`）。差量仍相对 `parent[0]`。
//! - **CRDT 知识合并** 使用 `uwu_crdt::LwwMap<String, String>`（与 MemoryVersionStore 一致语义）。
//! - **命名引用** 独立于 commit：`version_branches`、`version_tags`。
//!
//! # 表结构（详见 `migrations.rs` v5）
//!
//! - `version_commits(id, scope, tree_hash, author_json, message, timestamp, metadata_json)`
//! - `version_commit_parents(commit_id, parent_id, ordinal)`
//! - `version_branches(scope, name, head, branch_type, lifecycle_json, created_from, created_at)`
//! - `version_tags(scope, name, target, tag_type, message, timestamp)`
//! - `version_entry_deltas(commit_id, uri, op, entry_json, rename_from)`
//! - `version_heads(scope, commit_id)`

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use agent_context_db_core::{ContentLevel, ContentPayload, ContextEntry, ContextUri};
use agent_context_db_version::{
    AsOfTime, Author, Branch, BranchLifecycle, BranchName, BranchType, ChangeSet, Commit, CommitId,
    CommitMeta, ConflictItem, ConflictResolution, ConflictResolutionSet, ConflictSession,
    ConflictSessionId, ConflictSessionPersistence, ConflictStrategy, ConflictValueOp, ContentHash,
    ContradictionDetector, GcPolicy, GcReport, ImpactAnalysis, InteractiveOperation,
    InteractiveVersionStore, KnowledgeMergeStrategy, LogOpts, MergeResult, MergeStrategy,
    ProvenanceGraph, Result, SquashResult, StructuredDiff, Tag, TagName, TagType, TemporalVersion,
    TreeDiff, VersionAnalysisConfig, VersionError, VersionRef, VersionStore,
    detect_snapshot_contradictions,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::Row;
use uuid::Uuid;
use uwu_database::{Cache, DbPool};

// ===========================================================================
// PgVersionStore
// ===========================================================================

/// PG 后端 VersionStore，基于 `uwu_database::DbPool` + `sqlx` 直接访问。
///
/// # 快照缓存（三级）
///
/// `reconstruct_snapshot` 查找顺序：
/// 1. **L1**：内存缓存 `uwu_database::Cache`（通过 `with_cache()` 注入，key `snap:{commit_id}`）
/// 2. **L2**：`version_commit_checkpoints` 表 —— 深链兜底，commit 不可变故无失效问题
/// 3. **回退**：沿 `first_parent` 链遍历应用 delta；链长超过 `checkpoint_threshold` 时
///    自动写入 L2 checkpoint（默认阈值 32）
///
/// L2 checkpoint 通过 `ON DELETE CASCADE` 与 `version_commits` 联动，`gc()` 删 commit 时
/// 自动清理。写入 checkpoint 后按容量上限驱逐冷 checkpoint：branch/tag/head 与最近/高频访问
/// checkpoint 优先保留。L1 需手动 `cache_del_snapshot`。
#[derive(Clone)]
pub struct PgVersionStore {
    pool: sqlx::PgPool,
    cache: Option<Arc<dyn Cache>>,
    cache_ttl: Option<Duration>,
    /// 链长超过此值时写入 L2 checkpoint。0 表示禁用 L2 写入。
    checkpoint_threshold: usize,
    /// L2 checkpoint 总容量上限。0 表示不保留 L2 checkpoint。
    checkpoint_max_rows: usize,
    /// 超出容量时额外保护最近写入/访问的 checkpoint 数量。
    checkpoint_hot_rows: usize,
    /// 交互式冲突 session 的默认持久化策略。
    conflict_session_persistence: ConflictSessionPersistence,
    contradiction_detector: Option<Arc<dyn ContradictionDetector>>,
    analysis_config: VersionAnalysisConfig,
}

impl PgVersionStore {
    /// 构造 —— 构造时验证后端是 postgres。
    pub fn new(pool: Arc<DbPool>, analysis_config: VersionAnalysisConfig) -> Result<Self> {
        analysis_config.validate()?;
        let pool = pool.as_postgres().map_err(Self::storage_err)?.clone();
        Ok(Self {
            pool,
            cache: None,
            cache_ttl: Some(Duration::from_secs(3600)),
            checkpoint_threshold: 32,
            checkpoint_max_rows: 4096,
            checkpoint_hot_rows: 256,
            conflict_session_persistence: ConflictSessionPersistence::Disabled,
            contradiction_detector: None,
            analysis_config,
        })
    }

    pub fn with_contradiction_detector(mut self, detector: Arc<dyn ContradictionDetector>) -> Self {
        self.contradiction_detector = Some(detector);
        self
    }

    /// 注入快照缓存 —— 大幅降低深链下 `read_at` / `merge` 的 I/O 次数。
    pub fn with_cache(mut self, cache: Arc<dyn Cache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// 覆盖缓存 TTL（默认 1 小时）。传 `None` 表示永久缓存。
    pub fn with_cache_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// 配置 L2 checkpoint 写入阈值（默认 32）。传 `0` 禁用 L2 写入（仅读取现有 checkpoint）。
    pub fn with_checkpoint_threshold(mut self, threshold: usize) -> Self {
        self.checkpoint_threshold = threshold;
        self
    }

    /// 配置 L2 checkpoint 容量策略。
    ///
    /// - `max_rows = 0`：写入后立即驱逐，等价于不保留新 checkpoint。
    /// - branch/tag/head 指向的 checkpoint 永远保护。
    /// - `hot_rows` 额外保护最近写入/访问的 checkpoint，剩余冷 checkpoint 按热度驱逐。
    pub fn with_checkpoint_policy(mut self, max_rows: usize, hot_rows: usize) -> Self {
        self.checkpoint_max_rows = max_rows;
        self.checkpoint_hot_rows = hot_rows.min(max_rows);
        self
    }

    /// 设置交互式冲突 session 默认是否持久化。
    pub fn with_conflict_session_persistence(
        mut self,
        persistence: ConflictSessionPersistence,
    ) -> Self {
        self.conflict_session_persistence = persistence;
        self
    }

    fn pg(&self) -> &sqlx::PgPool {
        &self.pool
    }

    fn scope_key(scope: &ContextUri) -> String {
        scope.to_string()
    }

    fn should_persist_session(&self, requested: ConflictSessionPersistence) -> bool {
        matches!(requested, ConflictSessionPersistence::Enabled)
            || matches!(
                self.conflict_session_persistence,
                ConflictSessionPersistence::Enabled
            )
    }

    fn storage_err<E: std::fmt::Display>(e: E) -> VersionError {
        VersionError::Storage(e.to_string())
    }

    // ---- Commit 读取 ---------------------------------------------------------

    async fn load_commit(&self, id: &CommitId) -> Result<Option<Commit>> {
        let row = sqlx::query(
            r#"SELECT id, scope, tree_hash, author_json, message, timestamp, metadata_json
               FROM version_commits WHERE id = $1"#,
        )
        .bind(id.0)
        .fetch_optional(self.pg())
        .await
        .map_err(Self::storage_err)?;
        let row = match row {
            Some(r) => r,
            None => return Ok(None),
        };
        let parents = self.load_parents(id).await?;
        let author_val: serde_json::Value =
            row.try_get("author_json").map_err(Self::storage_err)?;
        let author: Author = serde_json::from_value(author_val).map_err(Self::storage_err)?;
        let meta_val: serde_json::Value =
            row.try_get("metadata_json").map_err(Self::storage_err)?;
        let metadata: CommitMeta = serde_json::from_value(meta_val).map_err(Self::storage_err)?;
        let tree_hash_str: String = row.try_get("tree_hash").map_err(Self::storage_err)?;
        let timestamp: DateTime<Utc> = row.try_get("timestamp").map_err(Self::storage_err)?;
        let message: String = row.try_get("message").map_err(Self::storage_err)?;
        Ok(Some(Commit {
            id: id.clone(),
            parents,
            tree_hash: ContentHash(tree_hash_str),
            author,
            message,
            timestamp,
            metadata,
        }))
    }

    async fn load_parents(&self, id: &CommitId) -> Result<Vec<CommitId>> {
        let rows = sqlx::query(
            r#"SELECT parent_id FROM version_commit_parents
               WHERE commit_id = $1 ORDER BY ordinal ASC"#,
        )
        .bind(id.0)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        rows.into_iter()
            .map(|r| {
                r.try_get::<Uuid, _>("parent_id")
                    .map(CommitId)
                    .map_err(Self::storage_err)
            })
            .collect()
    }

    async fn first_parent(&self, id: &CommitId) -> Result<Option<CommitId>> {
        let row = sqlx::query(
            r#"SELECT parent_id FROM version_commit_parents
               WHERE commit_id = $1 ORDER BY ordinal ASC LIMIT 1"#,
        )
        .bind(id.0)
        .fetch_optional(self.pg())
        .await
        .map_err(Self::storage_err)?;
        row.map(|r| {
            r.try_get::<Uuid, _>("parent_id")
                .map(CommitId)
                .map_err(Self::storage_err)
        })
        .transpose()
    }

    async fn commit_timestamp(&self, id: &CommitId) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query("SELECT timestamp FROM version_commits WHERE id = $1")
            .bind(id.0)
            .fetch_optional(self.pg())
            .await
            .map_err(Self::storage_err)?;
        row.map(|r| r.try_get("timestamp").map_err(Self::storage_err))
            .transpose()
    }

    // ---- 差量 ---------------------------------------------------------------

    /// 拉取单个 commit 的 delta 列表。
    async fn load_deltas(&self, id: &CommitId) -> Result<Vec<DeltaRow>> {
        let rows = sqlx::query(
            r#"SELECT uri, op, entry_json, rename_from
               FROM version_entry_deltas WHERE commit_id = $1"#,
        )
        .bind(id.0)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        rows.into_iter()
            .map(|r| {
                Ok(DeltaRow {
                    uri: r.try_get("uri").map_err(Self::storage_err)?,
                    op: r.try_get("op").map_err(Self::storage_err)?,
                    entry_json: r.try_get("entry_json").map_err(Self::storage_err)?,
                    rename_from: r.try_get("rename_from").map_err(Self::storage_err)?,
                })
            })
            .collect()
    }

    /// 快照缓存 key —— commit_id 全局唯一，无需 scope 前缀。
    fn snapshot_cache_key(id: &CommitId) -> String {
        format!("snap:{}", id.0)
    }

    /// 尝试从缓存加载快照。
    async fn cache_get_snapshot(&self, id: &CommitId) -> Option<HashMap<String, String>> {
        let cache = self.cache.as_ref()?;
        let key = Self::snapshot_cache_key(id);
        let bytes = cache.get(&key).await.ok()??;
        serde_json::from_slice(&bytes).ok()
    }

    /// 回填缓存。该优化为 best-effort；失败按 operation/key/error 记录，不影响快照正确性。
    async fn cache_put_snapshot(&self, id: &CommitId, snap: &HashMap<String, String>) {
        let Some(cache) = self.cache.as_ref() else {
            return;
        };
        let Ok(bytes) = serde_json::to_vec(snap) else {
            return;
        };
        let key = Self::snapshot_cache_key(id);
        if let Err(error) = cache.set(&key, &bytes, self.cache_ttl).await {
            tracing::warn!(operation = "snapshot_cache_put", %key, %error, "best-effort snapshot cache operation failed");
        }
    }

    /// gc 时 best-effort 清理已删 commit 的缓存条目；失败只影响缓存空间。
    async fn cache_del_snapshot(&self, id: &CommitId) {
        let Some(cache) = self.cache.as_ref() else {
            return;
        };
        let key = Self::snapshot_cache_key(id);
        if let Err(error) = cache.del(&key).await {
            tracing::warn!(operation = "snapshot_cache_delete", %key, %error, "best-effort snapshot cache operation failed");
        }
    }

    // ---- L2 checkpoint --------------------------------------------------------

    /// 尝试从 `version_commit_checkpoints` 读取 L2 快照。
    async fn checkpoint_get(&self, id: &CommitId) -> Option<HashMap<String, String>> {
        let row = sqlx::query(
            r#"SELECT snapshot_json FROM version_commit_checkpoints WHERE commit_id = $1"#,
        )
        .bind(id.0)
        .fetch_optional(self.pg())
        .await
        .ok()
        .flatten()?;
        let val: serde_json::Value = row.try_get("snapshot_json").ok()?;
        let snap = serde_json::from_value(val).ok()?;
        self.checkpoint_touch(id).await;
        Some(snap)
    }

    /// L2 checkpoint 命中后更新热度（best-effort）。
    async fn checkpoint_touch(&self, id: &CommitId) {
        if let Err(error) = sqlx::query(
            r#"UPDATE version_commit_checkpoints
               SET last_accessed_at = now(), access_count = access_count + 1
               WHERE commit_id = $1"#,
        )
        .bind(id.0)
        .execute(self.pg())
        .await
        {
            tracing::warn!(operation = "checkpoint_touch", key = %id.0, %error, "best-effort checkpoint heat update failed");
        }
    }

    /// 写入 L2 checkpoint（best-effort，冲突时更新）。
    async fn checkpoint_put(&self, id: &CommitId, snap: &HashMap<String, String>) {
        if self.checkpoint_max_rows == 0 {
            return;
        }
        let Ok(val) = serde_json::to_value(snap) else {
            return;
        };
        let inserted = sqlx::query(
            r#"INSERT INTO version_commit_checkpoints
                   (commit_id, snapshot_json, last_accessed_at, access_count)
               VALUES ($1, $2, now(), 1)
               ON CONFLICT (commit_id) DO UPDATE SET
                   snapshot_json = EXCLUDED.snapshot_json,
                   last_accessed_at = now(),
                   access_count = version_commit_checkpoints.access_count + 1"#,
        )
        .bind(id.0)
        .bind(val)
        .execute(self.pg())
        .await
        .is_ok();
        if inserted {
            self.checkpoint_prune().await;
        }
    }

    /// 驱逐冷 checkpoint，保证 L2 表有容量上限。
    async fn checkpoint_prune(&self) {
        let max_rows = self.checkpoint_max_rows as i64;
        let hot_rows = self.checkpoint_hot_rows.min(self.checkpoint_max_rows) as i64;
        if let Err(error) = sqlx::query(
            r#"
            WITH protected_refs AS (
                SELECT commit_id FROM version_heads
                UNION SELECT commit_id FROM version_branches
                UNION SELECT commit_id FROM version_tags
            ), ranked AS (
                SELECT
                    cp.commit_id,
                    row_number() OVER (
                        ORDER BY cp.last_accessed_at DESC, cp.access_count DESC, cp.created_at DESC
                    ) AS hot_rank,
                    row_number() OVER (
                        ORDER BY
                            CASE WHEN pr.commit_id IS NOT NULL THEN 1 ELSE 0 END ASC,
                            cp.last_accessed_at ASC,
                            cp.access_count ASC,
                            cp.created_at ASC
                    ) AS cold_rank,
                    count(*) OVER () AS total_rows,
                    pr.commit_id IS NOT NULL AS protected
                FROM version_commit_checkpoints cp
                LEFT JOIN protected_refs pr ON pr.commit_id = cp.commit_id
            ), victims AS (
                SELECT commit_id
                FROM ranked
                WHERE total_rows > $1
                  AND NOT protected
                  AND hot_rank > $2
                ORDER BY cold_rank ASC
                LIMIT GREATEST((SELECT max(total_rows) FROM ranked) - $1, 0)
            )
            DELETE FROM version_commit_checkpoints cp
            USING victims v
            WHERE cp.commit_id = v.commit_id
            "#,
        )
        .bind(max_rows)
        .bind(hot_rows)
        .execute(self.pg())
        .await
        {
            tracing::warn!(operation = "checkpoint_prune", key = "version_commit_checkpoints", %error, "best-effort checkpoint pruning failed");
        }
    }

    /// 沿 parent[0] 链重建 commit 的完整快照（URI → 序列化 entry JSON）。
    ///
    /// 三级查找：
    /// 1. L1 内存缓存 —— 命中自身或祖先
    /// 2. L2 checkpoint 表 —— 命中自身或祖先
    /// 3. 沿链遍历应用 delta；链长 ≥ `checkpoint_threshold` 时回写 L2 checkpoint
    async fn reconstruct_snapshot(&self, id: &CommitId) -> Result<HashMap<String, String>> {
        // 快路径 1：L1 自身
        if let Some(snap) = self.cache_get_snapshot(id).await {
            return Ok(snap);
        }
        // 快路径 2：L2 自身
        if let Some(snap) = self.checkpoint_get(id).await {
            self.cache_put_snapshot(id, &snap).await; // 回填 L1
            return Ok(snap);
        }

        // 沿 first_parent 链收集，遇到 L1 或 L2 命中就停
        let mut chain = vec![id.clone()];
        let mut cur = id.clone();
        let mut seeded: Option<HashMap<String, String>> = None;
        while let Some(p) = self.first_parent(&cur).await? {
            if chain.contains(&p) {
                break; // 防环
            }
            // L1 命中？
            if let Some(snap) = self.cache_get_snapshot(&p).await {
                seeded = Some(snap);
                break;
            }
            // L2 命中？
            if let Some(snap) = self.checkpoint_get(&p).await {
                // 顺手回填 L1
                self.cache_put_snapshot(&p, &snap).await;
                seeded = Some(snap);
                break;
            }
            chain.push(p.clone());
            cur = p;
        }
        let chain_len = chain.len();
        // 从种子（或空）开始，沿链正向应用 delta
        let mut snapshot = seeded.unwrap_or_default();
        for cid in chain.iter().rev() {
            let deltas = self.load_deltas(cid).await?;
            apply_deltas(&mut snapshot, &deltas);
        }
        // 回填 L1
        self.cache_put_snapshot(id, &snapshot).await;
        // 深链 → 写 L2 checkpoint（避免下次冷启动再走同样长的链）
        if self.checkpoint_threshold > 0 && chain_len >= self.checkpoint_threshold {
            self.checkpoint_put(id, &snapshot).await;
        }
        Ok(snapshot)
    }

    /// 从两个快照计算 delta：dst 相对 src 的变更。
    fn diff_snapshots(
        src: &HashMap<String, String>,
        dst: &HashMap<String, String>,
    ) -> Vec<DeltaRow> {
        let mut out = Vec::new();
        for (uri, dv) in dst {
            match src.get(uri) {
                None => out.push(DeltaRow {
                    uri: uri.clone(),
                    op: "add".into(),
                    entry_json: serde_json::from_str(dv).ok(),
                    rename_from: None,
                }),
                Some(sv) if sv != dv => out.push(DeltaRow {
                    uri: uri.clone(),
                    op: "update".into(),
                    entry_json: serde_json::from_str(dv).ok(),
                    rename_from: None,
                }),
                _ => {}
            }
        }
        for uri in src.keys() {
            if !dst.contains_key(uri) {
                out.push(DeltaRow {
                    uri: uri.clone(),
                    op: "delete".into(),
                    entry_json: None,
                    rename_from: None,
                });
            }
        }
        out
    }

    fn snapshot_to_payload_entries(
        snapshot: &HashMap<String, String>,
    ) -> Vec<(String, ContentPayload)> {
        let mut entries = Vec::with_capacity(snapshot.len());
        for (uri, raw) in snapshot {
            if let Ok(payload) = serde_json::from_str(raw) {
                entries.push((uri.clone(), payload));
            }
        }
        entries.sort_by_key(|entry| entry.0.clone());
        entries
    }

    fn payload_entries_to_snapshot(
        entries: Vec<(String, ContentPayload)>,
    ) -> HashMap<String, String> {
        entries
            .into_iter()
            .filter_map(|(uri, payload)| serde_json::to_string(&payload).ok().map(|raw| (uri, raw)))
            .collect()
    }

    fn payload_to_raw(payload: &ContentPayload) -> Result<String> {
        serde_json::to_string(payload).map_err(Self::storage_err)
    }

    fn structured_entity_diff(
        uri: &ContextUri,
        old_raw: Option<&str>,
        new_raw: Option<&str>,
    ) -> Vec<agent_context_db_version::EntityChange> {
        use agent_context_db_version::{ChangeType, EntityChange};

        let old_value = old_raw.and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
        let new_value = new_raw.and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
        match (old_value, new_value) {
            (None, None) => Vec::new(),
            (None, Some(value)) => vec![EntityChange {
                entity_uri: uri.clone(),
                field: "*".into(),
                old_value: None,
                new_value: Some(value),
                change_type: ChangeType::Set,
            }],
            (Some(value), None) => vec![EntityChange {
                entity_uri: uri.clone(),
                field: "*".into(),
                old_value: Some(value),
                new_value: None,
                change_type: ChangeType::Remove,
            }],
            (Some(old), Some(new)) if old == new => Vec::new(),
            (Some(old), Some(new)) => {
                let mut changes = Vec::new();
                Self::diff_json_value(uri, "", &old, &new, &mut changes);
                if changes.is_empty() {
                    changes.push(EntityChange {
                        entity_uri: uri.clone(),
                        field: "*".into(),
                        old_value: Some(old),
                        new_value: Some(new),
                        change_type: ChangeType::Set,
                    });
                }
                changes
            }
        }
    }

    fn diff_json_value(
        uri: &ContextUri,
        path: &str,
        old: &serde_json::Value,
        new: &serde_json::Value,
        changes: &mut Vec<agent_context_db_version::EntityChange>,
    ) {
        use agent_context_db_version::{ChangeType, EntityChange};

        match (old, new) {
            (serde_json::Value::Object(old_map), serde_json::Value::Object(new_map)) => {
                let mut keys = old_map.keys().chain(new_map.keys()).collect::<Vec<_>>();
                keys.sort();
                keys.dedup();
                for key in keys {
                    let child_path = if path.is_empty() {
                        key.to_string()
                    } else {
                        format!("{path}.{key}")
                    };
                    match (old_map.get(key), new_map.get(key)) {
                        (Some(old_child), Some(new_child)) => {
                            Self::diff_json_value(uri, &child_path, old_child, new_child, changes)
                        }
                        (Some(old_child), None) => changes.push(EntityChange {
                            entity_uri: uri.clone(),
                            field: child_path,
                            old_value: Some(old_child.clone()),
                            new_value: None,
                            change_type: ChangeType::Remove,
                        }),
                        (None, Some(new_child)) => changes.push(EntityChange {
                            entity_uri: uri.clone(),
                            field: child_path,
                            old_value: None,
                            new_value: Some(new_child.clone()),
                            change_type: ChangeType::Set,
                        }),
                        (None, None) => {}
                    }
                }
            }
            (serde_json::Value::Array(old_items), serde_json::Value::Array(new_items)) => {
                let common = old_items.len().min(new_items.len());
                for idx in 0..common {
                    let child_path = format!("{}[{}]", path_or_root(path), idx);
                    Self::diff_json_value(
                        uri,
                        &child_path,
                        &old_items[idx],
                        &new_items[idx],
                        changes,
                    );
                }
                for item in &new_items[common..] {
                    changes.push(EntityChange {
                        entity_uri: uri.clone(),
                        field: path_or_root(path).into(),
                        old_value: None,
                        new_value: Some(item.clone()),
                        change_type: ChangeType::ArrayAppend,
                    });
                }
                for item in &old_items[common..] {
                    changes.push(EntityChange {
                        entity_uri: uri.clone(),
                        field: path_or_root(path).into(),
                        old_value: Some(item.clone()),
                        new_value: None,
                        change_type: ChangeType::ArrayRemove,
                    });
                }
            }
            _ if old != new => changes.push(EntityChange {
                entity_uri: uri.clone(),
                field: path_or_root(path).into(),
                old_value: Some(old.clone()),
                new_value: Some(new.clone()),
                change_type: ChangeType::Set,
            }),
            _ => {}
        }
    }

    fn raw_to_payload(raw: &str) -> Option<ContentPayload> {
        serde_json::from_str(raw).ok()
    }

    fn apply_session_resolutions(
        session: &ConflictSession,
        resolutions: &ConflictResolutionSet,
    ) -> Result<HashMap<String, String>> {
        let mut snapshot = Self::payload_entries_to_snapshot(session.clean_snapshot.clone());
        for conflict in &session.conflicts {
            let Some(resolution) = resolutions.get(&conflict.uri) else {
                return Err(VersionError::ConflictSessionIncomplete(
                    conflict.uri.to_string(),
                ));
            };
            match resolution {
                ConflictResolution::Ours => match &conflict.ours {
                    Some(payload) => {
                        snapshot.insert(conflict.uri.to_string(), Self::payload_to_raw(payload)?);
                    }
                    None => {
                        snapshot.remove(&conflict.uri.to_string());
                    }
                },
                ConflictResolution::Theirs => match &conflict.theirs {
                    Some(payload) => {
                        snapshot.insert(conflict.uri.to_string(), Self::payload_to_raw(payload)?);
                    }
                    None => {
                        snapshot.remove(&conflict.uri.to_string());
                    }
                },
                ConflictResolution::Delete => {
                    snapshot.remove(&conflict.uri.to_string());
                }
                ConflictResolution::Manual(payload) => {
                    snapshot.insert(conflict.uri.to_string(), Self::payload_to_raw(payload)?);
                }
            }
        }
        Ok(snapshot)
    }

    async fn persist_conflict_session(&self, session: &ConflictSession) -> Result<()> {
        let session_json = serde_json::to_value(session).map_err(Self::storage_err)?;
        let operation_json = serde_json::to_value(&session.operation).map_err(Self::storage_err)?;
        sqlx::query(
            r#"INSERT INTO version_conflict_sessions
               (id, scope, operation_json, session_json, status, created_at, updated_at)
               VALUES ($1, $2, $3, $4, 'open', now(), now())
               ON CONFLICT (id) DO UPDATE SET
                   scope = EXCLUDED.scope,
                   operation_json = EXCLUDED.operation_json,
                   session_json = EXCLUDED.session_json,
                   status = 'open',
                   updated_at = now()"#,
        )
        .bind(session.id.0)
        .bind(session.scope.to_string())
        .bind(operation_json)
        .bind(session_json)
        .execute(self.pg())
        .await
        .map_err(Self::storage_err)?;
        Ok(())
    }

    async fn mark_conflict_session_status(
        &self,
        id: &ConflictSessionId,
        status: &str,
    ) -> Result<()> {
        let affected = sqlx::query(
            r#"UPDATE version_conflict_sessions
               SET status = $2, updated_at = now(), finished_at = now()
               WHERE id = $1 AND status = 'open'"#,
        )
        .bind(id.0)
        .bind(status)
        .execute(self.pg())
        .await
        .map_err(Self::storage_err)?
        .rows_affected();
        if affected == 0 {
            return Err(VersionError::ConflictSessionUnavailable(id.to_string()));
        }
        Ok(())
    }

    // ---- 祖先关系 ------------------------------------------------------------

    /// Finds the nearest common ancestor by walking all ancestors of `a` and then breadth-first
    /// from `b`. Breadth-first traversal makes the first intersection the merge base nearest `b`.
    async fn merge_base(&self, a: &CommitId, b: &CommitId) -> Result<Option<CommitId>> {
        let mut ancestors = HashSet::new();
        let mut stack = vec![a.clone()];
        while let Some(id) = stack.pop() {
            if ancestors.insert(id.clone()) {
                stack.extend(self.load_parents(&id).await?);
            }
        }
        let mut queue = std::collections::VecDeque::from([b.clone()]);
        let mut seen = HashSet::new();
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if ancestors.contains(&id) {
                return Ok(Some(id));
            }
            queue.extend(self.load_parents(&id).await?);
        }
        Ok(None)
    }

    async fn is_ancestor(&self, ancestor: &CommitId, candidate: &CommitId) -> Result<bool> {
        if ancestor == candidate {
            return Ok(true);
        }
        let mut visited: HashSet<CommitId> = HashSet::new();
        let mut stack = vec![candidate.clone()];
        while let Some(cur) = stack.pop() {
            if !visited.insert(cur.clone()) {
                continue;
            }
            if &cur == ancestor {
                return Ok(true);
            }
            let ps = self.load_parents(&cur).await?;
            stack.extend(ps);
        }
        Ok(false)
    }

    async fn commits_to_rebase(
        &self,
        branch_head: &CommitId,
        onto_head: &CommitId,
    ) -> Result<Vec<CommitId>> {
        let mut own = Vec::new();
        let mut cur = branch_head.clone();
        loop {
            if self.is_ancestor(&cur, onto_head).await? {
                break;
            }
            own.push(cur.clone());
            match self.first_parent(&cur).await? {
                Some(p) => cur = p,
                None => break,
            }
        }
        own.reverse();
        Ok(own)
    }

    // ---- 写入 commit ---------------------------------------------------------

    async fn insert_commit_row(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        commit: &Commit,
        scope: &str,
    ) -> Result<()> {
        let author_json = serde_json::to_value(&commit.author).unwrap_or(serde_json::json!({}));
        let meta_json = serde_json::to_value(&commit.metadata).unwrap_or(serde_json::json!({}));
        sqlx::query(
            r#"INSERT INTO version_commits
               (id, scope, tree_hash, author_json, message, timestamp, metadata_json)
               VALUES ($1, $2, $3, $4, $5, $6, $7)"#,
        )
        .bind(commit.id.0)
        .bind(scope)
        .bind(&commit.tree_hash.0)
        .bind(author_json)
        .bind(&commit.message)
        .bind(commit.timestamp)
        .bind(meta_json)
        .execute(&mut **tx)
        .await
        .map_err(Self::storage_err)?;
        for (i, p) in commit.parents.iter().enumerate() {
            sqlx::query(
                r#"INSERT INTO version_commit_parents (commit_id, parent_id, ordinal)
                   VALUES ($1, $2, $3)"#,
            )
            .bind(commit.id.0)
            .bind(p.0)
            .bind(i as i16)
            .execute(&mut **tx)
            .await
            .map_err(Self::storage_err)?;
        }
        Ok(())
    }

    async fn insert_deltas(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        commit_id: &CommitId,
        deltas: &[DeltaRow],
    ) -> Result<()> {
        for d in deltas {
            sqlx::query(
                r#"INSERT INTO version_entry_deltas
                   (commit_id, uri, op, entry_json, rename_from)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (commit_id, uri) DO UPDATE
                     SET op = EXCLUDED.op,
                         entry_json = EXCLUDED.entry_json,
                         rename_from = EXCLUDED.rename_from"#,
            )
            .bind(commit_id.0)
            .bind(&d.uri)
            .bind(&d.op)
            .bind(&d.entry_json)
            .bind(&d.rename_from)
            .execute(&mut **tx)
            .await
            .map_err(Self::storage_err)?;
        }
        Ok(())
    }

    // ---- HEAD / 分支 ---------------------------------------------------------

    async fn get_head(&self, scope: &str) -> Result<Option<CommitId>> {
        let row = sqlx::query("SELECT commit_id FROM version_heads WHERE scope = $1")
            .bind(scope)
            .fetch_optional(self.pg())
            .await
            .map_err(Self::storage_err)?;
        Ok(row
            .and_then(|r| r.try_get::<Uuid, _>("commit_id").ok())
            .map(CommitId))
    }

    async fn set_head(&self, scope: &str, commit: &CommitId) -> Result<()> {
        sqlx::query(
            r#"INSERT INTO version_heads (scope, commit_id) VALUES ($1, $2)
               ON CONFLICT (scope) DO UPDATE SET commit_id = EXCLUDED.commit_id"#,
        )
        .bind(scope)
        .bind(commit.0)
        .execute(self.pg())
        .await
        .map_err(Self::storage_err)?;
        Ok(())
    }

    async fn advance_head_if_matches(
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        scope: &str,
        expected: &CommitId,
        commit: &CommitId,
    ) -> Result<bool> {
        let affected = sqlx::query(
            "UPDATE version_heads SET commit_id = $1 WHERE scope = $2 AND commit_id = $3",
        )
        .bind(commit.0)
        .bind(scope)
        .bind(expected.0)
        .execute(&mut **tx)
        .await
        .map_err(Self::storage_err)?;
        Ok(affected.rows_affected() == 1)
    }

    async fn update_head_in_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        scope: &str,
        expected: &CommitId,
        commit: &CommitId,
    ) -> Result<()> {
        let affected = sqlx::query(
            "UPDATE version_heads SET commit_id = $1 WHERE scope = $2 AND commit_id = $3",
        )
        .bind(commit.0)
        .bind(scope)
        .bind(expected.0)
        .execute(&mut **tx)
        .await
        .map_err(Self::storage_err)?;
        if affected.rows_affected() != 1 {
            return Err(VersionError::Storage(format!(
                "HEAD for {scope} changed concurrently"
            )));
        }
        Ok(())
    }

    async fn get_branch(&self, scope: &str, name: &BranchName) -> Result<Option<Branch>> {
        let row = sqlx::query(
            r#"SELECT name, head, branch_type, lifecycle_json, created_from, created_at
               FROM version_branches WHERE scope = $1 AND name = $2"#,
        )
        .bind(scope)
        .bind(name.as_str())
        .fetch_optional(self.pg())
        .await
        .map_err(Self::storage_err)?;
        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(branch_from_row(row)?))
    }

    async fn update_branch_head(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        scope: &str,
        name: &BranchName,
        expected: &CommitId,
        head: &CommitId,
    ) -> Result<()> {
        let affected = sqlx::query(
            "UPDATE version_branches SET head = $1 WHERE scope = $2 AND name = $3 AND head = $4",
        )
        .bind(head.0)
        .bind(scope)
        .bind(name.as_str())
        .bind(expected.0)
        .execute(&mut **tx)
        .await
        .map_err(Self::storage_err)?;
        if affected.rows_affected() != 1 {
            return Err(VersionError::Storage(format!(
                "branch {} changed concurrently",
                name.as_str()
            )));
        }
        Ok(())
    }
}

// ===========================================================================
// 内部辅助
// ===========================================================================

#[derive(Debug, Clone)]
struct DeltaRow {
    uri: String,
    op: String,
    entry_json: Option<serde_json::Value>,
    rename_from: Option<String>,
}

/// 把 delta 应用到快照上（就地修改）。
fn apply_deltas(snapshot: &mut HashMap<String, String>, deltas: &[DeltaRow]) {
    for d in deltas {
        match d.op.as_str() {
            "add" | "update" => {
                if let Some(v) = &d.entry_json {
                    snapshot.insert(d.uri.clone(), v.to_string());
                }
            }
            "delete" => {
                snapshot.remove(&d.uri);
            }
            "rename" => {
                if let Some(from) = &d.rename_from
                    && let Some(v) = snapshot.remove(from)
                {
                    snapshot.insert(d.uri.clone(), v);
                }
            }
            _ => {} // 未知 op 忽略
        }
    }
}

fn branch_from_row(row: sqlx::postgres::PgRow) -> Result<Branch> {
    let name: String = row.try_get("name").map_err(PgVersionStore::storage_err)?;
    let head: Uuid = row.try_get("head").map_err(PgVersionStore::storage_err)?;
    let created_from: Uuid = row
        .try_get("created_from")
        .map_err(PgVersionStore::storage_err)?;
    let created_at: DateTime<Utc> = row
        .try_get("created_at")
        .map_err(PgVersionStore::storage_err)?;
    let bt_str: String = row
        .try_get("branch_type")
        .map_err(PgVersionStore::storage_err)?;
    let lc_val: serde_json::Value = row
        .try_get("lifecycle_json")
        .map_err(PgVersionStore::storage_err)?;
    Ok(Branch {
        name: BranchName::parse(name).map_err(|error| {
            VersionError::Storage(format!("invalid persisted branch name: {error}"))
        })?,
        head: CommitId(head),
        created_from: CommitId(created_from),
        created_at,
        branch_type: parse_branch_type(&bt_str),
        lifecycle: serde_json::from_value(lc_val).unwrap_or(BranchLifecycle::Active),
    })
}

fn parse_branch_type(s: &str) -> BranchType {
    match s {
        "StateFork" => BranchType::StateFork,
        "Experiment" => BranchType::Experiment,
        "Collaboration" => BranchType::Collaboration,
        "Staging" => BranchType::Staging,
        _ => BranchType::Main,
    }
}

fn branch_type_str(bt: BranchType) -> &'static str {
    match bt {
        BranchType::Main => "Main",
        BranchType::StateFork => "StateFork",
        BranchType::Experiment => "Experiment",
        BranchType::Collaboration => "Collaboration",
        BranchType::Staging => "Staging",
    }
}

// ===========================================================================
// VersionStore 实现
// ===========================================================================

#[async_trait]
impl VersionStore for PgVersionStore {
    // ---- Commit -----------------------------------------------------------

    #[tracing::instrument(skip(self, changes, meta), fields(scope = %scope, adds = changes.adds.len(), updates = changes.updates.len(), deletes = changes.deletes.len()))]
    async fn commit(
        &self,
        scope: &ContextUri,
        changes: ChangeSet,
        meta: CommitMeta,
    ) -> Result<CommitId> {
        let scope_key = Self::scope_key(scope);
        let parent = self.get_head(&scope_key).await?;
        let parents: Vec<CommitId> = parent.iter().cloned().collect();
        let commit_id = CommitId::new();
        let now = Utc::now();

        // 把 ChangeSet 转成 DeltaRow
        let mut deltas = Vec::new();
        for entry in &changes.adds {
            deltas.push(DeltaRow {
                uri: entry.uri.to_string(),
                op: "add".into(),
                entry_json: Some(
                    serde_json::to_value(entry).map_err(|e| {
                        VersionError::Storage(format!("serialize added entry: {e}"))
                    })?,
                ),
                rename_from: None,
            });
        }
        for upd in &changes.updates {
            if upd.entry.uri != upd.uri {
                return Err(VersionError::Storage(format!(
                    "update URI {} does not match entry URI {}",
                    upd.uri, upd.entry.uri
                )));
            }
            deltas.push(DeltaRow {
                uri: upd.uri.to_string(),
                op: "update".into(),
                entry_json: Some(
                    serde_json::to_value(&upd.entry).map_err(|e| {
                        VersionError::Storage(format!("serialize updated entry: {e}"))
                    })?,
                ),
                rename_from: None,
            });
        }
        for uri in &changes.deletes {
            deltas.push(DeltaRow {
                uri: uri.to_string(),
                op: "delete".into(),
                entry_json: None,
                rename_from: None,
            });
        }
        for r in &changes.renames {
            deltas.push(DeltaRow {
                uri: r.to.to_string(),
                op: "rename".into(),
                entry_json: None,
                rename_from: Some(r.from.to_string()),
            });
        }

        let commit = Commit {
            id: commit_id.clone(),
            parents,
            tree_hash: ContentHash(format!("tree-{}", commit_id.0)),
            author: Author {
                agent_id: None,
                user_id: None,
                system: true,
            },
            message: String::new(),
            timestamp: now,
            metadata: meta,
        };

        let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
        // Scope 级事务锁覆盖首次提交（HEAD 行尚不存在）和后续 CAS，防止并发提交分叉后静默覆盖。
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(&scope_key)
            .execute(&mut *tx)
            .await
            .map_err(Self::storage_err)?;
        let locked_parent: Option<Uuid> =
            sqlx::query_scalar("SELECT commit_id FROM version_heads WHERE scope = $1 FOR UPDATE")
                .bind(&scope_key)
                .fetch_optional(&mut *tx)
                .await
                .map_err(Self::storage_err)?;
        if locked_parent.map(CommitId) != parent {
            return Err(VersionError::Storage(format!(
                "HEAD for {scope_key} changed concurrently"
            )));
        }
        self.insert_commit_row(&mut tx, &commit, &scope_key).await?;
        self.insert_deltas(&mut tx, &commit_id, &deltas).await?;
        if let Some(expected) = &parent {
            Self::update_head_in_tx(&mut tx, &scope_key, expected, &commit_id).await?;
        } else {
            sqlx::query("INSERT INTO version_heads (scope, commit_id) VALUES ($1, $2)")
                .bind(&scope_key)
                .bind(commit_id.0)
                .execute(&mut *tx)
                .await
                .map_err(Self::storage_err)?;
        }
        tx.commit().await.map_err(Self::storage_err)?;
        Ok(commit_id)
    }

    async fn log(&self, scope: &ContextUri, opts: &LogOpts) -> Result<Vec<Commit>> {
        let scope_key = Self::scope_key(scope);
        let start = if let Some(branch) = &opts.branch {
            self.get_branch(&scope_key, branch).await?.map(|b| b.head)
        } else {
            self.get_head(&scope_key).await?
        };
        let Some(head) = start else {
            return Ok(vec![]);
        };

        // 拓扑遍历（BFS 沿 parents）
        let mut out = Vec::new();
        let mut visited: HashSet<CommitId> = HashSet::new();
        let mut stack = vec![head];
        while let Some(id) = stack.pop() {
            if !visited.insert(id.clone()) {
                continue;
            }
            if let Some(commit) = self.load_commit(&id).await? {
                for p in &commit.parents {
                    stack.push(p.clone());
                }
                out.push(commit);
            }
            if let Some(max) = opts.max_count
                && out.len() >= max
            {
                break;
            }
        }
        out.sort_by_key(|commit| std::cmp::Reverse(commit.timestamp));
        Ok(out)
    }

    // ---- Branch -----------------------------------------------------------

    async fn create_branch(
        &self,
        scope: &ContextUri,
        name: BranchName,
        from: CommitId,
        bt: BranchType,
    ) -> Result<Branch> {
        let scope_key = Self::scope_key(scope);
        if self.get_branch(&scope_key, &name).await?.is_some() {
            return Err(VersionError::BranchExists(name.as_str().to_string()));
        }
        let now = Utc::now();
        sqlx::query(
            r#"INSERT INTO version_branches
               (scope, name, head, branch_type, lifecycle_json, created_from, created_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7)"#,
        )
        .bind(&scope_key)
        .bind(name.as_str())
        .bind(from.0)
        .bind(branch_type_str(bt))
        .bind(serde_json::to_value(BranchLifecycle::Active).unwrap_or(serde_json::json!({})))
        .bind(from.0)
        .bind(now)
        .execute(self.pg())
        .await
        .map_err(Self::storage_err)?;
        Ok(Branch {
            name,
            head: from.clone(),
            created_from: from,
            created_at: now,
            branch_type: bt,
            lifecycle: BranchLifecycle::Active,
        })
    }

    async fn list_branches(&self, scope: &ContextUri) -> Result<Vec<Branch>> {
        let scope_key = Self::scope_key(scope);
        let rows = sqlx::query(
            r#"SELECT name, head, branch_type, lifecycle_json, created_from, created_at
               FROM version_branches WHERE scope = $1 ORDER BY created_at ASC"#,
        )
        .bind(&scope_key)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        rows.into_iter().map(branch_from_row).collect()
    }

    async fn delete_branch(&self, scope: &ContextUri, name: &BranchName) -> Result<()> {
        let scope_key = Self::scope_key(scope);
        sqlx::query("DELETE FROM version_branches WHERE scope = $1 AND name = $2")
            .bind(&scope_key)
            .bind(name.as_str())
            .execute(self.pg())
            .await
            .map_err(Self::storage_err)?;
        Ok(())
    }

    async fn switch_head(&self, scope: &ContextUri, branch: &BranchName) -> Result<()> {
        let scope_key = Self::scope_key(scope);
        let b = self
            .get_branch(&scope_key, branch)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", branch.as_str())))?;
        self.set_head(&scope_key, &b.head).await
    }

    // ---- Tag --------------------------------------------------------------

    async fn create_tag(&self, scope: &ContextUri, tag: Tag) -> Result<()> {
        let scope_key = Self::scope_key(scope);
        let (tt, cond_expr) = match &tag.tag_type {
            TagType::Immutable => ("Immutable", None),
            TagType::Mutable => ("Mutable", None),
            TagType::Semantic { condition } => ("Semantic", Some(condition.expr.clone())),
        };
        sqlx::query(
            r#"INSERT INTO version_tags (scope, name, target, tag_type, message, timestamp, condition_expr)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (scope, name) DO UPDATE
               SET target = EXCLUDED.target,
                   message = EXCLUDED.message,
                   tag_type = EXCLUDED.tag_type,
                   condition_expr = EXCLUDED.condition_expr"#,
        )
        .bind(&scope_key)
        .bind(tag.name.as_str())
        .bind(tag.target.0)
        .bind(tt)
        .bind(&tag.message)
        .bind(tag.created_at)
        .bind(cond_expr)
        .execute(self.pg())
        .await
        .map_err(Self::storage_err)?;
        Ok(())
    }

    async fn list_tags(&self, scope: &ContextUri) -> Result<Vec<Tag>> {
        let scope_key = Self::scope_key(scope);
        let rows = sqlx::query(
            r#"SELECT name, target, tag_type, message, timestamp, condition_expr
               FROM version_tags WHERE scope = $1"#,
        )
        .bind(&scope_key)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        let mut out = Vec::new();
        for row in rows {
            let name: String = row.try_get("name").map_err(Self::storage_err)?;
            let target: Uuid = row.try_get("target").map_err(Self::storage_err)?;
            let tt: String = row.try_get("tag_type").map_err(Self::storage_err)?;
            let message: Option<String> = row.try_get("message").ok();
            let timestamp: DateTime<Utc> = row.try_get("timestamp").map_err(Self::storage_err)?;
            let cond_expr: Option<String> = row.try_get("condition_expr").ok().flatten();
            let tag_type = match tt.as_str() {
                "Immutable" => TagType::Immutable,
                "Semantic" => TagType::Semantic {
                    condition: agent_context_db_version::SemanticCondition {
                        expr: cond_expr.unwrap_or_default(),
                    },
                },
                _ => TagType::Mutable,
            };
            out.push(Tag {
                name: TagName::parse(name).map_err(|e| {
                    VersionError::Storage(format!("invalid persisted tag name: {e}"))
                })?,
                target: CommitId(target),
                tag_type,
                message: message.unwrap_or_default(),
                created_by: Author {
                    agent_id: None,
                    user_id: None,
                    system: true,
                },
                created_at: timestamp,
            });
        }
        Ok(out)
    }

    // ---- 读取 / 时间旅行 -----------------------------------------------------

    async fn read_at(
        &self,
        uri: &ContextUri,
        ref_: VersionRef,
        _level: ContentLevel,
    ) -> Result<ContentPayload> {
        let scope_key = Self::scope_key(uri);
        let commit_id = match ref_ {
            VersionRef::Commit(c) => c,
            VersionRef::Branch(name) => {
                self.get_branch(&scope_key, &name)
                    .await?
                    .ok_or_else(|| VersionError::NotFound(format!("branch {}", name.as_str())))?
                    .head
            }
            VersionRef::Tag(name) => {
                let row =
                    sqlx::query("SELECT target FROM version_tags WHERE scope = $1 AND name = $2")
                        .bind(&scope_key)
                        .bind(name.as_str())
                        .fetch_optional(self.pg())
                        .await
                        .map_err(Self::storage_err)?
                        .ok_or_else(|| VersionError::NotFound(format!("tag {}", name.as_str())))?;
                let target: Uuid = row.try_get("target").map_err(Self::storage_err)?;
                CommitId(target)
            }
            VersionRef::Head => self
                .get_head(&scope_key)
                .await?
                .ok_or_else(|| VersionError::NotFound(format!("HEAD of {scope_key}")))?,
        };
        let snapshot = self.reconstruct_snapshot(&commit_id).await?;
        let json = snapshot
            .get(&uri.to_string())
            .ok_or_else(|| VersionError::NotFound(uri.to_string()))?;
        Ok(payload_from_json(json))
    }

    async fn asof_read(
        &self,
        uri: &ContextUri,
        when: AsOfTime,
        level: ContentLevel,
    ) -> Result<ContentPayload> {
        let commit_id = match when {
            AsOfTime::Commit(c) => c,
            AsOfTime::Timestamp(ts) => {
                let scope_key = Self::scope_key(uri);
                let row = sqlx::query(
                    r#"SELECT id FROM version_commits
                       WHERE scope = $1 AND timestamp <= $2
                       ORDER BY timestamp DESC LIMIT 1"#,
                )
                .bind(&scope_key)
                .bind(ts)
                .fetch_optional(self.pg())
                .await
                .map_err(Self::storage_err)?
                .ok_or_else(|| VersionError::NotFound(format!("no commit before {ts}")))?;
                let id: Uuid = row.try_get("id").map_err(Self::storage_err)?;
                CommitId(id)
            }
        };
        self.read_at(uri, VersionRef::Commit(commit_id), level)
            .await
    }

    // ---- 合并 / Diff ---------------------------------------------------------

    #[tracing::instrument(skip(self), fields(scope = %scope, from = %from, into = %into, strategy = ?strategy))]
    async fn merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: MergeStrategy,
    ) -> Result<MergeResult> {
        let scope_key = Self::scope_key(scope);
        let from_head = self
            .get_branch(&scope_key, from)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", from.as_str())))?
            .head;
        let into_head = self
            .get_branch(&scope_key, into)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", into.as_str())))?
            .head;

        // Fast-forward 检测
        if self.is_ancestor(&into_head, &from_head).await? {
            let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
            self.update_branch_head(&mut tx, &scope_key, into, &into_head, &from_head)
                .await?;
            if !Self::advance_head_if_matches(&mut tx, &scope_key, &into_head, &from_head).await? {
                return Err(VersionError::Storage(
                    "HEAD changed concurrently during merge".into(),
                ));
            }
            tx.commit().await.map_err(Self::storage_err)?;
            return Ok(MergeResult {
                commit: from_head,
                conflicts: vec![],
            });
        }

        let base = self.merge_base(&from_head, &into_head).await?;
        let base_snap = match base {
            Some(id) => self.reconstruct_snapshot(&id).await?,
            None => HashMap::new(),
        };
        let from_snap = self.reconstruct_snapshot(&from_head).await?;
        let into_snap = self.reconstruct_snapshot(&into_head).await?;

        // True three-way merge over the union is required for delete propagation and all
        // add/update/delete intersections. A conflict exists only when both sides changed the
        // merge-base value differently.
        let mut merged = into_snap.clone();
        let mut conflicts = Vec::new();
        let keys: HashSet<String> = base_snap
            .keys()
            .chain(from_snap.keys())
            .chain(into_snap.keys())
            .cloned()
            .collect();
        for uri in keys {
            let base_value = base_snap.get(&uri);
            let ours = into_snap.get(&uri);
            let theirs = from_snap.get(&uri);
            if theirs == base_value {
                continue;
            }
            if ours == base_value || ours == theirs {
                match theirs {
                    Some(value) => {
                        merged.insert(uri, value.clone());
                    }
                    None => {
                        merged.remove(&uri);
                    }
                }
                continue;
            }
            match strategy {
                MergeStrategy::Theirs | MergeStrategy::FastForward => match theirs {
                    Some(value) => {
                        merged.insert(uri, value.clone());
                    }
                    None => {
                        merged.remove(&uri);
                    }
                },
                MergeStrategy::Ours => {}
                MergeStrategy::ThreeWay => {
                    conflicts.push(ContextUri::parse(&uri).map_err(Self::storage_err)?)
                }
            }
        }

        // 创建 merge commit（两个 parent，delta 相对 into_head）
        let deltas = Self::diff_snapshots(&into_snap, &merged);
        let commit_id = CommitId::new();
        let now = Utc::now();
        let merge_commit = Commit {
            id: commit_id.clone(),
            parents: vec![into_head.clone(), from_head.clone()],
            tree_hash: ContentHash(format!("tree-{}", commit_id.0)),
            author: Author {
                agent_id: None,
                user_id: None,
                system: true,
            },
            message: format!(
                "merge {} <- {} conflicts={}",
                into.as_str(),
                from.as_str(),
                conflicts.len()
            ),
            timestamp: now,
            metadata: CommitMeta::default(),
        };

        let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
        self.insert_commit_row(&mut tx, &merge_commit, &scope_key)
            .await?;
        self.insert_deltas(&mut tx, &commit_id, &deltas).await?;
        self.update_branch_head(&mut tx, &scope_key, into, &into_head, &commit_id)
            .await?;
        if !Self::advance_head_if_matches(&mut tx, &scope_key, &into_head, &commit_id).await? {
            return Err(VersionError::Storage(
                "HEAD changed concurrently during merge".into(),
            ));
        }
        tx.commit().await.map_err(Self::storage_err)?;
        Ok(MergeResult {
            commit: commit_id,
            conflicts,
        })
    }

    async fn diff_commits(
        &self,
        _scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<TreeDiff> {
        let sa = self.reconstruct_snapshot(a).await?;
        let sb = self.reconstruct_snapshot(b).await?;
        let mut diff = TreeDiff::default();
        for uri in sb.keys() {
            if !sa.contains_key(uri) {
                if let Ok(u) = ContextUri::parse(uri.as_str()) {
                    diff.adds.push(u);
                }
            } else if sa.get(uri) != sb.get(uri)
                && let Ok(u) = ContextUri::parse(uri.as_str())
            {
                diff.updates.push(u);
            }
        }
        for uri in sa.keys() {
            if !sb.contains_key(uri)
                && let Ok(u) = ContextUri::parse(uri.as_str())
            {
                diff.deletes.push(u);
            }
        }
        Ok(diff)
    }

    // ---- 历史改写 ------------------------------------------------------------

    #[tracing::instrument(skip(self), fields(scope = %scope, commit = %commit.0, onto = %onto, strategy = ?strategy))]
    async fn cherry_pick(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        onto: &BranchName,
        strategy: ConflictStrategy,
    ) -> Result<CommitId> {
        let scope_key = Self::scope_key(scope);
        let target = self
            .get_branch(&scope_key, onto)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", onto.as_str())))?
            .head;
        self.cherry_pick_at_branch(scope, commit, &target, Some(onto), strategy)
            .await
    }

    #[tracing::instrument(skip(self), fields(scope = %scope, branch = %branch, onto = %onto, strategy = ?strategy))]
    async fn rebase(
        &self,
        scope: &ContextUri,
        branch: &BranchName,
        onto: &BranchName,
        strategy: ConflictStrategy,
    ) -> Result<Vec<CommitId>> {
        let scope_key = Self::scope_key(scope);
        let branch_head = self
            .get_branch(&scope_key, branch)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", branch.as_str())))?
            .head;
        let onto_head = self
            .get_branch(&scope_key, onto)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", onto.as_str())))?
            .head;

        // 收集 branch 独有 commit 序列（旧 → 新）。
        let own = self.commits_to_rebase(&branch_head, &onto_head).await?;

        // 逐个 cherry-pick 到 onto —— 冲突时立即停止并向上抛错，
        // 已应用的 commit 不会被回滚（Git 的默认行为，交由调用方决定 abort/continue）
        let mut applied = Vec::new();
        let mut target = onto_head;
        for cid in &own {
            let new = self
                .cherry_pick_at_branch(scope, cid, &target, None, strategy)
                .await
                .map_err(|e| match e {
                    VersionError::MergeConflict(msg) => {
                        VersionError::MergeConflict(format!("rebase halted at {}: {msg}", cid.0))
                    }
                    other => other,
                })?;
            applied.push(new.clone());
            target = new;
        }
        // 更新 branch HEAD
        let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
        self.update_branch_head(&mut tx, &scope_key, branch, &branch_head, &target)
            .await?;
        tx.commit().await.map_err(Self::storage_err)?;
        Ok(applied)
    }

    async fn squash(
        &self,
        scope: &ContextUri,
        commits: Vec<CommitId>,
        message: &str,
    ) -> Result<SquashResult> {
        if commits.len() < 2 {
            return Err(VersionError::Storage("squash needs >= 2 commits".into()));
        }
        let scope_key = Self::scope_key(scope);
        // 假设 commits 按新旧顺序给出（最新在末尾）
        let newest = commits.last().cloned().ok_or_else(|| {
            VersionError::Storage("squash received no commits after validation".into())
        })?;
        let oldest = commits.first().cloned().ok_or_else(|| {
            VersionError::Storage("squash received no commits after validation".into())
        })?;
        let parent = self.first_parent(&oldest).await?;
        let base_snap = if let Some(p) = &parent {
            self.reconstruct_snapshot(p).await?
        } else {
            HashMap::new()
        };
        let tip_snap = self.reconstruct_snapshot(&newest).await?;
        let deltas = Self::diff_snapshots(&base_snap, &tip_snap);

        let new_id = CommitId::new();
        let now = Utc::now();
        let parents = parent.iter().cloned().collect();
        let new_commit = Commit {
            id: new_id.clone(),
            parents,
            tree_hash: ContentHash(format!("tree-{}", new_id.0)),
            author: Author {
                agent_id: None,
                user_id: None,
                system: true,
            },
            message: message.to_string(),
            timestamp: now,
            metadata: CommitMeta::default(),
        };

        let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
        self.insert_commit_row(&mut tx, &new_commit, &scope_key)
            .await?;
        self.insert_deltas(&mut tx, &new_id, &deltas).await?;
        tx.commit().await.map_err(Self::storage_err)?;
        Ok(SquashResult {
            new_commit: new_id,
            squashed_count: commits.len(),
        })
    }

    // ---- GC / 语义标签 -------------------------------------------------------

    #[tracing::instrument(skip(self, policy), fields(scope = %scope))]
    async fn gc(&self, scope: &ContextUri, policy: &GcPolicy) -> Result<GcReport> {
        let scope_key = Self::scope_key(scope);
        let cutoff = Utc::now() - chrono::Duration::days(policy.max_age_days);
        // 只清理不可达且过期的 commit。branch/tag/head 可达祖先全部受保护，避免删掉
        // 仍被某个引用依赖的历史链；keep_recent 额外保护该 scope 最近写入的 commit。
        let rows = sqlx::query(
            r#"
            WITH RECURSIVE protected_refs AS (
                SELECT commit_id AS id FROM version_heads WHERE scope = $1
                UNION
                SELECT head AS id FROM version_branches WHERE scope = $1
                UNION
                SELECT target AS id FROM version_tags WHERE scope = $1
            ), protected AS (
                SELECT id FROM protected_refs
                UNION
                SELECT p.parent_id AS id
                FROM version_commit_parents p
                JOIN protected pr ON pr.id = p.commit_id
            ), recent AS (
                SELECT id FROM version_commits
                WHERE scope = $1
                ORDER BY timestamp DESC
                LIMIT $3
            )
            SELECT c.id
            FROM version_commits c
            LEFT JOIN protected p ON p.id = c.id
            LEFT JOIN recent r ON r.id = c.id
            WHERE c.scope = $1
              AND c.timestamp < $2
              AND p.id IS NULL
              AND r.id IS NULL
            ORDER BY c.timestamp ASC
            "#,
        )
        .bind(&scope_key)
        .bind(cutoff)
        .bind(policy.keep_recent as i64)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        let ids: Vec<Uuid> = rows
            .into_iter()
            .filter_map(|r| r.try_get("id").ok())
            .collect();
        let removed = ids.len();
        // 删除 commit + 级联 delta 与 parent 引用
        for id in &ids {
            sqlx::query("DELETE FROM version_commits WHERE id = $1")
                .bind(id)
                .execute(self.pg())
                .await
                .map_err(Self::storage_err)?;
            // 缓存失效
            self.cache_del_snapshot(&CommitId(*id)).await;
        }
        Ok(GcReport {
            removed_commits: removed,
            freed_snapshots: removed,
        })
    }

    async fn evaluate_semantic_tags(&self, scope: &ContextUri) -> Result<Vec<(TagName, CommitId)>> {
        use cel_interpreter::{Context as CelCtx, Program};

        let scope_key = Self::scope_key(scope);
        // 拉取所有 Semantic tag（含 expr）
        let tag_rows = sqlx::query(
            r#"SELECT name, condition_expr FROM version_tags
               WHERE scope = $1 AND tag_type = 'Semantic'
                 AND condition_expr IS NOT NULL AND length(condition_expr) > 0"#,
        )
        .bind(&scope_key)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        if tag_rows.is_empty() {
            return Ok(vec![]);
        }

        // 编译所有表达式（失败的跳过 —— 不阻塞其他）
        let mut compiled: Vec<(TagName, Program)> = Vec::new();
        for row in tag_rows {
            let name: String = row.try_get("name").map_err(Self::storage_err)?;
            let expr: String = row.try_get("condition_expr").map_err(Self::storage_err)?;
            match Program::compile(&expr) {
                Ok(prog) => match TagName::parse(&name) {
                    Ok(name) => compiled.push((name, prog)),
                    Err(e) => tracing::warn!(
                        "evaluate_semantic_tags: skip invalid persisted tag {name}: {e}"
                    ),
                },
                Err(e) => {
                    tracing::warn!("evaluate_semantic_tags: skip tag {name} — invalid CEL: {e}");
                }
            }
        }
        if compiled.is_empty() {
            return Ok(vec![]);
        }

        // 遍历该 scope 最近 500 个 commit，逐个求值
        let max_commits = self.analysis_config.sql_limit();
        let commit_rows = sqlx::query(
            r#"SELECT id, message, timestamp, metadata_json
               FROM version_commits WHERE scope = $1
               ORDER BY timestamp DESC LIMIT $2"#,
        )
        .bind(&scope_key)
        .bind(max_commits)
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;

        let mut matched: Vec<(TagName, CommitId)> = Vec::new();
        for row in commit_rows {
            let id: Uuid = row.try_get("id").map_err(Self::storage_err)?;
            let message: String = row.try_get("message").map_err(Self::storage_err)?;
            let timestamp: DateTime<Utc> = row.try_get("timestamp").map_err(Self::storage_err)?;
            let meta_val: serde_json::Value = row
                .try_get::<serde_json::Value, _>("metadata_json")
                .unwrap_or(serde_json::json!({}));
            let parents = self.load_parents(&CommitId(id)).await?;
            let parents_json: Vec<serde_json::Value> = parents
                .iter()
                .map(|p| serde_json::Value::String(p.0.to_string()))
                .collect();
            let commit_json = serde_json::json!({
                "id": id.to_string(),
                "message": message,
                "timestamp": timestamp.timestamp(),
                "parents": parents_json,
                "metadata": meta_val,
            });

            for (tag_name, program) in &compiled {
                let mut ctx = CelCtx::default();
                if ctx.add_variable("commit", commit_json.clone()).is_err() {
                    continue;
                }
                match program.execute(&ctx) {
                    Ok(v) => {
                        if matches!(v, cel_interpreter::Value::Bool(true)) {
                            matched.push((tag_name.clone(), CommitId(id)));
                        }
                    }
                    Err(_) => {
                        // 求值失败（类型错误等）—— 静默跳过
                    }
                }
            }
        }
        Ok(matched)
    }

    // ---- 因果分析 ------------------------------------------------------------

    async fn provenance(&self, uri: &ContextUri) -> Result<ProvenanceGraph> {
        // 语义：只返回**显式证据溯源** —— CommitMeta.provenance 中写入方主动声明的链接。
        // 不沿 commit parent DAG 回溯（那是 CausalDag / impact_analysis / evolution 的职责）。
        // 详见 VersionStore::provenance 的文档注释。
        //
        // 收集所有修改此 URI 的 commit，按时间倒序 —— 最新在前
        let rows = sqlx::query(
            r#"SELECT d.commit_id AS cid, c.metadata_json AS meta
               FROM version_entry_deltas d
               JOIN version_commits c ON c.id = d.commit_id
               WHERE d.uri = $1
               ORDER BY c.timestamp DESC"#,
        )
        .bind(uri.to_string())
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;

        let mut nodes: Vec<agent_context_db_version::ProvenanceLink> = Vec::new();
        for row in &rows {
            let meta_val: serde_json::Value = row
                .try_get::<serde_json::Value, _>("meta")
                .unwrap_or(serde_json::json!({}));
            if let Ok(meta) = serde_json::from_value::<CommitMeta>(meta_val) {
                nodes.extend(meta.provenance);
            }
        }
        Ok(ProvenanceGraph {
            root_uri: uri.clone(),
            nodes,
            depth: rows.len(),
        })
    }

    async fn impact_analysis(&self, commit: &CommitId) -> Result<ImpactAnalysis> {
        // 找到 commit 修改的所有 URI，作为 downstream_uris
        let rows =
            sqlx::query(r#"SELECT DISTINCT uri FROM version_entry_deltas WHERE commit_id = $1"#)
                .bind(commit.0)
                .fetch_all(self.pg())
                .await
                .map_err(Self::storage_err)?;
        let downstream_uris: Vec<ContextUri> = rows
            .into_iter()
            .filter_map(|r| r.try_get::<String, _>("uri").ok())
            .filter_map(|s| ContextUri::parse(s.as_str()).ok())
            .collect();
        // 找到含此 commit 的分支
        let branch_rows = sqlx::query(r#"SELECT name FROM version_branches WHERE head = $1"#)
            .bind(commit.0)
            .fetch_all(self.pg())
            .await
            .map_err(Self::storage_err)?;
        let affected_branches = branch_rows
            .into_iter()
            .map(|row| {
                let name = row
                    .try_get::<String, _>("name")
                    .map_err(Self::storage_err)?;
                BranchName::parse(name).map_err(|error| {
                    VersionError::Storage(format!("invalid persisted branch name: {error}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ImpactAnalysis {
            commit: commit.clone(),
            downstream_uris,
            affected_branches,
        })
    }

    // ---- 语义 diff -----------------------------------------------------------

    async fn semantic_diff(
        &self,
        _scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<StructuredDiff> {
        let before = self.reconstruct_snapshot(a).await?;
        let after = self.reconstruct_snapshot(b).await?;
        let mut uris = before
            .keys()
            .chain(after.keys())
            .cloned()
            .collect::<Vec<_>>();
        uris.sort();
        uris.dedup();

        let mut entity_changes = Vec::new();
        for uri_str in uris {
            let Ok(uri) = ContextUri::parse(&uri_str) else {
                continue;
            };
            entity_changes.extend(Self::structured_entity_diff(
                &uri,
                before.get(&uri_str).map(String::as_str),
                after.get(&uri_str).map(String::as_str),
            ));
        }

        let summary = if entity_changes.is_empty() {
            "no semantic entity changes".to_string()
        } else {
            format!(
                "{} semantic field change(s) across commit {}..{}",
                entity_changes.len(),
                a.0,
                b.0
            )
        };

        Ok(StructuredDiff {
            entity_changes,
            relation_changes: vec![],
            fact_corrections: vec![],
            confidence_delta: 0.0,
            summary,
        })
    }

    // ---- 时态演化 ------------------------------------------------------------

    async fn evolution(&self, uri: &ContextUri) -> Result<Vec<TemporalVersion>> {
        let rows = sqlx::query(
            r#"SELECT d.commit_id AS cid, c.timestamp AS ts
               FROM version_entry_deltas d
               JOIN version_commits c ON c.id = d.commit_id
               WHERE d.uri = $1
               ORDER BY c.timestamp ASC"#,
        )
        .bind(uri.to_string())
        .fetch_all(self.pg())
        .await
        .map_err(Self::storage_err)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| {
                let cid: Uuid = r.try_get("cid").ok()?;
                let ts: DateTime<Utc> = r.try_get("ts").ok()?;
                Some(TemporalVersion {
                    commit_id: CommitId(cid),
                    timestamp: ts,
                    content_hash: ContentHash(format!("hash-{cid}")),
                    valid_from: ts,
                    valid_until: None,
                })
            })
            .collect())
    }

    // ---- 知识图谱合并（uwu-crdt LwwMap）------------------------------------

    async fn knowledge_merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: KnowledgeMergeStrategy,
    ) -> Result<MergeResult> {
        use uwu_crdt::LwwMap;

        let scope_key = Self::scope_key(scope);
        let from_head = self
            .get_branch(&scope_key, from)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", from.as_str())))?
            .head;
        let into_head = self
            .get_branch(&scope_key, into)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", into.as_str())))?
            .head;

        // Fast-forward
        if self.is_ancestor(&into_head, &from_head).await? {
            let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
            self.update_branch_head(&mut tx, &scope_key, into, &into_head, &from_head)
                .await?;
            if !Self::advance_head_if_matches(&mut tx, &scope_key, &into_head, &from_head).await? {
                return Err(VersionError::Storage(
                    "HEAD changed concurrently during merge".into(),
                ));
            }
            tx.commit().await.map_err(Self::storage_err)?;
            return Ok(MergeResult {
                commit: from_head,
                conflicts: vec![],
            });
        }

        let from_snap = self.reconstruct_snapshot(&from_head).await?;
        let into_snap = self.reconstruct_snapshot(&into_head).await?;
        let from_clock = self
            .commit_timestamp(&from_head)
            .await?
            .map(|t| t.timestamp() as u64)
            .unwrap_or(0);
        let into_clock = self
            .commit_timestamp(&into_head)
            .await?
            .map(|t| t.timestamp() as u64)
            .unwrap_or(0);

        let mut from_map: LwwMap<String, String> = LwwMap::new();
        for (k, v) in &from_snap {
            from_map.set(k.clone(), v.clone(), from_clock, from.as_str());
        }
        let mut into_map: LwwMap<String, String> = LwwMap::new();
        for (k, v) in &into_snap {
            into_map.set(k.clone(), v.clone(), into_clock, into.as_str());
        }
        let merged = uwu_crdt::merge(&from_map, &into_map);

        let conflicts: Vec<ContextUri> = match strategy {
            KnowledgeMergeStrategy::ContradictionDetection { threshold } => {
                let detector = self.contradiction_detector.as_ref().ok_or_else(|| {
                    VersionError::MergeConflict(
                        "ContradictionDetection requires an injected detector".into(),
                    )
                })?;
                detect_snapshot_contradictions(detector, &from_snap, &into_snap, threshold).await?
            }
            KnowledgeMergeStrategy::EntityAutoMerge | KnowledgeMergeStrategy::GraphMerge { .. } => {
                Vec::new()
            }
        };

        let merged_snap: HashMap<String, String> =
            merged.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let deltas = Self::diff_snapshots(&into_snap, &merged_snap);

        let new_id = CommitId::new();
        let now = Utc::now();
        let merge_commit = Commit {
            id: new_id.clone(),
            parents: vec![into_head.clone(), from_head.clone()],
            tree_hash: ContentHash(format!("tree-{}", new_id.0)),
            author: Author {
                agent_id: None,
                user_id: None,
                system: true,
            },
            message: format!(
                "knowledge_merge {} <- {} conflicts={}",
                into.as_str(),
                from.as_str(),
                conflicts.len()
            ),
            timestamp: now,
            metadata: CommitMeta::default(),
        };

        let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
        self.insert_commit_row(&mut tx, &merge_commit, &scope_key)
            .await?;
        self.insert_deltas(&mut tx, &new_id, &deltas).await?;
        self.update_branch_head(&mut tx, &scope_key, into, &into_head, &new_id)
            .await?;
        if !Self::advance_head_if_matches(&mut tx, &scope_key, &into_head, &new_id).await? {
            return Err(VersionError::Storage(
                "HEAD changed concurrently during rewrite".into(),
            ));
        }
        tx.commit().await.map_err(Self::storage_err)?;
        Ok(MergeResult {
            commit: new_id,
            conflicts,
        })
    }
}

// ===========================================================================
// 私有辅助（PgVersionStore 上不便加为 trait 方法的）
// ===========================================================================

impl PgVersionStore {
    /// cherry_pick 的内部形态：直接指定 target commit，可选携带目标分支名（用于更新 HEAD）。
    ///
    /// 冲突检测：对被 pick 的每个 URI，若 target 当前值既不等于 base（parent），
    /// 也不等于 commit 的新值，说明 target 独立修改过。
    async fn build_cherry_pick_session(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        target: &CommitId,
        onto: &BranchName,
    ) -> Result<ConflictSession> {
        let scope_key = Self::scope_key(scope);
        for id in [commit, target] {
            let found: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM version_commits WHERE id = $1 AND scope = $2)",
            )
            .bind(id.0)
            .bind(&scope_key)
            .fetch_one(self.pg())
            .await
            .map_err(Self::storage_err)?;
            if !found {
                return Err(VersionError::NotFound(format!("commit {}", id.0)));
            }
        }
        let parent = self.first_parent(commit).await?;
        let base_snap = if let Some(p) = &parent {
            self.reconstruct_snapshot(p).await?
        } else {
            HashMap::new()
        };
        let commit_snap = self.reconstruct_snapshot(commit).await?;
        let cherry_deltas = Self::diff_snapshots(&base_snap, &commit_snap);
        let mut clean_snap = self.reconstruct_snapshot(target).await?;
        let mut clean_deltas = Vec::new();
        let mut conflicts = Vec::new();

        for d in cherry_deltas {
            let target_val = clean_snap.get(&d.uri).cloned();
            let base_val = base_snap.get(&d.uri).cloned();
            let commit_val = commit_snap.get(&d.uri).cloned();
            if target_val != base_val && target_val != commit_val {
                let Some(uri) = ContextUri::parse(&d.uri).ok() else {
                    continue;
                };
                conflicts.push(ConflictItem {
                    uri,
                    base: base_val.as_deref().and_then(Self::raw_to_payload),
                    ours: target_val.as_deref().and_then(Self::raw_to_payload),
                    theirs: commit_val.as_deref().and_then(Self::raw_to_payload),
                    op: if commit_val.is_some() {
                        ConflictValueOp::Set
                    } else {
                        ConflictValueOp::Delete
                    },
                });
            } else {
                clean_deltas.push(d);
            }
        }
        apply_deltas(&mut clean_snap, &clean_deltas);

        Ok(ConflictSession {
            id: ConflictSessionId::new(),
            scope: scope.clone(),
            operation: InteractiveOperation::CherryPick {
                commit: commit.clone(),
                onto: onto.clone(),
            },
            base: parent,
            target: target.clone(),
            commits: vec![commit.clone()],
            conflicts,
            clean_snapshot: Self::snapshot_to_payload_entries(&clean_snap),
        })
    }

    async fn commit_cherry_pick_snapshot(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        target: &CommitId,
        onto_branch: Option<&BranchName>,
        final_snap: HashMap<String, String>,
    ) -> Result<CommitId> {
        let scope_key = Self::scope_key(scope);
        let target_before = self.reconstruct_snapshot(target).await?;
        let final_deltas = Self::diff_snapshots(&target_before, &final_snap);
        let new_id = CommitId::new();
        let now = Utc::now();
        let src_commit = self.load_commit(commit).await?;
        let new_commit = Commit {
            id: new_id.clone(),
            parents: vec![target.clone()],
            tree_hash: ContentHash(format!("tree-{}", new_id.0)),
            author: src_commit
                .as_ref()
                .map(|c| c.author.clone())
                .unwrap_or(Author {
                    agent_id: None,
                    user_id: None,
                    system: true,
                }),
            message: format!(
                "cherry-pick {} onto {}",
                commit.0,
                onto_branch
                    .map(|b| b.as_str().to_string())
                    .unwrap_or_else(|| target.0.to_string())
            ),
            timestamp: now,
            metadata: CommitMeta::default(),
        };
        let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
        self.insert_commit_row(&mut tx, &new_commit, &scope_key)
            .await?;
        self.insert_deltas(&mut tx, &new_id, &final_deltas).await?;
        if let Some(b) = onto_branch {
            self.update_branch_head(&mut tx, &scope_key, b, target, &new_id)
                .await?;
        }
        tx.commit().await.map_err(Self::storage_err)?;
        Ok(new_id)
    }

    async fn cherry_pick_at_branch(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        target: &CommitId,
        onto_branch: Option<&BranchName>,
        strategy: ConflictStrategy,
    ) -> Result<CommitId> {
        let onto = match onto_branch {
            Some(branch) => branch.clone(),
            None => BranchName::parse(target.0.to_string())?,
        };
        let session = self
            .build_cherry_pick_session(scope, commit, target, &onto)
            .await?;
        if strategy == ConflictStrategy::Fail && !session.conflicts.is_empty() {
            let conflicts = session
                .conflicts
                .iter()
                .map(|c| c.uri.to_string())
                .collect::<Vec<_>>();
            return Err(VersionError::MergeConflict(format!(
                "cherry-pick {} onto {}: {} URI(s) conflicted: {}",
                commit.0,
                target.0,
                conflicts.len(),
                conflicts.join(", ")
            )));
        }

        let mut resolutions = ConflictResolutionSet::default();
        for conflict in &session.conflicts {
            resolutions.insert(
                &conflict.uri,
                match strategy {
                    ConflictStrategy::Fail => ConflictResolution::Ours,
                    ConflictStrategy::Ours => ConflictResolution::Ours,
                    ConflictStrategy::Theirs => ConflictResolution::Theirs,
                },
            );
        }
        let final_snap = Self::apply_session_resolutions(&session, &resolutions)?;
        self.commit_cherry_pick_snapshot(scope, commit, target, onto_branch, final_snap)
            .await
    }
}

#[async_trait]
impl InteractiveVersionStore for PgVersionStore {
    async fn begin_cherry_pick(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        onto: &BranchName,
        persistence: ConflictSessionPersistence,
    ) -> Result<ConflictSession> {
        let scope_key = Self::scope_key(scope);
        let target = self
            .get_branch(&scope_key, onto)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", onto.as_str())))?
            .head;
        let session = self
            .build_cherry_pick_session(scope, commit, &target, onto)
            .await?;
        if self.should_persist_session(persistence) {
            self.persist_conflict_session(&session).await?;
        }
        Ok(session)
    }

    async fn begin_rebase(
        &self,
        scope: &ContextUri,
        branch: &BranchName,
        onto: &BranchName,
        persistence: ConflictSessionPersistence,
    ) -> Result<ConflictSession> {
        let scope_key = Self::scope_key(scope);
        let source_head = self
            .get_branch(&scope_key, branch)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", branch.as_str())))?
            .head;
        let target_head = self
            .get_branch(&scope_key, onto)
            .await?
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", onto.as_str())))?
            .head;
        let commits = self.commits_to_rebase(&source_head, &target_head).await?;
        if commits.is_empty() {
            return Ok(ConflictSession {
                id: ConflictSessionId::new(),
                scope: scope.clone(),
                operation: InteractiveOperation::Rebase {
                    branch: branch.clone(),
                    onto: onto.clone(),
                },
                base: Some(target_head.clone()),
                target: target_head,
                commits: Vec::new(),
                conflicts: Vec::new(),
                clean_snapshot: Vec::new(),
            });
        }

        let first = commits[0].clone();
        let mut session = self
            .build_cherry_pick_session(scope, &first, &target_head, branch)
            .await?;
        session.operation = InteractiveOperation::Rebase {
            branch: branch.clone(),
            onto: onto.clone(),
        };
        session.commits = commits;
        if self.should_persist_session(persistence) {
            self.persist_conflict_session(&session).await?;
        }
        Ok(session)
    }

    async fn continue_conflict_session(
        &self,
        session: ConflictSession,
        resolutions: ConflictResolutionSet,
    ) -> Result<Vec<CommitId>> {
        let final_snap = Self::apply_session_resolutions(&session, &resolutions)?;
        match &session.operation {
            InteractiveOperation::CherryPick { commit, onto } => {
                let new_id = self
                    .commit_cherry_pick_snapshot(
                        &session.scope,
                        commit,
                        &session.target,
                        Some(onto),
                        final_snap,
                    )
                    .await?;
                Ok(vec![new_id])
            }
            InteractiveOperation::Rebase { branch, .. } => {
                let first = session.commits.first().ok_or_else(|| {
                    VersionError::ConflictSessionUnavailable(session.id.to_string())
                })?;
                let first_new = self
                    .commit_cherry_pick_snapshot(
                        &session.scope,
                        first,
                        &session.target,
                        None,
                        final_snap,
                    )
                    .await?;
                let mut applied = vec![first_new.clone()];
                let mut target = first_new;
                for cid in session.commits.iter().skip(1) {
                    let new = self
                        .cherry_pick_at_branch(
                            &session.scope,
                            cid,
                            &target,
                            None,
                            ConflictStrategy::Fail,
                        )
                        .await
                        .map_err(|e| match e {
                            VersionError::MergeConflict(msg) => VersionError::MergeConflict(
                                format!("rebase halted at {}: {msg}", cid.0),
                            ),
                            other => other,
                        })?;
                    applied.push(new.clone());
                    target = new;
                }
                let scope_key = Self::scope_key(&session.scope);
                let mut tx = self.pg().begin().await.map_err(Self::storage_err)?;
                self.update_branch_head(&mut tx, &scope_key, branch, &session.target, &target)
                    .await?;
                tx.commit().await.map_err(Self::storage_err)?;
                Ok(applied)
            }
        }
    }

    async fn load_conflict_session(&self, id: &ConflictSessionId) -> Result<ConflictSession> {
        let row = sqlx::query(
            r#"SELECT session_json FROM version_conflict_sessions
               WHERE id = $1 AND status = 'open'"#,
        )
        .bind(id.0)
        .fetch_optional(self.pg())
        .await
        .map_err(Self::storage_err)?
        .ok_or_else(|| VersionError::ConflictSessionUnavailable(id.to_string()))?;
        let value: serde_json::Value = row.try_get("session_json").map_err(Self::storage_err)?;
        serde_json::from_value(value).map_err(Self::storage_err)
    }

    async fn continue_conflict_session_by_id(
        &self,
        id: &ConflictSessionId,
        resolutions: ConflictResolutionSet,
    ) -> Result<Vec<CommitId>> {
        let session = self.load_conflict_session(id).await?;
        let result = self.continue_conflict_session(session, resolutions).await?;
        self.mark_conflict_session_status(id, "completed").await?;
        Ok(result)
    }

    async fn abort_conflict_session(&self, id: &ConflictSessionId) -> Result<()> {
        self.mark_conflict_session_status(id, "aborted").await
    }
}

fn path_or_root(path: &str) -> &str {
    if path.is_empty() { "*" } else { path }
}

/// Rebuild payloads from current full-entry snapshots and legacy payload-only snapshots.
fn payload_from_json(json: &str) -> ContentPayload {
    if let Ok(entry) = serde_json::from_str::<ContextEntry>(json) {
        return entry.payload;
    }
    serde_json::from_str::<ContentPayload>(json).unwrap_or(ContentPayload::Text {
        sparse: json.to_string(),
        dense: String::new(),
        full: json.to_string(),
    })
}
