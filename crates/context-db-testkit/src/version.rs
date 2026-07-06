//! `MemoryVersionStore`：类 Git DAG 版本存储的内存实现。
//!
//! - 完整 Commit DAG（多 parent 支持 merge commit）
//! - 命名分支 + StateFork/Experiment/Main 分支类型
//! - Tag（Immutable/Mutable）
//! - FastForward 合并
//! - 时间旅行（按 CommitId / Timestamp）
//! - 每个 commit 存储内容快照（uri → ContentPayload），用于 read_at/asof_read
//!
//! ## 架构说明
//!
//! 本实现将内容快照存储在 commit 内（`commit → {uri → entry_json}`），
//! 与生产 PG 的 `context_versions` 表语义一致：每个版本独立存储条目快照。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContextEntry, ContextUri,
};
use agent_context_db_version::{
    AsOfTime, Author, Branch, BranchLifecycle, BranchName, BranchType, ChangeSet, Commit,
    CommitId, CommitMeta, CommitTrigger, ConflictStrategy, ContentHash, GcPolicy, GcReport,
    ImpactAnalysis, LogOpts, MergeResult, MergeStrategy, ProvenanceGraph, Result,
    KnowledgeMergeStrategy, SquashResult, StructuredDiff, Tag, TagName, TagType, TemporalVersion,
    TreeDiff, VersionRef, VersionStore, VersionError,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::collections::HashMap;

// ===========================================================================
// MemoryVersionStore
// ===========================================================================

pub struct MemoryVersionStore {
    /// commit_id → Commit
    commits: Mutex<HashMap<CommitId, Commit>>,
    /// commit_id → { uri_str → serialized ContextEntry JSON }
    entry_snapshots: Mutex<HashMap<CommitId, HashMap<String, String>>>,
    /// (scope_uri, branch_name) → Branch
    branches: Mutex<HashMap<(String, String), Branch>>,
    /// (scope_uri, tag_name) → Tag
    tags: Mutex<HashMap<(String, String), Tag>>,
    /// scope_uri → HEAD CommitId
    heads: Mutex<HashMap<String, CommitId>>,
}

impl MemoryVersionStore {
    pub fn new() -> Self {
        Self {
            commits: Mutex::new(HashMap::new()),
            entry_snapshots: Mutex::new(HashMap::new()),
            branches: Mutex::new(HashMap::new()),
            tags: Mutex::new(HashMap::new()),
            heads: Mutex::new(HashMap::new()),
        }
    }

    /// 获取 scope 的 HEAD commit（不存在时返回 None）。
    fn head(&self, scope: &str) -> Option<CommitId> {
        self.heads.lock().get(scope).cloned()
    }

    fn set_head(&self, scope: &str, id: CommitId) {
        self.heads.lock().insert(scope.to_string(), id);
    }

    fn scope_key(scope: &ContextUri) -> String {
        scope.to_string()
    }

    /// 在指定 parent commit 上创建新 commit（不依赖 scope HEAD）。
    ///
    /// 用于模拟并行推演场景（两个 commit 从同一祖先分叉）。
    /// 生产环境中由 `VersionStore::commit()` 通过 scope HEAD 管理。
    pub fn commit_on_parent(
        &self,
        parent: &CommitId,
        meta: CommitMeta,
    ) -> CommitId {
        let commit_id = CommitId::new();
        let now = Utc::now();

        let parent_snapshot = self
            .entry_snapshots
            .lock()
            .get(parent)
            .cloned()
            .unwrap_or_default();

        let commit = Commit {
            id: commit_id.clone(),
            parents: vec![parent.clone()],
            tree_hash: ContentHash(format!("tree-{}", commit_id.0)),
            author: Author {
                agent_id: None,
                user_id: None,
                system: matches!(meta.trigger, CommitTrigger::AutoConsolidation),
            },
            message: format!("{:?}", meta.trigger),
            timestamp: now,
            metadata: meta,
        };

        self.commits.lock().insert(commit_id.clone(), commit);
        self.entry_snapshots
            .lock()
            .insert(commit_id.clone(), parent_snapshot);
        commit_id
    }

    /// 协助测试：预先注册条目内容，使 read_at 可工作。
    ///
    /// 生产环境由 `ContentRepo::write()` 同步写入 `context_versions` 表；
    /// 内存版通过此方法显式注入条目的各个版本。
    pub fn put_entry_version(
        &self,
        commit_id: &CommitId,
        uri: &ContextUri,
        entry: &ContextEntry,
    ) {
        let json = serde_json::to_string(entry).unwrap_or_default();
        self.entry_snapshots
            .lock()
            .entry(commit_id.clone())
            .or_default()
            .insert(uri.to_string(), json);
    }
}

impl Default for MemoryVersionStore {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// VersionStore 实现
// ===========================================================================

#[async_trait]
impl VersionStore for MemoryVersionStore {
    // ── commit ──────────────────────────────────────────────────────

    async fn commit(
        &self,
        scope: &ContextUri,
        changes: ChangeSet,
        meta: CommitMeta,
    ) -> Result<CommitId> {
        let scope_key = Self::scope_key(scope);
        let parents = self
            .head(&scope_key)
            .map(|h| vec![h])
            .unwrap_or_default();

        let commit_id = CommitId::new();
        let now = Utc::now();

        // 构建 commit 的内容快照：从 parent 继承 → 应用 changes
        let mut snapshot: HashMap<String, String> = if let Some(parent) = parents.first() {
            self.entry_snapshots
                .lock()
                .get(parent)
                .cloned()
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        // 应用变更（标记 add/update，实际内容通过 put_entry_version 注入）
        for add_uri in &changes.adds {
            snapshot.entry(add_uri.to_string()).or_insert_with(|| "{}".to_string());
        }
        for upd in &changes.updates {
            snapshot.insert(upd.uri.to_string(), "{}".to_string());
        }
        for del in &changes.deletes {
            snapshot.remove(&del.to_string());
        }
        for rename in &changes.renames {
            if let Some(v) = snapshot.remove(&rename.from.to_string()) {
                snapshot.insert(rename.to.to_string(), v);
            }
        }

        let tree_hash = ContentHash(format!("tree-{}", commit_id.0));

        // 推进当前 HEAD 所在分支（如果有）
        let first_parent = parents.first().cloned();
        {
            let mut branches = self.branches.lock();
            for ((s, _name), branch) in branches.iter_mut() {
                if s == &scope_key {
                    if let Some(ref p) = first_parent {
                        if branch.head == *p {
                            branch.head = commit_id.clone();
                        }
                    }
                }
            }
        }

        let commit = Commit {
            id: commit_id.clone(),
            parents: parents.clone(),
            tree_hash,
            author: Author {
                agent_id: None,
                user_id: None,
                system: matches!(meta.trigger, CommitTrigger::AutoConsolidation),
            },
            message: format!("{:?}", meta.trigger),
            timestamp: now,
            metadata: meta,
        };

        self.commits.lock().insert(commit_id.clone(), commit);
        self.entry_snapshots
            .lock()
            .insert(commit_id.clone(), snapshot);
        self.set_head(&scope_key, commit_id.clone());

        Ok(commit_id)
    }

    // ── branch ──────────────────────────────────────────────────────

    async fn create_branch(
        &self,
        scope: &ContextUri,
        name: BranchName,
        from: CommitId,
        bt: BranchType,
    ) -> Result<Branch> {
        let scope_key = Self::scope_key(scope);
        let key = (scope_key.clone(), name.as_str().to_string());
        let mut branches = self.branches.lock();
        if branches.contains_key(&key) {
            return Err(VersionError::BranchExists(name.as_str().to_string()));
        }

        let branch = Branch {
            name: name.clone(),
            head: from.clone(),
            created_from: from,
            created_at: Utc::now(),
            branch_type: bt,
            lifecycle: BranchLifecycle::Active,
        };

        self.branches.lock().insert(key, branch.clone());
        Ok(branch)
    }

    async fn list_branches(&self, scope: &ContextUri) -> Result<Vec<Branch>> {
        let scope_key = Self::scope_key(scope);
        Ok(self
            .branches
            .lock()
            .iter()
            .filter(|((s, _), _)| s == &scope_key)
            .map(|(_, b)| b.clone())
            .collect())
    }

    async fn delete_branch(&self, scope: &ContextUri, name: &BranchName) -> Result<()> {
        let scope_key = Self::scope_key(scope);
        let key = (scope_key, name.as_str().to_string());
        let mut branches = self.branches.lock();
        branches
            .remove(&key)
            .map(|_| ())
            .ok_or_else(|| VersionError::NotFound(format!("branch {}", name)))
    }

    // ── tag ─────────────────────────────────────────────────────────

    async fn create_tag(&self, scope: &ContextUri, tag: Tag) -> Result<()> {
        let scope_key = Self::scope_key(scope);
        let key = (scope_key, tag.name.as_str().to_string());
        self.tags.lock().insert(key, tag);
        Ok(())
    }

    async fn list_tags(&self, scope: &ContextUri) -> Result<Vec<Tag>> {
        let scope_key = Self::scope_key(scope);
        Ok(self
            .tags
            .lock()
            .iter()
            .filter(|((s, _), _)| s == &scope_key)
            .map(|(_, t)| t.clone())
            .collect())
    }

    // ── log ─────────────────────────────────────────────────────────

    async fn log(&self, scope: &ContextUri, opts: &LogOpts) -> Result<Vec<Commit>> {
        let scope_key = Self::scope_key(scope);

        // 确定起点：指定 branch → 分支 HEAD；否则 → scope HEAD
        let start = if let Some(ref branch_name) = opts.branch {
            let key = (scope_key.clone(), branch_name.as_str().to_string());
            self.branches
                .lock()
                .get(&key)
                .map(|b| b.head.clone())
                .unwrap_or_else(|| CommitId::new())
        } else {
            self.head(&scope_key).unwrap_or_else(CommitId::new)
        };

        let commits = self.commits.lock();
        let max = opts.max_count.unwrap_or(20);

        // 沿 parents 链回溯
        let mut result = Vec::new();
        let mut current = start;
        for _ in 0..max {
            if let Some(c) = commits.get(&current) {
                result.push(c.clone());
                if let Some(parent) = c.parents.first() {
                    current = parent.clone();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(result)
    }

    // ── read_at / asof_read ─────────────────────────────────────────

    async fn read_at(
        &self,
        uri: &ContextUri,
        ref_: VersionRef,
        level: ContentLevel,
    ) -> Result<ContentPayload> {
        let commit_id = match ref_ {
            VersionRef::Commit(id) => id,
            VersionRef::Head => {
                // Resolve Head from URI: extract scope as the URI's first segments
                // e.g. "uwu://t1/agent/a1/state/mid" -> try to find head for this scope
                let uri_str = uri.to_string();
                // Try progressively shorter scope prefixes
                let segments: Vec<&str> = uri_str.split('/').collect();
                let mut found = None;
                for i in (2..=segments.len()).rev() {
                    let candidate_scope = segments[..i].join("/");
                    if let Some(head_id) = self.head(&candidate_scope) {
                        found = Some(head_id);
                        break;
                    }
                }
                match found {
                    Some(id) => id,
                    None => {
                        return Err(VersionError::NotFound(format!(
                            "Head ref could not resolve from uri: {uri_str}"
                        )));
                    }
                }
            }
            ref other => {
                return Err(VersionError::NotFound(format!(
                    "unsupported VersionRef for read_at: {:?}",
                    other
                )));
            }
        };

        let snapshots = self.entry_snapshots.lock();
        if let Some(snapshot) = snapshots.get(&commit_id) {
            if let Some(entry_json) = snapshot.get(&uri.to_string()) {
                let entry: ContextEntry = serde_json::from_str(entry_json)
                    .map_err(|e| VersionError::Storage(format!("deserialize entry: {e}")))?;
                return Ok(match level {
                    ContentLevel::L0 => {
                        let s = entry.l0_text().to_string();
                        ContentPayload::Text { sparse: s.clone(), dense: s.clone(), full: s }
                    }
                    ContentLevel::L1 => {
                        let dense = match &entry.payload {
                            ContentPayload::Text { dense, .. } => dense.clone(),
                            _ => String::new(),
                        };
                        ContentPayload::Text { sparse: entry.l0_text().to_string(), dense: dense.clone(), full: dense }
                    }
                    ContentLevel::L2 => entry.payload.clone(),
                });
            }
        }

        Err(VersionError::NotFound(format!(
            "uri {} not found at commit {:?}",
            uri, commit_id
        )))
    }

    async fn asof_read(
        &self,
        uri: &ContextUri,
        when: AsOfTime,
        level: ContentLevel,
    ) -> Result<ContentPayload> {
        match when {
            AsOfTime::Commit(id) => {
                self.read_at(uri, VersionRef::Commit(id), level).await
            }
            AsOfTime::Timestamp(ts) => {
                // 找到该时间点之前的最后一个 commit（锁作用域内完成，不含 await）
                let best_id: Option<CommitId> = {
                    let commits = self.commits.lock();
                    let mut best: Option<(DateTime<Utc>, CommitId)> = None;
                    for (id, c) in commits.iter() {
                        if c.timestamp <= ts {
                            if best.is_none() || c.timestamp > best.as_ref().unwrap().0 {
                                best = Some((c.timestamp, id.clone()));
                            }
                        }
                    }
                    best.map(|(_, id)| id)
                };
                match best_id {
                    Some(id) => {
                        self.read_at(uri, VersionRef::Commit(id), level).await
                    }
                    None => Err(VersionError::NotFound(format!(
                        "no commit before {}",
                        ts
                    ))),
                }
            }
        }
    }

    // ── merge ───────────────────────────────────────────────────────

    async fn merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: MergeStrategy,
    ) -> Result<MergeResult> {
        let scope_key = Self::scope_key(scope);
        let from_key = (scope_key.clone(), from.as_str().to_string());
        let into_key = (scope_key.clone(), into.as_str().to_string());

        // 提取分支数据（克隆后释放锁）
        let (from_head, into_head) = {
            let branches = self.branches.lock();
            let from_branch = branches
                .get(&from_key)
                .ok_or_else(|| VersionError::NotFound(format!("branch {}", from)))?
                .clone();
            let into_branch = branches
                .get(&into_key)
                .ok_or_else(|| VersionError::NotFound(format!("branch {}", into)))?
                .clone();
            (from_branch.head, into_branch.head)
        };

        let is_ancestor = self.is_ancestor(&into_head, &from_head);

        match strategy {
            MergeStrategy::FastForward => {
                if !is_ancestor {
                    return Err(VersionError::MergeConflict(
                        "cannot fast-forward: branches have diverged".into(),
                    ));
                }
                let mut branches = self.branches.lock();
                if let Some(b) = branches.get_mut(&into_key) {
                    b.head = from_head.clone();
                    b.lifecycle = BranchLifecycle::Active;
                }
                Ok(MergeResult {
                    commit: from_head,
                    conflicts: vec![],
                })
            }
            MergeStrategy::ThreeWay | MergeStrategy::Ours | MergeStrategy::Theirs => {
                if is_ancestor {
                    let mut branches = self.branches.lock();
                    if let Some(b) = branches.get_mut(&into_key) {
                        b.head = from_head.clone();
                    }
                    return Ok(MergeResult {
                        commit: from_head,
                        conflicts: vec![],
                    });
                }
                // 分歧合并：创建 merge commit
                let merge_id = CommitId::new();
                let now = Utc::now();

                let merge_commit = Commit {
                    id: merge_id.clone(),
                    parents: vec![into_head.clone(), from_head.clone()],
                    tree_hash: ContentHash(format!("merge-{}", merge_id.0)),
                    author: Author {
                        agent_id: None,
                        user_id: None,
                        system: true,
                    },
                    message: format!("merge {} into {}", from, into),
                    timestamp: now,
                    metadata: CommitMeta {
                        trigger: CommitTrigger::Merge {
                            branches: vec![from.clone(), into.clone()],
                        },
                        changes: ChangeSet::default(),
                        provenance: vec![],
                    },
                };

                // 合并快照
                let into_snapshot = self
                    .entry_snapshots
                    .lock()
                    .get(&into_head)
                    .cloned()
                    .unwrap_or_default();
                let from_snapshot = self
                    .entry_snapshots
                    .lock()
                    .get(&from_head)
                    .cloned()
                    .unwrap_or_default();

                let mut merged = into_snapshot;
                for (k, v) in from_snapshot {
                    merged.insert(k, v);
                }

                self.commits.lock().insert(merge_id.clone(), merge_commit);
                self.entry_snapshots.lock().insert(merge_id.clone(), merged);

                let mut branches = self.branches.lock();
                if let Some(b) = branches.get_mut(&into_key) {
                    b.head = merge_id.clone();
                }
                self.set_head(&scope_key, merge_id.clone());

                Ok(MergeResult {
                    commit: merge_id,
                    conflicts: vec![],
                })
            }
        }
    }

    // ── diff ────────────────────────────────────────────────────────

    async fn diff_commits(
        &self,
        _scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<TreeDiff> {
        let snapshots = self.entry_snapshots.lock();
        let snap_a = snapshots.get(a);
        let snap_b = snapshots.get(b);

        let mut adds = Vec::new();
        let mut updates = Vec::new();
        let mut deletes = Vec::new();

        let map_a = snap_a.cloned().unwrap_or_default();
        let map_b = snap_b.cloned().unwrap_or_default();

        for (uri_str, _) in &map_b {
            if !map_a.contains_key(uri_str) {
                if let Ok(u) = ContextUri::parse(uri_str.clone()) { adds.push(u); }
            }
        }
        for (uri_str, content_b) in &map_b {
            if let Some(content_a) = map_a.get(uri_str) {
                if content_a != content_b {
                    if let Ok(u) = ContextUri::parse(uri_str.clone()) { updates.push(u); }
                }
            }
        }
        for uri_str in map_a.keys() {
            if !map_b.contains_key(uri_str) {
                if let Ok(u) = ContextUri::parse(uri_str.clone()) { deletes.push(u); }
            }
        }

        Ok(TreeDiff {
            adds,
            updates,
            deletes,
        })
    }

    async fn switch_head(&self, scope: &ContextUri, branch: &BranchName) -> Result<()> {
        let scope_key = Self::scope_key(scope);
        let key = (scope_key.clone(), branch.as_str().to_string());
        let branches = self.branches.lock();
        if let Some(b) = branches.get(&key) {
            self.set_head(&scope_key, b.head.clone());
            Ok(())
        } else {
            Err(VersionError::NotFound(format!("branch {}", branch)))
        }
    }

    async fn cherry_pick(&self, scope: &ContextUri, commit: &CommitId, onto: &BranchName, strategy: ConflictStrategy) -> Result<CommitId> {
        let scope_key = Self::scope_key(scope);
        let source_commit = self.commits.lock().get(commit).cloned()
            .ok_or_else(|| VersionError::NotFound(format!("commit {:?}", commit)))?;
        let source_snapshot = self.entry_snapshots.lock().get(commit).cloned().unwrap_or_default();

        // 三方冲突检测：base = source 的 parent[0]，target = 当前分支 head
        let base_snapshot = source_commit
            .parents
            .first()
            .and_then(|p| self.entry_snapshots.lock().get(p).cloned())
            .unwrap_or_default();
        let target_head = {
            let branches = self.branches.lock();
            branches.get(&(scope_key.clone(), onto.as_str().to_string())).map(|b| b.head.clone())
        };
        let mut target_snapshot = target_head
            .as_ref()
            .and_then(|h| self.entry_snapshots.lock().get(h).cloned())
            .unwrap_or_default();

        // 收集 source 相对 base 的变更 URIs
        let changed_uris: Vec<String> = {
            let mut uris: std::collections::HashSet<String> = std::collections::HashSet::new();
            for k in source_snapshot.keys() { uris.insert(k.clone()); }
            for k in base_snapshot.keys() { uris.insert(k.clone()); }
            uris.into_iter().filter(|u| source_snapshot.get(u) != base_snapshot.get(u)).collect()
        };

        let mut conflicts: Vec<String> = Vec::new();
        let mut skip: std::collections::HashSet<String> = std::collections::HashSet::new();
        for uri in &changed_uris {
            let tv = target_snapshot.get(uri);
            let bv = base_snapshot.get(uri);
            let sv = source_snapshot.get(uri);
            if tv != bv && tv != sv {
                match strategy {
                    ConflictStrategy::Fail => conflicts.push(uri.clone()),
                    ConflictStrategy::Ours => { skip.insert(uri.clone()); }
                    ConflictStrategy::Theirs => { /* apply below */ }
                }
            }
        }
        if !conflicts.is_empty() {
            return Err(VersionError::MergeConflict(format!(
                "cherry-pick {:?} onto {}: {} URI(s) conflicted: {}",
                commit, onto, conflicts.len(), conflicts.join(", ")
            )));
        }
        // 应用（跳过 Ours 保留的 URI）
        for uri in &changed_uris {
            if skip.contains(uri) { continue; }
            match source_snapshot.get(uri) {
                Some(v) => { target_snapshot.insert(uri.clone(), v.clone()); }
                None => { target_snapshot.remove(uri); }
            }
        }

        let new_id = CommitId::new();
        let cherry = Commit {
            id: new_id.clone(),
            parents: target_head.map(|h| vec![h]).unwrap_or_else(|| vec![commit.clone()]),
            tree_hash: ContentHash(format!("cherry-{}", new_id.0)),
            author: source_commit.author.clone(),
            message: format!("cherry-pick: {}", source_commit.message),
            timestamp: Utc::now(),
            metadata: source_commit.metadata.clone(),
        };

        self.commits.lock().insert(new_id.clone(), cherry);
        self.entry_snapshots.lock().insert(new_id.clone(), target_snapshot);

        let key = (scope_key.clone(), onto.as_str().to_string());
        let mut branches = self.branches.lock();
        if let Some(b) = branches.get_mut(&key) {
            b.head = new_id.clone();
        }
        self.set_head(&scope_key, new_id.clone());
        Ok(new_id)
    }

    async fn rebase(&self, scope: &ContextUri, branch: &BranchName, onto: &BranchName, strategy: ConflictStrategy) -> Result<Vec<CommitId>> {
        let scope_key = Self::scope_key(scope);
        let (branch_head, _onto_head) = {
            let branches = self.branches.lock();
            let b = branches.get(&(scope_key.clone(), branch.as_str().to_string()))
                .ok_or_else(|| VersionError::NotFound(format!("branch {}", branch)))?;
            let o = branches.get(&(scope_key.clone(), onto.as_str().to_string()))
                .ok_or_else(|| VersionError::NotFound(format!("branch {}", onto)))?;
            (b.head.clone(), o.head.clone())
        };

        let new_ids = vec![self.cherry_pick(scope, &branch_head, onto, strategy).await?];
        Ok(new_ids)
    }

    async fn squash(&self, scope: &ContextUri, commits: Vec<CommitId>, message: &str) -> Result<SquashResult> {
        let count = commits.len();
        let merged_snapshot = {
            let snapshots = self.entry_snapshots.lock();
            let mut merged = HashMap::new();
            for cid in &commits {
                if let Some(s) = snapshots.get(cid) {
                    for (k, v) in s { merged.insert(k.clone(), v.clone()); }
                }
            }
            merged
        };

        let new_id = CommitId::new();
        let squash = Commit {
            id: new_id.clone(),
            parents: commits,
            tree_hash: ContentHash(format!("squash-{}", new_id.0)),
            author: Author { agent_id: None, user_id: None, system: true },
            message: message.to_string(),
            timestamp: Utc::now(),
            metadata: CommitMeta::default(),
        };

        self.commits.lock().insert(new_id.clone(), squash);
        self.entry_snapshots.lock().insert(new_id.clone(), merged_snapshot);
        self.set_head(&Self::scope_key(scope), new_id.clone());

        Ok(SquashResult { new_commit: new_id, squashed_count: count })
    }

    async fn gc(&self, scope: &ContextUri, policy: &GcPolicy) -> Result<GcReport> {
        let log = self.log(scope, &LogOpts { max_count: None, ..Default::default() }).await?;
        let cutoff = log.len().saturating_sub(policy.keep_recent);
        let mut removed = 0;
        let mut freed = 0;

        for commit in log.iter().skip(policy.keep_recent) {
            self.commits.lock().remove(&commit.id);
            if self.entry_snapshots.lock().remove(&commit.id).is_some() { freed += 1; }
            removed += 1;
        }
        let _ = cutoff;
        Ok(GcReport { removed_commits: removed, freed_snapshots: freed })
    }

    async fn evaluate_semantic_tags(&self, scope: &ContextUri) -> Result<Vec<(TagName, CommitId)>> {
        let tags = self.list_tags(scope).await?;
        let mut updates = Vec::new();
        for tag in tags {
            if let TagType::Semantic { ref condition } = tag.tag_type {
                let _ = condition;
                updates.push((tag.name, tag.target));
            }
        }
        Ok(updates)
    }

    async fn provenance(&self, uri: &ContextUri) -> Result<ProvenanceGraph> {
        let commits = self.commits.lock();
        let mut nodes = Vec::new();
        for (_cid, commit) in commits.iter() {
            for link in &commit.metadata.provenance {
                if link.source_uri == *uri {
                    nodes.push(link.clone());
                }
            }
        }
        Ok(ProvenanceGraph {
            root_uri: uri.clone(),
            nodes,
            depth: 0,
        })
    }

    async fn impact_analysis(&self, commit: &CommitId) -> Result<ImpactAnalysis> {
        let commits = self.commits.lock();
        let mut downstream = Vec::new();
        let target = commit.clone();
        for (cid, c) in commits.iter() {
            if c.parents.contains(&target) {
                if let Ok(u) = ContextUri::parse(format!("commit-{}", cid.0)) {
                    downstream.push(u);
                }
            }
        }
        let branches = self.branches.lock();
        let affected: Vec<BranchName> = branches.iter()
            .filter(|(_, b)| b.head == target)
            .map(|((_, name), _)| BranchName::new(name.clone()))
            .collect();

        Ok(ImpactAnalysis {
            commit: commit.clone(),
            downstream_uris: downstream,
            affected_branches: affected,
        })
    }

    async fn semantic_diff(
        &self,
        _scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<StructuredDiff> {
        Ok(StructuredDiff {
            entity_changes: vec![],
            relation_changes: vec![],
            fact_corrections: vec![],
            confidence_delta: 0.0,
            summary: format!("diff from {a:?} to {b:?}"),
        })
    }

    async fn evolution(&self, uri: &ContextUri) -> Result<Vec<TemporalVersion>> {
        let commits = self.commits.lock();
        let snapshots = self.entry_snapshots.lock();
        let mut versions: Vec<TemporalVersion> = commits
            .iter()
            .filter(|(cid, _)| snapshots.get(cid).map(|s| s.contains_key(&uri.to_string())).unwrap_or(false))
            .map(|(cid, c)| TemporalVersion {
                commit_id: cid.clone(),
                timestamp: c.timestamp,
                content_hash: ContentHash(format!("hash-{}", cid.0)),
                valid_from: c.timestamp,
                valid_until: None,
            })
            .collect();
        versions.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(versions)
    }

    /// 语义级分支合并 —— 基于 uwu-crdt `LwwMap` 的 CRDT 合并。
    ///
    /// 策略语义：
    /// - `EntityAutoMerge`：纯 LWW，冲突时高时钟胜、tie-break 用分支名
    /// - `ContradictionDetection { threshold }`：先检查两侧修改的 URI Jaccard 相似度，
    ///   低于 threshold 的记为 conflict，但仍 LWW 合并（策略给上层决定丢弃/重做）
    /// - `GraphMerge`：暂等价 EntityAutoMerge（关系边合并留给 GraphStore 层）
    ///
    /// 时钟：以 commit 的 timestamp（Unix seconds）作为 LwwMap clock，
    /// tie-break node_id 用分支名，保证同 clock 时确定性可复现。
    async fn knowledge_merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: KnowledgeMergeStrategy,
    ) -> Result<MergeResult> {
        use uwu_crdt::LwwMap;

        let scope_key = Self::scope_key(scope);
        let from_key = (scope_key.clone(), from.as_str().to_string());
        let into_key = (scope_key.clone(), into.as_str().to_string());

        let (from_head, into_head) = {
            let branches = self.branches.lock();
            let from_branch = branches.get(&from_key).ok_or_else(|| {
                VersionError::NotFound(format!("branch {}", from.as_str()))
            })?;
            let into_branch = branches.get(&into_key).ok_or_else(|| {
                VersionError::NotFound(format!("branch {}", into.as_str()))
            })?;
            (from_branch.head.clone(), into_branch.head.clone())
        };

        // 若已是 fast-forward（into 是 from 的祖先），直接推进 into 到 from
        if self.is_ancestor(&into_head, &from_head) {
            let mut branches = self.branches.lock();
            if let Some(b) = branches.get_mut(&into_key) {
                b.head = from_head.clone();
            }
            return Ok(MergeResult {
                commit: from_head,
                conflicts: vec![],
            });
        }

        // 加载两侧快照 + 时钟（commit timestamp 秒数）
        let (from_snap, into_snap, from_clock, into_clock) = {
            let snapshots = self.entry_snapshots.lock();
            let commits = self.commits.lock();
            let from_snap = snapshots.get(&from_head).cloned().unwrap_or_default();
            let into_snap = snapshots.get(&into_head).cloned().unwrap_or_default();
            let from_clock = commits
                .get(&from_head)
                .map(|c| c.timestamp.timestamp() as u64)
                .unwrap_or(0);
            let into_clock = commits
                .get(&into_head)
                .map(|c| c.timestamp.timestamp() as u64)
                .unwrap_or(0);
            (from_snap, into_snap, from_clock, into_clock)
        };

        // 转成 LwwMap
        let mut from_map: LwwMap<String, String> = LwwMap::new();
        for (uri, json) in &from_snap {
            from_map.set(uri.clone(), json.clone(), from_clock, from.as_str());
        }
        let mut into_map: LwwMap<String, String> = LwwMap::new();
        for (uri, json) in &into_snap {
            into_map.set(uri.clone(), json.clone(), into_clock, into.as_str());
        }

        // CRDT 合并 —— from ∨ into，交换律 & 幂等
        let merged = uwu_crdt::merge(&from_map, &into_map);

        // 冲突检测（仅 ContradictionDetection 策略）
        let conflicts: Vec<ContextUri> = match strategy {
            KnowledgeMergeStrategy::ContradictionDetection { threshold } => {
                let mut cs = Vec::new();
                for (uri_str, from_val) in &from_snap {
                    if let Some(into_val) = into_snap.get(uri_str) {
                        if from_val == into_val {
                            continue;
                        }
                        let sim = jaccard_similarity(from_val, into_val);
                        if sim < threshold as f64 {
                            if let Ok(u) = ContextUri::parse(uri_str.as_str()) {
                                cs.push(u);
                            }
                        }
                    }
                }
                cs
            }
            KnowledgeMergeStrategy::EntityAutoMerge
            | KnowledgeMergeStrategy::GraphMerge { .. } => Vec::new(),
        };

        // 从 LwwMap 提取快照（跳过墓碑）
        let merged_snapshot: HashMap<String, String> = merged
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // 创建 merge commit（两个 parent）
        let now = Utc::now();
        let commit_id = CommitId::new();
        let merge_commit = Commit {
            id: commit_id.clone(),
            parents: vec![into_head.clone(), from_head.clone()],
            tree_hash: ContentHash(format!("tree-{}", commit_id.0)),
            author: Author { agent_id: None, user_id: None, system: true },
            message: format!(
                "knowledge_merge {} <- {} conflicts={}",
                into.as_str(),
                from.as_str(),
                conflicts.len()
            ),
            timestamp: now,
            metadata: CommitMeta { trigger: CommitTrigger::default(), ..Default::default() },
        };

        {
            let mut commits = self.commits.lock();
            commits.insert(commit_id.clone(), merge_commit);
        }
        {
            let mut snapshots = self.entry_snapshots.lock();
            snapshots.insert(commit_id.clone(), merged_snapshot);
        }
        {
            let mut branches = self.branches.lock();
            if let Some(b) = branches.get_mut(&into_key) {
                b.head = commit_id.clone();
            }
        }
        // 推进 scope HEAD 到新 merge commit
        {
            let mut heads = self.heads.lock();
            heads.insert(scope_key, commit_id.clone());
        }

        Ok(MergeResult { commit: commit_id, conflicts })
    }
}

/// Jaccard 相似度（词袋，忽略大小写和标点）—— 用于冲突检测。
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let tokens = |s: &str| -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect()
    };
    let sa = tokens(a);
    let sb = tokens(b);
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 { 0.0 } else { inter / union }
}

// ===========================================================================
// 内部辅助
// ===========================================================================

impl MemoryVersionStore {
    /// 检查 candidate 是否为 ancestor 的（祖先的）后代。
    /// 沿 candidate 的 parent 链回溯，看是否能到达 ancestor。
    fn is_ancestor(&self, ancestor: &CommitId, candidate: &CommitId) -> bool {
        if ancestor == candidate {
            return true;
        }
        let commits = self.commits.lock();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![candidate.clone()];

        while let Some(id) = stack.pop() {
            if &id == ancestor {
                return true;
            }
            if !visited.insert(id.clone()) {
                continue;
            }
            if let Some(c) = commits.get(&id) {
                for p in &c.parents {
                    stack.push(p.clone());
                }
            }
        }
        false
    }
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContextEntry, TenantId};

    fn scope() -> ContextUri {
        ContextUri::parse("uwu://t1/agent/a1/state/mid").unwrap()
    }

    fn entry_uri(s: &str) -> ContextUri {
        ContextUri::parse(s).unwrap()
    }

    #[test]
    fn new_store_has_no_head() {
        let store = MemoryVersionStore::new();
        assert!(store.head("uwu://t1/agent/a1/state/mid").is_none());
    }

    #[tokio::test]
    async fn commit_creates_dag_node() {
        let store = MemoryVersionStore::new();
        let s = scope();

        let id = store
            .commit(
                &s,
                ChangeSet {
                    adds: vec![entry_uri("uwu://t1/agent/a1/state/mid/s1")],
                    ..Default::default()
                },
                CommitMeta {
                    trigger: CommitTrigger::AutoConsolidation,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let log = store
            .log(&s, &LogOpts { max_count: Some(10), ..Default::default() })
            .await
            .unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].id, id);
    }

    #[tokio::test]
    async fn branch_and_merge_fast_forward() {
        let store = MemoryVersionStore::new();
        let s = scope();

        // commit on main
        let c1 = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();

        // create a feature branch from c1
        let main = BranchName::new("main");
        let feat = BranchName::new("feat");

        store
            .create_branch(&s, main.clone(), c1.clone(), BranchType::Main)
            .await
            .unwrap();
        store
            .create_branch(&s, feat.clone(), c1.clone(), BranchType::Experiment)
            .await
            .unwrap();

        // commit on feat
        let c2 = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();

        // feat head should be c2
        let branches = store.list_branches(&s).await.unwrap();
        let feat_branch = branches.iter().find(|b| b.name.as_str() == "feat").unwrap();
        assert_eq!(feat_branch.head, c2);

        // fast-forward merge feat into main
        let result = store
            .merge(&s, &feat, &main, MergeStrategy::FastForward)
            .await
            .unwrap();
        assert_eq!(result.commit, c2);
        assert!(result.conflicts.is_empty());
    }

    #[tokio::test]
    async fn fast_forward_rejects_divergent() {
        let store = MemoryVersionStore::new();
        let s = scope();

        // 构造真实分叉：两个 commit 各自以 root 为 parent（模拟两种策略的并行推演）
        let root = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();

        // 直接注入两个以 root 为 parent 的独立 commit
        let c_a = CommitId::new();
        let c_b = CommitId::new();

        store.commits.lock().insert(c_a.clone(), Commit {
            id: c_a.clone(),
            parents: vec![root.clone()],
            tree_hash: ContentHash("a".into()),
            author: Author { agent_id: None, user_id: None, system: false },
            message: "strategy A".into(),
            timestamp: Utc::now(),
            metadata: CommitMeta::default(),
        });
        store.commits.lock().insert(c_b.clone(), Commit {
            id: c_b.clone(),
            parents: vec![root.clone()],
            tree_hash: ContentHash("b".into()),
            author: Author { agent_id: None, user_id: None, system: false },
            message: "strategy B".into(),
            timestamp: Utc::now(),
            metadata: CommitMeta::default(),
        });

        // 创建分别指向两个 commit 的分支
        let a = BranchName::new("a");
        let b = BranchName::new("b");
        store.create_branch(&s, a.clone(), c_a, BranchType::Experiment).await.unwrap();
        store.create_branch(&s, b.clone(), c_b, BranchType::Experiment).await.unwrap();

        // 不能快进：A 和 B 没有祖先关系
        let result = store.merge(&s, &a, &b, MergeStrategy::FastForward).await;
        assert!(result.is_err(), "divergent branches should not fast-forward");
    }

    #[tokio::test]
    async fn read_at_retrieves_entry_from_snapshot() {
        let store = MemoryVersionStore::new();
        let s = scope();
        let uri = entry_uri("uwu://t1/agent/a1/memories/cases/c1");

        let c1 = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();

        // 注入条目到 c1 的快照
        let entry = ContextEntry::new_text(
            uri.clone(),
            TenantId(uuid::Uuid::nil()),
            "fixed a memory leak",
        );
        store.put_entry_version(&c1, &uri, &entry);

        // read_at
        let payload = store
            .read_at(&uri, agent_context_db_version::VersionRef::Commit(c1), ContentLevel::L0)
            .await
            .unwrap();
        assert!(matches!(payload, ContentPayload::Text { sparse, .. } if sparse.contains("memory leak")));
    }

    #[tokio::test]
    async fn asof_read_by_timestamp() {
        let store = MemoryVersionStore::new();
        let s = scope();
        let uri = entry_uri("uwu://t1/agent/a1/memories/events/e1");

        let c1 = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();

        let entry = ContextEntry::new_text(
            uri.clone(),
            TenantId(uuid::Uuid::nil()),
            "v1: initial",
        );
        store.put_entry_version(&c1, &uri, &entry);

        // 稍后读
        let payload = store
            .asof_read(
                &uri,
                AsOfTime::Timestamp(Utc::now()),
                ContentLevel::L0,
            )
            .await
            .unwrap();
        assert!(matches!(payload, ContentPayload::Text { sparse, .. } if sparse.contains("v1")));
    }

    #[tokio::test]
    async fn diff_between_commits() {
        let store = MemoryVersionStore::new();
        let s = scope();
        let uri_a = entry_uri("uwu://t1/agent/a1/memories/cases/c1");
        let uri_b = entry_uri("uwu://t1/agent/a1/memories/cases/c2");

        let c1 = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();
        store.put_entry_version(
            &c1,
            &uri_a,
            &ContextEntry::new_text(uri_a.clone(), TenantId(uuid::Uuid::nil()), "case A"),
        );

        let c2 = store
            .commit(&s, ChangeSet::default(), CommitMeta::default())
            .await
            .unwrap();
        store.put_entry_version(
            &c2,
            &uri_a,
            &ContextEntry::new_text(uri_a.clone(), TenantId(uuid::Uuid::nil()), "case A modified"),
        );
        store.put_entry_version(
            &c2,
            &uri_b,
            &ContextEntry::new_text(uri_b.clone(), TenantId(uuid::Uuid::nil()), "case B"),
        );

        let diff = store.diff_commits(&s, &c1, &c2).await.unwrap();
        assert!(diff.updates.contains(&uri_a), "uri_a should be updated");
        assert!(diff.adds.contains(&uri_b), "uri_b should be added");
    }
}
