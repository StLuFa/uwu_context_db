//! Tenant-safe wrappers for the version subsystem.

use crate::{FacadeError, InteractiveVersions, Result, Versions};
use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri};
use agent_context_db_version as v;
use std::sync::Arc;

impl Versions {
    fn validate(&self, uri: &ContextUri) -> Result<()> {
        self.db.validate_uri(&self.ctx, uri)
    }
    async fn finish<T>(
        &self,
        op: &str,
        detail: String,
        future: impl std::future::Future<Output = v::Result<T>>,
    ) -> Result<T> {
        let result = self.db.timed(&self.ctx, future).await;
        self.db.audit(&self.ctx, op, result.is_ok(), detail)?;
        result
    }
    pub async fn commit(
        &self,
        s: &ContextUri,
        c: v::ChangeSet,
        m: v::CommitMeta,
    ) -> Result<v::CommitId> {
        self.validate(s)?;
        self.finish(
            "version.commit",
            s.to_string(),
            self.db.0.parts.versions.commit(s, c, m),
        )
        .await
    }
    pub async fn create_branch(
        &self,
        s: &ContextUri,
        n: v::BranchName,
        f: v::CommitId,
        t: v::BranchType,
    ) -> Result<v::Branch> {
        self.validate(s)?;
        self.finish(
            "version.create_branch",
            s.to_string(),
            self.db.0.parts.versions.create_branch(s, n, f, t),
        )
        .await
    }
    pub async fn list_branches(&self, s: &ContextUri) -> Result<Vec<v::Branch>> {
        self.validate(s)?;
        self.finish(
            "version.list_branches",
            s.to_string(),
            self.db.0.parts.versions.list_branches(s),
        )
        .await
    }
    pub async fn delete_branch(&self, s: &ContextUri, n: &v::BranchName) -> Result<()> {
        self.validate(s)?;
        self.finish(
            "version.delete_branch",
            s.to_string(),
            self.db.0.parts.versions.delete_branch(s, n),
        )
        .await
    }
    pub async fn create_tag(&self, s: &ContextUri, t: v::Tag) -> Result<()> {
        self.validate(s)?;
        self.finish(
            "version.create_tag",
            s.to_string(),
            self.db.0.parts.versions.create_tag(s, t),
        )
        .await
    }
    pub async fn list_tags(&self, s: &ContextUri) -> Result<Vec<v::Tag>> {
        self.validate(s)?;
        self.finish(
            "version.list_tags",
            s.to_string(),
            self.db.0.parts.versions.list_tags(s),
        )
        .await
    }
    pub async fn log(&self, s: &ContextUri, o: &v::LogOpts) -> Result<Vec<v::Commit>> {
        self.validate(s)?;
        self.finish(
            "version.log",
            s.to_string(),
            self.db.0.parts.versions.log(s, o),
        )
        .await
    }
    pub async fn read_at(
        &self,
        u: &ContextUri,
        r: v::VersionRef,
        l: ContentLevel,
    ) -> Result<ContentPayload> {
        self.validate(u)?;
        self.finish(
            "version.read_at",
            u.to_string(),
            self.db.0.parts.versions.read_at(u, r, l),
        )
        .await
    }
    pub async fn asof_read(
        &self,
        u: &ContextUri,
        w: v::AsOfTime,
        l: ContentLevel,
    ) -> Result<ContentPayload> {
        self.validate(u)?;
        self.finish(
            "version.asof_read",
            u.to_string(),
            self.db.0.parts.versions.asof_read(u, w, l),
        )
        .await
    }
    pub async fn merge(
        &self,
        s: &ContextUri,
        f: &v::BranchName,
        i: &v::BranchName,
        x: v::MergeStrategy,
    ) -> Result<v::MergeResult> {
        self.validate(s)?;
        self.finish(
            "version.merge",
            s.to_string(),
            self.db.0.parts.versions.merge(s, f, i, x),
        )
        .await
    }
    pub async fn diff_commits(
        &self,
        s: &ContextUri,
        a: &v::CommitId,
        b: &v::CommitId,
    ) -> Result<v::TreeDiff> {
        self.validate(s)?;
        self.finish(
            "version.diff_commits",
            s.to_string(),
            self.db.0.parts.versions.diff_commits(s, a, b),
        )
        .await
    }
    pub async fn switch_head(&self, s: &ContextUri, b: &v::BranchName) -> Result<()> {
        self.validate(s)?;
        self.finish(
            "version.switch_head",
            s.to_string(),
            self.db.0.parts.versions.switch_head(s, b),
        )
        .await
    }
    pub async fn cherry_pick(
        &self,
        s: &ContextUri,
        c: &v::CommitId,
        o: &v::BranchName,
        x: v::ConflictStrategy,
    ) -> Result<v::CommitId> {
        self.validate(s)?;
        self.finish(
            "version.cherry_pick",
            s.to_string(),
            self.db.0.parts.versions.cherry_pick(s, c, o, x),
        )
        .await
    }
    pub async fn rebase(
        &self,
        s: &ContextUri,
        b: &v::BranchName,
        o: &v::BranchName,
        x: v::ConflictStrategy,
    ) -> Result<Vec<v::CommitId>> {
        self.validate(s)?;
        self.finish(
            "version.rebase",
            s.to_string(),
            self.db.0.parts.versions.rebase(s, b, o, x),
        )
        .await
    }
    pub async fn squash(
        &self,
        s: &ContextUri,
        c: Vec<v::CommitId>,
        m: &str,
    ) -> Result<v::SquashResult> {
        self.validate(s)?;
        self.finish(
            "version.squash",
            s.to_string(),
            self.db.0.parts.versions.squash(s, c, m),
        )
        .await
    }
    pub async fn gc(&self, s: &ContextUri, p: &v::GcPolicy) -> Result<v::GcReport> {
        self.validate(s)?;
        self.finish(
            "version.gc",
            s.to_string(),
            self.db.0.parts.versions.gc(s, p),
        )
        .await
    }
    pub async fn evaluate_semantic_tags(
        &self,
        s: &ContextUri,
    ) -> Result<Vec<(v::TagName, v::CommitId)>> {
        self.validate(s)?;
        self.finish(
            "version.evaluate_semantic_tags",
            s.to_string(),
            self.db.0.parts.versions.evaluate_semantic_tags(s),
        )
        .await
    }
    pub async fn provenance(&self, u: &ContextUri) -> Result<v::ProvenanceGraph> {
        self.validate(u)?;
        self.finish(
            "version.provenance",
            u.to_string(),
            self.db.0.parts.versions.provenance(u),
        )
        .await
    }
    /// Commit IDs carry no tenant. The returned URI is therefore checked before release.
    pub async fn impact_analysis(&self, c: &v::CommitId) -> Result<v::ImpactAnalysis> {
        let r = self
            .finish(
                "version.impact_analysis",
                "commit".into(),
                self.db.0.parts.versions.impact_analysis(c),
            )
            .await?;
        for u in &r.downstream_uris {
            self.validate(u)?;
        }
        Ok(r)
    }
    pub async fn semantic_diff(
        &self,
        s: &ContextUri,
        a: &v::CommitId,
        b: &v::CommitId,
    ) -> Result<v::StructuredDiff> {
        self.validate(s)?;
        self.finish(
            "version.semantic_diff",
            s.to_string(),
            self.db.0.parts.versions.semantic_diff(s, a, b),
        )
        .await
    }
    pub async fn evolution(&self, u: &ContextUri) -> Result<Vec<v::TemporalVersion>> {
        self.validate(u)?;
        self.finish(
            "version.evolution",
            u.to_string(),
            self.db.0.parts.versions.evolution(u),
        )
        .await
    }
    pub async fn knowledge_merge(
        &self,
        s: &ContextUri,
        f: &v::BranchName,
        i: &v::BranchName,
        x: v::KnowledgeMergeStrategy,
    ) -> Result<v::MergeResult> {
        self.validate(s)?;
        self.finish(
            "version.knowledge_merge",
            s.to_string(),
            self.db.0.parts.versions.knowledge_merge(s, f, i, x),
        )
        .await
    }
}

impl InteractiveVersions {
    fn validate(&self, u: &ContextUri) -> Result<()> {
        self.db.validate_uri(&self.ctx, u)
    }
    fn bind(&self, session: &v::ConflictSession) -> Result<()> {
        self.validate(&session.scope)?;
        for conflict in &session.conflicts {
            self.validate(&conflict.uri)?;
        }
        let mut bindings =
            self.db.0.conflict_sessions.lock().map_err(|_| {
                FacadeError::InvalidConfig("conflict session bindings poisoned".into())
            })?;
        if !bindings.tenants.contains_key(&session.id) {
            bindings.insertion_order.push_back(session.id.clone());
        }
        let now = std::time::Instant::now();
        let ttl = self.db.0.config.conflict_session_ttl;
        bindings
            .tenants
            .retain(|_, (_, seen)| now.duration_since(*seen) <= ttl);
        let live_ids: std::collections::HashSet<_> = bindings.tenants.keys().cloned().collect();
        bindings.insertion_order.retain(|id| live_ids.contains(id));
        bindings
            .tenants
            .insert(session.id.clone(), (Arc::from(self.ctx.tenant_name()), now));
        while bindings.tenants.len() > crate::facade::MAX_CONFLICT_SESSION_BINDINGS {
            if let Some(id) = bindings.insertion_order.pop_front() {
                bindings.tenants.remove(&id);
            }
        }
        Ok(())
    }
    fn owns(&self, id: &v::ConflictSessionId) -> Result<()> {
        let b =
            self.db.0.conflict_sessions.lock().map_err(|_| {
                FacadeError::InvalidConfig("conflict session bindings poisoned".into())
            })?;
        match b.tenants.get(id) {
            Some((tenant, seen))
                if &**tenant == self.ctx.tenant_name()
                    && seen.elapsed() <= self.db.0.config.conflict_session_ttl =>
            {
                Ok(())
            }
            _ => Err(FacadeError::TenantViolation(
                "unbound conflict session".into(),
            )),
        }
    }
    fn forget(&self, id: &v::ConflictSessionId) {
        if let Ok(mut b) = self.db.0.conflict_sessions.lock() {
            b.tenants.remove(id);
        }
    }
    async fn finish<T>(
        &self,
        op: &str,
        f: impl std::future::Future<Output = v::Result<T>>,
    ) -> Result<T> {
        let r = self.db.timed(&self.ctx, f).await;
        self.db
            .audit(&self.ctx, op, r.is_ok(), "conflict session")?;
        r
    }
    pub async fn begin_cherry_pick(
        &self,
        s: &ContextUri,
        c: &v::CommitId,
        o: &v::BranchName,
        p: v::ConflictSessionPersistence,
    ) -> Result<v::ConflictSession> {
        self.validate(s)?;
        let x = self
            .finish(
                "version.conflict.begin_cherry_pick",
                self.db
                    .0
                    .parts
                    .interactive_versions
                    .begin_cherry_pick(s, c, o, p),
            )
            .await?;
        self.bind(&x)?;
        Ok(x)
    }
    pub async fn begin_rebase(
        &self,
        s: &ContextUri,
        b: &v::BranchName,
        o: &v::BranchName,
        p: v::ConflictSessionPersistence,
    ) -> Result<v::ConflictSession> {
        self.validate(s)?;
        let x = self
            .finish(
                "version.conflict.begin_rebase",
                self.db
                    .0
                    .parts
                    .interactive_versions
                    .begin_rebase(s, b, o, p),
            )
            .await?;
        self.bind(&x)?;
        Ok(x)
    }
    pub async fn continue_conflict_session(
        &self,
        s: v::ConflictSession,
        r: v::ConflictResolutionSet,
    ) -> Result<Vec<v::CommitId>> {
        self.validate(&s.scope)?;
        self.owns(&s.id)?;
        let id = s.id.clone();
        let x = self
            .finish(
                "version.conflict.continue",
                self.db
                    .0
                    .parts
                    .interactive_versions
                    .continue_conflict_session(s, r),
            )
            .await;
        if x.is_ok() {
            self.forget(&id)
        }
        x
    }
    pub async fn load_conflict_session(
        &self,
        id: &v::ConflictSessionId,
    ) -> Result<v::ConflictSession> {
        let x = self
            .finish(
                "version.conflict.load",
                self.db
                    .0
                    .parts
                    .interactive_versions
                    .load_conflict_session(id),
            )
            .await?;
        self.bind(&x)?;
        Ok(x)
    }
    pub async fn continue_conflict_session_by_id(
        &self,
        id: &v::ConflictSessionId,
        r: v::ConflictResolutionSet,
    ) -> Result<Vec<v::CommitId>> {
        self.owns(id)?;
        let x = self
            .finish(
                "version.conflict.continue_by_id",
                self.db
                    .0
                    .parts
                    .interactive_versions
                    .continue_conflict_session_by_id(id, r),
            )
            .await;
        if x.is_ok() {
            self.forget(id)
        }
        x
    }
    pub async fn abort_conflict_session(&self, id: &v::ConflictSessionId) -> Result<()> {
        self.owns(id)?;
        let x = self
            .finish(
                "version.conflict.abort",
                self.db
                    .0
                    .parts
                    .interactive_versions
                    .abort_conflict_session(id),
            )
            .await;
        if x.is_ok() {
            self.forget(id)
        }
        x
    }
    /// Backward-compatible alias for [`Self::abort_conflict_session`].
    pub async fn abort(&self, id: &v::ConflictSessionId) -> Result<()> {
        self.abort_conflict_session(id).await
    }
}
