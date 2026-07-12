use crate::{
    AuditEvent, ContextDbConfig, ContextDbParts, FacadeError, RequestContext, Result,
    ports::reaction_with_context,
};
use agent_context_db_core::*;
use agent_context_db_retrieve::{
    ContextRetriever, PlanRetriever, Query, RetrievalResult, RetrieveContext,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};
use tokio::sync::{Mutex as AsyncMutex, Notify};

pub(crate) const MAX_CONFLICT_SESSION_BINDINGS: usize = 4_096;

#[derive(Default)]
pub(crate) struct ConflictSessionBindings {
    pub(crate) tenants: HashMap<agent_context_db_version::ConflictSessionId, (Arc<str>, Instant)>,
    pub(crate) insertion_order: VecDeque<agent_context_db_version::ConflictSessionId>,
}

struct InFlightGuard(Arc<Inner>);
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if self.0.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.drained.notify_waiters();
        }
    }
}

pub(crate) struct Inner {
    pub(crate) config: ContextDbConfig,
    pub(crate) parts: ContextDbParts,
    retriever: Arc<ContextRetriever>,
    accepting: AtomicBool,
    in_flight: AtomicUsize,
    drained: Notify,
    shutdown_result: AsyncMutex<Option<std::result::Result<(), String>>>,
    shutdown_lock: AsyncMutex<()>,
    pub(crate) conflict_sessions: Mutex<ConflictSessionBindings>,
}
/// Cheap-clone immutable composition root. No raw storage port is exposed publicly.
#[derive(Clone)]
pub struct ContextDb(pub(crate) Arc<Inner>);
impl ContextDb {
    pub(crate) fn assemble(
        config: ContextDbConfig,
        parts: ContextDbParts,
        retriever: Arc<ContextRetriever>,
    ) -> Result<Self> {
        Ok(Self(Arc::new(Inner {
            config,
            parts,
            retriever,
            accepting: AtomicBool::new(true),
            in_flight: AtomicUsize::new(0),
            drained: Notify::new(),
            shutdown_result: AsyncMutex::new(None),
            shutdown_lock: AsyncMutex::new(()),
            conflict_sessions: Mutex::new(ConflictSessionBindings::default()),
        })))
    }
    fn admit(&self) -> Result<InFlightGuard> {
        if !self.0.accepting.load(Ordering::Acquire) {
            return Err(FacadeError::NotConfigured("facade is shutting down"));
        }
        self.0.in_flight.fetch_add(1, Ordering::AcqRel);
        if !self.0.accepting.load(Ordering::Acquire) {
            if self.0.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
                self.0.drained.notify_waiters();
            }
            return Err(FacadeError::NotConfigured("facade is shutting down"));
        }
        Ok(InFlightGuard(self.0.clone()))
    }
    pub(crate) fn validate_uri(&self, ctx: &RequestContext, uri: &ContextUri) -> Result<()> {
        ctx.remaining()?;
        if uri.tenant() != ctx.tenant_name() {
            Err(FacadeError::TenantViolation(uri.to_string()))
        } else {
            Ok(())
        }
    }
    pub(crate) fn audit(
        &self,
        ctx: &RequestContext,
        op: &str,
        allowed: bool,
        detail: impl Into<String>,
    ) -> Result<()> {
        let mut detail = detail.into();
        while detail.len() > self.0.config.max_audit_detail_bytes {
            detail.pop();
        }
        let e = AuditEvent {
            tenant: ctx.tenant_name().into(),
            actor: ctx.actor().into(),
            request_id: ctx.request_id().into(),
            operation: op.into(),
            allowed,
            detail,
        };
        self.0.parts.audit.record(e).map_err(FacadeError::Audit)?;
        if let Some(s) = &self.0.parts.reactions {
            s.emit(reaction_with_context(Some(ctx), op, op, allowed));
        }
        Ok(())
    }
    pub(crate) async fn timed<T, E>(
        &self,
        ctx: &RequestContext,
        future: impl std::future::Future<Output = std::result::Result<T, E>>,
    ) -> Result<T>
    where
        E: Into<FacadeError>,
    {
        let _admission = self.admit()?;
        let left = ctx.remaining()?.min(self.0.config.default_timeout);
        tokio::select! {
            biased;
            () = ctx.cancelled() => Err(FacadeError::Cancelled),
            result = tokio::time::timeout(left, future) => result
                .map_err(|_| FacadeError::Timeout(left))?
                .map_err(Into::into),
        }
    }
    pub async fn read(
        &self,
        ctx: &RequestContext,
        uri: &ContextUri,
        level: ContentLevel,
    ) -> Result<ContentPayload> {
        self.validate_uri(ctx, uri)?;
        if let Some(lifecycle) = &self.0.parts.lifecycle {
            self.timed(ctx, lifecycle.restore_before_read(uri)).await?;
        }
        let r = self.timed(ctx, self.0.parts.fs.read(uri, level)).await;
        self.audit(ctx, "content.read", r.is_ok(), format!("{uri}"))?;
        r
    }
    pub async fn write(&self, ctx: &RequestContext, entry: ContextEntry) -> Result<MvccVersion> {
        self.validate_uri(ctx, &entry.uri)?;
        if entry.tenant != ctx.tenant() {
            return Err(FacadeError::TenantViolation(entry.uri.to_string()));
        }
        let lifecycle_entry = entry.clone();
        let r = self.timed(ctx, self.0.parts.content.write(entry)).await;
        if let (Ok(_), Some(lifecycle)) = (&r, &self.0.parts.lifecycle)
            && let Err(error) = self
                .timed(ctx, lifecycle.route_after_write(&lifecycle_entry))
                .await
        {
            let _ = self.audit(ctx, "content.write", false, "committed; lifecycle failed");
            return Err(FacadeError::CommittedWithPostCommitFailure {
                failure: error.to_string(),
            });
        }
        if let Err(error) = self.audit(ctx, "content.write", r.is_ok(), "write") {
            if r.is_ok() {
                return Err(FacadeError::CommittedWithPostCommitFailure {
                    failure: error.to_string(),
                });
            }
            return Err(error);
        }
        r
    }
    pub async fn delete(&self, ctx: &RequestContext, uri: &ContextUri) -> Result<()> {
        self.validate_uri(ctx, uri)?;
        self.timed(ctx, self.0.parts.content.delete(uri)).await
    }
    pub async fn rename(
        &self,
        ctx: &RequestContext,
        from: &ContextUri,
        to: &ContextUri,
    ) -> Result<()> {
        self.validate_uri(ctx, from)?;
        self.validate_uri(ctx, to)?;
        self.timed(ctx, self.0.parts.content.rename(from, to)).await
    }
    pub async fn batch_write(
        &self,
        ctx: &RequestContext,
        entries: &[ContextEntry],
    ) -> Result<Vec<MvccVersion>> {
        if entries.len() > self.0.config.max_batch_size {
            return Err(FacadeError::InvalidConfig(
                "batch exceeds configured limit".into(),
            ));
        }
        for e in entries {
            self.validate_uri(ctx, &e.uri)?;
            if e.tenant != ctx.tenant() {
                return Err(FacadeError::TenantViolation(e.uri.to_string()));
            }
        }
        self.timed(ctx, self.0.parts.content.batch_write(entries))
            .await
    }
    pub async fn list(
        &self,
        ctx: &RequestContext,
        dir: &ContextUri,
        page: PageRequest,
    ) -> Result<Page<DirEntry>> {
        self.validate_uri(ctx, dir)?;
        self.timed(ctx, self.0.parts.fs.ls(dir, page)).await
    }
    pub async fn find(
        &self,
        ctx: &RequestContext,
        pattern: &FindPattern,
        page: PageRequest,
    ) -> Result<Page<ContextUri>> {
        let scope = pattern
            .scope
            .as_ref()
            .ok_or_else(|| FacadeError::TenantViolation("find requires tenant scope".into()))?;
        self.validate_uri(ctx, scope)?;
        self.timed(ctx, self.0.parts.fs.find(pattern, page)).await
    }
    pub async fn grep(
        &self,
        ctx: &RequestContext,
        pattern: &str,
        scope: &ContextUri,
    ) -> Result<Vec<GrepHit>> {
        self.validate_uri(ctx, scope)?;
        self.timed(ctx, self.0.parts.fs.grep(pattern, scope)).await
    }
    pub async fn tree(
        &self,
        ctx: &RequestContext,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        self.validate_uri(ctx, root)?;
        if depth > self.0.config.max_tree_depth {
            return Err(FacadeError::InvalidConfig(
                "tree depth exceeds limit".into(),
            ));
        }
        self.timed(ctx, self.0.parts.fs.tree(root, depth, page))
            .await
    }
    pub async fn scan(
        &self,
        ctx: &RequestContext,
        prefix: &ContextUri,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        self.validate_uri(ctx, prefix)?;
        self.timed(
            ctx,
            self.0
                .parts
                .content_store
                .scan_by_prefix(&prefix.to_string(), page),
        )
        .await
    }
    pub async fn scan_by_type(
        &self,
        ctx: &RequestContext,
        prefix: &ContextUri,
        content_type: ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        self.validate_uri(ctx, prefix)?;
        let result = self
            .timed(
                ctx,
                self.0
                    .parts
                    .content_store
                    .scan_by_type(&prefix.to_string(), content_type, page),
            )
            .await;
        self.audit(
            ctx,
            "content.scan_by_type",
            result.is_ok(),
            prefix.to_string(),
        )?;
        result
    }
    pub async fn llm_complete(
        &self,
        ctx: &RequestContext,
        prompt: &str,
        opts: &LlmOpts,
    ) -> Result<String> {
        ctx.remaining()?;
        let execution = ctx.execution(ExecutionKind::Tool {
            name: "llm.complete".into(),
        });
        let decision = self
            .0
            .parts
            .gate
            .preflight(&execution, ExecutionRequest::new("llm.complete", prompt))
            .map_err(|e| FacadeError::PolicyDenied(e.to_string()))?;
        if !decision.allowed {
            self.audit(ctx, "llm.complete", false, "gate denied")?;
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let llm = self
            .0
            .parts
            .llm
            .as_ref()
            .ok_or(FacadeError::NotConfigured("llm"))?;
        let left = ctx.remaining()?.min(self.0.config.default_timeout);
        let result = tokio::select! { biased; () = ctx.cancelled() => Err(FacadeError::Cancelled), value = tokio::time::timeout(left, llm.complete(prompt, opts)) => value.map_err(|_| FacadeError::Timeout(left))?.map_err(Into::into) };
        self.audit(ctx, "llm.complete", result.is_ok(), "completion")?;
        result
    }
    pub async fn llm_complete_json(
        &self,
        ctx: &RequestContext,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String> {
        ctx.remaining()?;
        let execution = ctx.execution(ExecutionKind::Tool {
            name: "llm.complete_json".into(),
        });
        let decision = self
            .0
            .parts
            .gate
            .preflight(
                &execution,
                ExecutionRequest::new("llm.complete_json", prompt),
            )
            .map_err(|e| FacadeError::PolicyDenied(e.to_string()))?;
        if !decision.allowed {
            self.audit(ctx, "llm.complete_json", false, "gate denied")?;
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let llm = self
            .0
            .parts
            .llm
            .as_ref()
            .ok_or(FacadeError::NotConfigured("llm"))?;
        let result = self
            .timed(ctx, llm.complete_json(prompt, schema, opts))
            .await;
        self.audit(
            ctx,
            "llm.complete_json",
            result.is_ok(),
            "structured completion",
        )?;
        result
    }
    pub async fn llm_embed(&self, ctx: &RequestContext, text: &str) -> Result<EmbeddingVector> {
        ctx.remaining()?;
        let execution = ctx.execution(ExecutionKind::Tool {
            name: "llm.embed".into(),
        });
        let decision = self
            .0
            .parts
            .gate
            .preflight(&execution, ExecutionRequest::new("llm.embed", text))
            .map_err(|e| FacadeError::PolicyDenied(e.to_string()))?;
        if !decision.allowed {
            self.audit(ctx, "llm.embed", false, "gate denied")?;
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let llm = self
            .0
            .parts
            .llm
            .as_ref()
            .ok_or(FacadeError::NotConfigured("llm"))?;
        let result = self.timed(ctx, llm.embed(text)).await;
        self.audit(ctx, "llm.embed", result.is_ok(), "embedding")?;
        result
    }
    pub async fn federate(
        &self,
        ctx: &RequestContext,
        request: ExecutionRequest,
    ) -> Result<ExecutionResponse> {
        ctx.remaining()?;
        let gateway = self
            .0
            .parts
            .federation
            .as_ref()
            .ok_or(FacadeError::NotConfigured("federation"))?;
        let execution = ctx.execution(ExecutionKind::Tool {
            name: "federation".into(),
        });
        let decision = self
            .0
            .parts
            .gate
            .preflight(&execution, request.clone())
            .map_err(|error| FacadeError::PolicyDenied(error.to_string()))?;
        if !decision.allowed {
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let result = self
            .timed(ctx, gateway.execute(&execution, request))
            .await
            .map_err(|error| FacadeError::Federation(error.to_string()))
            .and_then(|response| {
                self.0
                    .parts
                    .gate
                    .postflight(&execution, &decision, response)
                    .map_err(|error| FacadeError::PolicyDenied(error.to_string()))
            });
        self.audit(
            ctx,
            "federation.execute",
            result.is_ok(),
            "tenant-bound federation",
        )?;
        result
    }
    pub async fn retrieve(
        &self,
        ctx: &RequestContext,
        query: &Query,
        options: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        ctx.remaining()?;
        let result = self
            .timed(ctx, self.0.retriever.retrieve_plan(query, options))
            .await;
        self.audit(ctx, "retrieve", result.is_ok(), "query")?;
        result
    }
    pub fn watch(&self, ctx: &RequestContext, mut options: WatchOptions) -> Result<TenantWatch> {
        ctx.remaining()?;
        let root = format!("uwu://{}/", ctx.tenant_name());
        if let Some(p) = &options.prefix {
            if !p.starts_with(&root) {
                return Err(FacadeError::TenantViolation(p.clone()));
            }
        } else {
            options.prefix = Some(root);
        }
        Ok(TenantWatch {
            ctx: ctx.clone(),
            inner: self.0.parts.watch.watch(options),
        })
    }
    /// Returns the tenant-guarded version API. The underlying storage port is never exposed.
    pub fn versions(&self, ctx: &RequestContext) -> Versions {
        Versions {
            ctx: ctx.clone(),
            db: self.clone(),
        }
    }
    /// Returns the tenant-guarded interactive conflict API.
    pub fn interactive_versions(&self, ctx: &RequestContext) -> InteractiveVersions {
        InteractiveVersions {
            ctx: ctx.clone(),
            db: self.clone(),
        }
    }
    pub async fn execute_tool(
        &self,
        ctx: &RequestContext,
        name: &str,
        request: ExecutionRequest,
    ) -> Result<ExecutionResponse> {
        self.execute(
            ctx,
            ExecutionKind::Tool { name: name.into() },
            request,
            self.0.parts.tool_executor.as_ref(),
        )
        .await
    }
    pub async fn execute_skill(
        &self,
        ctx: &RequestContext,
        name: &str,
        request: ExecutionRequest,
    ) -> Result<ExecutionResponse> {
        self.execute(
            ctx,
            ExecutionKind::Skill { name: name.into() },
            request,
            self.0.parts.skill_executor.as_ref(),
        )
        .await
    }
    async fn execute(
        &self,
        ctx: &RequestContext,
        kind: ExecutionKind,
        request: ExecutionRequest,
        executor: Option<&Arc<dyn crate::TypedExecutor>>,
    ) -> Result<ExecutionResponse> {
        let ex = ctx.execution(kind);
        let decision = self
            .0
            .parts
            .gate
            .preflight(&ex, request.clone())
            .map_err(|e| FacadeError::PolicyDenied(e.to_string()))?;
        if !decision.allowed {
            self.audit(ctx, "execute", false, "gate denied")?;
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let response = self
            .timed(
                ctx,
                executor
                    .ok_or(FacadeError::NotConfigured("executor"))?
                    .execute(&ex, request),
            )
            .await?;
        let result = self
            .0
            .parts
            .gate
            .postflight(&ex, &decision, response)
            .map_err(|e| FacadeError::PolicyDenied(e.to_string()));
        self.audit(ctx, "execute", result.is_ok(), "tool or skill")?;
        result
    }
    pub async fn wasm_register_tenant(
        &self,
        ctx: &RequestContext,
        policy: agent_context_db_wasm::TenantSandboxPolicy,
    ) -> Result<()> {
        ctx.remaining()?;
        let gateway = self
            .0
            .parts
            .wasm
            .as_ref()
            .ok_or(FacadeError::NotConfigured("wasm"))?;
        let result = gateway
            .register_tenant(
                &ctx.execution(ExecutionKind::Tool {
                    name: "wasm.register".into(),
                }),
                policy,
            )
            .await
            .map_err(FacadeError::Wasm);
        self.audit(
            ctx,
            "wasm.register",
            result.is_ok(),
            "tenant-bound registration",
        )?;
        result
    }
    pub async fn wasm_install(
        &self,
        ctx: &RequestContext,
        module: &str,
        bytes: Vec<u8>,
    ) -> Result<[u8; 32]> {
        ctx.remaining()?;
        if bytes.len() > self.0.config.max_wasm_module_bytes {
            return Err(FacadeError::InvalidConfig(
                "WASM module exceeds configured limit".into(),
            ));
        }
        let gateway = self
            .0
            .parts
            .wasm
            .as_ref()
            .ok_or(FacadeError::NotConfigured("wasm"))?;
        let execution = ctx.execution(ExecutionKind::Tool {
            name: "wasm.install".into(),
        });
        let gate_request = ExecutionRequest::new("wasm.install", module);
        let decision = self
            .0
            .parts
            .gate
            .preflight(&execution, gate_request)
            .map_err(|error| FacadeError::PolicyDenied(error.to_string()))?;
        if !decision.allowed {
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let result = self
            .timed(ctx, gateway.install(&execution, module, bytes))
            .await
            .map_err(|error| FacadeError::Wasm(error.to_string()));
        let result = result.and_then(|hash| {
            self.0
                .parts
                .gate
                .postflight(&execution, &decision, ExecutionResponse::new("installed"))
                .map_err(|error| FacadeError::PolicyDenied(error.to_string()))?;
            Ok(hash)
        });
        self.audit(ctx, "wasm.install", result.is_ok(), module)?;
        result
    }
    pub async fn wasm_invoke(
        &self,
        ctx: &RequestContext,
        token: &agent_context_db_wasm::CapabilityToken,
        module: &str,
        function: &str,
        request: Vec<u8>,
    ) -> Result<Vec<u8>> {
        ctx.remaining()?;
        if request.len() > self.0.config.max_wasm_request_bytes {
            return Err(FacadeError::InvalidConfig(
                "WASM request exceeds configured limit".into(),
            ));
        }
        let gateway = self
            .0
            .parts
            .wasm
            .as_ref()
            .ok_or(FacadeError::NotConfigured("wasm"))?;
        let execution = ctx.execution(ExecutionKind::Tool {
            name: "wasm.invoke".into(),
        });
        let decision = self
            .0
            .parts
            .gate
            .preflight(
                &execution,
                ExecutionRequest::new("wasm.invoke", format!("{module}::{function}")),
            )
            .map_err(|error| FacadeError::PolicyDenied(error.to_string()))?;
        if !decision.allowed {
            return Err(FacadeError::PolicyDenied("preflight denied".into()));
        }
        let result = self
            .timed(
                ctx,
                gateway.invoke(&execution, token, module, function, request),
            )
            .await
            .map_err(|error| FacadeError::Wasm(error.to_string()))
            .and_then(|output| {
                if output.len() > self.0.config.max_wasm_output_bytes {
                    return Err(FacadeError::Wasm(
                        "WASM output exceeds configured limit".into(),
                    ));
                }
                self.0
                    .parts
                    .gate
                    .postflight(&execution, &decision, ExecutionResponse::new("invoked"))
                    .map_err(|error| FacadeError::PolicyDenied(error.to_string()))?;
                Ok(output)
            });
        self.audit(
            ctx,
            "wasm.invoke",
            result.is_ok(),
            format!("{module}::{function}"),
        )?;
        result
    }
    pub async fn shutdown(&self) -> Result<()> {
        self.0.accepting.store(false, Ordering::Release);
        while self.0.in_flight.load(Ordering::Acquire) != 0 {
            self.0.drained.notified().await;
        }
        let _lock = self.0.shutdown_lock.lock().await;
        if matches!(&*self.0.shutdown_result.lock().await, Some(Ok(()))) {
            return Ok(());
        }
        for guard in &self.0.parts.runtime_guards {
            if let Err(error) = guard.shutdown().await {
                *self.0.shutdown_result.lock().await = Some(Err(error.to_string()));
                return Err(error.into());
            }
        }
        *self.0.shutdown_result.lock().await = Some(Ok(()));
        Ok(())
    }
}
pub struct TenantWatch {
    ctx: RequestContext,
    inner: WatchStream,
}
impl TenantWatch {
    pub async fn recv(&mut self) -> Result<ChangeEvent> {
        let left = self.ctx.remaining()?;
        let e = tokio::time::timeout(left, self.inner.recv())
            .await
            .map_err(|_| FacadeError::Timeout(left))??;
        if e.tenant != self.ctx.tenant_name() {
            return Err(FacadeError::TenantViolation(e.uri.to_string()));
        }
        Ok(e)
    }
}
/// Tenant-bound wrapper around all version-store operations.
pub struct Versions {
    pub(crate) ctx: RequestContext,
    pub(crate) db: ContextDb,
}

/// Tenant-bound wrapper around interactive conflict sessions.
pub struct InteractiveVersions {
    pub(crate) ctx: RequestContext,
    pub(crate) db: ContextDb,
}
