use crate::{
    AuditSink, ContextDb, FacadeError, FederationGateway, LifecyclePort, Result, RuntimeGuard,
    TypedExecutor, WasmGateway,
};
use agent_context_db_core::{
    ContentRepo, ContentStore, ExecutionGate, FsOps, LlmClient, ReactionSink, VectorIndex,
    WatchSource,
};
use agent_context_db_retrieve::{ContextRetriever, ContextRetrieverBuilder};
use std::{sync::Arc, time::Duration};

/// Validated facade runtime limits.
#[derive(Debug, Clone)]
pub struct ContextDbConfig {
    pub default_timeout: Duration,
    pub max_batch_size: usize,
    pub max_tree_depth: usize,
    /// Maximum UTF-8 bytes retained in an audit detail field.
    pub max_audit_detail_bytes: usize,
    /// Maximum accepted WASM module size in bytes.
    pub max_wasm_module_bytes: usize,
    /// Maximum accepted WASM invocation request size in bytes.
    pub max_wasm_request_bytes: usize,
    /// Maximum accepted WASM invocation response size in bytes.
    pub max_wasm_output_bytes: usize,
    /// Maximum lifetime of an in-memory conflict-session ownership binding.
    pub conflict_session_ttl: Duration,
}
impl Default for ContextDbConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            max_batch_size: 1_000,
            max_tree_depth: 64,
            max_audit_detail_bytes: 4_096,
            max_wasm_module_bytes: 16 * 1024 * 1024,
            max_wasm_request_bytes: 1024 * 1024,
            max_wasm_output_bytes: 1024 * 1024,
            conflict_session_ttl: Duration::from_secs(30 * 60),
        }
    }
}
impl ContextDbConfig {
    pub fn validate(&self) -> Result<()> {
        if self.default_timeout.is_zero()
            || self.max_batch_size == 0
            || self.max_tree_depth == 0
            || self.max_audit_detail_bytes == 0
            || self.max_wasm_module_bytes == 0
            || self.max_wasm_request_bytes == 0
            || self.max_wasm_output_bytes == 0
            || self.conflict_session_ttl.is_zero()
        {
            Err(FacadeError::InvalidConfig(
                "timeouts and resource limits must be non-zero".into(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Explicit application ports. Required ports cannot silently fall back to fake adapters.
pub struct ContextDbParts {
    pub fs: Arc<dyn FsOps>,
    pub content: Arc<dyn ContentRepo>,
    pub content_store: Arc<dyn ContentStore>,
    pub vector: Arc<dyn VectorIndex>,
    pub watch: Arc<dyn WatchSource>,
    pub versions: Arc<dyn agent_context_db_version::VersionStore>,
    pub interactive_versions: Arc<dyn agent_context_db_version::InteractiveVersionStore>,
    pub gate: Arc<dyn ExecutionGate>,
    pub lifecycle: Option<Arc<dyn LifecyclePort>>,
    pub tool_executor: Option<Arc<dyn TypedExecutor>>,
    pub skill_executor: Option<Arc<dyn TypedExecutor>>,
    pub llm: Option<Arc<dyn LlmClient>>,
    pub reactions: Option<Arc<dyn ReactionSink>>,
    /// Mandatory durable/observable audit boundary. There is deliberately no no-op fallback.
    pub audit: Arc<dyn AuditSink>,
    pub wasm: Option<Arc<dyn WasmGateway>>,
    pub federation: Option<Arc<dyn FederationGateway>>,
    pub runtime_guards: Vec<Arc<dyn RuntimeGuard>>,
}

/// Builder supporting deterministic injected ports and production storage assembly.
pub struct ContextDbBuilder {
    config: ContextDbConfig,
    parts: ContextDbParts,
}
impl ContextDbBuilder {
    pub fn injected(parts: ContextDbParts) -> Self {
        Self {
            config: ContextDbConfig::default(),
            parts,
        }
    }
    pub fn config(mut self, config: ContextDbConfig) -> Self {
        self.config = config;
        self
    }
    pub fn build(self) -> Result<ContextDb> {
        self.config.validate()?;
        let retriever: ContextRetriever = ContextRetrieverBuilder::new(self.parts.fs.clone())
            .with_content_store(self.parts.content_store.clone())
            .with_vector_index(self.parts.vector.clone())
            .build()?;
        ContextDb::assemble(self.config, self.parts, Arc::new(retriever))
    }
}
