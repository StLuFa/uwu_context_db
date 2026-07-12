use agent_context_db_core::{
    CausalAttribution, ContextEntry, ContextUri, ExecutionContext, ExecutionRequest,
    ExecutionResponse, Reaction,
};
use async_trait::async_trait;

/// Runs after durable writes and before reads that may require cold restoration.
#[async_trait]
pub trait LifecyclePort: Send + Sync {
    async fn route_after_write(&self, entry: &ContextEntry) -> agent_context_db_core::Result<()>;
    async fn restore_before_read(&self, uri: &ContextUri) -> agent_context_db_core::Result<()>;

    /// Compatibility entry point for callers still using the former metacog-specific name.
    async fn route_metacog(&self, entry: &ContextEntry) -> agent_context_db_core::Result<()> {
        self.route_after_write(entry).await
    }

    /// Compatibility entry point for callers still using the former metacog-specific name.
    async fn restore_metacog(&self, uri: &ContextUri) -> agent_context_db_core::Result<()> {
        self.restore_before_read(uri).await
    }
}

/// Typed tool or skill implementation. The facade always applies its gate around this call.
#[async_trait]
pub trait TypedExecutor: Send + Sync {
    async fn execute(
        &self,
        context: &ExecutionContext,
        request: ExecutionRequest,
    ) -> agent_context_db_core::Result<ExecutionResponse>;
}

/// Optional tenant-aware federation boundary.
#[async_trait]
pub trait FederationGateway: Send + Sync {
    async fn execute(
        &self,
        context: &ExecutionContext,
        request: ExecutionRequest,
    ) -> agent_context_db_core::Result<ExecutionResponse>;
}

/// Object-safe WASM boundary; callers never receive the raw runtime or registry.
#[async_trait]
pub trait WasmGateway: Send + Sync {
    async fn register_tenant(
        &self,
        context: &ExecutionContext,
        policy: agent_context_db_wasm::TenantSandboxPolicy,
    ) -> std::result::Result<(), String>;

    async fn install(
        &self,
        context: &ExecutionContext,
        module: &str,
        bytes: Vec<u8>,
    ) -> std::result::Result<[u8; 32], String>;

    async fn invoke(
        &self,
        context: &ExecutionContext,
        token: &agent_context_db_wasm::CapabilityToken,
        module: &str,
        function: &str,
        request: Vec<u8>,
    ) -> std::result::Result<Vec<u8>, String>;
}

/// A background runtime owned by the facade and shut down explicitly or on drop.
#[async_trait]
pub trait RuntimeGuard: Send + Sync {
    async fn shutdown(&self) -> agent_context_db_core::Result<()>;
}

/// Structured audit event. Details are truncated by the facade before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEvent {
    pub tenant: String,
    pub actor: String,
    pub request_id: String,
    pub operation: String,
    pub allowed: bool,
    pub detail: String,
}

/// Audit failures are observable and can fail the operation rather than disappearing silently.
pub trait AuditSink: Send + Sync {
    fn record(&self, event: AuditEvent) -> std::result::Result<(), String>;
}

/// In-memory bounded audit sink suitable for embedded deployments and tests.
pub struct BoundedAuditSink {
    capacity: usize,
    events: std::sync::Mutex<std::collections::VecDeque<AuditEvent>>,
}
impl BoundedAuditSink {
    pub fn new(capacity: usize) -> std::result::Result<Self, String> {
        if capacity == 0 {
            return Err("audit capacity must be positive".into());
        }
        Ok(Self {
            capacity,
            events: Default::default(),
        })
    }
    pub fn snapshot(&self) -> Vec<AuditEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}
impl AuditSink for BoundedAuditSink {
    fn record(&self, event: AuditEvent) -> std::result::Result<(), String> {
        let mut events = self
            .events
            .lock()
            .map_err(|_| "audit buffer lock poisoned".to_string())?;
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        Ok(())
    }
}

pub(crate) fn reaction_with_context(
    context: Option<&crate::RequestContext>,
    operation: &str,
    subject: &str,
    success: bool,
) -> Reaction {
    let mut traits = std::collections::HashMap::new();
    traits.insert("success".into(), if success { 1.0 } else { 0.0 });
    Reaction {
        id: uuid::Uuid::new_v4().to_string(),
        subject_id: subject.into(),
        execution_id: context.map_or_else(
            || uuid::Uuid::new_v4().to_string(),
            |ctx| ctx.request_id().into(),
        ),
        outcome: if success { 1.0 } else { 0.0 },
        predicted_outcome: None,
        observed_at: chrono::Utc::now(),
        attributions: vec![CausalAttribution {
            cause_id: context.map_or_else(
                || operation.into(),
                |ctx| format!("{}:{}", ctx.actor(), operation),
            ),
            credit: 1.0,
            confidence: 1.0,
        }],
        traits,
    }
}
