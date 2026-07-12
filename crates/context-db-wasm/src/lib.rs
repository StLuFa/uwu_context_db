//! Tenant-isolated, policy-gated WebAssembly facade for context-db.

use agent_context_db_core::{
    ExecutionContext, ExecutionGate, ExecutionRequest, ExecutionResponse, PolicyDecision,
};
use hmac::{Hmac, Mac};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fmt::Debug,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use uwu_wasm::{Attestor, Capability, Policy, Sandbox, SandboxEngine, SandboxRegistry};
use wasmtime::component::{ComponentNamedList, Lift, Lower};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Error)]
pub enum WasmError {
    #[error("invalid sandbox policy: {0}")]
    InvalidPolicy(String),
    #[error("capability token rejected: {0}")]
    Token(String),
    #[error("execution policy denied: {0}")]
    Gate(String),
    #[error("wasm runtime: {0}")]
    Runtime(#[from] anyhow::Error),
    #[error("audit event exceeds configured bound")]
    AuditTooLarge,
}

#[derive(Clone, Debug)]
pub struct TenantSandboxPolicy {
    pub memory_pages: u32,
    pub table_elements: u32,
    pub fuel: u64,
    pub deadline: Duration,
    pub max_concurrency: usize,
    pub max_output_bytes: usize,
    pub max_trace_bytes: usize,
    pub wasi: bool,
    pub host_imports: BTreeSet<(String, String)>,
    pub allowed_digests: BTreeSet<[u8; 32]>,
}

impl TenantSandboxPolicy {
    pub fn validate(&self) -> Result<(), WasmError> {
        if self.memory_pages == 0
            || self.table_elements == 0
            || self.fuel == 0
            || self.deadline.is_zero()
            || self.max_concurrency == 0
            || self.max_output_bytes == 0
            || self.max_trace_bytes == 0
        {
            return Err(WasmError::InvalidPolicy(
                "resource limits must all be non-zero".into(),
            ));
        }
        if self.allowed_digests.is_empty() {
            return Err(WasmError::InvalidPolicy(
                "module digest admission list must not be empty".into(),
            ));
        }
        if self
            .host_imports
            .iter()
            .any(|(m, f)| m.is_empty() || f.is_empty())
        {
            return Err(WasmError::InvalidPolicy(
                "host import names must not be empty".into(),
            ));
        }
        Ok(())
    }
    fn runtime_policy(&self) -> Policy {
        let mut policy = Policy::builder()
            .memory_pages(self.memory_pages)
            .fuel(self.fuel)
            .deadline(self.deadline)
            .max_concurrent_instances(self.max_concurrency)
            .output_limit(self.max_output_bytes)
            .trace_limit(self.max_trace_bytes)
            .build();
        policy.max_table_elements = Some(self.table_elements);
        for digest in &self.allowed_digests {
            policy.allowed_digests.insert(*digest);
        }
        for (module, function) in &self.host_imports {
            policy
                .caps
                .insert(Capability::HostImport(module.clone(), function.clone()));
        }
        if self.wasi {
            policy.caps.insert(Capability::Wasi);
        }
        policy
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityClaims {
    pub tenant: String,
    pub subject: String,
    pub execution_kind: String,
    pub module: String,
    pub function: String,
    pub expires_unix_ms: u64,
    pub nonce: String,
    pub max_fuel: u64,
    pub max_memory_pages: u32,
    pub max_output_bytes: usize,
    pub max_trace_bytes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapabilityToken {
    pub claims: CapabilityClaims,
    pub tag: String,
}

#[derive(Clone)]
pub struct TokenAuthority {
    key: Arc<[u8]>,
    used_nonces: Arc<Mutex<HashMap<String, u64>>>,
    nonce_capacity: usize,
}
impl TokenAuthority {
    pub fn new(key: impl Into<Vec<u8>>) -> Result<Self, WasmError> {
        Self::with_nonce_capacity(key, 100_000)
    }
    pub fn with_nonce_capacity(
        key: impl Into<Vec<u8>>,
        nonce_capacity: usize,
    ) -> Result<Self, WasmError> {
        let key = key.into();
        if key.len() < 32 {
            return Err(WasmError::Token(
                "authentication key must contain at least 32 bytes".into(),
            ));
        }
        if nonce_capacity == 0 {
            return Err(WasmError::Token("nonce capacity must be positive".into()));
        }
        Ok(Self {
            key: key.into(),
            used_nonces: Default::default(),
            nonce_capacity,
        })
    }
    pub fn issue(&self, claims: CapabilityClaims) -> Result<CapabilityToken, WasmError> {
        Ok(CapabilityToken {
            tag: self.tag(&claims)?,
            claims,
        })
    }
    fn tag(&self, claims: &CapabilityClaims) -> Result<String, WasmError> {
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .map_err(|e| WasmError::Token(format!("invalid authentication key: {e}")))?;
        let encoded = serde_json::to_vec(claims).map_err(|e| WasmError::Token(e.to_string()))?;
        mac.update(&encoded);
        Ok(hex::encode(mac.finalize().into_bytes()))
    }
    fn verify(
        &self,
        token: &CapabilityToken,
        ctx: &ExecutionContext,
        module: &str,
        function: &str,
        policy: &TenantSandboxPolicy,
    ) -> Result<(), WasmError> {
        let supplied = hex::decode(&token.tag)
            .map_err(|_| WasmError::Token("malformed authentication tag".into()))?;
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .map_err(|e| WasmError::Token(format!("invalid authentication key: {e}")))?;
        mac.update(
            &serde_json::to_vec(&token.claims).map_err(|e| WasmError::Token(e.to_string()))?,
        );
        mac.verify_slice(&supplied)
            .map_err(|_| WasmError::Token("authentication failed".into()))?;
        let c = &token.claims;
        let kind = serde_json::to_string(&ctx.kind).map_err(|e| WasmError::Token(e.to_string()))?;
        if c.tenant != ctx.tenant_id
            || c.subject != ctx.actor_id
            || c.execution_kind != kind
            || c.module != module
            || c.function != function
        {
            return Err(WasmError::Token("binding mismatch".into()));
        }
        if c.expires_unix_ms < now_ms() {
            return Err(WasmError::Token("expired".into()));
        }
        if c.max_fuel != policy.fuel
            || c.max_memory_pages != policy.memory_pages
            || c.max_output_bytes != policy.max_output_bytes
            || c.max_trace_bytes != policy.max_trace_bytes
        {
            return Err(WasmError::Token(
                "quota must exactly match the per-call sandbox policy".into(),
            ));
        }
        let now = now_ms();
        let key = format!("{}:{}:{}", c.tenant, c.subject, c.nonce);
        let mut used = self.used_nonces.lock();
        used.retain(|_, expires| *expires >= now);
        if used.contains_key(&key) {
            return Err(WasmError::Token("nonce replay".into()));
        }
        if used.len() >= self.nonce_capacity {
            return Err(WasmError::Token(
                "nonce cache capacity exhausted while tokens remain valid".into(),
            ));
        }
        used.insert(key, c.expires_unix_ms);
        Ok(())
    }
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmAuditEvent {
    pub tenant: String,
    pub subject: String,
    pub request_id: String,
    pub module: String,
    pub function: String,
    pub phase: String,
    pub allowed: bool,
    pub detail: String,
}
pub trait WasmAuditSink: Send + Sync {
    fn record(&self, event: WasmAuditEvent) -> Result<(), WasmError>;
}
pub struct BoundedAuditSink {
    events: Mutex<VecDeque<WasmAuditEvent>>,
    capacity: usize,
    max_event_bytes: usize,
}
impl BoundedAuditSink {
    pub fn new(capacity: usize, max_event_bytes: usize) -> Result<Self, WasmError> {
        if capacity == 0 || max_event_bytes == 0 {
            return Err(WasmError::InvalidPolicy(
                "audit bounds must be non-zero".into(),
            ));
        }
        Ok(Self {
            events: Default::default(),
            capacity,
            max_event_bytes,
        })
    }
    pub fn snapshot(&self) -> Vec<WasmAuditEvent> {
        self.events.lock().iter().cloned().collect()
    }
}
impl WasmAuditSink for BoundedAuditSink {
    fn record(&self, event: WasmAuditEvent) -> Result<(), WasmError> {
        if serde_json::to_vec(&event)
            .map_err(|e| WasmError::Gate(e.to_string()))?
            .len()
            > self.max_event_bytes
        {
            return Err(WasmError::AuditTooLarge);
        }
        let mut q = self.events.lock();
        if q.len() == self.capacity {
            q.pop_front();
        }
        q.push_back(event);
        Ok(())
    }
}

pub struct ContextDbWasm {
    registry: SandboxRegistry,
    gate: Arc<dyn ExecutionGate>,
    authority: TokenAuthority,
    audit: Arc<dyn WasmAuditSink>,
    policies: RwLock<HashMap<String, TenantSandboxPolicy>>,
}
impl ContextDbWasm {
    pub fn new(
        gate: Arc<dyn ExecutionGate>,
        authority: TokenAuthority,
        audit: Arc<dyn WasmAuditSink>,
    ) -> Result<Self, WasmError> {
        let engine = Arc::new(SandboxEngine::new()?);
        Ok(Self {
            registry: SandboxRegistry::new(engine),
            gate,
            authority,
            audit,
            policies: Default::default(),
        })
    }
    pub fn register_tenant(
        &self,
        context: &ExecutionContext,
        policy: TenantSandboxPolicy,
    ) -> Result<(), WasmError> {
        policy.validate()?;
        let tenant = context.tenant_id.clone();
        let sandbox = Sandbox::new(
            &tenant,
            self.registry.engine().clone(),
            policy.runtime_policy(),
            Arc::new(Attestor::ephemeral()),
        )
        .with_wasi(policy.wasi);
        self.registry.register(sandbox)?;
        self.policies.write().insert(tenant, policy);
        Ok(())
    }
    pub fn install(
        &self,
        context: &ExecutionContext,
        module: &str,
        bytes: Vec<u8>,
    ) -> Result<[u8; 32], WasmError> {
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let policy = self
            .policies
            .read()
            .get(&context.tenant_id)
            .cloned()
            .ok_or_else(|| WasmError::InvalidPolicy("tenant is not registered".into()))?;
        if !policy.allowed_digests.contains(&digest) {
            return Err(WasmError::InvalidPolicy("module digest denied".into()));
        }
        Ok(self
            .registry
            .install_for(&context.tenant_id, module, bytes)?)
    }
    pub fn invoke<P, R>(
        &self,
        context: &ExecutionContext,
        token: &CapabilityToken,
        module: &str,
        function: &str,
        args: P,
    ) -> Result<R, WasmError>
    where
        P: ComponentNamedList + Lower + Send + Sync + 'static + Debug,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + Debug,
    {
        let policy = self
            .policies
            .read()
            .get(&context.tenant_id)
            .cloned()
            .ok_or_else(|| WasmError::InvalidPolicy("tenant is not registered".into()))?;
        self.authority
            .verify(token, context, module, function, &policy)?;
        let request = ExecutionRequest::new("wasm.invoke", format!("{module}::{function}"));
        let decision = self
            .gate
            .preflight(context, request)
            .map_err(|e| WasmError::Gate(e.to_string()))?;
        if !decision.allowed {
            return self.denied(context, module, function, &decision);
        }
        self.audit.record(event(
            context,
            module,
            function,
            "preflight",
            true,
            "authorized",
        ))?;
        let receipt = self
            .registry
            .call::<P, R>(&context.tenant_id, module, function, args)?;
        let response = ExecutionResponse::new(format!(
            "fuel={:?};elapsed_ms={}",
            receipt.fuel_consumed, receipt.elapsed_ms
        ));
        self.gate
            .postflight(context, &decision, response)
            .map_err(|e| WasmError::Gate(e.to_string()))?;
        self.audit.record(event(
            context,
            module,
            function,
            "postflight",
            true,
            "completed",
        ))?;
        Ok(receipt.returns)
    }
    fn denied<R>(
        &self,
        c: &ExecutionContext,
        m: &str,
        f: &str,
        d: &PolicyDecision,
    ) -> Result<R, WasmError> {
        self.audit.record(event(
            c,
            m,
            f,
            "preflight",
            false,
            d.selected_rule.as_deref().unwrap_or("denied"),
        ))?;
        Err(WasmError::Gate("preflight denied".into()))
    }
}
fn event(c: &ExecutionContext, m: &str, f: &str, p: &str, a: bool, d: &str) -> WasmAuditEvent {
    WasmAuditEvent {
        tenant: c.tenant_id.clone(),
        subject: c.actor_id.clone(),
        request_id: c.request_id.clone(),
        module: m.into(),
        function: f.into(),
        phase: p.into(),
        allowed: a,
        detail: d.into(),
    }
}

/// Public application facade: depend on this module rather than on `uwu_wasm` directly.
pub mod facade {
    pub use super::{
        BoundedAuditSink, CapabilityClaims, CapabilityToken, ContextDbWasm, TenantSandboxPolicy,
        TokenAuthority, WasmAuditEvent, WasmAuditSink, WasmError,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ExecutionKind, PolicyError};
    use std::collections::BTreeMap;

    struct Gate(bool);
    impl ExecutionGate for Gate {
        fn version(&self) -> u64 {
            1
        }
        fn preflight(
            &self,
            _context: &ExecutionContext,
            request: ExecutionRequest,
        ) -> Result<PolicyDecision, PolicyError> {
            Ok(PolicyDecision {
                allowed: self.0,
                required: false,
                policy_version: 1,
                request,
                response: None,
                selected_rule: (!self.0).then(|| "deny-wasm".into()),
                audit: vec![],
            })
        }
        fn postflight(
            &self,
            _context: &ExecutionContext,
            _decision: &PolicyDecision,
            response: ExecutionResponse,
        ) -> Result<ExecutionResponse, PolicyError> {
            Ok(response)
        }
    }
    fn context(tenant: &str) -> ExecutionContext {
        ExecutionContext {
            tenant_id: tenant.into(),
            actor_id: "actor".into(),
            session_id: None,
            request_id: "request".into(),
            kind: ExecutionKind::Tool {
                name: "wasm".into(),
            },
            attributes: BTreeMap::new(),
        }
    }
    fn claims(ctx: &ExecutionContext) -> CapabilityClaims {
        CapabilityClaims {
            tenant: ctx.tenant_id.clone(),
            subject: ctx.actor_id.clone(),
            execution_kind: serde_json::to_string(&ctx.kind).unwrap(),
            module: "module".into(),
            function: "run".into(),
            expires_unix_ms: now_ms() + 60_000,
            nonce: "unique".into(),
            max_fuel: 10,
            max_memory_pages: 2,
            max_output_bytes: 10,
            max_trace_bytes: 10,
        }
    }
    fn policy(digest: [u8; 32]) -> TenantSandboxPolicy {
        TenantSandboxPolicy {
            memory_pages: 2,
            table_elements: 2,
            fuel: 10,
            deadline: Duration::from_millis(10),
            max_concurrency: 1,
            max_output_bytes: 10,
            max_trace_bytes: 10,
            wasi: false,
            host_imports: BTreeSet::new(),
            allowed_digests: BTreeSet::from([digest]),
        }
    }

    #[test]
    fn capability_is_authenticated_and_tenant_bound() -> Result<(), WasmError> {
        let authority = TokenAuthority::new([7; 32])?;
        let a = context("tenant-a");
        let b = context("tenant-b");
        let token = authority.issue(claims(&a))?;
        assert!(
            authority
                .verify(&token, &b, "module", "run", &policy([1; 32]))
                .is_err()
        );
        let mut tampered = token.clone();
        tampered.claims.subject = "other".into();
        assert!(matches!(
            authority.verify(&tampered, &a, "module", "run", &policy([1; 32])),
            Err(WasmError::Token(_))
        ));
        Ok(())
    }

    #[test]
    fn tenant_namespaces_and_digest_policy_are_enforced() {
        let audit = Arc::new(BoundedAuditSink::new(8, 1024).unwrap());
        let runtime = ContextDbWasm::new(
            Arc::new(Gate(true)),
            TokenAuthority::new([9; 32]).unwrap(),
            audit,
        )
        .unwrap();
        let bytes = b"not-a-component".to_vec();
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        runtime
            .register_tenant(&context("tenant-a"), policy(digest))
            .unwrap();
        runtime
            .register_tenant(&context("tenant-b"), policy([2; 32]))
            .unwrap();
        assert!(
            runtime
                .install(&context("tenant-b"), "module", bytes.clone())
                .is_err()
        );
        assert!(runtime.registry.get("tenant-a").is_some());
        assert!(runtime.registry.get("tenant-b").is_some());
    }

    #[test]
    fn denial_is_structurally_audited() -> Result<(), WasmError> {
        let audit = Arc::new(BoundedAuditSink::new(8, 1024)?);
        let runtime = ContextDbWasm::new(
            Arc::new(Gate(false)),
            TokenAuthority::new([3; 32]).unwrap(),
            audit.clone(),
        )
        .unwrap();
        let ctx = context("tenant-a");
        let p = policy([1; 32]);
        runtime.register_tenant(&ctx, p.clone()).unwrap();
        let token = runtime.authority.issue(claims(&ctx))?;
        let result = runtime.invoke::<(), ()>(&ctx, &token, "module", "run", ());
        assert!(matches!(result, Err(WasmError::Gate(_))));
        assert!(audit.snapshot().last().is_some_and(|event| !event.allowed));
        Ok(())
    }
}
