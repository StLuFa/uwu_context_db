use crate::{FacadeError, Result};
use agent_context_db_core::{ExecutionContext, ExecutionKind, TenantId};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[derive(Default)]
struct CancellationState {
    cancelled: AtomicBool,
    notify: Notify,
}

/// Cheap-clone cooperative cancellation primitive owned by callers.
#[derive(Clone, Default)]
pub struct CancellationToken(Arc<CancellationState>);
impl CancellationToken {
    pub fn cancel(&self) {
        if !self.0.cancelled.swap(true, Ordering::AcqRel) {
            self.0.notify.notify_waiters();
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
    }
    async fn cancelled(&self) {
        loop {
            let notified = self.0.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Canonical tenant identity used consistently by UUID-bearing records and URI authorities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantIdentity {
    id: TenantId,
    name: Arc<str>,
}
impl TenantIdentity {
    /// Creates a validated canonical tenant identity.
    pub fn new(id: TenantId, name: impl Into<Arc<str>>) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(FacadeError::InvalidConfig(
                "tenant name must be non-empty".into(),
            ));
        }
        Ok(Self { id, name })
    }
    /// Returns the stable tenant identifier.
    pub fn id(&self) -> TenantId {
        self.id
    }
    /// Returns the canonical URI authority name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Immutable tenant/actor/request binding with an absolute monotonic deadline.
#[derive(Clone)]
pub struct RequestContext {
    tenant: TenantId,
    tenant_name: Arc<str>,
    actor: Arc<str>,
    request_id: Arc<str>,
    deadline: Instant,
    cancellation: CancellationToken,
}
impl RequestContext {
    pub fn new(
        tenant: TenantIdentity,
        actor: impl Into<Arc<str>>,
        request_id: impl Into<Arc<str>>,
        deadline: Instant,
        cancellation: CancellationToken,
    ) -> Result<Self> {
        let actor = actor.into();
        let request_id = request_id.into();
        if actor.is_empty() || request_id.is_empty() {
            return Err(FacadeError::InvalidConfig(
                "request identity fields must be non-empty".into(),
            ));
        }
        Ok(Self {
            tenant: tenant.id(),
            tenant_name: tenant.name,
            actor,
            request_id,
            deadline,
            cancellation,
        })
    }
    pub fn tenant(&self) -> TenantId {
        self.tenant
    }
    pub fn tenant_name(&self) -> &str {
        &self.tenant_name
    }
    pub fn actor(&self) -> &str {
        &self.actor
    }
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
    pub fn remaining(&self) -> Result<Duration> {
        if self.cancellation.is_cancelled() {
            return Err(FacadeError::Cancelled);
        }
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or(FacadeError::DeadlineExceeded)
    }
    pub(crate) async fn cancelled(&self) {
        self.cancellation.cancelled().await;
    }
    pub(crate) fn execution(&self, kind: ExecutionKind) -> ExecutionContext {
        ExecutionContext {
            tenant_id: self.tenant_name.to_string(),
            actor_id: self.actor.to_string(),
            session_id: None,
            request_id: self.request_id.to_string(),
            kind,
            attributes: BTreeMap::new(),
        }
    }
}
