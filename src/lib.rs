//! Safe application facade for agent-context-db.

mod config;
mod context;
mod error;
mod facade;
mod ports;
mod version_facade;

#[cfg(test)]
mod tests;

pub use config::{ContextDbBuilder, ContextDbConfig, ContextDbParts};
pub use context::{CancellationToken, RequestContext, TenantIdentity};
pub use error::{FacadeError, Result};
pub use facade::{ContextDb, InteractiveVersions, TenantWatch, Versions};
pub use ports::{
    AuditEvent, AuditSink, BoundedAuditSink, FederationGateway, LifecyclePort, RuntimeGuard,
    TypedExecutor, WasmGateway,
};

pub use agent_context_db_core::{
    ChangeEvent, ContentLevel, ContentPayload, ContentType, ContextEntry, ContextUri, DirEntry,
    FindPattern, GrepHit, MvccVersion, Page, PageRequest, Reaction, TreeNode, WatchCheckpoint,
    WatchOptions,
};
pub use agent_context_db_retrieve::{Query, RetrievalResult, RetrieveContext};
pub use agent_context_db_version as version;
pub use agent_context_db_wasm::facade as wasm;
