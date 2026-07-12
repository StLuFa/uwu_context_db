use std::time::Duration;

/// Errors produced by the guarded application facade.
#[derive(Debug, thiserror::Error)]
pub enum FacadeError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("tenant boundary violation: {0}")]
    TenantViolation(String),
    #[error("request cancelled")]
    Cancelled,
    #[error("request deadline exceeded")]
    DeadlineExceeded,
    #[error("capability not configured: {0}")]
    NotConfigured(&'static str),
    #[error("policy denied: {0}")]
    PolicyDenied(String),
    #[error("context operation failed: {0}")]
    Context(#[from] agent_context_db_core::ContextError),
    #[error("version operation failed: {0}")]
    Version(#[from] agent_context_db_version::VersionError),
    #[error("retrieval configuration failed: {0}")]
    RetrieveConfig(#[from] agent_context_db_retrieve::RetrieveConfigError),
    #[error("LLM operation failed: {0}")]
    Llm(#[from] agent_context_db_core::LlmError),
    #[error("WASM operation failed: {0}")]
    Wasm(String),
    #[error("federation operation failed: {0}")]
    Federation(String),
    #[error("audit sink rejected event: {0}")]
    Audit(String),
    /// The durable mutation committed, but a required post-commit lifecycle or audit step failed.
    #[error("mutation committed but post-commit step failed: {failure}")]
    CommittedWithPostCommitFailure { failure: String },
    #[error("operation timed out after {0:?}")]
    Timeout(Duration),
}

impl From<String> for FacadeError {
    fn from(error: String) -> Self {
        Self::Wasm(error)
    }
}

pub type Result<T> = std::result::Result<T, FacadeError>;
