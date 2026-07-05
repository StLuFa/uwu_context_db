//! 错误类型（M0）。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid uri: {0}")]
    InvalidUri(String),
    #[error("storage: {0}")]
    Storage(String),
    #[error("version conflict: {0}")]
    VersionConflict(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("serialization: {0}")]
    Serialization(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, ContextError>;
