//! 错误类型（M0 + 重构扩展）。
//!
//! 所有错误变体通过 `#[from]` 自动派生，消除调用方的 `map_err` 样板。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid uri: {0}")]
    InvalidUri(String),
    #[error("storage: {0}")]
    Storage(String),
    #[error("LLM: {0}")]
    Llm(String),
    #[error("version conflict: {0}")]
    VersionConflict(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("serialization: {0}")]
    Serialization(String),
    #[error("IO: {0}")]
    Io(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl From<serde_json::Error> for ContextError {
    fn from(e: serde_json::Error) -> Self {
        ContextError::Serialization(e.to_string())
    }
}

impl From<std::io::Error> for ContextError {
    fn from(e: std::io::Error) -> Self {
        ContextError::Io(e.to_string())
    }
}

impl From<crate::LlmError> for ContextError {
    fn from(e: crate::LlmError) -> Self {
        ContextError::Llm(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ContextError>;
