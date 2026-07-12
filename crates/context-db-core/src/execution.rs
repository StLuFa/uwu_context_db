use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionKind {
    Llm,
    Tool { name: String },
    Skill { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub tenant_id: String,
    pub actor_id: String,
    pub session_id: Option<String>,
    pub request_id: String,
    pub kind: ExecutionKind,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub operation: String,
    pub content: String,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

impl ExecutionRequest {
    pub fn new(operation: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            content: content.into(),
            attributes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionResponse {
    pub content: String,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

impl ExecutionResponse {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            attributes: BTreeMap::new(),
        }
    }
}
