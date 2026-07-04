//! LLM 客户端端口（M0 抽象；具体 provider 由宿主注入）。
//!
//! context-db 的语义处理（L0/L1 生成、去重、意图分析）依赖此端口，
//! 但核心不绑定任何具体 LLM 引擎。

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("provider: {0}")]
    Provider(String),
    #[error("timeout")]
    Timeout,
    #[error("rate limited")]
    RateLimited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmOpts {
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for LlmOpts {
    fn default() -> Self {
        Self {
            model: None,
            max_tokens: Some(1024),
            temperature: Some(0.2),
        }
    }
}

/// LLM 结构化输出 schema 描述。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    pub schema: serde_json::Value,
}

impl JsonSchema {
    pub fn new(schema: serde_json::Value) -> Self {
        Self { schema }
    }
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    /// 文本补全。
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError>;
    /// 生成 embedding。
    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError>;

    /// 结构化 JSON 输出（返回原始 JSON 字符串，调用方自行解析）。
    async fn complete_json(
        &self, prompt: &str, schema: &JsonSchema, opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        let full_prompt = format!("{prompt}\n\nRespond with ONLY valid JSON matching this schema: {}", schema.schema);
        self.complete(&full_prompt, opts).await
    }

    /// 流式生成（默认 fallback 到 complete）。
    async fn stream_complete(
        &self, prompt: &str, opts: &LlmOpts,
    ) -> Result<Box<dyn LlmStream + Send>, LlmError> {
        let text = self.complete(prompt, opts).await?;
        Ok(Box::new(BufferedStream { chunks: vec![text], index: 0 }))
    }

    /// 批量补全（默认逐条调用 complete）。
    async fn batch_complete(
        &self, prompts: &[String], opts: &LlmOpts,
    ) -> Result<Vec<String>, LlmError> {
        let mut results = Vec::with_capacity(prompts.len());
        for p in prompts {
            results.push(self.complete(p, opts).await?);
        }
        Ok(results)
    }

    /// 投机执行（默认 fallback 到 complete）。
    async fn speculative_complete(
        &self, prompt: &str, opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        self.complete(prompt, opts).await
    }
}

/// 流式响应迭代器。
pub trait LlmStream: Send {
    fn next_chunk(&mut self) -> Option<Result<String, LlmError>>;
}

struct BufferedStream {
    chunks: Vec<String>,
    index: usize,
}

impl LlmStream for BufferedStream {
    fn next_chunk(&mut self) -> Option<Result<String, LlmError>> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            Some(Ok(chunk))
        } else {
            None
        }
    }
}
