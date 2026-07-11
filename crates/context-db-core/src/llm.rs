//! LLM 客户端端口（M0 抽象；具体 provider 由宿主注入）。
//!
//! context-db 的语义处理（L0/L1 生成、去重、意图分析）依赖此端口，
//! 但核心不绑定任何具体 LLM 引擎。

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::prompt::{LlmTaskKind, PromptOptimization, optimize_prompt};

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
    #[serde(default)]
    pub task: LlmTaskKind,
    #[serde(default)]
    pub prompt: PromptOptimization,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingVector {
    pub vector: Vec<f32>,
    pub model_id: String,
    pub dim: usize,
    pub version: u64,
}

impl EmbeddingVector {
    pub fn new(vector: Vec<f32>, model_id: impl Into<String>, version: u64) -> Self {
        let dim = vector.len();
        Self {
            vector,
            model_id: model_id.into(),
            dim,
            version,
        }
    }
}

impl Default for LlmOpts {
    fn default() -> Self {
        Self {
            model: None,
            max_tokens: Some(1024),
            temperature: Some(0.2),
            task: LlmTaskKind::General,
            prompt: PromptOptimization::default(),
        }
    }
}

impl LlmOpts {
    fn cache_key_part(&self) -> String {
        format!(
            "model={:?}|max={:?}|temp={:?}|task={:?}|prompt={:?}",
            self.model, self.max_tokens, self.temperature, self.task, self.prompt
        )
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
    /// 生成带模型元数据的 embedding。
    async fn embed(&self, text: &str) -> Result<EmbeddingVector, LlmError> {
        let mut embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .pop()
            .ok_or_else(|| LlmError::Provider("embedding provider returned no vectors".into()))
    }

    /// 批量生成 embedding。默认实现逐条调用，provider 应覆盖为单次批量请求。
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, LlmError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// 结构化 JSON 输出 — G.4: 无默认实现，强制后端提供。
    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError>;

    /// 流式生成（默认 fallback 到 complete）。
    async fn stream_complete(
        &self,
        prompt: &str,
        opts: &LlmOpts,
    ) -> Result<Box<dyn LlmStream + Send>, LlmError> {
        let text = self.complete(prompt, opts).await?;
        Ok(Box::new(BufferedStream {
            chunks: vec![text],
            index: 0,
        }))
    }

    /// 批量补全（默认逐条调用 complete）。
    async fn batch_complete(
        &self,
        prompts: &[String],
        opts: &LlmOpts,
    ) -> Result<Vec<String>, LlmError> {
        let mut results = Vec::with_capacity(prompts.len());
        for p in prompts {
            results.push(self.complete(p, opts).await?);
        }
        Ok(results)
    }

    /// 投机执行（默认 fallback 到 complete）。
    async fn speculative_complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        self.complete(prompt, opts).await
    }
}

/// 流式响应迭代器 — E.2: async trait。
#[async_trait::async_trait]
pub trait LlmStream: Send {
    async fn next_chunk(&mut self) -> Option<Result<String, LlmError>>;
}

struct BufferedStream {
    chunks: Vec<String>,
    index: usize,
}

#[async_trait::async_trait]
impl LlmStream for BufferedStream {
    async fn next_chunk(&mut self) -> Option<Result<String, LlmError>> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            Some(Ok(chunk))
        } else {
            None
        }
    }
}

pub struct PromptOptimizingLlmClient {
    inner: Arc<dyn LlmClient>,
}

impl PromptOptimizingLlmClient {
    pub fn new(inner: Arc<dyn LlmClient>) -> Self {
        Self { inner }
    }

    pub fn into_arc(self) -> Arc<dyn LlmClient> {
        Arc::new(self)
    }
}

#[async_trait]
impl LlmClient for PromptOptimizingLlmClient {
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        let optimized = optimize_prompt(prompt, &opts.prompt)
            .map_err(|error| LlmError::Provider(error.to_string()))?;
        self.inner.complete(&optimized.text, opts).await
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingVector, LlmError> {
        self.inner.embed(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, LlmError> {
        self.inner.embed_batch(texts).await
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        let optimized = optimize_prompt(prompt, &opts.prompt)
            .map_err(|error| LlmError::Provider(error.to_string()))?;
        self.inner
            .complete_json(&optimized.text, schema, opts)
            .await
    }

    async fn stream_complete(
        &self,
        prompt: &str,
        opts: &LlmOpts,
    ) -> Result<Box<dyn LlmStream + Send>, LlmError> {
        let optimized = optimize_prompt(prompt, &opts.prompt)
            .map_err(|error| LlmError::Provider(error.to_string()))?;
        self.inner.stream_complete(&optimized.text, opts).await
    }

    async fn batch_complete(
        &self,
        prompts: &[String],
        opts: &LlmOpts,
    ) -> Result<Vec<String>, LlmError> {
        let optimized = prompts
            .iter()
            .map(|prompt| {
                optimize_prompt(prompt, &opts.prompt)
                    .map(|optimized| optimized.text)
                    .map_err(|error| LlmError::Provider(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.inner.batch_complete(&optimized, opts).await
    }

    async fn speculative_complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        self.complete(prompt, opts).await
    }
}

#[derive(Debug, Clone)]
pub struct CascadeLlmConfig {
    pub cheap_model: Option<String>,
    pub strong_model: Option<String>,
    pub upgrade_token_threshold: usize,
}

impl Default for CascadeLlmConfig {
    fn default() -> Self {
        Self {
            cheap_model: None,
            strong_model: None,
            upgrade_token_threshold: 4_000,
        }
    }
}

pub struct CascadeLlmClient {
    inner: Arc<dyn LlmClient>,
    config: CascadeLlmConfig,
}

impl CascadeLlmClient {
    pub fn new(inner: Arc<dyn LlmClient>, config: CascadeLlmConfig) -> Self {
        Self { inner, config }
    }

    pub fn into_arc(self) -> Arc<dyn LlmClient> {
        Arc::new(self)
    }

    fn routed_opts(&self, prompt: &str, opts: &LlmOpts) -> Result<LlmOpts, LlmError> {
        let mut routed = opts.clone();
        if routed.model.is_some() {
            return Ok(routed);
        }
        let prompt_tokens = crate::tokenizer::count_tokens(prompt)
            .map_err(|error| LlmError::Provider(error.to_string()))?;
        let high_value = matches!(
            routed.task,
            LlmTaskKind::Arbitration | LlmTaskKind::Merge | LlmTaskKind::Synthesis
        );
        routed.model = if high_value || prompt_tokens >= self.config.upgrade_token_threshold {
            self.config
                .strong_model
                .clone()
                .or(self.config.cheap_model.clone())
        } else {
            self.config
                .cheap_model
                .clone()
                .or(self.config.strong_model.clone())
        };
        Ok(routed)
    }

    async fn complete_with_upgrade(
        &self,
        prompt: &str,
        opts: &LlmOpts,
        json_schema: Option<&JsonSchema>,
    ) -> Result<String, LlmError> {
        let routed = self.routed_opts(prompt, opts)?;
        let first = match json_schema {
            Some(schema) => self.inner.complete_json(prompt, schema, &routed).await,
            None => self.inner.complete(prompt, &routed).await,
        };
        let should_upgrade = matches!(first, Err(LlmError::Provider(_)))
            && routed.model == self.config.cheap_model
            && self.config.strong_model.is_some();
        if should_upgrade {
            let mut upgraded = routed;
            upgraded.model = self.config.strong_model.clone();
            return match json_schema {
                Some(schema) => self.inner.complete_json(prompt, schema, &upgraded).await,
                None => self.inner.complete(prompt, &upgraded).await,
            };
        }
        first
    }
}

#[async_trait]
impl LlmClient for CascadeLlmClient {
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        self.complete_with_upgrade(prompt, opts, None).await
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingVector, LlmError> {
        self.inner.embed(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, LlmError> {
        self.inner.embed_batch(texts).await
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        self.complete_with_upgrade(prompt, opts, Some(schema)).await
    }

    async fn batch_complete(
        &self,
        prompts: &[String],
        opts: &LlmOpts,
    ) -> Result<Vec<String>, LlmError> {
        let mut out = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            out.push(self.complete(prompt, opts).await?);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct CachingLlmClientConfig {
    pub completion_capacity: u64,
    pub embedding_capacity: u64,
    pub completion_ttl: Duration,
    pub embedding_ttl: Duration,
}

impl Default for CachingLlmClientConfig {
    fn default() -> Self {
        Self {
            completion_capacity: 10_000,
            embedding_capacity: 100_000,
            completion_ttl: Duration::from_secs(60 * 60),
            embedding_ttl: Duration::from_secs(60 * 60 * 24),
        }
    }
}

impl CachingLlmClientConfig {
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.completion_capacity == 0 || self.embedding_capacity == 0 {
            return Err(LlmError::Provider(
                "cache capacities must be greater than zero".into(),
            ));
        }
        if self.completion_ttl.is_zero() || self.embedding_ttl.is_zero() {
            return Err(LlmError::Provider(
                "cache TTLs must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Transparent LLM cache for repeated completions and embeddings.
///
/// The cache key includes provider options and schema content, so callers with
/// different models, temperatures, output limits, or JSON schemas do not collide.
pub struct CachingLlmClient {
    inner: Arc<dyn LlmClient>,
    completions: Cache<String, String>,
    embeddings: Cache<String, EmbeddingVector>,
}

impl CachingLlmClient {
    pub fn new(inner: Arc<dyn LlmClient>) -> Result<Self, LlmError> {
        Self::with_config(inner, CachingLlmClientConfig::default())
    }

    pub fn with_config(
        inner: Arc<dyn LlmClient>,
        config: CachingLlmClientConfig,
    ) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            inner,
            completions: Cache::builder()
                .max_capacity(config.completion_capacity)
                .time_to_live(config.completion_ttl)
                .build(),
            embeddings: Cache::builder()
                .max_capacity(config.embedding_capacity)
                .time_to_live(config.embedding_ttl)
                .build(),
        })
    }

    pub fn into_arc(self) -> Arc<dyn LlmClient> {
        Arc::new(self)
    }
}

#[async_trait]
impl LlmClient for CachingLlmClient {
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        let key = completion_cache_key("complete", prompt, None, opts);
        if let Some(value) = self.completions.get(&key).await {
            return Ok(value);
        }
        let value = self.inner.complete(prompt, opts).await?;
        self.completions.insert(key, value.clone()).await;
        Ok(value)
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingVector, LlmError> {
        let mut embeddings = self.embed_batch(&[text.to_string()]).await?;
        embeddings
            .pop()
            .ok_or_else(|| LlmError::Provider("embedding provider returned no vectors".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, LlmError> {
        let mut results: Vec<Option<EmbeddingVector>> = vec![None; texts.len()];
        let mut miss_order = Vec::new();
        let mut miss_texts = Vec::new();
        let mut first_index_by_key = HashMap::new();

        for (idx, text) in texts.iter().enumerate() {
            let key = embedding_cache_key(text);
            if let Some(value) = self.embeddings.get(&key).await {
                results[idx] = Some(value);
            } else if let Some(first_idx) = first_index_by_key.get(&key).copied() {
                miss_order.push((idx, key, first_idx));
            } else {
                let first_idx = miss_texts.len();
                first_index_by_key.insert(key.clone(), first_idx);
                miss_order.push((idx, key, first_idx));
                miss_texts.push(text.clone());
            }
        }

        let loaded = if miss_texts.is_empty() {
            Vec::new()
        } else {
            self.inner.embed_batch(&miss_texts).await?
        };
        if loaded.len() != miss_texts.len() {
            return Err(LlmError::Provider(format!(
                "embedding provider returned {} vectors for {} inputs",
                loaded.len(),
                miss_texts.len()
            )));
        }

        for (idx, key, first_idx) in miss_order {
            let embedding = loaded[first_idx].clone();
            self.embeddings.insert(key, embedding.clone()).await;
            results[idx] = Some(embedding);
        }

        results
            .into_iter()
            .map(|item| item.ok_or_else(|| LlmError::Provider("missing cached embedding".into())))
            .collect()
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        let schema_key = schema.schema.to_string();
        let key = completion_cache_key("json", prompt, Some(&schema_key), opts);
        if let Some(value) = self.completions.get(&key).await {
            return Ok(value);
        }
        let value = self.inner.complete_json(prompt, schema, opts).await?;
        self.completions.insert(key, value.clone()).await;
        Ok(value)
    }

    async fn stream_complete(
        &self,
        prompt: &str,
        opts: &LlmOpts,
    ) -> Result<Box<dyn LlmStream + Send>, LlmError> {
        self.inner.stream_complete(prompt, opts).await
    }

    async fn batch_complete(
        &self,
        prompts: &[String],
        opts: &LlmOpts,
    ) -> Result<Vec<String>, LlmError> {
        let mut out = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            out.push(self.complete(prompt, opts).await?);
        }
        Ok(out)
    }

    async fn speculative_complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        self.complete(prompt, opts).await
    }
}

fn completion_cache_key(kind: &str, prompt: &str, schema: Option<&str>, opts: &LlmOpts) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(kind.as_bytes());
    hasher.update(b"\0");
    hasher.update(opts.cache_key_part().as_bytes());
    hasher.update(b"\0");
    if let Some(schema) = schema {
        hasher.update(schema.as_bytes());
    }
    hasher.update(b"\0");
    hasher.update(prompt.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn embedding_cache_key(text: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"embedding\0");
    hasher.update(text.as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingLlm {
        completes: AtomicUsize,
        embeds: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for CountingLlm {
        async fn complete(&self, prompt: &str, _opts: &LlmOpts) -> Result<String, LlmError> {
            self.completes.fetch_add(1, Ordering::SeqCst);
            Ok(format!("answer:{prompt}"))
        }

        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, LlmError> {
            self.embeds.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|text| EmbeddingVector::new(vec![text.len() as f32], "test", 1))
                .collect())
        }

        async fn complete_json(
            &self,
            prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> Result<String, LlmError> {
            self.completes.fetch_add(1, Ordering::SeqCst);
            Ok(format!(r#"{{"answer":"{prompt}"}}"#))
        }
    }

    #[tokio::test]
    async fn caching_client_reuses_completion_by_options() {
        let inner = Arc::new(CountingLlm {
            completes: AtomicUsize::new(0),
            embeds: AtomicUsize::new(0),
        });
        let cached = CachingLlmClient::new(inner.clone()).unwrap();
        let opts = LlmOpts::default();

        assert_eq!(cached.complete("same", &opts).await.unwrap(), "answer:same");
        assert_eq!(cached.complete("same", &opts).await.unwrap(), "answer:same");
        assert_eq!(inner.completes.load(Ordering::SeqCst), 1);

        let mut different_opts = opts.clone();
        different_opts.temperature = Some(0.9);
        cached.complete("same", &different_opts).await.unwrap();
        assert_eq!(inner.completes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn caching_client_batches_unique_embedding_misses() {
        let inner = Arc::new(CountingLlm {
            completes: AtomicUsize::new(0),
            embeds: AtomicUsize::new(0),
        });
        let cached = CachingLlmClient::new(inner.clone()).unwrap();
        let texts = vec!["alpha".to_string(), "beta".to_string(), "alpha".to_string()];

        let first = cached.embed_batch(&texts).await.unwrap();
        let second = cached.embed_batch(&texts).await.unwrap();

        assert_eq!(
            first.iter().map(|v| v.vector[0]).collect::<Vec<_>>(),
            vec![5.0, 4.0, 5.0]
        );
        assert_eq!(second, first);
        assert_eq!(inner.embeds.load(Ordering::SeqCst), 1);
    }
}
