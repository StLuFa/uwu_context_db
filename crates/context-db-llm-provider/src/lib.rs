//! HTTP-backed LLM providers for `agent-context-db-core`.
//!
//! This crate keeps provider-specific HTTP protocols out of core while still
//! offering a config-driven factory for host applications.

use std::{collections::BTreeMap, env, sync::Arc, time::Duration};

use agent_context_db_core::{
    JsonSchema, LlmClient, LlmError, LlmOpts,
    config::{LlmConfig, UwuConfig},
};
use async_trait::async_trait;
use reqwest::{
    Client, StatusCode,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const OPENAI_BASE_URL: &str = "https://api.openai.com";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const OPENAI_CHAT_PATH: &str = "/v1/chat/completions";
const OPENAI_EMBEDDINGS_PATH: &str = "/v1/embeddings";
const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";
const DEFAULT_OPENAI_EMBEDDING_MODEL: &str = "text-embedding-3-small";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProviderKind {
    OpenAi,
    Anthropic,
    GenericHttp,
}

impl LlmProviderKind {
    pub fn parse(value: &str) -> Result<Self, LlmError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" | "open_ai" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "http" | "generic" | "generic_http" | "self_hosted" | "self-hosted" => {
                Ok(Self::GenericHttp)
            }
            other => Err(LlmError::Provider(format!("unknown llm provider: {other}"))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    pub provider: LlmProviderKind,
    pub model: String,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub embedding_base_url: Option<String>,
    pub embedding_model: Option<String>,
    pub completion_path: Option<String>,
    pub embedding_path: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub timeout_secs: Option<u64>,
}

impl LlmProviderConfig {
    pub fn from_llm_config(config: &LlmConfig) -> Result<Self, LlmError> {
        Ok(Self {
            provider: LlmProviderKind::parse(&config.provider)?,
            model: config.model.clone(),
            api_key: config.api_key.clone(),
            api_key_env: config.api_key_env.clone(),
            base_url: config.base_url.clone(),
            embedding_base_url: config.embedding_base_url.clone(),
            embedding_model: config.embedding_model.clone(),
            completion_path: config.completion_path.clone(),
            embedding_path: config.embedding_path.clone(),
            headers: config.headers.clone(),
            timeout_secs: config.timeout_secs,
        })
    }

    fn api_key(&self) -> Result<Option<String>, LlmError> {
        if let Some(key) = self.api_key.as_ref().filter(|v| !v.is_empty()) {
            return Ok(Some(key.clone()));
        }
        if let Some(env_name) = self.api_key_env.as_ref().filter(|v| !v.is_empty()) {
            return match env::var(env_name) {
                Ok(value) if !value.is_empty() => Ok(Some(value)),
                Ok(_) => Ok(None),
                Err(env::VarError::NotPresent) => Ok(None),
                Err(e) => Err(LlmError::Provider(format!(
                    "failed to read api key env {env_name}: {e}"
                ))),
            };
        }
        Ok(None)
    }

    fn client(&self) -> Result<Client, LlmError> {
        let timeout = Duration::from_secs(self.timeout_secs.unwrap_or(60));
        Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| LlmError::Provider(format!("build http client: {e}")))
    }
}

pub fn from_uwu_config(config: &UwuConfig) -> Result<Arc<dyn LlmClient>, LlmError> {
    from_llm_config(&config.llm)
}

pub fn from_llm_config(config: &LlmConfig) -> Result<Arc<dyn LlmClient>, LlmError> {
    let provider_config = LlmProviderConfig::from_llm_config(config)?;
    match provider_config.provider {
        LlmProviderKind::OpenAi => Ok(Arc::new(OpenAiLlmClient::new(provider_config)?)),
        LlmProviderKind::Anthropic => Ok(Arc::new(AnthropicLlmClient::new(provider_config)?)),
        LlmProviderKind::GenericHttp => Ok(Arc::new(GenericHttpLlmClient::new(provider_config)?)),
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiLlmClient {
    inner: OpenAiCompatibleClient,
}

impl OpenAiLlmClient {
    pub fn new(mut config: LlmProviderConfig) -> Result<Self, LlmError> {
        config.provider = LlmProviderKind::OpenAi;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| OPENAI_BASE_URL.into());
        Ok(Self {
            inner: OpenAiCompatibleClient::new(
                config,
                base_url,
                OPENAI_CHAT_PATH,
                OPENAI_EMBEDDINGS_PATH,
            )?,
        })
    }
}

#[async_trait]
impl LlmClient for OpenAiLlmClient {
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        self.inner.complete(prompt, opts).await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        self.inner.embed(text).await
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        self.inner.complete_json(prompt, schema, opts).await
    }
}

#[derive(Debug, Clone)]
pub struct GenericHttpLlmClient {
    inner: OpenAiCompatibleClient,
}

impl GenericHttpLlmClient {
    pub fn new(mut config: LlmProviderConfig) -> Result<Self, LlmError> {
        config.provider = LlmProviderKind::GenericHttp;
        let base_url = config
            .base_url
            .clone()
            .ok_or_else(|| LlmError::Provider("generic http llm requires base_url".into()))?;
        let chat_path = config
            .completion_path
            .clone()
            .unwrap_or_else(|| OPENAI_CHAT_PATH.into());
        let embedding_path = config
            .embedding_path
            .clone()
            .unwrap_or_else(|| OPENAI_EMBEDDINGS_PATH.into());
        Ok(Self {
            inner: OpenAiCompatibleClient::new(config, base_url, &chat_path, &embedding_path)?,
        })
    }
}

#[async_trait]
impl LlmClient for GenericHttpLlmClient {
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        self.inner.complete(prompt, opts).await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        self.inner.embed(text).await
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        self.inner.complete_json(prompt, schema, opts).await
    }
}

#[derive(Debug, Clone)]
pub struct AnthropicLlmClient {
    config: LlmProviderConfig,
    client: Client,
    base_url: String,
    messages_path: String,
}

impl AnthropicLlmClient {
    pub fn new(mut config: LlmProviderConfig) -> Result<Self, LlmError> {
        config.provider = LlmProviderKind::Anthropic;
        let client = config.client()?;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| ANTHROPIC_BASE_URL.into());
        let messages_path = config
            .completion_path
            .clone()
            .unwrap_or_else(|| ANTHROPIC_MESSAGES_PATH.into());
        Ok(Self {
            config,
            client,
            base_url,
            messages_path,
        })
    }

    async fn post_messages(&self, body: Value) -> Result<Value, LlmError> {
        let api_key = self
            .config
            .api_key()?
            .ok_or_else(|| LlmError::Provider("anthropic api key is not configured".into()))?;
        let headers = provider_headers(
            &self.config.headers,
            Some(("x-api-key", api_key.as_str())),
            &[(&"anthropic-version", "2023-06-01")],
        )?;
        post_json(
            &self.client,
            &join_url(&self.base_url, &self.messages_path),
            headers,
            body,
        )
        .await
    }

    async fn embed_via_configured_endpoint(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let base_url = self.config.embedding_base_url.as_ref().ok_or_else(|| {
            LlmError::Provider(
                "anthropic does not expose a first-party embedding endpoint; set embedding_base_url for an OpenAI-compatible embedding service".into(),
            )
        })?;
        let model = self.config.embedding_model.as_deref().ok_or_else(|| {
            LlmError::Provider("embedding_model is required with embedding_base_url".into())
        })?;
        let path = self
            .config
            .embedding_path
            .as_deref()
            .unwrap_or(OPENAI_EMBEDDINGS_PATH);
        let headers = bearer_headers(&self.config.headers, self.config.api_key()?.as_deref())?;
        let body = json!({ "model": model, "input": text });
        let value = post_json(&self.client, &join_url(base_url, path), headers, body).await?;
        extract_embedding(&value)
    }
}

#[async_trait]
impl LlmClient for AnthropicLlmClient {
    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        let body = json!({
            "model": model(&self.config.model, opts),
            "max_tokens": opts.max_tokens.unwrap_or(1024),
            "temperature": opts.temperature.unwrap_or(0.2),
            "messages": [{ "role": "user", "content": prompt }],
        });
        let value = self.post_messages(body).await?;
        extract_anthropic_text(&value)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        self.embed_via_configured_endpoint(text).await
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        let body = json!({
            "model": model(&self.config.model, opts),
            "max_tokens": opts.max_tokens.unwrap_or(1024),
            "temperature": opts.temperature.unwrap_or(0.2),
            "messages": [{ "role": "user", "content": prompt }],
            "tools": [{
                "name": "emit_json",
                "description": "Return the requested result as JSON.",
                "input_schema": schema.schema,
            }],
            "tool_choice": { "type": "tool", "name": "emit_json" },
        });
        let value = self.post_messages(body).await?;
        extract_anthropic_tool_json(&value).or_else(|_| extract_anthropic_text(&value))
    }
}

#[derive(Debug, Clone)]
struct OpenAiCompatibleClient {
    config: LlmProviderConfig,
    client: Client,
    base_url: String,
    chat_path: String,
    embedding_path: String,
}

impl OpenAiCompatibleClient {
    fn new(
        config: LlmProviderConfig,
        base_url: String,
        chat_path: &str,
        embedding_path: &str,
    ) -> Result<Self, LlmError> {
        let client = config.client()?;
        Ok(Self {
            config,
            client,
            base_url,
            chat_path: chat_path.into(),
            embedding_path: embedding_path.into(),
        })
    }

    async fn complete(&self, prompt: &str, opts: &LlmOpts) -> Result<String, LlmError> {
        let body = self.chat_body(prompt, opts, None);
        let value = self.post_chat(body).await?;
        extract_openai_text(&value)
    }

    async fn complete_json(
        &self,
        prompt: &str,
        schema: &JsonSchema,
        opts: &LlmOpts,
    ) -> Result<String, LlmError> {
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "context_db_result",
                "schema": schema.schema,
                "strict": false,
            }
        });
        let body = self.chat_body(prompt, opts, Some(response_format));
        let value = self.post_chat(body).await?;
        extract_openai_text(&value)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let model = self
            .config
            .embedding_model
            .as_deref()
            .unwrap_or(DEFAULT_OPENAI_EMBEDDING_MODEL);
        let body = json!({ "model": model, "input": text });
        let value = self.post_embeddings(body).await?;
        extract_embedding(&value)
    }

    fn chat_body(&self, prompt: &str, opts: &LlmOpts, response_format: Option<Value>) -> Value {
        let mut body = json!({
            "model": model(&self.config.model, opts),
            "messages": [{ "role": "user", "content": prompt }],
        });
        if let Some(max_tokens) = opts.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = opts.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(response_format) = response_format {
            body["response_format"] = response_format;
        }
        body
    }

    async fn post_chat(&self, body: Value) -> Result<Value, LlmError> {
        let headers = bearer_headers(&self.config.headers, self.config.api_key()?.as_deref())?;
        post_json(
            &self.client,
            &join_url(&self.base_url, &self.chat_path),
            headers,
            body,
        )
        .await
    }

    async fn post_embeddings(&self, body: Value) -> Result<Value, LlmError> {
        let base_url = self
            .config
            .embedding_base_url
            .as_ref()
            .unwrap_or(&self.base_url);
        let headers = bearer_headers(&self.config.headers, self.config.api_key()?.as_deref())?;
        post_json(
            &self.client,
            &join_url(base_url, &self.embedding_path),
            headers,
            body,
        )
        .await
    }
}

fn model(default_model: &str, opts: &LlmOpts) -> String {
    opts.model.clone().unwrap_or_else(|| default_model.into())
}

fn bearer_headers(
    custom: &BTreeMap<String, String>,
    api_key: Option<&str>,
) -> Result<HeaderMap, LlmError> {
    let auth = api_key.map(|key| (AUTHORIZATION.as_str(), format!("Bearer {key}")));
    provider_headers(custom, auth.as_ref().map(|(k, v)| (*k, v.as_str())), &[])
}

fn provider_headers(
    custom: &BTreeMap<String, String>,
    auth: Option<(&str, &str)>,
    required: &[(&&str, &str)],
) -> Result<HeaderMap, LlmError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Some((name, value)) = auth {
        insert_header(&mut headers, name, value)?;
    }
    for (name, value) in required {
        insert_header(&mut headers, name, value)?;
    }
    for (name, value) in custom {
        insert_header(&mut headers, name, value)?;
    }
    Ok(headers)
}

fn insert_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), LlmError> {
    let name = HeaderName::from_bytes(name.as_bytes())
        .map_err(|e| LlmError::Provider(format!("invalid header name {name}: {e}")))?;
    let value = HeaderValue::from_str(value)
        .map_err(|e| LlmError::Provider(format!("invalid header value for {name}: {e}")))?;
    headers.insert(name, value);
    Ok(())
}

async fn post_json(
    client: &Client,
    url: &str,
    headers: HeaderMap,
    body: Value,
) -> Result<Value, LlmError> {
    let response = client
        .post(url)
        .headers(headers)
        .json(&body)
        .send()
        .await
        .map_err(reqwest_error)?;
    let status = response.status();
    let text = response.text().await.map_err(reqwest_error)?;
    if !status.is_success() {
        return Err(status_error(status, text));
    }
    serde_json::from_str(&text).map_err(|e| {
        LlmError::Provider(format!("invalid provider json response: {e}; body={text}"))
    })
}

fn reqwest_error(error: reqwest::Error) -> LlmError {
    if error.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Provider(error.to_string())
    }
}

fn status_error(status: StatusCode, body: String) -> LlmError {
    if status == StatusCode::TOO_MANY_REQUESTS {
        LlmError::RateLimited
    } else {
        LlmError::Provider(format!("provider http {status}: {body}"))
    }
}

fn join_url(base: &str, path: &str) -> String {
    if path.starts_with("http://") || path.starts_with("https://") {
        return path.into();
    }
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

fn extract_openai_text(value: &Value) -> Result<String, LlmError> {
    value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/choices/0/text").and_then(Value::as_str))
        .map(str::to_owned)
        .ok_or_else(|| LlmError::Provider(format!("missing OpenAI-compatible text: {value}")))
}

fn extract_anthropic_text(value: &Value) -> Result<String, LlmError> {
    let content = value
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| LlmError::Provider(format!("missing Anthropic content: {value}")))?;
    let text = content
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");
    if text.is_empty() {
        Err(LlmError::Provider(format!(
            "missing Anthropic text: {value}"
        )))
    } else {
        Ok(text)
    }
}

fn extract_anthropic_tool_json(value: &Value) -> Result<String, LlmError> {
    let content = value
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| LlmError::Provider(format!("missing Anthropic content: {value}")))?;
    let input = content
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("tool_use"))
        .and_then(|item| item.get("input"))
        .ok_or_else(|| LlmError::Provider(format!("missing Anthropic tool_use input: {value}")))?;
    Ok(input.to_string())
}

fn extract_embedding(value: &Value) -> Result<Vec<f32>, LlmError> {
    let array = value
        .pointer("/data/0/embedding")
        .and_then(Value::as_array)
        .ok_or_else(|| LlmError::Provider(format!("missing embedding vector: {value}")))?;
    array
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| LlmError::Provider(format!("non-number embedding value: {v}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_text() {
        let value = json!({"choices":[{"message":{"content":"ok"}}]});
        assert_eq!(extract_openai_text(&value).unwrap(), "ok");
    }

    #[test]
    fn parses_anthropic_tool_json() {
        let value = json!({"content":[{"type":"tool_use","input":{"answer":42}}]});
        assert_eq!(
            extract_anthropic_tool_json(&value).unwrap(),
            r#"{"answer":42}"#
        );
    }

    #[test]
    fn parses_embedding() {
        let value = json!({"data":[{"embedding":[0.5,-1.25]}]});
        assert_eq!(extract_embedding(&value).unwrap(), vec![0.5, -1.25]);
    }
}
