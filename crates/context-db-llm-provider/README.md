# agent-context-db-llm-provider

`agent-context-db-core` 的 HTTP LLM 客户端实现：OpenAI、Anthropic，以及任何 OpenAI 兼容 / 自托管端点。

## 定位

- 目的：把 provider 相关的 HTTP 协议**挡在 core 外面**
- core 只定义 `LlmClient` / `LlmError` / `LlmOpts` / `JsonSchema` 等抽象
- 这个 crate 是配置驱动工厂，让宿主应用可以用 `UwuConfig` 一行代码拿到 `Arc<dyn LlmClient>`

## 单模块结构

只有 `src/lib.rs`：

- `LlmProviderKind`：`OpenAi` / `Anthropic` / `GenericHttp`
- `LlmProviderConfig`：从 `LlmConfig` 转换而来的运行时配置（provider、model、api_key / api_key_env、base_url、headers、timeout 等）
- 具体 provider 实现：OpenAI 的 `chat/completions` + `embeddings`，Anthropic 的 `messages`
- 工厂函数：从 `UwuConfig` / `LlmConfig` 构造 `Arc<dyn LlmClient>`

## 关键导出

- `LlmProviderKind`
- `LlmProviderConfig`
- 工厂构造函数（从 `UwuConfig` / `LlmConfig` 生成实现 `LlmClient` 的具体类型）

## 依赖

- `agent-context-db-core`
- `reqwest`（HTTP 客户端）
- `async-trait`、`serde`、`serde_json`、`thiserror`

## 用法

```rust
use agent_context_db_core::{LlmClient, LlmOpts, config::UwuConfig};
use agent_context_db_llm_provider::{LlmProviderConfig, LlmProviderKind};
use std::sync::Arc;

// 1) 从配置构造 provider
let cfg = LlmProviderConfig {
    provider: LlmProviderKind::OpenAi,
    model: "gpt-4o-mini".into(),
    api_key: None,
    api_key_env: Some("OPENAI_API_KEY".into()),
    base_url: None,
    embedding_base_url: None,
    embedding_model: Some("text-embedding-3-small".into()),
    completion_path: None,
    embedding_path: None,
    headers: Default::default(),
    timeout_secs: Some(30),
};

// 2) 拿到 Arc<dyn LlmClient>，之后交给上层 crate（consolidation / retrieve / cdt）
let client: Arc<dyn LlmClient> = build_client(cfg)?;
let text = client.complete("hello", &LlmOpts::default()).await?;
```

同一份代码通过换 `provider` + `base_url` 即可切到 Anthropic 或任何 OpenAI 兼容的自托管端点（例如 vLLM、Ollama 的 OpenAI 兼容层）。

## 与其他 crate 的关系

- **依赖**：只依赖 `core`
- **被依赖**：`consolidation` / `retrieve` / `cdt` 及应用层，用来注入具体的 `LlmClient` 实现
- **不参与**：数据库 / 存储 / 检索逻辑（严格是 provider 适配层）
