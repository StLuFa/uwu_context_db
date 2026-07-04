# agent-context-db-parse

语义处理 + 记忆提取 + 轨迹归纳。

## 模块

| 模块 | 内容 |
|------|------|
| `semantic` | `SemanticProcessorImpl` — LlmClient 驱动 L0(100 tokens) / L1(2k tokens) 生成 |
| `extractor` | `MemoryExtractorImpl` — LlmClient 驱动 8 类记忆提取 + LLM 去重决策(Skip/Create/Merge) |
| `trajectory` | `TrajectoryExtractorImpl` — 会话→Trajectory(did_what/how/result) + 多轨迹→Experience(situation/approach/reflect) |

## 核心 trait

```rust
pub trait SemanticProcessor: Send + Sync {
    async fn generate_abstract(&self, uri) -> Result<String>;
    async fn generate_overview(&self, uri) -> Result<String>;
    async fn aggregate_upward(&self, root) -> Result<()>;
    async fn multimodal_to_text(&self, uri) -> Result<(String, String)>;
}
pub trait MemoryExtractor: Send + Sync {
    async fn extract(&self, archive) -> Result<Vec<MemoryCandidate>>;
    async fn deduplicate(&self, candidates) -> Result<Vec<DedupDecision>>;
}
pub trait TrajectoryExtractor: Send + Sync {
    async fn extract_trajectory(&self, archive) -> Result<Trajectory>;
    async fn induce_experience(&self, trajectories) -> Result<Experience>;
}
```

全部通过 `LlmClient` trait 注入，Mock 测试 / HTTP 生产可互换。
