# agent-context-db-retrieve

分层检索管线 + 质量闸门 + 预测性预加载。

## 检索流程

```
query → IntentAnalyzer(LlmIntentAnalyzer|RuleBasedIntentAnalyzer)
      → VectorIndex.search() 向量召回定位目录
      → FsOps.ls() + grep() 目录内搜索
      → FsOps.read() 递归深入子目录
      → Reranker.rerank() 精排
      → HallucinationDetector.evaluate() 质量闸门
      → CompressionAwareLoader 按预算加载 L0/L1/L2
      → RetrievalResult + RetrievalTrace
```

## 模块

| 模块 | 内容 |
|------|------|
| `intent` | `RuleBasedIntentAnalyzer`(关键词) + `LlmIntentAnalyzer`(LLM 结构化分类) |
| `retriever` | `HierarchicalRetrieverImpl`(6阶段管线，可选向量召回+LLM embedding) |
| `quality` | `HallucinationDetector`(F20 幻觉检测) + `CompressionAwareLoader`(F17 压缩感知) |
| `innovation` | `PredictivePrefetcher`(F16 预测预加载) + `IncrementalRetrievalLearner`(F28 增量学习) |

## 核心 trait

```rust
pub trait HierarchicalRetriever: Send + Sync {
    async fn retrieve(&self, query, ctx) -> Result<RetrievalResult>;
    async fn retrieve_typed(&self, query, class, ctx) -> Result<RetrievalResult>;
}
pub trait IntentAnalyzer: Send + Sync {
    async fn analyze(&self, query, ctx) -> Result<Vec<TypedQuery>>;
}
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query, hits) -> Result<Vec<RetrievalHit>>;
}
```

## 端口依赖

仅依赖 `FsOps` + `VectorIndex` + `LlmClient`（全部来自 core），不依赖具体后端。
