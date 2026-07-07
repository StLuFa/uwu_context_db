# agent-context-db-consolidation

高级学习巩固层：把原始记忆条目变成"经过决策、带血缘、有信心校准、可跨 Agent 流通"的巩固产物（`ConsolidationProduct`）。

## 定位

- 输入：`ContextEntry` / 观测流 / 反馈
- 输出：`ConsolidationProduct` + 决策动作（ADD / UPDATE / INVALIDATE / NOOP）+ 血缘 / 质量 / 校准信号
- 是 CDT 训练的上游数据源，也是 Marketplace 联邦流通的数据源
- 依赖 `retrieve` 做语义检索，`knowledge-network` 做联邦发现，`version` 做时序版本管理

## 主要模块

### 单 Agent 侧

| 模块 | 作用 |
| --- | --- |
| `lib.rs` | 顶层类型：`ConsolidationProduct`、`MemoryResolver`（ADD/UPDATE/INVALIDATE/NOOP）、`EpistemicTyper`、`IgnoranceMap`、`ConfidenceCalibrator` |
| `batch.rs` | 批量巩固处理与队列管理 |
| `halflife.rs` | Ebbinghaus 遗忘曲线 / 半衰期 |
| `rif.rs` | Retrieval-Induced Forgetting：新采纳抑制冗余邻居 |
| `entanglement.rs` | 多维关系纠缠与上下文继承 |
| `lineage.rs` | 血缘链条与来源追踪 |
| `opportunity.rs` | 边际效用与机会成本评估 |
| `quality.rs` | `HorizonAwareQualityScorer`：短 / 中 / 长期不同权重的七维质量重评 |
| `semantic_axis.rs` / `relational_axis.rs` | 语义轴与关系轴（三轴模型的两条轴，第三条 URI/type 轴由 `retrieve` 承担） |
| `tiered_cache.rs` | L1 内存 / L2 快照 / 兜底存储的三级缓存 |
| `patcher.rs` | 增量补丁 |
| `loader.rs` | 上下文加载 + 预算感知 |
| `security.rs` / `guard.rs` | 写入前安全检查与免疫记忆 |
| `explainable.rs` | 决策可解释性溯源 |

### Marketplace 子模块（`src/marketplace/*.rs`）

| 模块 | 作用 |
| --- | --- |
| `publish.rs` | `PublishGate`：发布决策与门控 |
| `discovery.rs` | `DiscoveryEngine`：本地 → 缓存 → 联邦的三级搜索 |
| `registry.rs` | `FederatedRegistry`：联邦注册表 + 向量索引 |
| `feedback.rs` | `ReputationEngine`：反馈驱动的声誉计算 |
| `consensus.rs` | 共识与投票聚合 |
| `crdt.rs` | `SemanticCrdtMerger`：语义感知 CRDT 合并 |
| `voting.rs` | `SocialVoter`：社会投票 |
| `conflict.rs` | 冲突解决 |
| `cap.rs` | `CapPolicyEngine`：一致性级别策略 |
| `immune.rs` | `ImmuneProtocol`：抗体记忆与免疫防护 |
| `influence.rs` / `phylogeny.rs` / `community.rs` | 影响力、演化谱系、社区检测 |

## 关键导出

- `ConsolidationProduct`、`ConsolidationMeta`、`ConsolidationStatus`
- `MemoryResolver`、`ResolveAction`
- `EpistemicTyper`、`HypothesisOutcome`
- `IgnoranceMap`、`BlindSpot`
- `ConfidenceCalibrator`
- `HorizonAwareQualityScorer`、`QualityRoute`
- `SleeptimeExecutor`、`SleeptimeTask`、`SleeptimeReport`
- Marketplace：`PublishGate`、`DiscoveryEngine`、`FederatedRegistry`、`ReputationEngine`、`SemanticCrdtMerger`、`SocialVoter` 等

## 依赖

- `agent-context-db-core`
- `agent-context-db-retrieve` — 语义检索 / 三轴查询
- `agent-context-db-knowledge-network` — 联邦发现
- `agent-context-db-version` — 版本状态
- `agent-context-db-marketplace-types` — 与联邦网络共享的窄边界 DTO
- `moka`（异步 LRU）、`parking_lot`、`tokio`、`chrono`、`serde`、`tracing`

## 用法

```rust
use agent_context_db_consolidation::{
    ConsolidationEngine, MemoryResolver, ResolveAction, SleeptimeExecutor, SleeptimeTask,
};

// 决策：新的候选产物应该 ADD / UPDATE / INVALIDATE 还是 NOOP
let resolver = MemoryResolver::new();
let action = resolver.resolve(&product, existing.as_ref(), similar_count, has_contradiction);

// 后台整理：quality 重评、一致性检查、纠缠检测、反向进化…
let sleep = SleeptimeExecutor::new(vec![
    SleeptimeTask::QualityReassessment,
    SleeptimeTask::ConsistencyCheck,
    SleeptimeTask::BackwardEvolve,
])
.with_store(store)
.with_graph(graph);
let report = sleep.run_once(&engine, &scope).await;
```

## 与其他 crate 的关系

- **上游**：`core`（类型 / 存储 trait）、`retrieve`（检索）、`knowledge-network`（联邦）
- **下游**：`cdt`（用巩固产物提取梯度和 Skill）
- **共享边界**：`marketplace-types`（跨 crate DTO 不放在这里）
