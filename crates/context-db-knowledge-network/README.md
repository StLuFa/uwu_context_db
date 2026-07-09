# agent-context-db-knowledge-network

隐私保护型联邦知识网络：跨 Agent 的能力发现、信任路由、拓扑优化、差分隐私查询与流式聚合，用来支撑 `consolidation/marketplace` 的联邦搜索与跨 Agent 学习。

## 定位

- 只做**联邦网络与治理**：能力索引、隐私预算、访问授权、路由、聚合、拓扑
- **不承载业务 payload**：与 Marketplace / Consolidation 通信只通过 `agent-context-db-marketplace` 的窄边界 DTO
- 传输解耦：`MeshTransport` trait 是唯一网络出口，配合 `uwu_event_mesh`

## 主要模块

| 模块 | 作用 |
| --- | --- |
| `fabric.rs` | `FederatedKnowledgeFabric`：聚合入口，编排隐私 / 能力 / 路由 / 拓扑 / 治理各子系统 |
| `privacy.rs` | `PrivacyGuard`、`DpPolicy`、`PrivacyReceipt`：差分隐私预算与消费凭证 |
| `capability.rs` | `CapabilityIndex`、`CapabilitySketch`：Agent 能力建模与索引 |
| `access.rs` | `AccessGrantManager`、`AccessGrant`：细粒度授权 |
| `trust.rs` | `TrustRouter`：信任评分与路由决策 |
| `topology.rs` | `TopologyOptimizer`：拓扑优化与跳数最小化 |
| `identity.rs` | `IdentityRegistry`：Agent 身份与签名验证 |
| `governance.rs` | `GovernanceEngine`：查询授权与策略执行 |
| `persistence.rs` | `KnowledgeNetworkPersistence` trait + 内存实现 |
| `planner.rs` | `MeshQueryPlanner`：Probe / Fetch 查询计划 |
| `learning.rs` | `RouteOutcomeLearning`：反馈驱动的路由学习 |
| `intent.rs` | `QueryIntentClassifier`：FastLookup / HighPrecision 等意图 |
| `aggregation.rs` | `StreamingTopKAggregator`：流式 top-k 聚合 |
| `transport.rs` | `MeshTransport` trait：事件网络传输适配层 |
| `semantic_graph.rs` | 语义能力图 |

## 关键导出

- `FederatedKnowledgeFabric`
- `PrivacyGuard`、`DpPolicy`、`PrivacyReceipt`
- `CapabilityIndex`、`CapabilitySketch`
- `AccessGrantManager`、`AccessGrant`
- `TrustRouter`、`TopologyOptimizer`、`IdentityRegistry`、`GovernanceEngine`
- `MeshQueryPlanner`、`MeshTransport`
- `KnowledgeNetworkPersistence`、`InMemoryKnowledgeNetworkPersistence`
- `StreamingTopKAggregator`、`QueryIntentClassifier`、`RouteOutcomeLearning`

## 依赖

- `agent-context-db-core`
- `agent-context-db-marketplace` — 与 marketplace 共享的窄边界 DTO
- `uwu_event_mesh` — 底层事件网络（uwu 生态）
- `parking_lot`、`chrono`、`serde`、`uuid`、`thiserror`

## 用法

```rust
use agent_context_db_knowledge_network::{
    FederatedKnowledgeFabric, PrivacyGuard, DpPolicy,
    CapabilityIndex, TrustRouter, InMemoryKnowledgeNetworkPersistence,
};

// 组装联邦网络
let fabric = FederatedKnowledgeFabric::builder()
    .with_privacy(PrivacyGuard::new(DpPolicy::default()))
    .with_capabilities(CapabilityIndex::new())
    .with_router(TrustRouter::new())
    .with_persistence(InMemoryKnowledgeNetworkPersistence::default())
    .build();

// 联邦查询
let result = fabric.query(&intent, &authz).await?;
```

## 与其他 crate 的关系

- **依赖**：`core`、`marketplace`、`uwu_event_mesh`
- **被依赖**：`consolidation`（身份签名）、`marketplace` 的上层装配（联邦发现后端）
- **不依赖**：`retrieve`、`consolidation`——这个 crate 是它们的下游支撑，避免反向依赖
