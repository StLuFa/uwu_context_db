# agent-context-db-marketplace-types

Marketplace 与联邦发现共享的 DTO（Data Transfer Object）crate。故意做得很薄。

## 定位

- 是 `consolidation`（生产者）和 `knowledge-network`（联邦分发方）之间的**窄边界**
- 不包含任何巩固逻辑、检索逻辑、隐私逻辑
- 只放跨 crate 共享的 ID、枚举、证明结构、时间戳类型

之所以把这些类型单独抽出来，是为了避免 `knowledge-network` 反向依赖 `consolidation`。

## 单模块结构

只有 `src/lib.rs`，主要类型：

- `MarketId(Uuid)`：市场条目唯一 ID
- `AgentId(String)`：Agent 身份
- `BondLevel`：Observer / Contributor / Validator / Authority
- `MarketEntryType`：Fact / Skill / Procedure / Antibody / ErrorPattern
- `ThreatSeverity`：Low / Medium / High / Critical
- `CorroborationLevel`：Unverified / SingleSession / CrossSession / CrossAgent / Established
- `CorroborationProof`：跨 Agent 佐证结构（`corroborators: Vec<AgentId>`，时间戳等）
- 其他辅助 DTO

## 依赖

- `agent-context-db-core` — `ContentType` / `ContextUri` / `EpistemicType`
- `chrono`、`serde`、`uuid`

## 用法

作为公共类型直接引用：

```rust
use agent_context_db_marketplace_types::{
    AgentId, BondLevel, CorroborationLevel, CorroborationProof,
    MarketEntryType, MarketId, ThreatSeverity,
};

let entry_id = MarketId::new();
let proof = CorroborationProof {
    corroborators: vec![AgentId::new("agent-a"), AgentId::new("agent-b")],
    // …
};
```

## 与其他 crate 的关系

- **只依赖** `core`
- **被 `consolidation` 与 `knowledge-network` 共同引用**
- 这个 crate 应该**保持不变或极少变化**：任何在这里新增的字段都会同时改变两侧
