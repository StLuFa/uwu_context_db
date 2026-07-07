# agent-context-db-nats

`EventMesh` 的 NATS 桥接：把本地 `agent-context-db-core::EventMesh` 事件同步到 NATS / JetStream，并从 NATS 反向回灌回本地 mesh，用来做跨进程 / 跨节点事件同步。

## 定位

- **不实现事件系统本身**：`EventMesh` / `Envelope` / `FlowChannel` / `Bridge` trait 都在 `core`
- **不实现 NATS 协议**：NATS 客户端封装在 `uwu_nats_bridge`（uwu 生态兄弟仓库）
- 这个 crate 只做胶水：把 core 的 `Bridge` trait 接到 `uwu_nats_bridge` 的 `NatsPublisher / NatsSubscriber` 上

## 单模块结构

只有 `src/lib.rs`：

- `NatsBridgeConfig`：URL / correlation_id（映射 NATS subject `agent.{cid}.*`）/ 连接名 / 默认 channel
- `NatsBridge`：**出站桥**，实现 `Bridge`，`mesh.attach_bridge()` 之后 mesh publish 的 envelope 会转发到 NATS
- `NatsIngestor`：**入站桥**，从 NATS 订阅并回灌到本地 mesh
- `NatsBridgeError`：桥接错误

## 关键导出

- `NatsBridgeConfig`
- `NatsBridge`（`impl Bridge`）
- `NatsIngestor`（后台 spawn 消费）
- `NatsBridgeError`

## 依赖

- `agent-context-db-core` — `Bridge` / `Envelope` / `EventMesh` / `FlowChannel` / `SerializedEnvelope`
- `uwu_event_mesh`、`uwu_nats_bridge`（workspace 依赖，uwu 生态兄弟仓库）
- `async-trait`、`tokio`、`serde` / `serde_json`、`tracing`、`thiserror`

## 用法

```rust
use std::sync::Arc;
use agent_context_db_core::EventMesh;
use agent_context_db_nats::{NatsBridge, NatsBridgeConfig, NatsIngestor};

let mesh = EventMesh::new();
let cfg = NatsBridgeConfig {
    url: "nats://localhost:4222".into(),
    correlation_id: "session-42".into(),
    connection_name: "context-db".into(),
    default_channel: agent_context_db_core::FlowChannel::Consolidation,
};

// 1) 出站：本地 → NATS
let bridge = NatsBridge::connect(cfg.clone()).await?;
mesh.attach_bridge(Arc::new(bridge)).await?;

// 2) 入站：NATS → 本地
let ingestor = NatsIngestor::connect(cfg, mesh.clone()).await?;
ingestor.spawn(); // 后台 task
```

## Channel 路由建议

- `FlowChannel::Main`：低延迟事件流
- `FlowChannel::Consolidation`：走 JetStream，保证持久化
- `FlowChannel::Monitoring`：观测 / 指标事件

`NatsBridge::with_channel(...)` 可以覆盖默认 channel。

## 与其他 crate 的关系

- **依赖**：`core`、`uwu_event_mesh`、`uwu_nats_bridge`
- **被依赖**：应用层直接引用；`consolidation` / `retrieve` / `cdt` 都通过 `core::EventMesh` 间接受益，不需要直接依赖这个 crate
- **可选组件**：不接 NATS 时本 crate 完全无需引入
