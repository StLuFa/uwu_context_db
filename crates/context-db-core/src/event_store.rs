//! 事件系统 — 基于 `uwu_event_mesh` 的进程内 mesh + 持久化 + 跨进程桥接。
//!
//! 本模块只做两件事：
//! 1. **re-export** `uwu_event_mesh` 的核心类型（`Envelope`/`EventMesh`/`EventStore` 等）
//! 2. **定义 context-db 领域的 `TypeId` 常量**（`context.entry.created` 等）
//!
//! 存储实现（PG/Memory/JSONL）在 `context-db-storage`。
//! NATS 桥接在 `context-db-storage`（或独立 feature）通过 `uwu_nats_bridge` 完成。

use uwu_event_mesh::TypeId;

// ---------------------------------------------------------------------------
// 从 uwu_event_mesh 重导出常用类型
// ---------------------------------------------------------------------------
pub use uwu_event_mesh::{
    // 事件模型
    Envelope,
    SerializedEnvelope,
    EventMetadata,
    CorrelationId,
    ReplayId,
    TypeId as EventTypeId,
    TypeRegistry,

    // Mesh
    EventMesh,
    EventMeshBuilder,
    FlowChannel,
    FlowHandle,
    FlowReceiver,
    Subscription,
    SubscribeOptions,

    // 主题
    Topic,
    TopicPattern,

    // 存储 + 回放
    EventStore,
    ReplayFilter,
    MemoryStore,
    JsonlStore,
    JsonlStoreOptions,
    SegmentedStore,
    SegmentedStoreOptions,

    // 类型化事件集
    EventKind,
    EventSet,
    TypedSubscription,

    // 跨进程桥接
    Bridge,
    ChannelBridge,
    ChannelBridgePair,
};

// ---------------------------------------------------------------------------
// context-db 领域 TypeId 常量
// ---------------------------------------------------------------------------

/// 领域名 — 所有 context-db 事件的 domain 前缀。
pub const DOMAIN: &str = "context";

/// 内容条目相关事件。
pub mod entry {
    use super::*;

    pub fn created() -> TypeId { TypeId::new(DOMAIN, "entry.created") }
    pub fn updated() -> TypeId { TypeId::new(DOMAIN, "entry.updated") }
    pub fn deleted() -> TypeId { TypeId::new(DOMAIN, "entry.deleted") }
    pub fn archived() -> TypeId { TypeId::new(DOMAIN, "entry.archived") }
    pub fn merged() -> TypeId { TypeId::new(DOMAIN, "entry.merged") }
    pub fn branched() -> TypeId { TypeId::new(DOMAIN, "entry.branched") }
    pub fn tagged() -> TypeId { TypeId::new(DOMAIN, "entry.tagged") }
    pub fn rolled_back() -> TypeId { TypeId::new(DOMAIN, "entry.rolled_back") }
}

/// 巩固相关事件。
pub mod consolidation {
    use super::*;

    pub fn quality_assessed() -> TypeId { TypeId::new(DOMAIN, "consolidation.quality_assessed") }
    pub fn consolidated() -> TypeId { TypeId::new(DOMAIN, "consolidation.consolidated") }
    pub fn contradicted() -> TypeId { TypeId::new(DOMAIN, "consolidation.contradicted") }
    pub fn calibration_updated() -> TypeId { TypeId::new(DOMAIN, "consolidation.calibration_updated") }
}

/// Marketplace 联邦事件。
pub mod marketplace {
    use super::*;

    pub fn skill_published() -> TypeId { TypeId::new(DOMAIN, "marketplace.skill_published") }
    pub fn skill_adopted() -> TypeId { TypeId::new(DOMAIN, "marketplace.skill_adopted") }
    pub fn peer_discovered() -> TypeId { TypeId::new(DOMAIN, "marketplace.peer_discovered") }
    pub fn threat_detected() -> TypeId { TypeId::new(DOMAIN, "marketplace.threat_detected") }
}
