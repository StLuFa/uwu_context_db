//! # agent-context-db-core (M0 内核)
//!
//! Agent 上下文数据库的**通用核心**最小内核：
//!
//! - `uwu://` URI 强类型寻址（[`uri`]）
//! - 三层信息模型 + 8 种记忆分类 + 内容载荷（[`model`]）
//! - 存储窄端口 `FsOps` / `ContentRepo` / `VersionOps` / `TenantOps`（[`store`]）
//! - LLM 客户端端口（[`llm`]）
//!
//! ## 设计约束（见 ARCHITECTURE.md §0.5 / §2.0）
//!
//! - **无 uwu 依赖**：核心与具体 Agent 框架无关，可独立发布。
//! - **端口/适配器**：全部存储/LLM 能力以 trait 暴露，实现由宿主注入（零实现）。
//! - **接口隔离**：上层只依赖用到的窄端口，禁止依赖聚合 `ContextStore`。
//!
//! 内存版实现见 `agent-context-db-testkit`；生产由 PG + Qdrant 适配器注入。

pub mod config;
pub mod error;
pub mod event;
pub mod event_bus;
pub mod event_store;
pub mod lifecycle;
pub mod llm;
pub mod lsh;
pub mod model;
pub mod observability;
pub mod pack;
pub mod rate_limiter;
pub mod read_cache;
#[cfg(feature = "redis-backend")]
pub mod redis_backend;
pub mod similarity;
pub mod store;
pub mod uri;
pub mod vector;
pub mod zerocopy;

pub use error::{ContextError, Result};
pub use event::{
    CausalLink, ChangeEventStream, ChangeSource, ContextTemplate, EventEmitter,
    InheritanceChain, InheritanceNode, OverrideAction, OverrideRule, TemplateEngine,
    TemplateEntry,
};
pub use event_bus::{EventBus, EventStream as EventBusStream, MemoryEventBus};
pub use event_store::{
    CausalDag, CausalityType, DomainEvent, EventFilter as StoreEventFilter, EventId,
    EventKind as StoreEventKind, EventMetadata, EventPayload, EventStore, MemoryEventStore,
    SnapshotStore,
};
pub use lifecycle::{
    AccessEvent, AccessOutcome, DegradeAction, EbbinghausModel, ForgettingCurve,
    ForgettingModel, ImportanceScore, ImportanceWeights, LifecycleAction, LifecycleEngine,
    LifecyclePolicy, LifecycleRule, TokenBudget,
};
pub use lsh::LshIndex;
pub use rate_limiter::{MemoryRateLimiter, RateLimiter};
pub use read_cache::{MemoryReadCache, ReadCache};
pub use llm::{JsonSchema, LlmClient, LlmError, LlmOpts, LlmStream};
pub use observability::{
    ChangeEvent, ChangeEventType, ContextPubSub, ProvenanceEdge, ProvenanceGraph,
    ProvenanceNode, ProvenanceRelationType, QualityDimension, QualityScore, QualityScorer,
    SubscriptionFilter,
};
pub use pack::{AclRule, ContextPack, PackMeta, PathAcl, Permissions, Principal};
#[allow(deprecated)]
pub use similarity::{Cluster, CrossAgentDedup, DedupRecommendation, KnowledgeNetwork, LocalKnowledgeNetwork, SimilarityResult, VectorSimilarity};
pub use model::{
    BlobRef, ConsolidationMeta, ConsolidationStatus, ContentHash, ContentLevel, ContentPart,
    ContentPayload, ContentType, ContextDiff, ContextEntry, ContextMeta, DecodedContent,
    DerivationChain, DerivationRule, DirEntry, EpistemicType, FindPattern, GrepHit,
    LineageEntry, MediaType, MvccVersion, StateScope, TenantId, TreeNode, ValidityRecord,
    VersionEntry,
};
#[allow(deprecated)]
pub use model::{ContentRef, MemoryClass};
pub use store::{
    BlobStore, BrowsingOps, ContentRepo, ContentStore, ContextStore, FsOps, GraphRelation,
    GraphStore, StorageEngine, TenantOps, VersionOps,
};
pub use uri::{ContextUri, UriCategory, SCHEME};
pub use vector::{IndexHit, IndexPoint, VectorIndex};
