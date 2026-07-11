//! # agent-context-db-core (M0 内核)
//!
//! Agent 上下文数据库的**通用核心**最小内核：
//!
//! - `uwu://` URI 强类型寻址（[`uri`]）
//! - 三层信息模型 + 多种记忆分类 + 内容载荷（[`model`]）
//! - 存储窄端口 `FsOps` / `ContentRepo` / `VersionOps` / `TenantOps`（[`store`]）
//! - LLM 客户端端口（[`llm`]）
//!
//! ## 设计约束（见 ARCHITECTURE.md §0.5 / §2.0）
//!
//! - **端口/适配器**：全部存储/LLM 能力以 trait 暴露，实现由宿主注入（零实现）。
//! - **接口隔离**：上层只依赖用到的窄端口，禁止依赖聚合 `ContextStore`。
//!
//! 内存版实现见 `agent-context-db-testkit`；生产由 PG + Qdrant 适配器注入。

pub mod config;
pub mod embedding_cache;
pub mod embedding_migration;
pub mod error;
pub mod event;
pub mod event_store;
pub mod lifecycle;
pub mod llm;
pub mod lsh;
pub mod model;
pub mod observability;
pub mod pack;
pub mod prompt;
pub mod read_cache;
#[cfg(feature = "redis-backend")]
pub mod redis_backend;
pub mod similarity;
pub mod store;
pub mod tokenizer;
pub mod uri;
pub mod vector;
pub mod watch;
pub mod write_dedup;
pub mod write_security;
pub mod zerocopy;

pub use embedding_cache::{EmbeddingCache, MemoryEmbeddingCache, embedding_content_hash};
pub use embedding_migration::{
    EmbeddingMigrationAction, EmbeddingMigrationExecutor, EmbeddingMigrationPhase,
    EmbeddingMigrationPlan, EmbeddingMigrationReport, EmbeddingModelVersion,
};
pub use error::{ContextError, Result};
pub use event::{
    ContextTemplate, InheritanceChain, InheritanceNode, OverrideAction, OverrideRule,
    TemplateEngine, TemplateEntry,
};
pub use event_store::EventMetadata;
pub use event_store::{
    Bridge, CorrelationId, Envelope, EventKind, EventMesh, EventMeshBuilder, EventSet, EventStore,
    EventTypeId, FlowChannel, FlowHandle, FlowReceiver, JsonlStore, JsonlStoreOptions, MemoryStore,
    ReplayFilter, ReplayId, SegmentedStore, SegmentedStoreOptions, SerializedEnvelope,
    Subscription, Topic, TopicPattern, TypeRegistry, TypedSubscription,
};
pub use lifecycle::{
    AccessEvent, AccessOutcome, EbbinghausModel, ForgettingModel, ImportanceScore,
    ImportanceWeights, LifecycleAction, LifecycleEngine, LifecycleRule, TokenBudget,
};
pub use llm::{
    CachingLlmClient, CachingLlmClientConfig, CascadeLlmClient, CascadeLlmConfig, EmbeddingVector,
    JsonSchema, LlmClient, LlmError, LlmOpts, LlmStream, PromptOptimizingLlmClient,
};
pub use lsh::LshIndex;
pub use model::{
    BlobRef, ConsolidationMeta, ConsolidationStatus, ContentHash, ContentIndexProjection,
    ContentLevel, ContentPart, ContentPayload, ContentType, ContextDiff, ContextEntry, ContextMeta,
    DecodedContent, DerivationChain, DerivationRule, DirEntry, EpistemicType, FindPattern, GrepHit,
    HalfLife, LineageEntry, MediaType, MvccVersion, SchemaRef, StateScope, TenantId, TreeNode,
    ValidityRecord, VersionEntry,
};
pub use observability::{
    MetricsExporter, MetricsExporterConfig, ProvenanceEdge, ProvenanceGraph, ProvenanceNode,
    ProvenanceRelationType, QualityDimension, QualityScore, QualityScorer,
    install_metrics_exporter, install_metrics_recorder,
};
pub use pack::{
    AclProtectedStore, AclRule, ContextPack, PackMeta, PackSignature, PackTrustPolicy, PathAcl,
    Permissions, Principal,
};
pub use prompt::{
    LlmTaskKind, OptimizedPrompt, PromptCacheMode, PromptCompressionMode, PromptOptimization,
    optimize_prompt,
};
pub use read_cache::{MemoryReadCache, ReadCache};
pub use similarity::{
    Cluster, CrossAgentDedup, CrossAgentSimilarityConfig, DedupRecommendation, KnowledgeNetwork,
    LocalKnowledgeNetwork, LocalKnowledgeNetworkConfig, SimilarityResult, VectorSimilarity,
};
pub use store::{
    BlobStore, BrowsingOps, ContentRepo, ContentStore, ContextStore, FsOps, GraphRelation,
    GraphStore, MAX_PAGE_SIZE, Page, PageRequest, StorageEngine, TenantOps, VersionOps,
};
pub use tokenizer::{count_tokens, count_tokens_with_floor};
pub use uri::{AsOfTime, ContextUri, QueryParams, SCHEME, UriCategory};
pub use vector::{IndexHit, IndexPoint, IndexVector, VectorIndex};
pub use watch::{
    ChangeEvent, ChangeKind, WatchCheckpoint, WatchHub, WatchOptions, WatchSource, WatchStream,
    WatchableStore,
};
pub use write_dedup::{SemanticWriteDedupConfig, SemanticWriteDedupStore, WriteDedupDecision};
pub use write_security::{
    SensitiveFinding, SensitiveKind, redact_sensitive_entry, sanitize_entry_for_write,
    scan_sensitive_entry,
};
