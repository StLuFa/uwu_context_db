//! # agent-context-db-marketplace — Agent-to-Agent 知识市场
//!
//! 从 `agent-context-db-consolidation` 拆分出的联邦流通层。
//!
//! 核心机制：C2D 硬边界 / 联邦注册表 / KPI 声誉 / 六阶段工作流 /
//! EventMesh 消息总线 / 三注册分离。
//!
//! 单 Agent 侧的巩固（`ConsolidationProduct` 等）仍在 consolidation crate，
//! 本 crate 消费其巩固产物，向外发布并做跨 agent 联邦流通。

pub mod cap;
pub mod community;
pub mod conflict;
pub mod consensus;
pub mod crdt;
pub mod discovery;
pub mod feedback;
pub mod immune;
pub mod influence;
pub mod memetic;
pub mod phylogeny;
pub mod publish;
pub mod registry;
pub mod secure_aggregation;
pub mod types;
pub mod voting;

pub use cap::{CapPolicyEngine, ConsistencyLevel};
pub use community::{CommunityDetector, SpeciationEvent, SpeciationFork, SpeciationTracker};
pub use conflict::{ConflictResolver, MarketConflictResolution};
pub use consensus::ConsensusTracker;
pub use crdt::{CrdtMergeStrategy, SemanticCrdtMerger};
pub use discovery::{DiscoveryEngine, FederatedDiscoveryBackend};
pub use feedback::{
    FeedbackError, FeedbackPayload, FeedbackRegistry, FeedbackSignatureVerifier, MarketFeedback,
    ReputationEngine,
};
pub use immune::ImmuneProtocol;
pub use influence::InfluenceAnalyzer;
pub use memetic::{
    EvolutionAction, EvolutionCandidate, EvolutionOffspring, EvolutionRunReport, FitnessScore,
    FitnessSignals, MemeticEvolutionConfig, MemeticEvolutionEngine, OffspringValidationStatus,
};
pub use phylogeny::CrossAgentPhylogeny;
pub use publish::{KnowledgeSigner, PublishGate, PublishableProduct};
pub use registry::FederatedRegistry;
pub use secure_aggregation::{
    ContributionCommitment, DpBudget, HashProvenanceVerifier, PrivateContribution,
    ProvenanceVerifier, SecretShare, SecureAggregateReport, SecureAggregationEngine,
    SecureAggregationError, SecureAggregationRejection, SharedContribution,
};
pub use types::*;
pub use voting::{
    SignedVote, SocialVoter, VoteError, VoteOp, VotePayload, VoteReputationPolicy,
    VoteSignatureVerifier, VoteTally,
};
