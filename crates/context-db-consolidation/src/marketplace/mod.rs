//! # Knowledge Marketplace — Agent-to-Agent 知识市场
//!
//! 核心机制：C2D硬边界 / 联邦注册表 / KPI声誉 / 六阶段工作流 / EventBus消息总线 / 三注册分离。

pub mod types;
pub mod registry;
pub mod publish;
pub mod discovery;
pub mod feedback;
pub mod conflict;
pub mod immune;
pub mod consensus;
pub mod voting;
pub mod crdt;
pub mod community;
pub mod influence;
pub mod phylogeny;
pub mod cap;

pub use types::*;
pub use registry::FederatedRegistry;
pub use publish::PublishGate;
pub use discovery::DiscoveryEngine;
pub use feedback::{ReputationEngine, MarketFeedback};
pub use conflict::ConflictResolver;
pub use immune::ImmuneProtocol;
pub use consensus::ConsensusTracker;
pub use voting::SocialVoter;
pub use crdt::SemanticCrdtMerger;
pub use community::{CommunityDetector, SpeciationTracker};
pub use influence::InfluenceAnalyzer;
pub use phylogeny::CrossAgentPhylogeny;
pub use cap::{CapPolicyEngine, ConsistencyLevel};
