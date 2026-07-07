//! # Knowledge Marketplace — Agent-to-Agent 知识市场
//!
//! 核心机制：C2D硬边界 / 联邦注册表 / KPI声誉 / 六阶段工作流 / EventMesh消息总线 / 三注册分离。

pub mod cap;
pub mod community;
pub mod conflict;
pub mod consensus;
pub mod crdt;
pub mod discovery;
pub mod feedback;
pub mod immune;
pub mod influence;
pub mod phylogeny;
pub mod publish;
pub mod registry;
pub mod types;
pub mod voting;

pub use cap::{CapPolicyEngine, ConsistencyLevel};
pub use community::{CommunityDetector, SpeciationTracker};
pub use conflict::ConflictResolver;
pub use consensus::ConsensusTracker;
pub use crdt::SemanticCrdtMerger;
pub use discovery::DiscoveryEngine;
pub use feedback::{MarketFeedback, ReputationEngine};
pub use immune::ImmuneProtocol;
pub use influence::InfluenceAnalyzer;
pub use phylogeny::CrossAgentPhylogeny;
pub use publish::PublishGate;
pub use registry::FederatedRegistry;
pub use types::*;
pub use voting::SocialVoter;
