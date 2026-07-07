use crate::capability::CapabilitySketch;
use crate::learning::RouteLearningState;
use crate::privacy::{BudgetScope, PrivacyCost, PrivacyReceipt};
use crate::types::{KnowledgeNetworkError, Result};
use agent_context_db_marketplace_types::AgentId;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;

#[async_trait]
pub trait KnowledgeNetworkPersistence: Send + Sync {
    async fn put_capability(&self, sketch: CapabilitySketch) -> Result<()>;
    async fn get_capability(&self, peer: &AgentId) -> Result<Option<CapabilitySketch>>;
    async fn record_budget_receipt(&self, receipt: PrivacyReceipt) -> Result<()>;
    async fn record_route_state(&self, peer: AgentId, state: RouteLearningState) -> Result<()>;
}

#[derive(Default)]
pub struct InMemoryKnowledgeNetworkPersistence {
    capabilities: RwLock<HashMap<AgentId, CapabilitySketch>>,
    receipts: RwLock<HashMap<uuid::Uuid, PrivacyReceipt>>,
    route_states: RwLock<HashMap<AgentId, RouteLearningState>>,
}

#[async_trait]
impl KnowledgeNetworkPersistence for InMemoryKnowledgeNetworkPersistence {
    async fn put_capability(&self, sketch: CapabilitySketch) -> Result<()> {
        self.capabilities
            .write()
            .insert(sketch.peer.clone(), sketch);
        Ok(())
    }

    async fn get_capability(&self, peer: &AgentId) -> Result<Option<CapabilitySketch>> {
        Ok(self.capabilities.read().get(peer).cloned())
    }

    async fn record_budget_receipt(&self, receipt: PrivacyReceipt) -> Result<()> {
        self.receipts.write().insert(receipt.id, receipt);
        Ok(())
    }

    async fn record_route_state(&self, peer: AgentId, state: RouteLearningState) -> Result<()> {
        self.route_states.write().insert(peer, state);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PersistentBudgetCharge {
    pub actor: AgentId,
    pub scope: BudgetScope,
    pub cost: PrivacyCost,
}

impl PersistentBudgetCharge {
    pub fn validate(&self) -> Result<()> {
        let total = self.cost.query_epsilon + self.cost.response_epsilon + self.cost.relay_epsilon;
        if total.is_finite() && total >= 0.0 {
            Ok(())
        } else {
            Err(KnowledgeNetworkError::PrivacyBudgetExhausted(
                "invalid privacy charge".into(),
            ))
        }
    }
}
