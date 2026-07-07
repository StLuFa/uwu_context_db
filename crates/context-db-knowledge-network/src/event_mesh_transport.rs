use crate::planner::MeshQueryPlan;
use crate::transport::MeshTransport;
use crate::types::{
    FetchResponse, KnowledgeNetworkError, PrivateQuerySketch, ProbeResponse, Result,
};
use agent_context_db_marketplace_types::AgentId;
use async_trait::async_trait;
use serde::Serialize;
use std::sync::Arc;
use uwu_event_mesh::{Envelope, EventMesh, Topic, TypeId};

pub const KN_PROBE_TOPIC: &str = "knowledge.network.probe";
pub const KN_FETCH_TOPIC: &str = "knowledge.network.fetch";
pub const KN_GOSSIP_CAPABILITY_TOPIC: &str = "knowledge.network.capability.gossip";

#[derive(Clone)]
pub struct EventMeshMeshTransport {
    mesh: EventMesh,
    delegate: Arc<dyn MeshTransport>,
    source: AgentId,
}

impl EventMeshMeshTransport {
    pub fn new(mesh: EventMesh, delegate: Arc<dyn MeshTransport>, source: AgentId) -> Self {
        Self {
            mesh,
            delegate,
            source,
        }
    }

    async fn publish_event<T: Serialize>(&self, topic: &str, event: &T) -> Result<()> {
        let topic =
            Topic::new(topic).map_err(|err| KnowledgeNetworkError::Transport(err.to_string()))?;
        let payload = serde_json::to_value(event)
            .map_err(|err| KnowledgeNetworkError::Transport(err.to_string()))?;
        let mut env = Envelope::new(&topic, payload);
        env.type_id = Some(TypeId::new(
            "knowledge-network",
            event_type_name(topic.as_str()),
        ));
        env.source = Some(self.source.to_string());
        self.mesh
            .publish(env)
            .await
            .map_err(|err| KnowledgeNetworkError::Transport(err.to_string()))?;
        Ok(())
    }
}

fn event_type_name(topic: &str) -> &'static str {
    match topic {
        KN_PROBE_TOPIC => "probe",
        KN_FETCH_TOPIC => "fetch",
        KN_GOSSIP_CAPABILITY_TOPIC => "capability_gossip",
        _ => "mesh_event",
    }
}

#[derive(Debug, Clone, Serialize)]
struct ProbeEvent<'a> {
    peer: &'a AgentId,
    sketch: &'a PrivateQuerySketch,
    plan: &'a MeshQueryPlan,
}

#[derive(Debug, Clone, Serialize)]
struct FetchEvent<'a> {
    peer: &'a AgentId,
    sketch: &'a PrivateQuerySketch,
    plan: &'a MeshQueryPlan,
}

#[derive(Debug, Clone, Serialize)]
struct CapabilityGossipEvent<'a> {
    peer: &'a AgentId,
    payload_len: usize,
}

#[async_trait]
impl MeshTransport for EventMeshMeshTransport {
    async fn probe(
        &self,
        peer: &AgentId,
        sketch: &PrivateQuerySketch,
        plan: &MeshQueryPlan,
    ) -> Result<ProbeResponse> {
        self.publish_event(KN_PROBE_TOPIC, &ProbeEvent { peer, sketch, plan })
            .await?;
        self.delegate.probe(peer, sketch, plan).await
    }

    async fn fetch(
        &self,
        peer: &AgentId,
        sketch: &PrivateQuerySketch,
        plan: &MeshQueryPlan,
    ) -> Result<FetchResponse> {
        self.publish_event(KN_FETCH_TOPIC, &FetchEvent { peer, sketch, plan })
            .await?;
        self.delegate.fetch(peer, sketch, plan).await
    }

    async fn gossip_capability(&self, peer: &AgentId, payload: Vec<u8>) -> Result<()> {
        self.publish_event(
            KN_GOSSIP_CAPABILITY_TOPIC,
            &CapabilityGossipEvent {
                peer,
                payload_len: payload.len(),
            },
        )
        .await?;
        self.delegate.gossip_capability(peer, payload).await
    }
}
