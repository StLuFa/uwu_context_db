use crate::privacy::{DpPolicy, PrivacyCost};
use crate::trust::PeerRouteScore;
use crate::types::{MeshDiscoveryOpts, PrivateQuerySketch, Result};
use agent_context_db_marketplace_types::AgentId;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshQueryPlan {
    pub query_id: Uuid,
    pub probe_peers: Vec<AgentId>,
    pub fetch_peers: Vec<AgentId>,
    pub deadline_ms: u64,
    pub top_k_per_peer: usize,
    pub final_top_k: usize,
    pub privacy_cost: PrivacyCost,
}

#[derive(Debug, Clone)]
pub struct PlanningContext {
    pub opts: MeshDiscoveryOpts,
    pub route_scores: Vec<PeerRouteScore>,
    pub dp_policy: DpPolicy,
}

#[async_trait]
pub trait MeshQueryPlanner: Send + Sync {
    async fn plan(&self, query: &PrivateQuerySketch, ctx: PlanningContext)
    -> Result<MeshQueryPlan>;
}

#[derive(Default)]
pub struct DefaultMeshQueryPlanner;

#[async_trait]
impl MeshQueryPlanner for DefaultMeshQueryPlanner {
    async fn plan(
        &self,
        _query: &PrivateQuerySketch,
        ctx: PlanningContext,
    ) -> Result<MeshQueryPlan> {
        let probe_peers = ctx
            .route_scores
            .iter()
            .take(ctx.opts.probe_peers)
            .map(|s| s.peer.clone())
            .collect::<Vec<_>>();
        let fetch_peers = ctx
            .route_scores
            .iter()
            .take(ctx.opts.fetch_peers)
            .map(|s| s.peer.clone())
            .collect::<Vec<_>>();
        Ok(MeshQueryPlan {
            query_id: Uuid::new_v4(),
            probe_peers,
            fetch_peers,
            deadline_ms: ctx.opts.deadline_ms,
            top_k_per_peer: (ctx.opts.final_top_k / ctx.opts.fetch_peers.max(1)).max(1),
            final_top_k: ctx.opts.final_top_k,
            privacy_cost: PrivacyCost::for_query(&ctx.dp_policy),
        })
    }
}
