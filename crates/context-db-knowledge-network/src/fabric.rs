use crate::access::AccessGrantManager;
use crate::aggregation::StreamingTopKAggregator;
use crate::capability::CapabilityIndex;
use crate::corroboration::CorroborationGraph;
use crate::governance::GovernanceEngine;
use crate::identity::IdentityRegistry;
use crate::intent::QueryIntentClassifier;
use crate::learning::{RouteOutcome, RouteOutcomeLearning};
use crate::persistence::{InMemoryKnowledgeNetworkPersistence, KnowledgeNetworkPersistence};
use crate::planner::{DefaultMeshQueryPlanner, MeshQueryPlanner, PlanningContext};
use crate::privacy::{DpPolicy, InMemoryPrivacyBudgetLedger, PrivacyGuard, has_k_anonymity};
use crate::semantic_graph::SemanticCapabilityGraph;
use crate::topology::TopologyOptimizer;
use crate::transport::MeshTransport;
use crate::trust::TrustRouter;
use crate::types::{MeshDiscoveryOpts, MeshResultPhase, ProgressiveMeshResult, Result};
use agent_context_db_marketplace_types::{
    AgentId, DiscoveryQuery, FederatedDiscoveryResult, SearchTier,
};
use chrono::Utc;
use std::sync::Arc;

pub struct FederatedKnowledgeFabric {
    pub local_agent: AgentId,
    pub capability_index: Arc<dyn CapabilityIndex>,
    pub planner: Arc<dyn MeshQueryPlanner>,
    pub transport: Arc<dyn MeshTransport>,
    pub privacy: Arc<PrivacyGuard>,
    pub trust_router: Arc<TrustRouter>,
    pub aggregator: Arc<StreamingTopKAggregator>,
    pub intent_classifier: Arc<QueryIntentClassifier>,
    pub semantic_graph: Arc<SemanticCapabilityGraph>,
    pub corroboration_graph: Arc<CorroborationGraph>,
    pub route_learning: Arc<RouteOutcomeLearning>,
    pub topology_optimizer: Arc<TopologyOptimizer>,
    pub governance: Arc<GovernanceEngine>,
    pub access_grants: Arc<AccessGrantManager>,
    pub identities: Arc<IdentityRegistry>,
    pub persistence: Arc<dyn KnowledgeNetworkPersistence>,
}

impl FederatedKnowledgeFabric {
    pub fn new(
        local_agent: AgentId,
        capability_index: Arc<dyn CapabilityIndex>,
        transport: Arc<dyn MeshTransport>,
    ) -> Self {
        let dp_policy = DpPolicy::default();
        Self {
            local_agent,
            capability_index,
            planner: Arc::new(DefaultMeshQueryPlanner),
            transport,
            privacy: Arc::new(PrivacyGuard::new(
                dp_policy,
                Arc::new(InMemoryPrivacyBudgetLedger::default()),
            )),
            trust_router: Arc::new(TrustRouter::default()),
            aggregator: Arc::new(StreamingTopKAggregator::default()),
            intent_classifier: Arc::new(QueryIntentClassifier::default()),
            semantic_graph: Arc::new(SemanticCapabilityGraph::default()),
            corroboration_graph: Arc::new(CorroborationGraph::default()),
            route_learning: Arc::new(RouteOutcomeLearning::default()),
            topology_optimizer: Arc::new(TopologyOptimizer::default()),
            governance: Arc::new(GovernanceEngine::default()),
            access_grants: Arc::new(AccessGrantManager::default()),
            identities: Arc::new(IdentityRegistry::default()),
            persistence: Arc::new(InMemoryKnowledgeNetworkPersistence::default()),
        }
    }

    pub fn with_governance(mut self, governance: Arc<GovernanceEngine>) -> Self {
        self.governance = governance;
        self
    }

    pub fn with_persistence(mut self, persistence: Arc<dyn KnowledgeNetworkPersistence>) -> Self {
        self.persistence = persistence;
        self
    }

    pub fn with_intelligence(
        mut self,
        intent_classifier: Arc<QueryIntentClassifier>,
        semantic_graph: Arc<SemanticCapabilityGraph>,
        corroboration_graph: Arc<CorroborationGraph>,
        route_learning: Arc<RouteOutcomeLearning>,
        topology_optimizer: Arc<TopologyOptimizer>,
    ) -> Self {
        self.intent_classifier = intent_classifier;
        self.semantic_graph = semantic_graph;
        self.corroboration_graph = corroboration_graph;
        self.route_learning = route_learning;
        self.topology_optimizer = topology_optimizer;
        self
    }

    pub async fn discover(
        &self,
        query: DiscoveryQuery,
        opts: MeshDiscoveryOpts,
    ) -> Result<ProgressiveMeshResult> {
        let opts = self.intent_classifier.tune_opts(&query, opts);
        self.governance.authorize_query(
            &self.local_agent,
            &query,
            opts.intent,
            &self.privacy.policy,
        )?;
        if self.access_grants.has_grants() {
            self.access_grants.authorize(
                &self.local_agent,
                &query.domains,
                self.privacy.policy.query_epsilon,
            )?;
        }

        let (sketch, receipt) = self
            .privacy
            .protect_query(&self.local_agent, &query)
            .await?;
        let mut candidates = self
            .capability_index
            .candidate_peers(&sketch, opts.max_peers)
            .await?;
        self.semantic_graph
            .boost_candidates(&query.domains, &sketch, &mut candidates);
        if !has_k_anonymity(candidates.len(), &self.privacy.policy) {
            self.privacy.budget_ledger.refund(&receipt).await?;
            return Ok(ProgressiveMeshResult {
                phase: MeshResultPhase::PartialFast,
                hits: Vec::new(),
                confidence: 0.0,
                missing_peer_count: self
                    .privacy
                    .policy
                    .min_k_anonymity
                    .saturating_sub(candidates.len()),
                privacy_receipts: vec![receipt],
            });
        }
        let mut route_scores = self
            .trust_router
            .route(candidates, opts.probe_peers.max(opts.fetch_peers));
        for score in &mut route_scores {
            score.final_score =
                (score.final_score + self.route_learning.route_bonus(&score.peer)).clamp(0.0, 1.0);
        }
        route_scores.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let plan = self
            .planner
            .plan(
                &sketch,
                PlanningContext {
                    opts: opts.clone(),
                    route_scores,
                    dp_policy: self.privacy.policy.clone(),
                },
            )
            .await?;
        self.governance.authorize_plan(&plan)?;

        let mut probe_peers = plan.probe_peers.clone();
        self.topology_optimizer.optimize_peer_order(
            &mut probe_peers,
            opts.intent,
            &self.route_learning,
        );

        let mut responsive = Vec::new();
        for peer in &probe_peers {
            let started = Utc::now();
            let probe = self.transport.probe(peer, &sketch, &plan).await?;
            let latency_ms = (Utc::now() - started).num_milliseconds().max(0) as u64;
            let success = probe.noisy_match_count > 0 || probe.noisy_max_score > 0.0;
            self.route_learning.record(RouteOutcome {
                peer: peer.clone(),
                success,
                latency_ms: latency_ms.max(probe.expected_latency_ms),
                hit_count: probe.noisy_match_count as usize,
                avg_score: probe.noisy_max_score,
                observed_at: Utc::now(),
            });
            if success {
                responsive.push(peer.clone());
            }
        }
        if responsive.is_empty() {
            self.privacy.budget_ledger.refund(&receipt).await?;
            return Ok(ProgressiveMeshResult {
                phase: MeshResultPhase::PartialFast,
                hits: Vec::new(),
                confidence: 0.0,
                missing_peer_count: plan.probe_peers.len(),
                privacy_receipts: vec![receipt],
            });
        }

        let mut fetch_peers = plan
            .fetch_peers
            .iter()
            .filter(|peer| responsive.contains(peer))
            .cloned()
            .collect::<Vec<_>>();
        self.topology_optimizer.optimize_peer_order(
            &mut fetch_peers,
            opts.intent,
            &self.route_learning,
        );

        let mut hits = Vec::new();
        for peer in &fetch_peers {
            let started = Utc::now();
            let response = self.transport.fetch(peer, &sketch, &plan).await?;
            let latency_ms = (Utc::now() - started).num_milliseconds().max(0) as u64;
            let avg_score = if response.hits.is_empty() {
                0.0
            } else {
                response.hits.iter().map(|hit| hit.final_score).sum::<f32>()
                    / response.hits.len() as f32
            };
            self.route_learning.record(RouteOutcome {
                peer: peer.clone(),
                success: !response.hits.is_empty(),
                latency_ms,
                hit_count: response.hits.len(),
                avg_score,
                observed_at: Utc::now(),
            });
            hits.extend(response.hits);
        }
        self.corroboration_graph.observe_hits(&hits);
        self.corroboration_graph.enrich_scores(&mut hits);
        let hits = self.governance.filter_hits(hits);
        let hits = self.aggregator.merge(hits, plan.final_top_k);
        let confidence = if hits.is_empty() {
            0.0
        } else {
            (hits.len() as f32 / plan.final_top_k.max(1) as f32).clamp(0.0, 1.0)
        };
        self.privacy.budget_ledger.commit(&receipt).await?;
        self.persistence
            .record_budget_receipt(receipt.clone())
            .await?;
        for (peer, state) in self.route_learning.snapshot() {
            self.persistence.record_route_state(peer, state).await?;
        }
        Ok(ProgressiveMeshResult {
            phase: MeshResultPhase::PartialFast,
            hits,
            confidence,
            missing_peer_count: plan.fetch_peers.len().saturating_sub(fetch_peers.len()),
            privacy_receipts: vec![receipt],
        })
    }

    pub async fn discover_result(
        &self,
        query: DiscoveryQuery,
        opts: MeshDiscoveryOpts,
    ) -> Result<FederatedDiscoveryResult> {
        let result = self.discover(query, opts).await?;
        let total_available = result.hits.len();
        let avg_quality = if result.hits.is_empty() {
            0.0
        } else {
            result
                .hits
                .iter()
                .map(|h| h.publication.quality_score)
                .sum::<f32>()
                / result.hits.len() as f32
        };
        let mut domains_covered = result
            .hits
            .iter()
            .map(|h| h.publication.domain.clone())
            .collect::<Vec<_>>();
        domains_covered.sort();
        domains_covered.dedup();
        Ok(FederatedDiscoveryResult {
            hits: result.hits,
            total_available,
            domains_covered,
            avg_quality,
            search_tier: SearchTier::Federation,
        })
    }
}
