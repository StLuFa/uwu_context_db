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
use crate::topology::{PeerConcurrencyLimits, TopologyOptimizer};
use crate::transport::MeshTransport;
use crate::trust::TrustRouter;
use crate::types::{MeshDiscoveryOpts, MeshResultPhase, ProgressiveMeshResult, Result};
use agent_context_db_marketplace::{
    AgentId, DiscoveryQuery, FederatedDiscoveryBackend, FederatedDiscoveryResult, SearchTier,
};
use async_trait::async_trait;
use chrono::Utc;
use futures::{StreamExt, stream};
use std::sync::Arc;
use tokio::time::{Duration, Instant, timeout_at};

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
    pub peer_limits: Option<Arc<PeerConcurrencyLimits>>,
    pub governance: Arc<GovernanceEngine>,
    pub access_grants: Arc<AccessGrantManager>,
    pub identities: Arc<IdentityRegistry>,
    pub persistence: Arc<dyn KnowledgeNetworkPersistence>,
    pub reaction_sink: Option<Arc<dyn agent_context_db_core::ReactionSink>>,
    pub route_update: agent_context_db_core::OnlineUpdateConfig,
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
            peer_limits: None,
            governance: Arc::new(GovernanceEngine::default()),
            access_grants: Arc::new(AccessGrantManager::default()),
            identities: Arc::new(IdentityRegistry::default()),
            persistence: Arc::new(InMemoryKnowledgeNetworkPersistence::default()),
            reaction_sink: None,
            route_update: agent_context_db_core::OnlineUpdateConfig::default(),
        }
    }

    pub fn with_governance(mut self, governance: Arc<GovernanceEngine>) -> Self {
        self.governance = governance;
        self
    }

    pub fn with_peer_limits(mut self, limits: Arc<PeerConcurrencyLimits>) -> Self {
        self.peer_limits = Some(limits);
        self
    }

    pub fn with_persistence(mut self, persistence: Arc<dyn KnowledgeNetworkPersistence>) -> Self {
        self.persistence = persistence;
        self
    }

    pub fn with_reaction_sink(
        mut self,
        sink: Arc<dyn agent_context_db_core::ReactionSink>,
        route_update: agent_context_db_core::OnlineUpdateConfig,
    ) -> Self {
        self.reaction_sink = Some(sink);
        self.route_update = route_update;
        self
    }

    fn record_route_outcome(&self, outcome: RouteOutcome, execution_id: &str) {
        if let Some(sink) = &self.reaction_sink {
            sink.emit(outcome.clone().into_reaction(execution_id));
        }
        self.route_learning.record(outcome);
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
        opts.validate()?;
        let _query_permit = self.governance.acquire_query(&self.local_agent)?;
        let deadline = Instant::now() + Duration::from_millis(opts.deadline_ms);
        let opts = self.intent_classifier.tune_opts(&query, opts);
        opts.validate()?;
        self.governance.authorize_query(
            &self.local_agent,
            &query,
            opts.intent,
            &self.privacy.policy,
        )?;
        self.access_grants.authorize(
            &self.local_agent,
            &query.domains,
            self.privacy.policy.query_epsilon,
        )?;

        let (sketch, reservation) = timeout_at(
            deadline,
            self.privacy.protect_query(&self.local_agent, &query),
        )
        .await
        .map_err(|_| {
            crate::types::KnowledgeNetworkError::Transport(
                "federated discovery deadline exceeded during privacy protection".into(),
            )
        })??;
        let mut candidates = match timeout_at(
            deadline,
            self.capability_index
                .candidate_peers(&sketch, opts.max_peers),
        )
        .await
        {
            Ok(Ok(candidates)) => candidates,
            Ok(Err(error)) => return Err(reservation.refund_after_error(error).await),
            Err(_) => {
                let error = crate::types::KnowledgeNetworkError::Transport(
                    "federated discovery deadline exceeded during candidate discovery".into(),
                );
                return Err(reservation.refund_after_error(error).await);
            }
        };
        self.semantic_graph
            .boost_candidates(&query.domains, &sketch, &mut candidates);
        if !has_k_anonymity(candidates.len(), &self.privacy.policy) {
            let receipt = reservation.refund().await?;
            return Ok(ProgressiveMeshResult {
                phase: MeshResultPhase::PartialFast,
                hits: Vec::new(),
                confidence: 0.0,
                missing_peer_count: self
                    .privacy
                    .policy
                    .min_k_anonymity
                    .saturating_sub(candidates.len()),
                failed_peer_count: 0,
                timed_out_peer_count: 0,
                cancelled_peer_count: 0,
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
        let plan = match timeout_at(
            deadline,
            self.planner.plan(
                &sketch,
                PlanningContext {
                    opts: opts.clone(),
                    route_scores,
                    dp_policy: self.privacy.policy.clone(),
                },
            ),
        )
        .await
        {
            Ok(Ok(plan)) => plan,
            Ok(Err(error)) => return Err(reservation.refund_after_error(error).await),
            Err(_) => {
                let error = crate::types::KnowledgeNetworkError::Planner(
                    "federated discovery deadline exceeded during planning".into(),
                );
                return Err(reservation.refund_after_error(error).await);
            }
        };
        if let Err(error) = self.governance.authorize_plan(&plan) {
            return Err(reservation.refund_after_error(error).await);
        }

        let mut probe_peers = plan.probe_peers.clone();
        self.topology_optimizer.optimize_peer_order(
            &mut probe_peers,
            opts.intent,
            &self.route_learning,
        );

        let peer_timeout = Duration::from_millis(opts.peer_timeout_ms);
        let probe_total = probe_peers.len();
        let mut probe_stream = stream::iter(probe_peers.into_iter().map(|peer| {
            let transport = Arc::clone(&self.transport);
            let sketch = sketch.clone();
            let plan = plan.clone();
            let peer_limits = self.peer_limits.clone();
            async move {
                let started = Instant::now();
                let peer_deadline = deadline.min(Instant::now() + peer_timeout);
                let result = timeout_at(peer_deadline, async {
                    let _permit = match peer_limits {
                        Some(limits) => Some(limits.acquire(&peer).await?),
                        None => None,
                    };
                    transport.probe(&peer, &sketch, &plan).await
                })
                .await;
                (peer, started.elapsed(), result)
            }
        }))
        .buffer_unordered(opts.max_concurrency);
        let mut responsive = Vec::new();
        let mut failed_peer_count = 0usize;
        let mut timed_out_peer_count = 0usize;
        let mut completed_probes = 0usize;
        while let Some((peer, elapsed, result)) = probe_stream.next().await {
            completed_probes += 1;
            match result {
                Ok(Ok(probe)) => {
                    let success = probe.noisy_match_count > 0 || probe.noisy_max_score > 0.0;
                    self.record_route_outcome(
                        RouteOutcome {
                            peer: peer.clone(),
                            success,
                            latency_ms: elapsed.as_millis() as u64,
                            hit_count: probe.noisy_match_count as usize,
                            avg_score: probe.noisy_max_score,
                            observed_at: Utc::now(),
                        },
                        &format!("mesh:{}", self.local_agent),
                    );
                    if success {
                        responsive.push(peer);
                    }
                }
                Ok(Err(_)) => failed_peer_count += 1,
                Err(_) => timed_out_peer_count += 1,
            }
            if Instant::now() >= deadline {
                break;
            }
        }
        let mut cancelled_peer_count = probe_total.saturating_sub(completed_probes);
        drop(probe_stream);
        if responsive.is_empty() {
            let receipt = reservation.refund().await?;
            return Ok(ProgressiveMeshResult {
                phase: MeshResultPhase::PartialFast,
                hits: Vec::new(),
                confidence: 0.0,
                missing_peer_count: plan.probe_peers.len(),
                failed_peer_count,
                timed_out_peer_count,
                cancelled_peer_count,
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

        let fetch_total = fetch_peers.len();
        let mut fetch_stream = stream::iter(fetch_peers.iter().cloned().map(|peer| {
            let transport = Arc::clone(&self.transport);
            let sketch = sketch.clone();
            let plan = plan.clone();
            let peer_limits = self.peer_limits.clone();
            async move {
                let started = Instant::now();
                let peer_deadline = deadline.min(Instant::now() + peer_timeout);
                let result = timeout_at(peer_deadline, async {
                    let _permit = match peer_limits {
                        Some(limits) => Some(limits.acquire(&peer).await?),
                        None => None,
                    };
                    transport.fetch(&peer, &sketch, &plan).await
                })
                .await;
                (peer, started.elapsed(), result)
            }
        }))
        .buffer_unordered(opts.max_concurrency);
        let mut hits = Vec::new();
        let mut completed_fetches = 0usize;
        let mut successful_fetches = 0usize;
        while let Some((peer, elapsed, result)) = fetch_stream.next().await {
            completed_fetches += 1;
            match result {
                Ok(Ok(response)) => {
                    successful_fetches += 1;
                    let avg_score = if response.hits.is_empty() {
                        0.0
                    } else {
                        response.hits.iter().map(|hit| hit.final_score).sum::<f32>()
                            / response.hits.len() as f32
                    };
                    self.record_route_outcome(
                        RouteOutcome {
                            peer,
                            success: !response.hits.is_empty(),
                            latency_ms: elapsed.as_millis() as u64,
                            hit_count: response.hits.len(),
                            avg_score,
                            observed_at: Utc::now(),
                        },
                        &format!("mesh:{}", self.local_agent),
                    );
                    hits.extend(response.hits);
                }
                Ok(Err(_)) => failed_peer_count += 1,
                Err(_) => timed_out_peer_count += 1,
            }
            if Instant::now() >= deadline {
                break;
            }
        }
        cancelled_peer_count += fetch_total.saturating_sub(completed_fetches);
        drop(fetch_stream);
        self.corroboration_graph.observe_hits(&hits);
        self.corroboration_graph.enrich_scores(&mut hits);
        let hits = self.governance.filter_hits(hits);
        let hits = self.aggregator.merge(hits, plan.final_top_k);
        let confidence = if hits.is_empty() {
            0.0
        } else {
            (hits.len() as f32 / plan.final_top_k.max(1) as f32).clamp(0.0, 1.0)
        };
        // Persist all fallible result state before committing the non-refundable charge.
        for (peer, state) in self.route_learning.snapshot() {
            match timeout_at(deadline, self.persistence.record_route_state(peer, state)).await {
                Ok(Ok(())) => {}
                Ok(Err(primary)) => return Err(reservation.refund_after_error(primary).await),
                Err(_) => {
                    let error = crate::types::KnowledgeNetworkError::Transport(
                        "federated discovery deadline exceeded while persisting route state".into(),
                    );
                    return Err(reservation.refund_after_error(error).await);
                }
            }
        }
        let receipt = reservation.receipt()?.clone();
        match timeout_at(deadline, self.persistence.record_budget_receipt(receipt)).await {
            Ok(Ok(())) => {}
            Ok(Err(primary)) => return Err(reservation.refund_after_error(primary).await),
            Err(_) => {
                let error = crate::types::KnowledgeNetworkError::Transport(
                    "federated discovery deadline exceeded while persisting budget receipt".into(),
                );
                return Err(reservation.refund_after_error(error).await);
            }
        }
        let receipt = reservation.commit().await?;
        Ok(ProgressiveMeshResult {
            phase: MeshResultPhase::PartialFast,
            hits,
            confidence,
            missing_peer_count: plan.fetch_peers.len().saturating_sub(successful_fetches),
            failed_peer_count,
            timed_out_peer_count,
            cancelled_peer_count,
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

#[async_trait]
impl FederatedDiscoveryBackend for FederatedKnowledgeFabric {
    async fn discover_federated(
        &self,
        query: DiscoveryQuery,
        limit: usize,
    ) -> agent_context_db_core::Result<FederatedDiscoveryResult> {
        let opts = MeshDiscoveryOpts {
            final_top_k: limit,
            ..Default::default()
        };
        self.discover_result(query, opts).await.map_err(|err| {
            agent_context_db_core::ContextError::Unsupported(format!("knowledge network: {err}"))
        })
    }
}
