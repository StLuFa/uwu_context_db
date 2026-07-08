//! DiscoveryEngine — 三级搜索 + 声誉排序。

use crate::marketplace::registry::FederatedRegistry;
use crate::marketplace::types::*;
use agent_context_db_core::{ContextError, Result};
use async_trait::async_trait;
use std::sync::Arc;

/// 发现查询。
#[derive(Debug, Clone)]
pub struct DiscoveryQuery {
    pub query_embedding: Vec<f32>,
    pub domains: Vec<String>,
    pub entry_types: Vec<MarketEntryType>,
    pub min_quality: f32,
    pub min_corroboration_level: CorroborationLevel,
    pub license_compatible: bool,
}

impl Default for DiscoveryQuery {
    fn default() -> Self {
        Self {
            query_embedding: vec![],
            domains: vec![],
            entry_types: vec![],
            min_quality: 0.7,
            min_corroboration_level: CorroborationLevel::CrossSession,
            license_compatible: true,
        }
    }
}

/// 发现结果。
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    pub hits: Vec<MarketHit>,
    pub total_available: usize,
    pub domains_covered: Vec<String>,
    pub avg_quality: f32,
    pub search_tier: SearchTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTier {
    Local,
    Cache,
    Federation,
}

#[derive(Debug, Clone)]
pub struct MarketHit {
    pub entry: MarketEntry,
    pub relevance: f32,
    pub reputation_bonus: f32,
    pub final_score: f32,
}

#[async_trait]
pub trait FederatedDiscoveryBackend: Send + Sync {
    async fn discover_federated(
        &self,
        query: agent_context_db_marketplace_types::DiscoveryQuery,
        limit: usize,
    ) -> Result<agent_context_db_marketplace_types::FederatedDiscoveryResult>;
}

#[async_trait]
impl FederatedDiscoveryBackend for agent_context_db_knowledge_network::FederatedKnowledgeFabric {
    async fn discover_federated(
        &self,
        query: agent_context_db_marketplace_types::DiscoveryQuery,
        limit: usize,
    ) -> Result<agent_context_db_marketplace_types::FederatedDiscoveryResult> {
        let opts = agent_context_db_knowledge_network::MeshDiscoveryOpts {
            final_top_k: limit,
            ..Default::default()
        };
        self.discover_result(query, opts)
            .await
            .map_err(|e| ContextError::Unsupported(format!("knowledge network: {e}")))
    }
}

/// 发现引擎 — 三级搜索策略。
pub struct DiscoveryEngine {
    registry: Arc<FederatedRegistry>,
    federation: Option<Arc<dyn FederatedDiscoveryBackend>>,
    /// 联邦搜索结果的 TTL 缓存。
    cache: parking_lot::RwLock<std::collections::HashMap<String, Vec<MarketEntry>>>,
}

impl DiscoveryEngine {
    pub fn new(registry: Arc<FederatedRegistry>) -> Self {
        Self {
            registry,
            federation: None,
            cache: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn with_federation(mut self, federation: Arc<dyn FederatedDiscoveryBackend>) -> Self {
        self.federation = Some(federation);
        self
    }

    /// 三级搜索：本地 → 缓存 → 联邦。
    pub async fn search(&self, query: &DiscoveryQuery, limit: usize) -> DiscoveryResult {
        // === Tier 1: 本地向量索引 ===
        let local = self
            .registry
            .search_local(&query.query_embedding, limit * 2)
            .await;
        if local.len() >= limit {
            let hits = self.rank(local, query, limit);
            return DiscoveryResult {
                avg_quality: avg_quality(&hits),
                domains_covered: domains_from(&hits),
                total_available: hits.len(),
                hits,
                search_tier: SearchTier::Local,
            };
        }

        // === Tier 2: 缓存命中 ===
        let cache_key = format!("{:?}", query.domains);
        if let Some(cached) = self.cache.read().get(&cache_key) {
            let hits = self.rank(cached.clone(), query, limit);
            if hits.len() >= limit / 2 {
                return DiscoveryResult {
                    avg_quality: avg_quality(&hits),
                    domains_covered: domains_from(&hits),
                    total_available: hits.len(),
                    hits,
                    search_tier: SearchTier::Cache,
                };
            }
        }

        // === Tier 3: 联邦查询 ===
        let mut all_results = local;
        if let Some(federation) = &self.federation {
            let federated_query = agent_context_db_marketplace_types::DiscoveryQuery {
                query_embedding: query.query_embedding.clone(),
                domains: query.domains.clone(),
                entry_types: query.entry_types.clone(),
                min_quality: query.min_quality,
                min_corroboration_level: query.min_corroboration_level,
                license_compatible: query.license_compatible,
            };
            if let Ok(remote) = federation.discover_federated(federated_query, limit).await {
                all_results.extend(
                    remote
                        .hits
                        .into_iter()
                        .map(|hit| market_entry_from_publication(hit.publication)),
                );
            }
        }

        // 缓存结果

        if !all_results.is_empty() {
            self.cache.write().insert(cache_key, all_results.clone());
        }

        let hits = self.rank(all_results, query, limit);
        DiscoveryResult {
            avg_quality: avg_quality(&hits),
            domains_covered: domains_from(&hits),
            total_available: hits.len(),
            hits,
            search_tier: SearchTier::Federation,
        }
    }

    /// 声誉排序：relevance × (1 + reputation_bonus)。
    fn rank(
        &self,
        entries: Vec<MarketEntry>,
        query: &DiscoveryQuery,
        limit: usize,
    ) -> Vec<MarketHit> {
        let mut scored: Vec<MarketHit> = entries
            .into_iter()
            .filter(|e| {
                // 领域 + 类型 + 质量 + 确认过滤
                (query.domains.is_empty() || query.domains.contains(&e.domain))
                    && (query.entry_types.is_empty() || query.entry_types.contains(&e.entry_type))
                    && e.quality_score >= query.min_quality
                    && e.corroboration.level >= query.min_corroboration_level
            })
            .map(|e| {
                let reputation_bonus = 0.0; // 在完整实现中查询 ReputationEngine
                let relevance = e.quality_score;
                MarketHit {
                    final_score: relevance * (1.0 + reputation_bonus),
                    relevance,
                    reputation_bonus,
                    entry: e,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);
        scored
    }
}

fn market_entry_from_publication(publication: PublicationMetadata) -> MarketEntry {
    MarketEntry {
        id: publication.id,
        publisher: publication.publisher,
        domain: publication.domain,
        entry_type: publication.entry_type,
        principle: format!("federated-publication:{}", publication.id.0),
        evidence_uris: publication.source_uri.into_iter().collect(),
        quality_score: publication.quality_score,
        confidence: publication.corroboration.level as u8 as f32
            / CorroborationLevel::Established as u8 as f32,
        corroboration: publication.corroboration,
        provenance: publication.provenance,
        license: match publication.license.scope {
            LicenseScope::Open if !publication.license.attribution_required => {
                KnowledgeLicense::PublicDomain
            }
            LicenseScope::Open => KnowledgeLicense::Attribution,
            LicenseScope::TenantOnly => KnowledgeLicense::TenantOnly,
            LicenseScope::NonCommercial | LicenseScope::RequiresApproval => {
                KnowledgeLicense::RequiresApproval
            }
        },
        epistemic_type: publication.epistemic_type,
        content_type: publication.content_type,
        half_life_days: publication.half_life_days,
        created_at: publication.created_at,
        expires_at: publication.expires_at,
    }
}

fn avg_quality(hits: &[MarketHit]) -> f32 {
    if hits.is_empty() {
        return 0.0;
    }
    hits.iter().map(|h| h.entry.quality_score).sum::<f32>() / hits.len() as f32
}

fn domains_from(hits: &[MarketHit]) -> Vec<String> {
    let mut domains: Vec<String> = hits.iter().map(|h| h.entry.domain.clone()).collect();
    domains.dedup();
    domains
}
