//! DiscoveryEngine — 三级搜索 + 声誉排序。

use crate::marketplace::types::*;
use crate::marketplace::registry::FederatedRegistry;
use crate::marketplace::publish::PublishGate;
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

/// 发现引擎 — 三级搜索策略。
pub struct DiscoveryEngine {
    registry: Arc<FederatedRegistry>,
    /// 联邦搜索结果的 TTL 缓存。
    cache: parking_lot::RwLock<std::collections::HashMap<String, Vec<MarketEntry>>>,
}

impl DiscoveryEngine {
    pub fn new(registry: Arc<FederatedRegistry>) -> Self {
        Self { registry, cache: parking_lot::RwLock::new(std::collections::HashMap::new()) }
    }

    /// 三级搜索：本地 → 缓存 → 联邦。
    pub async fn search(&self, query: &DiscoveryQuery, limit: usize) -> DiscoveryResult {
        // === Tier 1: 本地向量索引 ===
        let local = self.registry.search_local(&query.query_embedding, limit * 2).await;
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
        //
        // TODO(federation-dp): 生产级联邦 + 差分隐私
        //
        // 当前实现是 skeleton：直接把本地 registry.my_publications() 当作 peer 结果，
        // 未做任何跨进程通信、也无隐私保护。上生产前需要用户决策以下条目：
        //
        //   A. 联邦传输层
        //      - EventMesh request/reply（已有依赖，最省事）
        //      - 独立 gRPC / QUIC（自主但要维护）
        //      - 请求超时 / 部分失败聚合策略（k-of-n quorum？）
        //
        //   B. 信任 & 认证
        //      - peer 身份（DID / mTLS / uwu 生态既有方案？）
        //      - 请求签名 + 响应验签
        //      - 黑名单 / 声誉门槛（ReputationEngine 已有骨架）
        //
        //   C. 差分隐私（DP）
        //      - 噪声机制：Laplace（数值） / Gaussian / 离散指数
        //      - 隐私预算 (ε, δ) 记账：per-query 还是 per-peer/day？
        //      - 敏感度分析：query_embedding 泄露 / hit count 泄露 / 元数据泄露
        //      - Report noisy count vs Report noisy top-k（后者复杂但更实用）
        //
        //   D. 聚合协议
        //      - 明文聚合（简单，但 peer 能看到彼此结果）
        //      - 安全聚合（SecAgg / MPC，重）
        //      - 客户端本地 DP + 服务端聚合（推荐起点）
        //
        //   E. 缓存与失效
        //      - 联邦结果 TTL（当前无 TTL，永久缓存）
        //      - peer 上线/下线时的 cache invalidation
        //
        // 依赖决策（A/B/C/D）都需要用户拍板，不做自作主张的选型。
        let mut all_results = local;
        for domain in &query.domains {
            let peers = self.registry.peers_in_domain(domain);
            for _peer in peers {
                // 联邦查询：向每个同伴请求该领域的条目
                // 在完整实现中通过 EventMesh request/reply 模式
                // 这里先用本地 registry 的所有发布作为替代
                let domain_pubs: Vec<MarketEntry> = self.registry.my_publications()
                    .into_iter()
                    .filter(|e| e.domain == *domain)
                    .collect();
                all_results.extend(domain_pubs);
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
    fn rank(&self, entries: Vec<MarketEntry>, query: &DiscoveryQuery, limit: usize) -> Vec<MarketHit> {
        let mut scored: Vec<MarketHit> = entries.into_iter()
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

        scored.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }
}

fn avg_quality(hits: &[MarketHit]) -> f32 {
    if hits.is_empty() { return 0.0; }
    hits.iter().map(|h| h.entry.quality_score).sum::<f32>() / hits.len() as f32
}

fn domains_from(hits: &[MarketHit]) -> Vec<String> {
    let mut domains: Vec<String> = hits.iter().map(|h| h.entry.domain.clone()).collect();
    domains.dedup();
    domains
}
