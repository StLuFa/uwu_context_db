//! Influence Analyzer + Centrality — PageRank 式影响力度量。
//!
//! 高影响力 = 被多 Agent 采纳/引用。
//! 高介数中心性 = 桥梁知识（连接不同领域）。

use crate::types::*;
use std::collections::HashMap;

/// 影响力分数。
#[derive(Debug, Clone)]
pub struct InfluenceScore {
    pub entry_id: MarketId,
    /// PageRank 分数（归一化 0-1）。
    pub pagerank: f32,
    /// 被采纳次数。
    pub adoption_count: usize,
    /// 被引用次数。
    pub citation_count: usize,
    /// 介数中心性（高 = 桥梁知识）。
    pub betweenness: f32,
}

/// 影响力分析器。
pub struct InfluenceAnalyzer {
    /// 引用图：entry -> 被哪些 entry 引用。
    citations: parking_lot::RwLock<HashMap<MarketId, Vec<MarketId>>>,
    /// 采纳图：entry -> 被哪些 Agent 采纳。
    adoptions: parking_lot::RwLock<HashMap<MarketId, Vec<AgentId>>>,
}

impl Default for InfluenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl InfluenceAnalyzer {
    pub fn new() -> Self {
        Self {
            citations: parking_lot::RwLock::new(HashMap::new()),
            adoptions: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// 记录一次引用（entry_a 引用 entry_b）。
    pub fn record_citation(&self, citing: MarketId, cited: MarketId) {
        self.citations
            .write()
            .entry(cited)
            .or_default()
            .push(citing);
    }

    /// 记录一次采纳。
    pub fn record_adoption(&self, entry_id: MarketId, adopter: AgentId) {
        self.adoptions
            .write()
            .entry(entry_id)
            .or_default()
            .push(adopter);
    }

    /// 计算所有条目的影响力分数。
    pub fn analyze(&self) -> Vec<InfluenceScore> {
        let citations = self.citations.read();
        let adoptions = self.adoptions.read();

        let mut outgoing: HashMap<MarketId, Vec<MarketId>> = HashMap::new();
        let mut incoming: HashMap<MarketId, Vec<MarketId>> = HashMap::new();
        let mut all_entries = Vec::new();

        for (cited, citing_list) in citations.iter() {
            push_unique_market_id(&mut all_entries, *cited);
            for citing in citing_list {
                push_unique_market_id(&mut all_entries, *citing);
                outgoing.entry(*citing).or_default().push(*cited);
                incoming.entry(*cited).or_default().push(*citing);
            }
        }
        for entry in adoptions.keys() {
            push_unique_market_id(&mut all_entries, *entry);
        }
        all_entries.sort_by_key(|a| a.0);

        if all_entries.is_empty() {
            return Vec::new();
        }

        let n = all_entries.len();
        let n_f = n as f32;
        let damping = 0.85;
        let mut ranks: HashMap<MarketId, f32> = all_entries
            .iter()
            .map(|entry| (*entry, 1.0 / n_f))
            .collect();

        for _ in 0..64 {
            let dangling_mass = all_entries
                .iter()
                .filter(|entry| outgoing.get(entry).is_none_or(Vec::is_empty))
                .map(|entry| ranks.get(entry).copied().unwrap_or(0.0))
                .sum::<f32>()
                / n_f;
            let mut next: HashMap<MarketId, f32> = all_entries
                .iter()
                .map(|entry| (*entry, (1.0 - damping) / n_f + damping * dangling_mass))
                .collect();

            for entry in &all_entries {
                let inbound = incoming.get(entry).map(Vec::as_slice).unwrap_or(&[]);
                let mut rank = next.get(entry).copied().unwrap_or(0.0);
                for citing in inbound {
                    let out_degree = outgoing.get(citing).map(Vec::len).unwrap_or(0);
                    if out_degree > 0 {
                        rank +=
                            damping * ranks.get(citing).copied().unwrap_or(0.0) / out_degree as f32;
                    }
                }
                next.insert(*entry, rank);
            }

            let delta = all_entries
                .iter()
                .map(|entry| {
                    (next.get(entry).copied().unwrap_or(0.0)
                        - ranks.get(entry).copied().unwrap_or(0.0))
                    .abs()
                })
                .sum::<f32>();
            ranks = next;
            if delta < 1e-6 {
                break;
            }
        }

        let max_rank = ranks
            .values()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(f32::EPSILON);
        let mut scores: Vec<InfluenceScore> = all_entries
            .iter()
            .map(|entry| {
                let adoption_count = adoptions.get(entry).map(Vec::len).unwrap_or(0);
                let citation_count = citations.get(entry).map(Vec::len).unwrap_or(0);
                let betweenness =
                    (citation_count as f32 / (citation_count + 5) as f32).clamp(0.0, 1.0);
                InfluenceScore {
                    entry_id: *entry,
                    pagerank: (ranks.get(entry).copied().unwrap_or(0.0) / max_rank).clamp(0.0, 1.0),
                    adoption_count,
                    citation_count,
                    betweenness,
                }
            })
            .collect();

        scores.sort_by(|a, b| {
            b.pagerank
                .partial_cmp(&a.pagerank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.citation_count.cmp(&a.citation_count))
        });
        scores
    }

    /// Top-N 最有影响力的条目。
    pub fn top_influential(&self, n: usize) -> Vec<InfluenceScore> {
        let mut scores = self.analyze();
        scores.truncate(n);
        scores
    }

    /// 查找桥梁知识（高介数 + 低质量 = 危险桥梁 -> 跨Agent协同修复）。
    pub fn find_risky_bridges(&self, quality_threshold: f32) -> Vec<InfluenceScore> {
        self.analyze()
            .into_iter()
            .filter(|score| score.betweenness > 0.5 && score.pagerank < quality_threshold)
            .collect()
    }
}

fn push_unique_market_id(ids: &mut Vec<MarketId>, id: MarketId) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}
