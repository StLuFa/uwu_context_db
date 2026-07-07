//! Influence Analyzer + Centrality — PageRank 式影响力度量。
//!
//! 高影响力 = 被多 Agent 采纳/引用。
//! 高介数中心性 = 桥梁知识（连接不同领域）。

use crate::marketplace::types::*;
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
    /// 引用图：entry → 被哪些 entry 引用。
    citations: parking_lot::RwLock<HashMap<MarketId, Vec<MarketId>>>,
    /// 采纳图：entry → 被哪些 Agent 采纳。
    adoptions: parking_lot::RwLock<HashMap<MarketId, Vec<AgentId>>>,
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

        // 构建图
        let mut graph: HashMap<MarketId, Vec<MarketId>> = HashMap::new();
        let mut all_entries: Vec<MarketId> = Vec::new();

        for (cited, citing_list) in citations.iter() {
            graph.entry(*cited).or_default().extend(citing_list.clone());
            if !all_entries.contains(cited) {
                all_entries.push(*cited);
            }
            for c in citing_list {
                if !all_entries.contains(c) {
                    all_entries.push(*c);
                }
            }
        }

        if all_entries.is_empty() {
            return vec![];
        }

        // 简化 PageRank：迭代 20 轮
        let n = all_entries.len();
        let damping = 0.85;
        let mut ranks: HashMap<MarketId, f32> =
            all_entries.iter().map(|e| (*e, 1.0 / n as f32)).collect();

        for _ in 0..20 {
            let mut new_ranks: HashMap<MarketId, f32> = HashMap::new();
            for entry in &all_entries {
                let mut rank = (1.0 - damping) / n as f32;
                // 找到所有引用 entry 的节点
                for (citing, cited_list) in graph.iter() {
                    if cited_list.contains(entry) {
                        let out_degree = graph.get(citing).map(|l| l.len()).unwrap_or(1);
                        rank +=
                            damping * ranks.get(citing).unwrap_or(&0.0) / out_degree.max(1) as f32;
                    }
                }
                new_ranks.insert(*entry, rank);
            }
            ranks = new_ranks;
        }

        // 生成结果
        let mut scores: Vec<InfluenceScore> = all_entries
            .iter()
            .map(|e| {
                let adoption_count = adoptions.get(e).map(|l| l.len()).unwrap_or(0);
                let citation_count = citations.get(e).map(|l| l.len()).unwrap_or(0);
                let betweenness =
                    (citation_count as f32 / (citation_count + 5) as f32).clamp(0.0, 1.0);
                InfluenceScore {
                    entry_id: *e,
                    pagerank: ranks.get(e).copied().unwrap_or(0.0),
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
        });
        scores
    }

    /// Top-N 最有影响力的条目。
    pub fn top_influential(&self, n: usize) -> Vec<InfluenceScore> {
        let mut scores = self.analyze();
        scores.truncate(n);
        scores
    }

    /// 查找桥梁知识（高介数 + 低质量 = 危险桥梁 → 跨Agent协同修复）。
    pub fn find_risky_bridges(&self, quality_threshold: f32) -> Vec<InfluenceScore> {
        self.analyze()
            .into_iter()
            .filter(|s| s.betweenness > 0.5 && s.pagerank < quality_threshold)
            .collect()
    }
}
