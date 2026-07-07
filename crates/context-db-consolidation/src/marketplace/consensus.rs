//! Consensus + Schelling Point — 多源独立收敛检测。
//!
//! 两个独立 session/agent 蒸馏出语义相似的晶体 = Schelling 点 = "知识被确认"。
//! 达到 N 个独立源 = Established（已确立）。

use crate::marketplace::types::*;
use agent_context_db_core::{ContextUri, VectorIndex, VectorSimilarity};
use std::sync::Arc;

/// 共识追踪器 — 检测多源独立收敛。
pub struct ConsensusTracker {
    vector_index: Arc<dyn VectorIndex>,
    /// 收敛阈值（余弦相似度 > 此值 = 相同结论）。
    convergence_threshold: f32,
    /// 一致性阈值（达成此独立源数 → Established）。
    established_threshold: usize,
}

/// 收敛报告。
#[derive(Debug, Clone)]
pub struct ConvergenceReport {
    pub new_entry_id: MarketId,
    /// 已发现的独立相似条目。
    pub independent_sources: Vec<(AgentId, MarketId, f32)>,
    /// 确认总数（独立源数）。
    pub corroboration: usize,
    /// 推定状态。
    pub status: EstablishmentStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstablishmentStatus {
    Unverified,   // 0 个独立源
    Single,       // 1 个
    Corroborated, // 2 个
    Established,  // ≥3 个 = Schelling 点
}

impl ConsensusTracker {
    pub fn new(vector_index: Arc<dyn VectorIndex>) -> Self {
        Self {
            vector_index,
            convergence_threshold: 0.9,
            established_threshold: 3,
        }
    }

    /// 新条目写入时：检查是否已有独立 Agent 达成相同结论。
    /// 如果是 → boost corroboration。
    pub async fn check_convergence(
        &self,
        new_entry: &MarketEntry,
        existing: &[MarketEntry],
    ) -> ConvergenceReport {
        let mut independent_sources = Vec::new();
        let mut corroboration = 1; // 自身

        for existing_entry in existing {
            if existing_entry.publisher == new_entry.publisher {
                continue;
            }
            if existing_entry.domain != new_entry.domain {
                continue;
            }

            // Jaccard 相似度作为快速筛选
            let sim = jaccard_similarity(&new_entry.principle, &existing_entry.principle);
            if sim >= (self.convergence_threshold - 0.2) {
                // 高重叠 → 可能是独立收敛
                independent_sources.push((
                    existing_entry.publisher.clone(),
                    existing_entry.id,
                    sim,
                ));
                corroboration += 1;
            }
        }

        let status = if corroboration >= self.established_threshold {
            EstablishmentStatus::Established
        } else if corroboration >= 2 {
            EstablishmentStatus::Corroborated
        } else if corroboration >= 1 {
            EstablishmentStatus::Single
        } else {
            EstablishmentStatus::Unverified
        };

        ConvergenceReport {
            new_entry_id: new_entry.id,
            independent_sources,
            corroboration,
            status,
        }
    }

    /// Schelling 点：返回所有 ≥N 个独立源的"已确立"条目。
    pub fn find_established<'a>(
        &self,
        entries: &'a [MarketEntry],
        min_sources: usize,
    ) -> Vec<&'a MarketEntry> {
        let mut established = Vec::new();
        for entry in entries {
            if entry.corroboration.independent_sources >= min_sources {
                established.push(entry);
            }
        }
        established.sort_by(|a, b| {
            b.corroboration
                .independent_sources
                .cmp(&a.corroboration.independent_sources)
        });
        established
    }
}

fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let wa: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let wb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let intersection = wa.intersection(&wb).count();
    let union = wa.union(&wb).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}
