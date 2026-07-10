//! Consensus + Schelling Point — 多源独立收敛检测。
//!
//! 两个独立 session/agent 蒸馏出语义相似的晶体 = Schelling 点 = "知识被确认"。
//! 达到 N 个独立源 = Established（已确立）。

use crate::types::*;
use agent_context_db_core::VectorIndex;
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
        let vector_candidates = self.vector_candidates(new_entry, existing).await;
        let candidate_entries = if vector_candidates.is_empty() {
            existing.iter().collect::<Vec<_>>()
        } else {
            vector_candidates
        };

        for existing_entry in candidate_entries {
            if existing_entry.publisher == new_entry.publisher {
                continue;
            }
            if existing_entry.domain != new_entry.domain {
                continue;
            }

            let sim = jaccard_similarity(&new_entry.principle, &existing_entry.principle);
            if sim >= (self.convergence_threshold - 0.2) {
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

    async fn vector_candidates<'a>(
        &self,
        new_entry: &MarketEntry,
        existing: &'a [MarketEntry],
    ) -> Vec<&'a MarketEntry> {
        let hits = self
            .vector_index
            .search(
                "market",
                text_signature_vector(&new_entry.principle),
                64,
                Some(serde_json::json!({ "domain": new_entry.domain })),
            )
            .await
            .unwrap_or_default();
        if hits.is_empty() {
            return Vec::new();
        }
        let ids = hits
            .iter()
            .filter_map(|hit| hit.payload.get("market_id")?.as_str())
            .filter_map(|raw| uuid::Uuid::parse_str(raw).ok())
            .map(MarketId)
            .collect::<std::collections::HashSet<_>>();
        existing
            .iter()
            .filter(|entry| ids.contains(&entry.id))
            .collect()
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

fn text_signature_vector(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0_f32; 128];
    for token in text.split_whitespace() {
        let mut acc = 0xcbf29ce484222325_u64;
        for byte in token.as_bytes() {
            acc ^= *byte as u64;
            acc = acc.wrapping_mul(0x100000001b3);
        }
        let idx = (acc as usize) % vector.len();
        vector[idx] += 1.0;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
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
