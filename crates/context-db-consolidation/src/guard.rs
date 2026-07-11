//! ContextRotGuard — 主动裁剪 + 边际效用检查。

use crate::rif::RifSuppressor;
use agent_context_db_core::{ContextEntry, ContextUri};

/// Context Rot 守卫 — 容量超限时裁剪最低价值条目。
pub struct ContextRotGuard {
    max_capacity: usize,
    rif: Option<RifSuppressor>,
}

impl ContextRotGuard {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            max_capacity,
            rif: None,
        }
    }

    pub fn with_rif(mut self, rif: RifSuppressor) -> Self {
        self.rif = Some(rif);
        self
    }

    /// 检查容量，超限返回待裁剪 URI 列表。
    pub fn enforce_capacity(
        &self,
        entries: &[ContextEntry],
        access_counts: &std::collections::HashMap<String, u64>,
    ) -> Vec<ContextUri> {
        if entries.len() <= self.max_capacity {
            return vec![];
        }

        let excess = entries.len() - self.max_capacity;
        let mut scored: Vec<(&ContextEntry, f64)> = entries
            .iter()
            .map(|e| {
                let access = access_counts.get(&e.uri.to_string()).copied().unwrap_or(0) as f64;
                let quality = e.metadata.quality_score.unwrap_or(0.5) as f64;
                let score = access * 0.3 + quality * 0.7;
                (e, score)
            })
            .collect();

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored
            .iter()
            .take(excess)
            .map(|(e, _)| e.uri.clone())
            .collect()
    }

    /// 边际效用检查 — 新 entry 的 InfoGain 是否值得写入。
    pub fn marginal_utility(
        &self,
        new_entry: &ContextEntry,
        existing_similar: &[ContextEntry],
    ) -> bool {
        if existing_similar.is_empty() {
            return true; // 无相似条目，一定写入
        }

        let new_quality = new_entry.metadata.quality_score.unwrap_or(0.5);
        let best_existing = existing_similar
            .iter()
            .filter_map(|e| e.metadata.quality_score)
            .filter(|quality| quality.is_finite())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.5);

        // 新条目质量必须显著高于已有最好条目
        new_quality > best_existing + 0.1
    }
}
