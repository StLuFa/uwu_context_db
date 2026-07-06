//! 机会成本组合优化 — token 预算内覆盖最大化。
//!
//! 贪心 + 覆盖度惩罚：按 `relevance / token_cost` 性价比排序，
//! 每次选入一个候选后，与已选集合重叠越大的后续候选被降权。

use agent_context_db_core::{ContentLevel, ContextUri};

/// 机会成本加载器 — 替代贪心策略，预算内覆盖最大化。
pub struct OpportunityCostLoader {
    budget: usize,
    /// 覆盖度惩罚系数（0.0 = 完全忽略重叠；1.0 = 重叠即降到 0）。
    overlap_penalty: f32,
    /// 最低有效相关性（选入门槛，低于此值直接放弃剩余预算）。
    min_effective_value: f32,
    /// L1 升级阈值：effective_value 超过此值才尝试加载更详细内容。
    l1_upgrade_threshold: f32,
    /// L1 单条 token 估算成本。
    l1_token_cost: usize,
}

/// 候选条目。
#[derive(Debug, Clone)]
pub struct Candidate {
    pub uri: ContextUri,
    pub relevance: f32,
    pub token_cost: usize,
}

impl OpportunityCostLoader {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            overlap_penalty: 0.5,
            min_effective_value: 0.05,
            l1_upgrade_threshold: 0.7,
            l1_token_cost: 2000,
        }
    }

    /// 在 token 预算内选覆盖最大化的组合。
    ///
    /// 算法（贪心 + 覆盖度惩罚）：
    /// 1. 按 `relevance / token_cost` 性价比降序排列候选
    /// 2. 依次尝试选入，用与已选集合的重叠度惩罚 effective_value
    /// 3. effective_value 超过 `l1_upgrade_threshold` 且剩余预算足够 → 升级到 L1
    /// 4. token_cost 超过剩余预算 → 跳过；effective_value 低于阈值 → 停止
    pub fn select_optimal(&self, candidates: &[Candidate]) -> Vec<(ContextUri, ContentLevel)> {
        let mut ranked: Vec<&Candidate> = candidates.iter().collect();
        ranked.sort_by(|a, b| {
            let ratio_a = a.relevance / a.token_cost.max(1) as f32;
            let ratio_b = b.relevance / b.token_cost.max(1) as f32;
            ratio_b.partial_cmp(&ratio_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected: Vec<(ContextUri, ContentLevel)> = Vec::new();
        let mut remaining = self.budget;

        for cand in ranked {
            if remaining == 0 {
                break;
            }
            let overlap = self.estimate_overlap(cand, &selected);
            let effective_value = cand.relevance * (1.0 - overlap * self.overlap_penalty);

            if effective_value < self.min_effective_value {
                // 剩余候选性价比只会更差，提前终止
                break;
            }

            // 决定加载级别 —— 若剩余预算足够且值得，升级到 L1
            let (level, cost) = if effective_value >= self.l1_upgrade_threshold
                && remaining >= self.l1_token_cost
            {
                (ContentLevel::L1, self.l1_token_cost)
            } else {
                (ContentLevel::L0, cand.token_cost.max(1))
            };

            if cost > remaining {
                // 预算不够跳过该候选，继续尝试更便宜的
                continue;
            }
            selected.push((cand.uri.clone(), level));
            remaining -= cost;
        }

        selected
    }

    /// 估算候选与已选条目的重叠度（0.0-1.0）。
    fn estimate_overlap(
        &self,
        cand: &Candidate,
        selected: &[(ContextUri, ContentLevel)],
    ) -> f32 {
        if selected.is_empty() {
            return 0.0;
        }
        let cand_str = cand.uri.to_string();
        let cand_segs: Vec<&str> = cand_str.split('/').take(4).collect();
        let matches = selected
            .iter()
            .filter(|(s, _)| {
                let s_str = s.to_string();
                let s_segs: Vec<&str> = s_str.split('/').take(4).collect();
                cand_segs == s_segs
            })
            .count();
        matches as f32 / selected.len() as f32
    }
}

