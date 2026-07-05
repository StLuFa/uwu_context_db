//! 机会成本组合优化 — token 预算内覆盖最大化。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri};

/// 机会成本加载器 — 替代贪心策略，预算内覆盖最大化。
pub struct OpportunityCostLoader {
    budget: usize,
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
        Self { budget }
    }

    /// 在 token 预算内选覆盖最大化的组合。
    /// 按 relevance/token_cost 排序（性价比），
    /// 选了 A 后与 A 重叠的 B 降权（覆盖度奖励）。
    pub fn select_optimal(&self, candidates: &[Candidate]) -> Vec<(ContextUri, ContentLevel)> {
        let mut ranked: Vec<&Candidate> = candidates.iter().collect();
        ranked.sort_by(|a, b| {
            let ratio_a = a.relevance / a.token_cost.max(1) as f32;
            let ratio_b = b.relevance / b.token_cost.max(1) as f32;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });

        let mut selected = Vec::new();
        let mut remaining = self.budget;

        for cand in ranked {
            if remaining < 50 {
                break;
            }

            // 覆盖度奖励：与已选条目语义重叠度低的候选加分
            let overlap = self.estimate_overlap(cand, &selected);
            let effective_value = cand.relevance * (1.0 - overlap * 0.5);

            let level = if effective_value > 0.7 && remaining >= 2000 {
                ContentLevel::L1
            } else {
                ContentLevel::L0
            };

            let cost = match level {
                ContentLevel::L1 => 2000,
                _ => 100,
            };

            if remaining >= cost {
                selected.push((cand.uri.clone(), level));
                remaining -= cost;
            }
        }

        selected
    }

    /// 估算候选与已选条目的重叠度（0.0-1.0）。
    fn estimate_overlap(
        &self,
        _cand: &Candidate,
        _selected: &[(ContextUri, ContentLevel)],
    ) -> f32 {
        // 简化：URI 路径前缀匹配度
        if _selected.is_empty() {
            return 0.0;
        }
        let cand_str = _cand.uri.to_string();
        let matches = _selected
            .iter()
            .filter(|(s, _)| {
                let s_str = s.to_string();
                // 共享前 3 段视为重叠
                let cand_segs: Vec<&str> =
                    cand_str.split('/').take(3).collect();
                let s_segs: Vec<&str> = s_str.split('/').take(3).collect();
                cand_segs == s_segs
            })
            .count();
        matches as f32 / _selected.len().max(1) as f32
    }
}
