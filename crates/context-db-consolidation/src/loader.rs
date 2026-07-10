//! ProgressiveLoader — 四层渐进式加载（索引→摘要→完整→证据）。
//!
//! 预算分配使用轻量在线 bandit：冷启动时走固定阈值，积累反馈后按
//! UCB reward/cost 选择层级，避免把 token 长期浪费在“读到 Full 也无用”的条目上。

use std::collections::HashMap;

use agent_context_db_core::{ContentPayload, ContextEntry};

/// 加载层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoadLevel {
    Index = 0,    // name + 类型 (~20 tokens)
    Abstract = 1, // L0 摘要 (~100 tokens)
    Full = 2,     // L1 概览 (~2k tokens)
    Evidence = 3, // 完整证据树（按需）
}

impl LoadLevel {
    pub fn estimated_tokens(self) -> usize {
        match self {
            Self::Index => 24,
            Self::Abstract => 120,
            Self::Full => 2_000,
            Self::Evidence => 4_000,
        }
    }

    fn all() -> [Self; 4] {
        [Self::Index, Self::Abstract, Self::Full, Self::Evidence]
    }
}

/// 读取反馈，用来学习“某类条目读到某层是否值得”。
#[derive(Debug, Clone, Copy)]
pub struct LoadFeedback {
    pub level: LoadLevel,
    pub tokens_spent: usize,
    pub useful: bool,
    pub downstream_reward: f32,
}

impl LoadFeedback {
    pub fn reward(&self) -> f32 {
        let adoption = if self.useful { 1.0 } else { 0.0 };
        (adoption * 0.7 + self.downstream_reward.clamp(0.0, 1.0) * 0.3).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct ArmStats {
    pulls: u64,
    reward_sum: f32,
    token_sum: usize,
}

impl Default for ArmStats {
    fn default() -> Self {
        Self {
            pulls: 0,
            reward_sum: 0.0,
            token_sum: 0,
        }
    }
}

impl ArmStats {
    fn observe(&mut self, feedback: LoadFeedback) {
        self.pulls += 1;
        self.reward_sum += feedback.reward();
        self.token_sum = self.token_sum.saturating_add(feedback.tokens_spent.max(1));
    }

    fn mean_reward(self) -> f32 {
        if self.pulls == 0 {
            0.0
        } else {
            self.reward_sum / self.pulls as f32
        }
    }

    fn mean_cost(self, fallback: usize) -> f32 {
        if self.pulls == 0 {
            fallback as f32
        } else {
            (self.token_sum as f32 / self.pulls as f32).max(1.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct BanditBudgetPolicy {
    arms: HashMap<String, HashMap<LoadLevel, ArmStats>>,
    exploration: f32,
    min_samples_before_learning: u64,
}

impl Default for BanditBudgetPolicy {
    fn default() -> Self {
        Self {
            arms: HashMap::new(),
            exploration: 0.85,
            min_samples_before_learning: 4,
        }
    }
}

impl BanditBudgetPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn choose(&self, entry: &ContextEntry, remaining_budget: usize) -> LoadLevel {
        let feasible = LoadLevel::all()
            .into_iter()
            .filter(|level| level.estimated_tokens() <= remaining_budget)
            .collect::<Vec<_>>();
        if feasible.is_empty() {
            return LoadLevel::Index;
        }

        let key = feature_key(entry);
        let Some(stats) = self.arms.get(&key) else {
            return threshold_fallback(remaining_budget);
        };
        let total_pulls: u64 = stats.values().map(|s| s.pulls).sum();
        if total_pulls < self.min_samples_before_learning {
            return threshold_fallback(remaining_budget);
        }

        feasible
            .into_iter()
            .max_by(|a, b| {
                self.ucb_score(stats, *a, total_pulls, remaining_budget)
                    .total_cmp(&self.ucb_score(stats, *b, total_pulls, remaining_budget))
            })
            .unwrap_or(LoadLevel::Index)
    }

    pub fn observe(&mut self, entry: &ContextEntry, feedback: LoadFeedback) {
        self.arms
            .entry(feature_key(entry))
            .or_default()
            .entry(feedback.level)
            .or_default()
            .observe(feedback);
    }

    fn ucb_score(
        &self,
        stats: &HashMap<LoadLevel, ArmStats>,
        level: LoadLevel,
        total_pulls: u64,
        remaining_budget: usize,
    ) -> f32 {
        let arm = stats.get(&level).copied().unwrap_or_default();
        if arm.pulls == 0 {
            return 0.08 + self.exploration * 0.05 - budget_pressure(level, remaining_budget);
        }
        let exploit = arm.mean_reward() / arm.mean_cost(level.estimated_tokens()).sqrt();
        let explore = ((total_pulls as f32).ln() / arm.pulls as f32).sqrt() * self.exploration;
        exploit + explore - budget_pressure(level, remaining_budget)
    }
}

/// 渐进式加载器。
pub struct ProgressiveLoader {
    budget: usize,
    used: usize,
    policy: BanditBudgetPolicy,
}

impl ProgressiveLoader {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            policy: BanditBudgetPolicy::new(),
        }
    }

    pub fn with_policy(budget: usize, policy: BanditBudgetPolicy) -> Self {
        Self {
            budget,
            used: 0,
            policy,
        }
    }

    /// 按在线策略选择层级；历史不足时自动退回固定预算阈值。
    pub fn load_level(&self, entry: &ContextEntry) -> LoadLevel {
        self.policy.choose(entry, self.remaining())
    }

    /// 把一次读取结果反馈给 bandit 策略。
    pub fn observe_feedback(&mut self, entry: &ContextEntry, feedback: LoadFeedback) {
        self.policy.observe(entry, feedback);
    }

    pub fn policy(&self) -> &BanditBudgetPolicy {
        &self.policy
    }

    /// 按层级返回对应内容。
    pub fn content_at(&self, entry: &ContextEntry, level: LoadLevel) -> String {
        match level {
            LoadLevel::Index => {
                format!(
                    "{} ({})",
                    entry.uri,
                    entry
                        .content_type()
                        .map(|c| c.as_path_segment())
                        .unwrap_or("?"),
                )
            }
            LoadLevel::Abstract => entry.l0_text().to_string(),
            LoadLevel::Full => match &entry.payload {
                ContentPayload::Text { dense, .. } => dense.clone(),
                _ => entry.l0_text().to_string(),
            },
            LoadLevel::Evidence => {
                // 证据树的实际展开由调用方基于 GraphStore 注入，loader 只负责预算决策。
                format!("[evidence tree for {}]", entry.uri)
            }
        }
    }

    /// 消费 token 预算。
    pub fn consume(&mut self, tokens: usize) {
        self.used = self.used.saturating_add(tokens);
    }

    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.used)
    }
}

fn threshold_fallback(remaining: usize) -> LoadLevel {
    match remaining {
        r if r >= LoadLevel::Evidence.estimated_tokens() => LoadLevel::Evidence,
        r if r >= LoadLevel::Full.estimated_tokens() => LoadLevel::Full,
        r if r >= LoadLevel::Abstract.estimated_tokens() => LoadLevel::Abstract,
        _ => LoadLevel::Index,
    }
}

fn budget_pressure(level: LoadLevel, remaining: usize) -> f32 {
    let cost = level.estimated_tokens() as f32;
    let remaining = remaining.max(1) as f32;
    ((cost / remaining) * 0.15).clamp(0.0, 0.4)
}

fn feature_key(entry: &ContextEntry) -> String {
    let content_type = entry
        .content_type()
        .map(|ct| ct.as_path_segment().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let quality_bucket = match entry.metadata.quality_score.unwrap_or(0.5) {
        q if q >= 0.8 => "high",
        q if q >= 0.45 => "mid",
        _ => "low",
    };
    format!("{content_type}:{quality_bucket}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, ContextEntry, ContextUri, TenantId};
    use uuid::Uuid;

    fn entry(quality: f32) -> ContextEntry {
        let mut entry = ContextEntry::new_text(
            ContextUri::parse("uwu://tenant/agent/a/memory/fact/topic/entry").unwrap(),
            TenantId(Uuid::nil()),
            "summary",
        );
        entry.metadata.quality_score = Some(quality);
        entry.metadata.content_type = Some(ContentType::Fact);
        entry
    }

    #[test]
    fn cold_start_uses_budget_thresholds() {
        let loader = ProgressiveLoader::new(2_500);
        assert_eq!(loader.load_level(&entry(0.9)), LoadLevel::Full);
        let loader = ProgressiveLoader::new(80);
        assert_eq!(loader.load_level(&entry(0.9)), LoadLevel::Index);
    }

    #[test]
    fn bandit_learns_to_avoid_unrewarded_full_reads() {
        let e = entry(0.9);
        let mut loader = ProgressiveLoader::new(3_000);
        for _ in 0..8 {
            loader.observe_feedback(
                &e,
                LoadFeedback {
                    level: LoadLevel::Full,
                    tokens_spent: 2_000,
                    useful: false,
                    downstream_reward: 0.0,
                },
            );
            loader.observe_feedback(
                &e,
                LoadFeedback {
                    level: LoadLevel::Abstract,
                    tokens_spent: 120,
                    useful: true,
                    downstream_reward: 0.8,
                },
            );
        }
        assert_eq!(loader.load_level(&e), LoadLevel::Abstract);
    }
}
