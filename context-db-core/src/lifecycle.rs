//! 记忆生命周期（F22 遗忘曲线 + F26 上下文经济模型）。

use crate::{ContentLevel, ContextEntry, MemoryClass};
use chrono::{DateTime, Utc};

// ═══════════════════════════════════════════════════════════════════════════
// F22 遗忘曲线
// ═══════════════════════════════════════════════════════════════════════════

/// 艾宾浩斯遗忘曲线参数。
///
/// 记忆保留率 R = e^(-t/S)  其中 t = 距创建时间, S = 稳定性因子
#[derive(Debug, Clone)]
pub struct ForgettingCurve {
    /// 稳定性因子（越大遗忘越慢）
    pub stability: f64,
    /// 初始强化次数
    pub reinforcements: u32,
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self { stability: 7.0, reinforcements: 0 }
    }
}

impl ForgettingCurve {
    /// 计算当前保留率 (0-1)。
    pub fn retention(&self, created_at: DateTime<Utc>, now: DateTime<Utc>) -> f64 {
        let days = (now - created_at).num_hours() as f64 / 24.0;
        let effective_stability = self.stability * (1.0 + self.reinforcements as f64 * 0.5);
        (-days / effective_stability).exp()
    }

    /// 记忆被访问后强化。
    pub fn reinforce(&mut self) {
        self.reinforcements = self.reinforcements.saturating_add(1);
    }

    /// 判断是否需要降级（L2→L1→L0→归档）。
    pub fn should_degrade(&self, created_at: DateTime<Utc>, now: DateTime<Utc>) -> Option<ContentLevel> {
        let r = self.retention(created_at, now);
        match r {
            x if x < 0.1 => Some(ContentLevel::L0), // 几乎遗忘 → L0
            x if x < 0.3 => Some(ContentLevel::L1), // 模糊 → L1
            _ => None,                                // 清晰 → 保持
        }
    }

    /// 判断是否应完全归档。
    pub fn should_archive(&self, created_at: DateTime<Utc>, now: DateTime<Utc>) -> bool {
        self.retention(created_at, now) < 0.05
    }
}

/// 条目生命周期策略。
#[derive(Debug, Clone)]
pub struct LifecyclePolicy {
    /// 曲线参数
    pub curve: ForgettingCurve,
    /// 最大保留版本数
    pub max_versions: Option<usize>,
    /// 最大保留天数
    pub max_age_days: Option<i64>,
    /// 降级动作
    pub on_degrade: DegradeAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeAction {
    /// 仅降低加载级别
    Downgrade,
    /// 归档到冷存储
    Archive,
    /// 删除
    Delete,
    /// 保留
    Keep,
}

impl LifecyclePolicy {
    /// 按记忆类型的默认策略。
    pub fn for_class(class: MemoryClass) -> Self {
        match class {
            MemoryClass::Events | MemoryClass::Cases => Self {
                curve: ForgettingCurve { stability: 90.0, reinforcements: 0 },
                max_versions: None,
                max_age_days: None,
                on_degrade: DegradeAction::Keep,
            },
            MemoryClass::Profile | MemoryClass::Preferences => Self {
                curve: ForgettingCurve { stability: 30.0, reinforcements: 2 },
                max_versions: Some(20),
                max_age_days: Some(180),
                on_degrade: DegradeAction::Downgrade,
            },
            MemoryClass::Patterns | MemoryClass::Skills | MemoryClass::Tools => Self {
                curve: ForgettingCurve { stability: 60.0, reinforcements: 1 },
                max_versions: Some(50),
                max_age_days: Some(365),
                on_degrade: DegradeAction::Archive,
            },
            _ => Self {
                curve: ForgettingCurve::default(),
                max_versions: Some(100),
                max_age_days: Some(90),
                on_degrade: DegradeAction::Downgrade,
            },
        }
    }

    /// 对条目应用生命周期检查。
    pub fn evaluate(&self, entry: &ContextEntry, now: DateTime<Utc>) -> LifecycleAction {
        // 检查时间衰减
        if let Some(level) = self.curve.should_degrade(entry.created_at, now) {
            if self.curve.should_archive(entry.created_at, now) {
                return LifecycleAction::Archive;
            }
            return LifecycleAction::Downgrade(level);
        }

        // 检查最大天数
        if let Some(max_days) = self.max_age_days {
            let age = (now - entry.created_at).num_days();
            if age > max_days {
                return match self.on_degrade {
                    DegradeAction::Delete => LifecycleAction::Delete,
                    DegradeAction::Archive => LifecycleAction::Archive,
                    _ => LifecycleAction::Downgrade(ContentLevel::L0),
                };
            }
        }

        LifecycleAction::Keep
    }
}

#[derive(Debug, Clone)]
pub enum LifecycleAction {
    Keep,
    Downgrade(ContentLevel),
    Archive,
    Delete,
}

// ═══════════════════════════════════════════════════════════════════════════
// F26 上下文经济模型
// ═══════════════════════════════════════════════════════════════════════════

/// Token 消耗追踪器。
#[derive(Debug, Clone, Default)]
pub struct TokenBudget {
    /// 总预算
    pub total: usize,
    /// 已消耗
    pub used: usize,
    /// 按操作类别的消耗明细
    pub breakdown: TokenBreakdown,
}

#[derive(Debug, Clone, Default)]
pub struct TokenBreakdown {
    pub retrieval: usize,
    pub generation: usize,
    pub embedding: usize,
    pub storage: usize,
}

impl TokenBudget {
    pub fn new(total: usize) -> Self {
        Self { total, ..Default::default() }
    }

    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.used)
    }

    pub fn is_exhausted(&self) -> bool {
        self.used >= self.total
    }

    pub fn pressure(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { self.used as f32 / self.total as f32 }
    }

    /// 预留一批 token（返回是否成功）。
    pub fn reserve(&mut self, amount: usize) -> bool {
        if self.used + amount <= self.total {
            self.used += amount;
            true
        } else {
            false
        }
    }

    /// 记录检索消耗。
    pub fn spend_retrieval(&mut self, tokens: usize) {
        self.used = self.used.saturating_add(tokens).min(self.total);
        self.breakdown.retrieval += tokens;
    }

    /// 记录生成消耗。
    pub fn spend_generation(&mut self, tokens: usize) {
        self.used = self.used.saturating_add(tokens).min(self.total);
        self.breakdown.generation += tokens;
    }

    /// 记录 embedding 消耗。
    pub fn spend_embedding(&mut self, tokens: usize) {
        self.used = self.used.saturating_add(tokens).min(self.total);
        self.breakdown.embedding += tokens;
    }

    /// 成本报告（假设 $0.01/1K tokens）。
    pub fn cost_estimate(&self) -> f64 {
        self.used as f64 * 0.01 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn forgetting_curve_decays_over_time() {
        let fc = ForgettingCurve::default();
        let now = Utc::now();
        let past = now - Duration::days(30);
        assert!(fc.retention(past, now) < 0.1);
    }

    #[test]
    fn reinforcement_slows_decay() {
        let mut fc = ForgettingCurve { stability: 7.0, reinforcements: 5 };
        let now = Utc::now();
        let past = now - Duration::days(14);
        let r1 = fc.retention(past, now);
        fc.reinforce();
        let r2 = fc.retention(past, now);
        assert!(r2 > r1);
    }

    #[test]
    fn token_budget_tracks_spend() {
        let mut budget = TokenBudget::new(10000);
        budget.spend_retrieval(2000);
        budget.spend_generation(1000);
        assert_eq!(budget.used, 3000);
        assert_eq!(budget.remaining(), 7000);
    }

    #[test]
    fn token_budget_exhaustion() {
        let mut budget = TokenBudget::new(100);
        assert!(budget.reserve(90));
        assert!(!budget.reserve(20)); // 超出预算
        assert!(budget.reserve(10));  // 刚好
        assert!(budget.is_exhausted());
    }
}
