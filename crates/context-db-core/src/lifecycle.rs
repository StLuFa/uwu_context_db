//! 生命周期管理：per-entry 自适应遗忘 + 重要性评分 + 可组合规则引擎。
//!
//! 替代旧的全局固定 `ForgettingCurve` 和 `LifecyclePolicy::for_class` 硬编码。

use crate::{ContentLevel, ContentType, ContextEntry};
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

// ===========================================================================
// 访问事件
// ===========================================================================

#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub timestamp: DateTime<Utc>,
    pub accessor: String,
    pub context: String,
    pub outcome: AccessOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessOutcome {
    Adopted,
    Rejected,
    Ignored,
    Modified,
}

// ===========================================================================
// 遗忘模型 trait
// ===========================================================================

pub trait ForgettingModel: Send + Sync {
    fn retention(&self, now: DateTime<Utc>) -> f32;
    fn record_access(&mut self, access: AccessEvent);
    fn fit(&mut self);
    fn half_life(&self) -> Option<Duration>;
}

// ===========================================================================
// Ebbinghaus 遗忘模型
// ===========================================================================

#[derive(Debug, Clone)]
pub struct EbbinghausModel {
    pub stability: f64,
    pub reinforcements: u32,
    pub last_access: DateTime<Utc>,
    pub access_history: Vec<AccessEvent>,
}

impl EbbinghausModel {
    pub fn new() -> Self {
        Self {
            stability: 7.0,
            reinforcements: 0,
            last_access: Utc::now(),
            access_history: Vec::new(),
        }
    }
}

impl Default for EbbinghausModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ForgettingModel for EbbinghausModel {
    fn retention(&self, now: DateTime<Utc>) -> f32 {
        let elapsed = (now - self.last_access).num_seconds() as f64 / 86400.0;
        let s_eff = self.stability * (1.0 + 0.5 * self.reinforcements as f64);
        ((-elapsed / s_eff).exp() as f32).clamp(0.0, 1.0)
    }

    fn record_access(&mut self, access: AccessEvent) {
        self.access_history.push(access);
        self.reinforcements = self.reinforcements.saturating_add(1);
        self.last_access = Utc::now();
        if self.reinforcements % 10 == 0 {
            self.fit();
        }
    }

    fn fit(&mut self) {
        if self.access_history.len() < 3 {
            return;
        }
        let intervals: Vec<f64> = self
            .access_history
            .windows(2)
            .map(|w| (w[1].timestamp - w[0].timestamp).num_seconds() as f64 / 86400.0)
            .collect();
        let avg = intervals.iter().sum::<f64>() / intervals.len() as f64;
        self.stability = self.stability * 0.7 + avg * 0.3;
    }

    fn half_life(&self) -> Option<Duration> {
        let s_eff = self.stability * (1.0 + 0.5 * self.reinforcements as f64);
        Some(Duration::days((s_eff * 0.693) as i64))
    }
}

// ===========================================================================
// SM-2 间隔重复模型（Anki 风格）
// ===========================================================================

/// 间隔重复模型 —— 基于 SuperMemo SM-2，用于主动学习场景。
///
/// 状态：`interval_days` 是"下次复习的间隔天数"，`ease_factor` 是难度系数。
/// 每次 `record_access` 相当于一次复习，根据 outcome（Success/Miss/Partial）
/// 更新 interval 与 ease_factor。
#[derive(Debug, Clone)]
pub struct SpacedRepetitionModel {
    pub interval_days: f64,
    pub ease_factor: f64,
    pub repetitions: u32,
    pub last_access: DateTime<Utc>,
    pub access_history: Vec<AccessEvent>,
}

impl SpacedRepetitionModel {
    pub fn new() -> Self {
        Self {
            interval_days: 1.0,
            ease_factor: 2.5,
            repetitions: 0,
            last_access: Utc::now(),
            access_history: Vec::new(),
        }
    }

    /// 把 AccessOutcome 映射到 SM-2 质量分（0-5）。
    fn quality(&self, outcome: &AccessOutcome) -> u8 {
        match outcome {
            AccessOutcome::Adopted => 5,
            AccessOutcome::Modified => 3,
            AccessOutcome::Ignored => 2,
            AccessOutcome::Rejected => 1,
        }
    }
}

impl Default for SpacedRepetitionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ForgettingModel for SpacedRepetitionModel {
    fn retention(&self, now: DateTime<Utc>) -> f32 {
        // 到期越近保留度越高；超过 interval 后指数衰减
        let elapsed = (now - self.last_access).num_seconds() as f64 / 86400.0;
        if self.interval_days <= 0.0 {
            return 0.5;
        }
        let ratio = elapsed / self.interval_days;
        // ratio=0 → 1.0；ratio=1 → ~0.5；ratio=3 → ~0.05
        (0.5_f64.powf(ratio) as f32).clamp(0.0, 1.0)
    }

    fn record_access(&mut self, access: AccessEvent) {
        let q = self.quality(&access.outcome);
        self.access_history.push(access);
        if q < 3 {
            // 失败 —— 重置间隔，保留 ease_factor
            self.repetitions = 0;
            self.interval_days = 1.0;
        } else {
            self.repetitions = self.repetitions.saturating_add(1);
            self.interval_days = match self.repetitions {
                1 => 1.0,
                2 => 6.0,
                _ => self.interval_days * self.ease_factor,
            };
            // SM-2 ease factor 更新公式
            let qf = q as f64;
            let delta = 0.1 - (5.0 - qf) * (0.08 + (5.0 - qf) * 0.02);
            self.ease_factor = (self.ease_factor + delta).max(1.3);
        }
        self.last_access = Utc::now();
    }

    fn fit(&mut self) {
        // SM-2 是在线更新，无需批量拟合。可选做 outcome 频率统计。
        if self.access_history.len() < 5 {
            return;
        }
        let recent = &self.access_history[self.access_history.len().saturating_sub(20)..];
        let success_ratio = recent
            .iter()
            .filter(|a| matches!(a.outcome, AccessOutcome::Adopted))
            .count() as f64
            / recent.len() as f64;
        // 长期成功率高 → 允许 ease_factor 微升；反之下调
        let target = 1.3 + 1.7 * success_ratio;
        self.ease_factor = self.ease_factor * 0.9 + target * 0.1;
    }

    fn half_life(&self) -> Option<Duration> {
        // retention=0.5 恰好在 interval_days 时到达，可视为半衰期
        Some(Duration::seconds((self.interval_days * 86400.0) as i64))
    }
}

// ===========================================================================
// 贝叶斯遗忘模型（先验 + 观测更新）
// ===========================================================================

/// 贝叶斯遗忘模型 —— 用 Beta 分布对"检索时被采纳"事件建模。
///
/// - `alpha`（成功计数 + 先验）、`beta`（失败计数 + 先验）
/// - `retention` 返回后验均值 α/(α+β)，随时间以指数衰减率 `decay_rate` 消退
/// - `record_access` 观测到 Success/Partial 累加 α，Miss 累加 β
#[derive(Debug, Clone)]
pub struct BayesianModel {
    pub alpha: f64,
    pub beta: f64,
    /// 无访问情况下每天信念衰减率（0=不衰减，0.05=约每 14 天减半）
    pub decay_rate: f64,
    pub last_access: DateTime<Utc>,
    pub access_history: Vec<AccessEvent>,
}

impl BayesianModel {
    pub fn new() -> Self {
        // Beta(1, 1) 均匀先验
        Self {
            alpha: 1.0,
            beta: 1.0,
            decay_rate: 0.05,
            last_access: Utc::now(),
            access_history: Vec::new(),
        }
    }

    /// 用 Beta(a, b) 作为自定义先验。
    pub fn with_prior(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(0.1),
            beta: beta.max(0.1),
            ..Self::new()
        }
    }
}

impl Default for BayesianModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ForgettingModel for BayesianModel {
    fn retention(&self, now: DateTime<Utc>) -> f32 {
        let posterior_mean = self.alpha / (self.alpha + self.beta);
        let elapsed_days = ((now - self.last_access).num_seconds() as f64 / 86400.0).max(0.0);
        let decayed = posterior_mean * (-self.decay_rate * elapsed_days).exp();
        (decayed as f32).clamp(0.0, 1.0)
    }

    fn record_access(&mut self, access: AccessEvent) {
        match access.outcome {
            AccessOutcome::Adopted => self.alpha += 1.0,
            AccessOutcome::Modified => self.alpha += 0.5,
            AccessOutcome::Ignored => self.beta += 0.5,
            AccessOutcome::Rejected => self.beta += 1.0,
        }
        self.access_history.push(access);
        self.last_access = Utc::now();
    }

    fn fit(&mut self) {
        // 定期收紧过度积累的观测（软重置到有效样本数上限，防止先验被彻底压过）
        let total = self.alpha + self.beta;
        const MAX_EFFECTIVE: f64 = 200.0;
        if total > MAX_EFFECTIVE {
            let scale = MAX_EFFECTIVE / total;
            self.alpha *= scale;
            self.beta *= scale;
        }
    }

    fn half_life(&self) -> Option<Duration> {
        if self.decay_rate <= 0.0 {
            return None;
        }
        // retention 从 posterior_mean 衰减到其一半的天数 = ln(2)/decay_rate
        let days = std::f64::consts::LN_2 / self.decay_rate;
        Some(Duration::seconds((days * 86400.0) as i64))
    }
}

// ===========================================================================
// 重要性评分
// ===========================================================================

#[derive(Debug, Clone)]
pub struct ImportanceScore {
    pub access_frequency: f32,
    pub recency: f32,
    pub centrality: f32,
    pub confidence: f32,
    pub tenant_priority: f32,
    pub composite: f32,
}

impl ImportanceScore {
    pub fn compute(
        log: &[AccessEvent],
        meta: &crate::ContextMeta,
        weights: &ImportanceWeights,
    ) -> Self {
        let access_frequency = if log.is_empty() {
            0.1
        } else {
            (log.len() as f32 / (log.len() as f32 + 5.0)).clamp(0.0, 1.0)
        };
        let recency = match log.last() {
            Some(a) => {
                let d = (Utc::now() - a.timestamp).num_hours() as f32 / 24.0;
                (-d / 30.0).exp().clamp(0.05, 1.0)
            }
            None => 0.05,
        };
        let centrality = 0.5;
        let confidence = meta.quality_score.unwrap_or(0.5);
        let tenant_priority = 0.5;
        let composite = weights.access_freq * access_frequency
            + weights.recency * recency
            + weights.centrality * centrality
            + weights.confidence * confidence
            + weights.tenant_priority * tenant_priority;
        Self {
            access_frequency,
            recency,
            centrality,
            confidence,
            tenant_priority,
            composite,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImportanceWeights {
    pub access_freq: f32,
    pub recency: f32,
    pub centrality: f32,
    pub confidence: f32,
    pub tenant_priority: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            access_freq: 0.25,
            recency: 0.25,
            centrality: 0.15,
            confidence: 0.25,
            tenant_priority: 0.10,
        }
    }
}

// ===========================================================================
// LifecycleEngine（可组合规则）
// ===========================================================================

#[derive(Debug, Clone)]
pub enum LifecycleAction {
    Keep,
    Downgrade { to_level: ContentLevel },
    Archive,
    Consolidate,
    Delete,
    Freeze,
}

pub struct LifecycleRule {
    pub name: String,
    pub condition: Box<dyn Fn(&ImportanceScore, &crate::ContextMeta) -> bool + Send + Sync>,
    pub action: LifecycleAction,
    pub priority: u32,
}

pub struct LifecycleEngine {
    rules: Vec<LifecycleRule>,
    weights: ImportanceWeights,
}

impl LifecycleEngine {
    pub fn new(rules: Vec<LifecycleRule>, weights: ImportanceWeights) -> Self {
        Self { rules, weights }
    }

    pub fn evaluate(&self, score: &ImportanceScore, meta: &crate::ContextMeta) -> LifecycleAction {
        self.rules
            .iter()
            .filter(|r| (r.condition)(score, meta))
            .max_by_key(|r| r.priority)
            .map(|r| r.action.clone())
            .unwrap_or(LifecycleAction::Keep)
    }

    pub fn default_rules() -> Vec<LifecycleRule> {
        vec![
            LifecycleRule {
                name: "freeze".into(),
                condition: Box::new(|s, m| {
                    s.tenant_priority > 0.9 || m.tags.contains(&"pinned".to_string())
                }),
                action: LifecycleAction::Freeze,
                priority: 100,
            },
            LifecycleRule {
                name: "consolidate".into(),
                condition: Box::new(|s, _| s.composite < 0.2),
                action: LifecycleAction::Consolidate,
                priority: 50,
            },
            LifecycleRule {
                name: "archive".into(),
                condition: Box::new(|s, _| s.recency < 0.1 && s.access_frequency < 0.1),
                action: LifecycleAction::Archive,
                priority: 30,
            },
            LifecycleRule {
                name: "delete".into(),
                condition: Box::new(|s, _| s.composite < 0.05),
                action: LifecycleAction::Delete,
                priority: 20,
            },
        ]
    }
}

// ===========================================================================
// 旧类型（deprecated）
// ===========================================================================

#[deprecated(note = "使用 ForgettingModel trait 替代")]
#[derive(Debug, Clone)]
pub struct ForgettingCurve {
    pub stability: f64,
    pub reinforcements: u32,
}

#[allow(deprecated)]
impl ForgettingCurve {
    pub fn new() -> Self {
        Self {
            stability: 7.0,
            reinforcements: 0,
        }
    }
    pub fn retention(&self, created_at: DateTime<Utc>, now: DateTime<Utc>) -> f64 {
        let elapsed = (now - created_at).num_seconds() as f64 / 86400.0;
        let s_eff = self.stability * (1.0 + 0.5 * self.reinforcements as f64);
        (-elapsed / s_eff).exp().clamp(0.0, 1.0)
    }
    pub fn reinforce(&mut self) {
        self.reinforcements = self.reinforcements.saturating_add(1);
    }
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self::new()
    }
}

#[deprecated(note = "使用 LifecycleAction 替代")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradeAction {
    Downgrade,
    Archive,
    Delete,
    Keep,
}

#[deprecated(note = "使用 LifecycleEngine 替代")]
pub struct LifecyclePolicy {
    pub curve: ForgettingCurve,
}

#[allow(deprecated)]
impl LifecyclePolicy {
    pub fn new() -> Self {
        Self {
            curve: ForgettingCurve::new(),
        }
    }
    pub fn evaluate(&self, entry: &ContextEntry, now: DateTime<Utc>) -> LifecycleAction {
        let r = self.curve.retention(entry.created_at, now);
        if r < 0.05 {
            LifecycleAction::Delete
        } else if r < 0.15 {
            LifecycleAction::Archive
        } else if r < 0.35 {
            LifecycleAction::Downgrade {
                to_level: ContentLevel::L0,
            }
        } else {
            LifecycleAction::Keep
        }
    }
}

impl Default for LifecyclePolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Token 经济
// ===========================================================================

#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub total: usize,
    pub used: usize,
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
        Self {
            total,
            used: 0,
            breakdown: TokenBreakdown::default(),
        }
    }
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.used)
    }
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }
    pub fn pressure(&self) -> f32 {
        if self.total == 0 {
            1.0
        } else {
            (self.used as f32 / self.total as f32).clamp(0.0, 1.0)
        }
    }
    pub fn reserve(&mut self, amount: usize) -> bool {
        if self.remaining() >= amount {
            self.used += amount;
            true
        } else {
            false
        }
    }
}
