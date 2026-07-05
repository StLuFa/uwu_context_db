//! CAP Consistency Levels — 按认识论类型选择一致性级别。
//!
//! C (Consistency) / A (Availability) / P (Partition tolerance) 的工程化应用：
//! 不同类型的知识需要不同的一致性保证。

use crate::marketplace::types::*;
use agent_context_db_core::{ContentType, EpistemicType};

/// 一致性级别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyLevel {
    /// 强一致：patch 立即广播，所有 Agent 同步。
    /// 适用：Fact — 一个 Agent 修正了事实，所有 Agent 必须立即知道。
    Strong,
    /// 最终一致：异步传播，短暂不一致可接受。
    /// 适用：Heuristic/Belief — 可以在几轮训练中逐步收敛。
    Eventual,
    /// 会话边界一致：同一 session 内保证一致，跨 session 允许不一致。
    /// 适用：Hypothesis — 不同 Agent 可能有不同假设，等待验证。
    SessionBound,
    /// 无一致性要求：各 Agent 独立维护。
    /// 适用：Preference/Profile — 个人偏好无需全局同步。
    None,
}

/// 按认识论类型选择默认一致性级别。
impl ConsistencyLevel {
    pub fn for_epistemic(et: EpistemicType) -> Self {
        match et {
            EpistemicType::Fact => ConsistencyLevel::Strong,
            EpistemicType::Heuristic | EpistemicType::Belief => ConsistencyLevel::Eventual,
            EpistemicType::Hypothesis => ConsistencyLevel::SessionBound,
            EpistemicType::Procedure => ConsistencyLevel::Eventual,
        }
    }

    pub fn for_content_type(ct: ContentType) -> Self {
        match ct {
            ContentType::Fact | ContentType::Error => ConsistencyLevel::Strong,
            ContentType::Skill | ContentType::Procedure | ContentType::Heuristic => ConsistencyLevel::Eventual,
            ContentType::Hypothesis => ConsistencyLevel::SessionBound,
            ContentType::Preference | ContentType::Profile | ContentType::Goal => ConsistencyLevel::None,
            ContentType::Belief => ConsistencyLevel::Eventual,
            _ => ConsistencyLevel::Eventual,
        }
    }

    /// 是否需要立即广播。
    pub fn requires_broadcast(&self) -> bool {
        matches!(self, ConsistencyLevel::Strong)
    }

    /// 是否允许本地缓存。
    pub fn allows_caching(&self) -> bool {
        !matches!(self, ConsistencyLevel::Strong)
    }

    /// 冲突时可自动合并还是需要仲裁。
    pub fn auto_merge(&self) -> bool {
        matches!(self, ConsistencyLevel::Eventual | ConsistencyLevel::None)
    }
}

/// CAP 策略引擎 — 根据知识类型自动选择一致性策略。
pub struct CapPolicyEngine;

impl CapPolicyEngine {
    /// 对 MarketEntry 选择一致性策略。
    pub fn for_entry(entry: &MarketEntry) -> ConsistencyLevel {
        ConsistencyLevel::for_content_type(entry.content_type)
    }

    /// 是否需要为这个条目牺牲可用性以保持一致性。
    /// （CAP 定理中的 CP 选择）
    pub fn should_sacrifice_availability(entry: &MarketEntry) -> bool {
        matches!(ConsistencyLevel::for_content_type(entry.content_type), ConsistencyLevel::Strong)
    }

    /// 批量选择：取最严格的一致性级别。
    pub fn strictest(entries: &[MarketEntry]) -> ConsistencyLevel {
        entries.iter()
            .map(|e| ConsistencyLevel::for_content_type(e.content_type))
            .max_by_key(|l| match l {
                ConsistencyLevel::Strong => 3,
                ConsistencyLevel::Eventual => 2,
                ConsistencyLevel::SessionBound => 1,
                ConsistencyLevel::None => 0,
            })
            .unwrap_or(ConsistencyLevel::Eventual)
    }
}
