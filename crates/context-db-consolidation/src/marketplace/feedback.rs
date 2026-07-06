//! ReputationEngine + MarketFeedback — 采纳反馈→声誉更新（KPI 驱动）。

use crate::marketplace::types::*;
use std::collections::HashMap;
use std::sync::Arc;

/// 采纳结果。
#[derive(Debug, Clone)]
pub enum AdoptionOutcome {
    Adopted { confidence_gain: f32 },
    Rejected { reason: String },
    Outdated,
    Contradicted,
}

/// 市场反馈 — Agent 使用市场知识后提交。
#[derive(Debug, Clone)]
pub struct MarketFeedback {
    pub entry_id: MarketId,
    pub consumer: AgentId,
    pub outcome: AdoptionOutcome,
    pub evidence: Option<agent_context_db_core::ContextUri>,
}

/// 声誉引擎 — 多维 KPI 追踪 + ReputationBond 管理。
pub struct ReputationEngine {
    kpis: parking_lot::RwLock<HashMap<AgentId, ReputationKpi>>,
    bonds: parking_lot::RwLock<HashMap<AgentId, ReputationBond>>,
}

impl ReputationEngine {
    pub fn new() -> Self {
        Self {
            kpis: parking_lot::RwLock::new(HashMap::new()),
            bonds: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// 注册新 Agent。
    pub fn register(&self, agent: AgentId) {
        let mut kpis = self.kpis.write();
        kpis.entry(agent.clone()).or_insert_with(|| ReputationKpi {
            agent: agent.clone(),
            last_active: chrono::Utc::now(),
            ..Default::default()
        });
        self.bonds.write().entry(agent.clone()).or_insert_with(|| ReputationBond::new(agent));
    }

    /// 处理一条反馈。
    pub fn process_feedback(&self, feedback: &MarketFeedback) {
        let mut kpis = self.kpis.write();
        if let Some(kpi) = kpis.get_mut(&feedback.consumer) {
            kpi.last_active = chrono::Utc::now();
        }

        match &feedback.outcome {
            AdoptionOutcome::Adopted { confidence_gain: _ } => {
                self.boost(&feedback.consumer);
            }
            AdoptionOutcome::Rejected { .. } | AdoptionOutcome::Contradicted => {
                self.ding(&feedback.consumer);
            }
            AdoptionOutcome::Outdated => {
                // 无惩罚，自然衰减
            }
        }
    }

    /// 提升发布者声誉。
    pub fn boost(&self, agent: &str) {
        let mut kpis = self.kpis.write();
        if let Some(kpi) = kpis.get_mut(agent) {
            kpi.entries_published = kpi.entries_published.saturating_add(1);
            kpi.adoption_rate = (kpi.adoption_rate * 0.9 + 0.1).min(1.0);
            kpi.recompute();
        }
    }

    /// 降低发布者声誉。
    pub fn ding(&self, agent: &str) {
        let mut kpis = self.kpis.write();
        if let Some(kpi) = kpis.get_mut(agent) {
            kpi.downvote_count = kpi.downvote_count.saturating_add(1);
            kpi.adoption_rate = (kpi.adoption_rate * 0.9).max(0.0);
            kpi.recompute();
        }
    }

    /// 记录矛盾。
    pub fn record_contradiction(&self, agent: &str) {
        let mut kpis = self.kpis.write();
        if let Some(kpi) = kpis.get_mut(agent) {
            kpi.contradiction_count = kpi.contradiction_count.saturating_add(1);
            kpi.recompute();

            // 一次确认的矛盾 → 降级 ReputationBond
            let mut bonds = self.bonds.write();
            if let Some(bond) = bonds.get_mut(agent) {
                bond.demote(BondLevel::Contributor);
            }
        }
    }

    /// 记录抗体贡献。
    pub fn record_immune_contribution(&self, agent: &str) {
        let mut kpis = self.kpis.write();
        if let Some(kpi) = kpis.get_mut(agent) {
            kpi.immune_contributions = kpi.immune_contributions.saturating_add(1);
            kpi.recompute();
        }
    }

    /// 获取 Agent 的声誉 KPI。
    pub fn get_kpi(&self, agent: &str) -> Option<ReputationKpi> {
        self.kpis.read().get(agent).cloned()
    }

    /// 重新计算全部分声誉债券。
    pub fn recalc_bonds(&self) {
        let kpis = self.kpis.read();
        let mut bonds = self.bonds.write();
        for (agent, kpi) in kpis.iter() {
            if let Some(bond) = bonds.get_mut(agent) {
                bond.promote(kpi);
            }
        }
    }

    /// Top-N 高声誉 Agent。
    pub fn top_agents(&self, n: usize) -> Vec<(AgentId, f32)> {
        let kpis = self.kpis.read();
        let mut sorted: Vec<_> = kpis.iter()
            .map(|(a, kpi)| (a.clone(), kpi.composite))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }
}
