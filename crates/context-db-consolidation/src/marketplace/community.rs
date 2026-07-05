//! Community Detection + Speciation — 隐性知识领域发现 + 物种形成。
//!
//! 当多个 Agent 在相同领域发布 → 隐性的知识社区。
//! 某晶体 80%+ 命中来自同一 Agent/场景 → fork 特化版。

use crate::marketplace::types::*;
use std::collections::{HashMap, HashSet};

/// 知识社区。
#[derive(Debug, Clone)]
pub struct Community {
    pub id: String,
    /// 社区内的 Agent。
    pub agents: Vec<AgentId>,
    /// 核心领域。
    pub domains: Vec<String>,
    /// 社区内的条目数。
    pub entry_count: usize,
    /// 社区密度（边数 / 最大可能边数）。
    pub density: f32,
    /// 桥梁条目（连接不同社区的边界知识）。
    pub bridge_entries: Vec<MarketId>,
}

/// 物种形成事件。
#[derive(Debug, Clone)]
pub struct SpeciationEvent {
    pub original_entry: MarketId,
    pub forked_entry: Option<MarketId>,
    pub specialized_for_agent: AgentId,
    pub reason: String,
}

/// 社区检测器 — 基于 Agent-领域 共现的简化 Louvain 算法。
pub struct CommunityDetector {
    /// min_agents_per_community。
    min_agents: usize,
}

impl CommunityDetector {
    pub fn new() -> Self { Self { min_agents: 2 } }

    /// 从市场条目中检测社区。
    pub fn detect(&self, entries: &[MarketEntry]) -> Vec<Community> {
        // 构建 Agent-领域 共现图
        let mut agent_domains: HashMap<AgentId, HashSet<String>> = HashMap::new();
        for entry in entries {
            agent_domains.entry(entry.publisher.clone())
                .or_default()
                .insert(entry.domain.clone());
        }

        // 按领域分组 Agent
        let mut domain_agents: HashMap<String, Vec<AgentId>> = HashMap::new();
        for (agent, domains) in &agent_domains {
            for domain in domains {
                domain_agents.entry(domain.clone()).or_default().push(agent.clone());
            }
        }

        // 过滤：领域内 Agent 数 ≥ min_agents
        let mut communities = Vec::new();
        for (domain, agents) in &domain_agents {
            if agents.len() < self.min_agents { continue; }
            let mut unique_agents = agents.clone();
            unique_agents.dedup();

            let entry_count = entries.iter()
                .filter(|e| e.domain == *domain)
                .count();

            communities.push(Community {
                id: format!("community-{}", domain),
                agents: unique_agents.clone(),
                domains: vec![domain.clone()],
                entry_count,
                density: unique_agents.len() as f32 / entry_count.max(1) as f32,
                bridge_entries: vec![],
            });
        }

        // 找出桥梁条目（属于 ≥2 个社区的条目）
        for entry in entries {
            let count = communities.iter()
                .filter(|c| c.domains.contains(&entry.domain))
                .count();
            if count >= 2 {
                for c in &mut communities {
                    if c.domains.contains(&entry.domain) {
                        c.bridge_entries.push(entry.id);
                    }
                }
            }
        }

        communities
    }
}

/// 物种形成追踪器 — 检测何时需要 fork 特化版。
pub struct SpeciationTracker {
    hit_distribution: parking_lot::RwLock<HashMap<MarketId, HashMap<AgentId, usize>>>,
    speciation_threshold: f32,
}

impl SpeciationTracker {
    pub fn new() -> Self {
        Self {
            hit_distribution: parking_lot::RwLock::new(HashMap::new()),
            speciation_threshold: 0.8,
        }
    }

    /// 记录一次命中。
    pub fn record_hit(&self, entry_id: MarketId, agent: AgentId) {
        self.hit_distribution.write()
            .entry(entry_id)
            .or_default()
            .entry(agent)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// 检查是否需要物种形成。
    /// 某条目 80%+ 命中来自同一 Agent → fork 特化版。
    pub fn check_speciation(&self, entry_id: &MarketId) -> Option<SpeciationEvent> {
        let dist = self.hit_distribution.read();
        let hits = dist.get(entry_id)?;
        let total: usize = hits.values().sum();
        if total < 10 { return None; }

        for (agent, count) in hits {
            let ratio = *count as f32 / total as f32;
            if ratio >= self.speciation_threshold {
                return Some(SpeciationEvent {
                    original_entry: *entry_id,
                    forked_entry: None,
                    specialized_for_agent: agent.clone(),
                    reason: format!("{}% of hits from {}", (ratio * 100.0) as usize, agent),
                });
            }
        }
        None
    }
}
