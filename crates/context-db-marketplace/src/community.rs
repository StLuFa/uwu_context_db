//! Community Detection + Speciation — 隐性知识领域发现 + 物种形成。
//!
//! 当多个 Agent 在相同领域发布 → 隐性的知识社区。
//! 某晶体 80%+ 命中来自同一 Agent/场景 → fork 特化版。

use crate::types::*;
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

/// 实际生成的特化分叉，包含市场条目和谱系节点。
#[derive(Debug, Clone)]
pub struct SpeciationFork {
    pub event: SpeciationEvent,
    pub entry: MarketEntry,
    pub lineage: LineageNode,
}

/// 社区检测器 — 基于 Agent-领域共现图的标签传播社区发现。
pub struct CommunityDetector {
    /// min_agents_per_community。
    min_agents: usize,
}

impl CommunityDetector {
    pub fn new() -> Self {
        Self { min_agents: 2 }
    }

    /// 从市场条目中检测社区。
    pub fn detect(&self, entries: &[MarketEntry]) -> Vec<Community> {
        let mut agent_domains: HashMap<AgentId, HashMap<String, usize>> = HashMap::new();
        let mut domain_entries: HashMap<String, HashSet<MarketId>> = HashMap::new();
        for entry in entries {
            *agent_domains
                .entry(entry.publisher.clone())
                .or_default()
                .entry(entry.domain.clone())
                .or_default() += 1;
            domain_entries
                .entry(entry.domain.clone())
                .or_default()
                .insert(entry.id);
        }

        if agent_domains.len() < self.min_agents {
            return Vec::new();
        }

        let agents = agent_domains.keys().cloned().collect::<Vec<_>>();
        let mut edges: HashMap<AgentId, HashMap<AgentId, f32>> = HashMap::new();
        for (idx, left) in agents.iter().enumerate() {
            for right in agents.iter().skip(idx + 1) {
                let weight = shared_domain_weight(&agent_domains[left], &agent_domains[right]);
                if weight <= 0.0 {
                    continue;
                }
                edges
                    .entry(left.clone())
                    .or_default()
                    .insert(right.clone(), weight);
                edges
                    .entry(right.clone())
                    .or_default()
                    .insert(left.clone(), weight);
            }
        }

        let mut labels = agents
            .iter()
            .map(|agent| (agent.clone(), agent.clone()))
            .collect::<HashMap<_, _>>();
        for _ in 0..16 {
            let mut changed = false;
            let mut ordered = agents.clone();
            ordered.sort_by(|a, b| a.as_str().cmp(b.as_str()));
            for agent in ordered {
                let Some(neighbors) = edges.get(&agent) else {
                    continue;
                };
                let mut label_scores: HashMap<AgentId, f32> = HashMap::new();
                for (neighbor, weight) in neighbors {
                    let label = labels.get(neighbor).cloned().unwrap_or_else(|| neighbor.clone());
                    *label_scores.entry(label).or_default() += *weight;
                }
                let Some((best_label, _)) = label_scores.into_iter().max_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b.0.as_str().cmp(a.0.as_str()))
                }) else {
                    continue;
                };
                if labels.get(&agent) != Some(&best_label) {
                    labels.insert(agent, best_label);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut grouped: HashMap<AgentId, Vec<AgentId>> = HashMap::new();
        for agent in agents {
            let label = labels.get(&agent).cloned().unwrap_or_else(|| agent.clone());
            grouped.entry(label).or_default().push(agent);
        }

        let mut communities = grouped
            .into_iter()
            .filter_map(|(label, mut members)| {
                members.sort_by(|a, b| a.as_str().cmp(b.as_str()));
                members.dedup();
                if members.len() < self.min_agents {
                    return None;
                }

                let member_set = members.iter().collect::<HashSet<_>>();
                let mut domain_scores: HashMap<String, usize> = HashMap::new();
                for member in &members {
                    for (domain, count) in &agent_domains[member] {
                        *domain_scores.entry(domain.clone()).or_default() += *count;
                    }
                }
                let mut domains = domain_scores.into_iter().collect::<Vec<_>>();
                domains.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let domains = domains.into_iter().map(|(domain, _)| domain).collect::<Vec<_>>();
                let domain_set = domains.iter().collect::<HashSet<_>>();
                let entry_count = entries
                    .iter()
                    .filter(|entry| {
                        member_set.contains(&entry.publisher) && domain_set.contains(&entry.domain)
                    })
                    .count();
                let internal_weight = community_internal_weight(&members, &edges);
                let max_edges = members.len() * (members.len() - 1) / 2;
                let density = if max_edges == 0 {
                    0.0
                } else {
                    (internal_weight / max_edges as f32).clamp(0.0, 1.0)
                };

                Some(Community {
                    id: format!("community-{}", sanitize_domain_segment(label.as_str())),
                    agents: members,
                    domains,
                    entry_count,
                    density,
                    bridge_entries: Vec::new(),
                })
            })
            .collect::<Vec<_>>();

        let domain_to_communities = communities
            .iter()
            .enumerate()
            .flat_map(|(idx, community)| {
                community
                    .domains
                    .iter()
                    .cloned()
                    .map(move |domain| (domain, idx))
            })
            .fold(HashMap::<String, Vec<usize>>::new(), |mut acc, (domain, idx)| {
                acc.entry(domain).or_default().push(idx);
                acc
            });
        for (domain, ids) in domain_to_communities {
            if ids.len() < 2 {
                continue;
            }
            let Some(entry_ids) = domain_entries.get(&domain) else {
                continue;
            };
            for idx in ids {
                communities[idx].bridge_entries.extend(entry_ids.iter().copied());
                communities[idx]
                    .bridge_entries
                    .sort_by(|left, right| left.0.cmp(&right.0));
                communities[idx].bridge_entries.dedup();
            }
        }

        communities.sort_by(|a, b| b.entry_count.cmp(&a.entry_count).then_with(|| a.id.cmp(&b.id)));
        communities
    }
}

fn shared_domain_weight(left: &HashMap<String, usize>, right: &HashMap<String, usize>) -> f32 {
    let shared = left
        .iter()
        .filter_map(|(domain, left_count)| {
            right
                .get(domain)
                .map(|right_count| (*left_count).min(*right_count) as f32)
        })
        .sum::<f32>();
    let total = left.values().sum::<usize>() + right.values().sum::<usize>();
    if total == 0 {
        0.0
    } else {
        (2.0 * shared / total as f32).clamp(0.0, 1.0)
    }
}

fn community_internal_weight(
    members: &[AgentId],
    edges: &HashMap<AgentId, HashMap<AgentId, f32>>,
) -> f32 {
    let mut weight = 0.0;
    for (idx, left) in members.iter().enumerate() {
        for right in members.iter().skip(idx + 1) {
            weight += edges
                .get(left)
                .and_then(|neighbors| neighbors.get(right))
                .copied()
                .unwrap_or(0.0);
        }
    }
    weight
}

/// 物种形成追踪器 — 检测何时需要 fork 特化版。
pub struct SpeciationTracker {
    hit_distribution: parking_lot::RwLock<HashMap<MarketId, HashMap<AgentId, usize>>>,
    entries: parking_lot::RwLock<HashMap<MarketId, MarketEntry>>,
    forks: parking_lot::RwLock<HashMap<(MarketId, AgentId), MarketId>>,
    speciation_threshold: f32,
}

impl SpeciationTracker {
    pub fn new() -> Self {
        Self {
            hit_distribution: parking_lot::RwLock::new(HashMap::new()),
            entries: parking_lot::RwLock::new(HashMap::new()),
            forks: parking_lot::RwLock::new(HashMap::new()),
            speciation_threshold: 0.8,
        }
    }

    /// 注册可被物种分化的源条目。
    pub fn register_entry(&self, entry: MarketEntry) {
        self.entries.write().insert(entry.id, entry);
    }

    /// 记录一次命中。
    pub fn record_hit(&self, entry_id: MarketId, agent: AgentId) {
        self.hit_distribution
            .write()
            .entry(entry_id)
            .or_default()
            .entry(agent)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// 检查是否需要物种形成。
    /// 某条目 80%+ 命中来自同一 Agent → fork 特化版。
    pub fn check_speciation(&self, entry_id: &MarketId) -> Option<SpeciationEvent> {
        let (agent, ratio) = self.dominant_agent(entry_id)?;
        let forked_entry = self.forks.read().get(&(*entry_id, agent.clone())).copied();
        Some(SpeciationEvent {
            original_entry: *entry_id,
            forked_entry,
            specialized_for_agent: agent.clone(),
            reason: format!("{}% of hits from {}", (ratio * 100.0) as usize, agent),
        })
    }

    /// 检查并生成特化 fork。重复调用同一 entry/agent 会返回 None，避免重复分叉。
    pub fn check_and_fork(&self, entry_id: &MarketId) -> Option<SpeciationFork> {
        let (agent, ratio) = self.dominant_agent(entry_id)?;
        if self.forks.read().contains_key(&(*entry_id, agent.clone())) {
            return None;
        }
        let source = self.entries.read().get(entry_id)?.clone();
        if !LicenseInfo::from(source.license.clone()).derivative_allowed {
            return None;
        }
        let mut forked = source.clone();
        forked.id = MarketId::new();
        forked.publisher = agent.clone();
        forked.domain = format!(
            "{}/agent/{}",
            source.domain,
            sanitize_domain_segment(agent.as_str())
        );
        forked.principle = format!("[specialized for {}] {}", agent, source.principle);
        forked.quality_score = (source.quality_score * (0.92 + ratio * 0.08)).clamp(0.0, 1.0);
        forked.confidence = (source.confidence * (0.88 + ratio * 0.10)).clamp(0.0, 1.0);
        forked.created_at = chrono::Utc::now();
        forked.provenance = None;

        self.forks
            .write()
            .insert((*entry_id, agent.clone()), forked.id);
        self.entries.write().insert(forked.id, forked.clone());

        let event = SpeciationEvent {
            original_entry: *entry_id,
            forked_entry: Some(forked.id),
            specialized_for_agent: agent.clone(),
            reason: format!("{}% of hits from {}", (ratio * 100.0) as usize, agent),
        };
        let lineage = LineageNode {
            market_id: forked.id,
            publisher: agent,
            action: LineageAction::Derived,
            parent_ids: vec![*entry_id],
            timestamp: forked.created_at,
        };
        Some(SpeciationFork {
            event,
            entry: forked,
            lineage,
        })
    }

    fn dominant_agent(&self, entry_id: &MarketId) -> Option<(AgentId, f32)> {
        let dist = self.hit_distribution.read();
        let hits = dist.get(entry_id)?;
        let total: usize = hits.values().sum();
        if total < 10 {
            return None;
        }
        hits.iter()
            .map(|(agent, count)| (agent.clone(), *count as f32 / total as f32))
            .filter(|(_, ratio)| *ratio >= self.speciation_threshold)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

fn sanitize_domain_segment(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "agent".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, EpistemicType};

    fn market_entry() -> MarketEntry {
        MarketEntry {
            id: MarketId::new(),
            publisher: AgentId::new("source-agent"),
            domain: "rust/retrieval".into(),
            entry_type: MarketEntryType::Skill,
            principle: "Prefer bounded graph traversal for relationship recall".into(),
            evidence_uris: vec![],
            quality_score: 0.82,
            confidence: 0.76,
            corroboration: CorroborationProof::new(),
            provenance: None,
            license: KnowledgeLicense::Attribution,
            epistemic_type: EpistemicType::Heuristic,
            content_type: ContentType::Skill,
            half_life_days: Some(90.0),
            created_at: chrono::Utc::now(),
            expires_at: None,
        }
    }

    #[test]
    fn speciation_tracker_creates_derived_fork_once() {
        let tracker = SpeciationTracker::new();
        let entry = market_entry();
        let original = entry.id;
        tracker.register_entry(entry);
        for _ in 0..9 {
            tracker.record_hit(original, AgentId::new("agent-a"));
        }
        tracker.record_hit(original, AgentId::new("agent-b"));

        let fork = tracker
            .check_and_fork(&original)
            .expect("dominant agent should fork");
        assert_eq!(fork.event.original_entry, original);
        assert_eq!(fork.event.forked_entry, Some(fork.entry.id));
        assert_eq!(fork.event.specialized_for_agent, AgentId::new("agent-a"));
        assert_eq!(fork.lineage.action, LineageAction::Derived);
        assert_eq!(fork.lineage.parent_ids, vec![original]);
        assert!(fork.entry.domain.ends_with("/agent/agent-a"));
        assert!(fork.entry.principle.contains("specialized for agent-a"));

        let event = tracker.check_speciation(&original).unwrap();
        assert_eq!(event.forked_entry, Some(fork.entry.id));
        assert!(tracker.check_and_fork(&original).is_none());
    }
}
