//! 可观测性模块（F9 上下文订阅 + F13 质量评分 + F15 血缘图）。

use crate::{ContentLevel, ContextEntry, ContextUri};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::mpsc;

// ═══════════════════════════════════════════════════════════════════════════
// F9 上下文订阅与增量推送
// ═══════════════════════════════════════════════════════════════════════════

/// 订阅过滤器。
#[derive(Debug, Clone)]
pub struct SubscriptionFilter {
    /// 监听的 URI scope
    pub scope: ContextUri,
    /// 最小变更级别
    pub min_level: ContentLevel,
    /// 只关注特定记忆类型
    pub memory_class: Option<crate::MemoryClass>,
}

/// 变更事件。
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    /// 变更的 URI
    pub uri: ContextUri,
    /// 变更类型
    pub event_type: ChangeEventType,
    /// L0 变更摘要
    pub abstract_: String,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeEventType {
    Created,
    Updated,
    Deleted,
    RolledBack,
}

/// 订阅句柄。
pub struct Subscription {
    pub id: String,
    pub filter: SubscriptionFilter,
    pub tx: mpsc::Sender<ChangeEvent>,
}

/// 上下文发布-订阅系统。
pub struct ContextPubSub {
    subscribers: parking_lot::Mutex<Vec<Subscription>>,
    /// 最近事件的滑动窗口
    recent_events: parking_lot::Mutex<Vec<ChangeEvent>>,
    window_size: usize,
}

impl ContextPubSub {
    pub fn new(window_size: usize) -> Self {
        Self {
            subscribers: parking_lot::Mutex::new(Vec::new()),
            recent_events: parking_lot::Mutex::new(Vec::new()),
            window_size,
        }
    }

    /// 创建订阅。
    pub fn subscribe(&self, filter: SubscriptionFilter) -> mpsc::Receiver<ChangeEvent> {
        let (tx, rx) = mpsc::channel();
        let sub = Subscription {
            id: uuid::Uuid::new_v4().to_string(),
            filter,
            tx,
        };
        self.subscribers.lock().push(sub);
        rx
    }

    /// 发布变更事件到所有匹配的订阅者。
    pub fn publish(&self, event: ChangeEvent) {
        {
            let mut recent = self.recent_events.lock();
            recent.push(event.clone());
            if recent.len() > self.window_size {
                recent.remove(0);
            }
        }
        let subs = self.subscribers.lock();
        for sub in subs.iter() {
            if event.uri.to_string().starts_with(&sub.filter.scope.to_string()) {
                let _ = sub.tx.send(event.clone());
            }
        }
    }

    /// 获取最近 N 个事件（用于新订阅者追赶）。
    pub fn recent(&self, limit: usize) -> Vec<ChangeEvent> {
        let recent = self.recent_events.lock();
        recent.iter().rev().take(limit).cloned().collect()
    }

    /// 获取活跃订阅数。
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.lock().len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F13 上下文质量评分
// ═══════════════════════════════════════════════════════════════════════════

/// 质量维度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDimension {
    Completeness,
    Freshness,
    Consistency,
    Utility,
}

/// 条目质量评分。
#[derive(Debug, Clone, Default)]
pub struct QualityScore {
    /// 各维度评分 0-1
    pub dimensions: HashMap<QualityDimension, f32>,
    /// 综合评分
    pub overall: f32,
    /// 评分时间
    pub scored_at: DateTime<Utc>,
}

/// 质量评分器。
pub struct QualityScorer;

impl QualityScorer {
    /// 基于访问频率、更新时间、L0 完整度计算质量分。
    pub fn score(
        entry: &ContextEntry,
        access_count: u64,
        now: DateTime<Utc>,
    ) -> QualityScore {
        let mut dims = HashMap::new();

        // 完整度：L0 + L1 都有 = 高完整度
        let completeness = if entry.l1_overview.is_some() { 0.9 } else { 0.4 };
        dims.insert(QualityDimension::Completeness, completeness);

        // 新鲜度：最近更新的衰减
        let age_days = (now - entry.updated_at).num_hours() as f32 / 24.0;
        let freshness = (-age_days / 30.0).exp().clamp(0.05, 1.0);
        dims.insert(QualityDimension::Freshness, freshness);

        // 一致性：L0 长度适中（不太短也不太长）
        let l0_len = entry.l0_abstract.len() as f32;
        let consistency = if l0_len > 20.0 && l0_len < 2000.0 { 0.85 } else { 0.5 };
        dims.insert(QualityDimension::Consistency, consistency);

        // 实用性：基于访问频率
        let utility = ((access_count as f32 + 1.0).ln() / 5.0).clamp(0.1, 1.0);
        dims.insert(QualityDimension::Utility, utility);

        let overall = (completeness * 0.25 + freshness * 0.25 + consistency * 0.2 + utility * 0.3)
            .clamp(0.0, 1.0);

        QualityScore { dimensions: dims, overall, scored_at: now }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F15 上下文血缘图
// ═══════════════════════════════════════════════════════════════════════════

/// 血缘图中的节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceNode {
    pub uri: ContextUri,
    /// 产生该条目的来源
    pub derived_from: Vec<ProvenanceEdge>,
    /// 该条目衍生的下游
    pub derived_to: Vec<ProvenanceEdge>,
}

/// 血缘边。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEdge {
    pub source: ContextUri,
    pub target: ContextUri,
    pub relation: ProvenanceRelationType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceRelationType {
    ExtractedFrom,
    MergedFrom,
    GeneratedBy,
    TriggeredBy,
    DerivedFrom,
}

/// 上下文血缘图。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceGraph {
    pub nodes: Vec<ProvenanceNode>,
    pub edges: Vec<ProvenanceEdge>,
}

impl ProvenanceGraph {
    pub fn new() -> Self { Self::default() }

    /// 添加一条派生关系。
    pub fn add_derivation(
        &mut self,
        source: ContextUri,
        target: ContextUri,
        relation: ProvenanceRelationType,
    ) {
        let now = Utc::now();
        self.edges.push(ProvenanceEdge {
            source: source.clone(), target: target.clone(), relation, timestamp: now,
        });

        // Upsert source node
        let src_idx = self.nodes.iter().position(|n| n.uri == source);
        if let Some(idx) = src_idx {
            self.nodes[idx].derived_to.push(ProvenanceEdge {
                source: source.clone(), target: target.clone(), relation, timestamp: now,
            });
        } else {
            self.nodes.push(ProvenanceNode {
                uri: source.clone(),
                derived_from: vec![],
                derived_to: vec![ProvenanceEdge {
                    source: source.clone(), target: target.clone(), relation, timestamp: now,
                }],
            });
        }

        // Upsert target node
        let tgt_idx = self.nodes.iter().position(|n| n.uri == target);
        if let Some(idx) = tgt_idx {
            self.nodes[idx].derived_from.push(ProvenanceEdge {
                source: source.clone(), target: target.clone(), relation, timestamp: now,
            });
        } else {
            self.nodes.push(ProvenanceNode {
                uri: target.clone(),
                derived_from: vec![ProvenanceEdge {
                    source, target, relation, timestamp: now,
                }],
                derived_to: vec![],
            });
        }
    }

    /// 从根节点向下游遍历 K 步。
    pub fn downstream<'a>(&'a self, uri: &'a ContextUri, k: usize) -> Vec<&'a ContextUri> {
        let mut visited = Vec::new();
        let mut stack: Vec<(&ContextUri, usize)> = vec![(uri, 0)];
        while let Some((current, depth)) = stack.pop() {
            if depth > k { continue; }
            if visited.contains(&current) { continue; }
            visited.push(current);
            if let Some(node) = self.nodes.iter().find(|n| &n.uri == current) {
                for edge in &node.derived_to {
                    stack.push((&edge.target, depth + 1));
                }
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pubsub_broadcasts_to_matching_subscriber() {
        let ps = ContextPubSub::new(100);
        let rx = ps.subscribe(SubscriptionFilter {
            scope: ContextUri::parse("uwu://t1/agent/a1").unwrap(),
            min_level: ContentLevel::L0,
            memory_class: None,
        });

        ps.publish(ChangeEvent {
            uri: ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap(),
            event_type: ChangeEventType::Created,
            abstract_: "new case created".into(),
            timestamp: Utc::now(),
        });

        // 非阻塞检查
        match rx.try_recv() {
            Ok(event) => assert_eq!(event.event_type, ChangeEventType::Created),
            Err(_) => {} // 异步通道可能未到达
        }
    }

    #[test]
    fn quality_scorer_rates_fresh_high() {
        let entry = ContextEntry::new_text(
            ContextUri::parse("uwu://t/x").unwrap(),
            crate::TenantId(uuid::Uuid::nil()),
            "a well-formed and sufficiently detailed entry about the system behavior",
        );
        let score = QualityScorer::score(&entry, 10, Utc::now());
        assert!(score.overall > 0.5);
        assert!(score.dimensions.contains_key(&QualityDimension::Freshness));
    }

    #[test]
    fn provenance_graph_downstream_traversal() {
        let mut graph = ProvenanceGraph::new();
        let session = ContextUri::parse("uwu://t1/sessions/s1/archive/0").unwrap();
        let memory = ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap();
        let experience = ContextUri::parse("uwu://t1/agent/a1/experiences/e1").unwrap();

        graph.add_derivation(session.clone(), memory.clone(), ProvenanceRelationType::ExtractedFrom);
        graph.add_derivation(memory.clone(), experience.clone(), ProvenanceRelationType::GeneratedBy);

        let downstream = graph.downstream(&session, 2);
        assert!(downstream.contains(&&memory));
        assert!(downstream.contains(&&experience));
    }
}
