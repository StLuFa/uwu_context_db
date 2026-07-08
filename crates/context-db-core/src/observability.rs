//! 可观测性模块（F13 质量评分 + F15 血缘图 + Metrics）。
//!
//! F9 ContextPubSub 已删除，使用 `crate::event_store`（基于 `uwu_event_mesh`）替代。

use crate::{ContentPayload, ContextEntry, ContextUri};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// F13 质量评分
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDimension {
    Completeness,
    Freshness,
    Consistency,
    Utility,
}

#[derive(Debug, Clone)]
pub struct QualityScore {
    pub dimensions: HashMap<QualityDimension, f32>,
    pub overall: f32,
    pub scored_at: DateTime<Utc>,
}

pub struct QualityScorer;

impl QualityScorer {
    pub fn score(entry: &ContextEntry, access_count: u64, now: DateTime<Utc>) -> QualityScore {
        let mut dims = HashMap::new();

        let completeness = if matches!(&entry.payload, ContentPayload::Text { dense, .. } if !dense.is_empty())
        {
            0.9
        } else {
            0.4
        };
        dims.insert(QualityDimension::Completeness, completeness);

        let age_days = (now - entry.updated_at).num_hours() as f32 / 24.0;
        let freshness = (-age_days / 30.0).exp().clamp(0.05, 1.0);
        dims.insert(QualityDimension::Freshness, freshness);

        let l0_text = entry.payload.sparse_text();
        let l0_len = l0_text.len() as f32;
        let consistency = if l0_len > 20.0 && l0_len < 2000.0 {
            0.85
        } else {
            0.5
        };
        dims.insert(QualityDimension::Consistency, consistency);

        let utility = ((access_count as f32 + 1.0).ln() / 5.0).clamp(0.1, 1.0);
        dims.insert(QualityDimension::Utility, utility);

        let overall = (completeness * 0.25 + freshness * 0.25 + consistency * 0.2 + utility * 0.3)
            .clamp(0.0, 1.0);

        QualityScore {
            dimensions: dims,
            overall,
            scored_at: now,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F15 血缘图（Provenance Graph）
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ProvenanceNode {
    pub uri: ContextUri,
    pub derived_from: Vec<ProvenanceEdge>,
    pub derived_to: Vec<ProvenanceEdge>,
}

#[derive(Debug, Clone)]
pub struct ProvenanceEdge {
    pub source: ContextUri,
    pub target: ContextUri,
    pub relation: ProvenanceRelationType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceRelationType {
    ExtractedFrom,
    MergedFrom,
    GeneratedBy,
    TriggeredBy,
    DerivedFrom,
}

pub struct ProvenanceGraph {
    nodes: parking_lot::RwLock<HashMap<String, ProvenanceNode>>, // B.1: 读多写少
}

impl ProvenanceGraph {
    pub fn new() -> Self {
        Self {
            nodes: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn add_derivation(
        &self,
        source: &ContextUri,
        target: &ContextUri,
        relation: ProvenanceRelationType,
    ) {
        let mut nodes = self.nodes.write();
        let edge = ProvenanceEdge {
            source: source.clone(),
            target: target.clone(),
            relation,
            timestamp: Utc::now(),
        };
        nodes
            .entry(source.to_string())
            .or_insert_with(|| ProvenanceNode {
                uri: source.clone(),
                derived_from: vec![],
                derived_to: vec![],
            })
            .derived_to
            .push(edge.clone());
        nodes
            .entry(target.to_string())
            .or_insert_with(|| ProvenanceNode {
                uri: target.clone(),
                derived_from: vec![],
                derived_to: vec![],
            })
            .derived_from
            .push(edge);
    }

    pub fn downstream(&self, root: &ContextUri, k: usize) -> Vec<ContextUri> {
        let nodes = self.nodes.read(); // B.1: 读锁
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![(root.clone(), 0)];
        while let Some((uri, depth)) = stack.pop() {
            if depth > k || !visited.insert(uri.to_string()) {
                continue;
            }
            if depth > 0 {
                result.push(uri.clone());
            }
            if let Some(node) = nodes.get(&uri.to_string()) {
                for edge in &node.derived_to {
                    stack.push((edge.target.clone(), depth + 1));
                }
            }
        }
        result
    }
}

impl Default for ProvenanceGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// I.2: 可观测性 — metrics crate 真实集成
// ═══════════════════════════════════════════════════════════════════════════

/// 记录一次检索操作。
pub fn record_retrieval(hits: usize, duration_ms: u64, tokens: usize, cache_hit: bool) {
    metrics::counter!("uwu.retrieval.requests").increment(1);
    metrics::counter!("uwu.retrieval.hits").increment(hits as u64);
    metrics::histogram!("uwu.retrieval.duration_ms").record(duration_ms as f64);
    metrics::counter!("uwu.retrieval.tokens").increment(tokens as u64);
    if cache_hit {
        metrics::counter!("uwu.cache.hit").increment(1);
    } else {
        metrics::counter!("uwu.cache.miss").increment(1);
    }
}

/// 记录一次写入操作。
pub fn record_write(_uri: &ContextUri, success: bool) {
    metrics::counter!("uwu.write.requests").increment(1);
    if success {
        metrics::counter!("uwu.write.success").increment(1);
    } else {
        metrics::counter!("uwu.write.failure").increment(1);
    }
}

/// 记录一次巩固操作。
pub fn record_consolidation(entries: usize, products: usize, duration_ms: u64) {
    metrics::counter!("uwu.consolidation.requests").increment(1);
    metrics::counter!("uwu.consolidation.entries").increment(entries as u64);
    metrics::counter!("uwu.consolidation.products").increment(products as u64);
    metrics::histogram!("uwu.consolidation.duration_ms").record(duration_ms as f64);
}

/// 记录一次 LLM 调用。
pub fn record_llm_call(provider: &str, tokens: usize, duration_ms: u64, success: bool) {
    metrics::counter!("uwu.llm.calls", "provider" => provider.to_string()).increment(1);
    metrics::counter!("uwu.llm.tokens", "provider" => provider.to_string())
        .increment(tokens as u64);
    metrics::histogram!("uwu.llm.duration_ms", "provider" => provider.to_string())
        .record(duration_ms as f64);
    if !success {
        metrics::counter!("uwu.llm.errors", "provider" => provider.to_string()).increment(1);
    }
}

/// 记录缓存命中/未命中。
pub fn record_cache(hit: bool) {
    if hit {
        metrics::counter!("uwu.cache.hit").increment(1);
    } else {
        metrics::counter!("uwu.cache.miss").increment(1);
    }
}
