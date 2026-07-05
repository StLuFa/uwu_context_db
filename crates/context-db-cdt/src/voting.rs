//! 投票演化 — Insight 持续投票，低分自动淘汰。

use agent_context_db_core::{ContentType, ContextUri};
use chrono::{DateTime, Utc};

/// 可演化 Insight。
#[derive(Debug, Clone)]
pub struct EvolvableInsight {
    pub uri: ContextUri,
    pub content: String,
    pub epistemic_type: ContentType,
    pub votes: VoteRecord,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct VoteRecord {
    pub upvotes: Vec<Vote>,
    pub downvotes: Vec<Vote>,
    pub net_score: f32,
}

#[derive(Debug, Clone)]
pub struct Vote {
    pub voter_uri: ContextUri,
    pub weight: f32,
    pub timestamp: DateTime<Utc>,
    pub evidence: Option<String>,
}

/// 投票操作。
pub enum VoteOp { Add(EvolvableInsight), Upvote(ContextUri, Vote), Downvote(ContextUri, Vote), Edit(ContextUri, String) }

impl EvolvableInsight {
    /// 加权净分 = Σ(upvote.weight) - Σ(downvote.weight)。
    pub fn recompute_score(&mut self) {
        let up: f32 = self.votes.upvotes.iter().map(|v| v.weight).sum();
        let down: f32 = self.votes.downvotes.iter().map(|v| v.weight).sum();
        self.votes.net_score = up - down;
    }
    /// 净分 ≤ 0 → 淘汰。
    pub fn should_deprecate(&self) -> bool { self.votes.net_score <= 0.0 }
}

/// 投票演化引擎。
pub struct InsightEvolutionEngine;

impl InsightEvolutionEngine {
    pub fn new() -> Self { Self }
    pub fn vote(&self, insight: &mut EvolvableInsight, op: VoteOp) {
        match op {
            VoteOp::Add(i) => *insight = i,
            VoteOp::Upvote(_, v) => { insight.votes.upvotes.push(v); insight.recompute_score(); }
            VoteOp::Downvote(_, v) => { insight.votes.downvotes.push(v); insight.recompute_score(); }
            VoteOp::Edit(_, new) => { insight.content = new; insight.last_updated = Utc::now(); }
        }
    }
    pub fn cleanup(insights: &mut Vec<EvolvableInsight>) -> usize {
        let before = insights.len();
        insights.retain(|i| !i.should_deprecate());
        before - insights.len()
    }
}
