//! Social Voting — 市场条目的众包投票。

use crate::types::*;
use std::collections::HashMap;

/// 投票操作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteOp {
    Upvote,
    Downvote,
    FlagOutdated,
}

/// 一票。
#[derive(Debug, Clone)]
pub struct Vote {
    pub voter: AgentId,
    pub entry_id: MarketId,
    pub op: VoteOp,
    pub weight: f32,
    pub evidence: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 投票统计。
#[derive(Debug, Clone, Default)]
pub struct VoteTally {
    pub upvotes: u32,
    pub downvotes: u32,
    pub weighted_score: f32,
    pub flags: u32,
}

/// 社会投票器。
pub struct SocialVoter {
    votes: parking_lot::RwLock<HashMap<MarketId, Vec<Vote>>>,
}

impl SocialVoter {
    pub fn new() -> Self {
        Self {
            votes: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// 投一票。
    pub fn vote(&self, vote: Vote) {
        self.votes
            .write()
            .entry(vote.entry_id)
            .or_default()
            .push(vote);
    }

    /// 统计票数。
    pub fn tally(&self, entry_id: &MarketId) -> VoteTally {
        let votes = self.votes.read();
        let entry_votes = votes.get(entry_id);
        let mut tally = VoteTally::default();
        if let Some(list) = entry_votes {
            for v in list {
                match v.op {
                    VoteOp::Upvote => {
                        tally.upvotes += 1;
                        tally.weighted_score += v.weight;
                    }
                    VoteOp::Downvote => {
                        tally.downvotes += 1;
                        tally.weighted_score -= v.weight;
                    }
                    VoteOp::FlagOutdated => {
                        tally.flags += 1;
                    }
                }
            }
        }
        tally
    }

    /// 共识因子 = 1 - variance（高共识 = 明确可信/不可信；低共识 = 有争议）。
    pub fn consensus_factor(&self, entry_id: &MarketId) -> f32 {
        let tally = self.tally(entry_id);
        let total = tally.upvotes + tally.downvotes;
        if total == 0 {
            return 0.0;
        }
        let p = tally.upvotes as f32 / total as f32;
        1.0 - 2.0 * (p - 0.5).abs() // 0=完全有争议, 1=高度共识
    }

    /// 应下架？（downvote > upvote + flagged）
    pub fn should_delist(&self, entry_id: &MarketId) -> bool {
        let tally = self.tally(entry_id);
        tally.downvotes > tally.upvotes + 3 || tally.flags >= 5
    }
}
