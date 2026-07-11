//! Social Voting — authenticated, replay-safe marketplace voting.

use crate::secure_aggregation::PrivateContribution;
use crate::types::*;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// A voter's current choice. Submitting another valid request for the same
/// `(entry_id, voter)` replaces the old choice rather than creating a new vote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteOp {
    Upvote,
    Downvote,
    FlagOutdated,
}

/// The signed portion of a vote. Weight is deliberately absent: it is assigned
/// by the trusted server-side reputation policy after authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotePayload {
    pub voter: AgentId,
    pub entry_id: MarketId,
    pub op: VoteOp,
    pub evidence: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub replay_id: Uuid,
}

#[derive(Debug, Clone)]
pub struct SignedVote {
    pub payload: VotePayload,
    pub public_key: String,
    pub signature: String,
}

/// Persisted, server-authorized vote.
#[derive(Debug, Clone)]
pub struct Vote {
    pub voter: AgentId,
    pub entry_id: MarketId,
    pub op: VoteOp,
    pub weight: f32,
    pub evidence: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub replay_id: Uuid,
}

pub trait VoteSignatureVerifier: Send + Sync {
    /// Must validate both the signature and that `public_key` belongs to
    /// `payload.voter`; checking a signature without identity binding is unsafe.
    fn verify(&self, payload: &VotePayload, public_key: &str, signature: &str) -> bool;
}

pub trait VoteReputationPolicy: Send + Sync {
    /// Returns a finite weight in the inclusive range `[0, 1]`.
    fn weight_for(&self, voter: &AgentId) -> Result<f32, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum VoteError {
    InvalidSignature,
    Stale,
    FutureDated,
    Replay,
    InvalidWeight(f32),
    Reputation(String),
}

#[derive(Debug, Clone, Default)]
pub struct VoteTally {
    pub upvotes: u32,
    pub downvotes: u32,
    pub weighted_score: f32,
    pub flags: u32,
}

#[derive(Default)]
struct VoteState {
    votes: HashMap<(MarketId, AgentId), Vote>,
    replay_ids: HashSet<Uuid>,
}

#[derive(Debug, Clone)]
pub struct VotingConfig {
    pub max_age: Duration,
    pub max_future_skew: Duration,
    pub downvote_margin: u32,
    pub flag_threshold: u32,
    pub outdated_flag_weight: f32,
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            max_age: Duration::minutes(5),
            max_future_skew: Duration::seconds(30),
            downvote_margin: 3,
            flag_threshold: 5,
            outdated_flag_weight: 0.5,
        }
    }
}

impl VotingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_age <= Duration::zero() || self.max_future_skew < Duration::zero() {
            return Err("voting time bounds are invalid".into());
        }
        if self.flag_threshold == 0 {
            return Err("flag_threshold must be non-zero".into());
        }
        if !self.outdated_flag_weight.is_finite()
            || !(0.0..=1.0).contains(&self.outdated_flag_weight)
        {
            return Err("outdated_flag_weight must be finite and in [0, 1]".into());
        }
        Ok(())
    }
}

/// Authenticated social voter. All externally reachable voting goes through
/// `submit`; there is no API that accepts a caller-selected weight.
pub struct SocialVoter {
    state: parking_lot::RwLock<VoteState>,
    verifier: Box<dyn VoteSignatureVerifier>,
    reputation: Box<dyn VoteReputationPolicy>,
    config: VotingConfig,
}

impl SocialVoter {
    pub fn new(
        verifier: Box<dyn VoteSignatureVerifier>,
        reputation: Box<dyn VoteReputationPolicy>,
        config: VotingConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            state: parking_lot::RwLock::new(VoteState::default()),
            verifier,
            reputation,
            config,
        })
    }

    /// Verifies identity/signature, freshness and replay protection, computes
    /// weight from server reputation, then atomically inserts or replaces the
    /// unique `(entry_id, voter)` vote.
    pub fn submit(&self, signed: SignedVote, now: DateTime<Utc>) -> Result<(), VoteError> {
        let payload = signed.payload;
        if payload.timestamp < now - self.config.max_age {
            return Err(VoteError::Stale);
        }
        if payload.timestamp > now + self.config.max_future_skew {
            return Err(VoteError::FutureDated);
        }
        if !self
            .verifier
            .verify(&payload, &signed.public_key, &signed.signature)
        {
            return Err(VoteError::InvalidSignature);
        }
        let weight = self
            .reputation
            .weight_for(&payload.voter)
            .map_err(VoteError::Reputation)?;
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(VoteError::InvalidWeight(weight));
        }

        let mut state = self.state.write();
        if !state.replay_ids.insert(payload.replay_id) {
            return Err(VoteError::Replay);
        }
        let vote = Vote {
            voter: payload.voter.clone(),
            entry_id: payload.entry_id,
            op: payload.op,
            weight,
            evidence: payload.evidence,
            timestamp: payload.timestamp,
            replay_id: payload.replay_id,
        };
        state
            .votes
            .insert((vote.entry_id, vote.voter.clone()), vote);
        Ok(())
    }

    pub fn tally(&self, entry_id: &MarketId) -> VoteTally {
        let state = self.state.read();
        let mut tally = VoteTally::default();
        for vote in state
            .votes
            .values()
            .filter(|vote| &vote.entry_id == entry_id)
        {
            match vote.op {
                VoteOp::Upvote => {
                    tally.upvotes += 1;
                    tally.weighted_score += vote.weight;
                }
                VoteOp::Downvote => {
                    tally.downvotes += 1;
                    tally.weighted_score -= vote.weight;
                }
                VoteOp::FlagOutdated => tally.flags += 1,
            }
        }
        tally
    }

    /// Consensus is zero for an even split and one for unanimous directional votes.
    pub fn consensus_factor(&self, entry_id: &MarketId) -> f32 {
        let tally = self.tally(entry_id);
        let total = tally.upvotes + tally.downvotes;
        if total == 0 {
            return 0.0;
        }
        let p = tally.upvotes as f32 / total as f32;
        2.0 * (p - 0.5).abs()
    }

    pub fn private_vote_contributions(&self, entry: &MarketEntry) -> Vec<PrivateContribution> {
        self.state
            .read()
            .votes
            .values()
            .filter(|vote| vote.entry_id == entry.id)
            .cloned()
            .map(|vote| PrivateContribution {
                contributor: vote.voter,
                entry_id: entry.id,
                value: match vote.op {
                    VoteOp::Upvote => vote.weight,
                    VoteOp::Downvote => -vote.weight,
                    VoteOp::FlagOutdated => -self.config.outdated_flag_weight * vote.weight,
                },
                evidence_uris: entry.evidence_uris.clone(),
                quality_score: entry.quality_score,
                confidence: entry.confidence,
                epistemic_type: entry.epistemic_type,
                content_type: entry.content_type,
                created_at: vote.timestamp,
            })
            .collect()
    }

    pub fn should_delist(&self, entry_id: &MarketId) -> bool {
        let tally = self.tally(entry_id);
        tally.downvotes > tally.upvotes + self.config.downvote_margin
            || tally.flags >= self.config.flag_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Verifier;
    impl VoteSignatureVerifier for Verifier {
        fn verify(&self, p: &VotePayload, key: &str, signature: &str) -> bool {
            key == p.voter.as_str() && signature == "valid"
        }
    }
    struct Reputation(f32);
    impl VoteReputationPolicy for Reputation {
        fn weight_for(&self, _: &AgentId) -> Result<f32, String> {
            Ok(self.0)
        }
    }
    fn request(entry_id: MarketId, voter: &str, op: VoteOp, now: DateTime<Utc>) -> SignedVote {
        SignedVote {
            payload: VotePayload {
                voter: AgentId::new(voter),
                entry_id,
                op,
                evidence: None,
                timestamp: now,
                replay_id: Uuid::new_v4(),
            },
            public_key: voter.into(),
            signature: "valid".into(),
        }
    }

    #[test]
    fn repeat_voter_updates_instead_of_stacking() {
        let voter = SocialVoter::new(
            Box::new(Verifier),
            Box::new(Reputation(0.7)),
            VotingConfig::default(),
        )
        .unwrap();
        let now = Utc::now();
        let entry = MarketId::new();
        voter
            .submit(request(entry, "a", VoteOp::Upvote, now), now)
            .unwrap();
        voter
            .submit(request(entry, "a", VoteOp::Downvote, now), now)
            .unwrap();
        let tally = voter.tally(&entry);
        assert_eq!((tally.upvotes, tally.downvotes), (0, 1));
        assert!((tally.weighted_score + 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn consensus_is_zero_without_direction_or_for_even_split_and_one_when_unanimous() {
        let voter = SocialVoter::new(
            Box::new(Verifier),
            Box::new(Reputation(1.0)),
            VotingConfig::default(),
        )
        .unwrap();
        let now = Utc::now();
        let entry = MarketId::new();
        assert_eq!(voter.consensus_factor(&entry), 0.0);
        voter
            .submit(request(entry, "a", VoteOp::Upvote, now), now)
            .unwrap();
        assert_eq!(voter.consensus_factor(&entry), 1.0);
        voter
            .submit(request(entry, "b", VoteOp::Downvote, now), now)
            .unwrap();
        assert_eq!(voter.consensus_factor(&entry), 0.0);
        voter
            .submit(request(entry, "c", VoteOp::Upvote, now), now)
            .unwrap();
        assert!((voter.consensus_factor(&entry) - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn rejects_bad_identity_stale_future_replay_and_illegal_weights() {
        let now = Utc::now();
        let entry = MarketId::new();
        let voter = SocialVoter::new(
            Box::new(Verifier),
            Box::new(Reputation(1.0)),
            VotingConfig::default(),
        )
        .unwrap();
        let mut bad = request(entry, "a", VoteOp::Upvote, now);
        bad.public_key = "b".into();
        assert_eq!(voter.submit(bad, now), Err(VoteError::InvalidSignature));
        assert_eq!(
            voter.submit(
                request(entry, "a", VoteOp::Upvote, now - Duration::minutes(6)),
                now
            ),
            Err(VoteError::Stale)
        );
        assert_eq!(
            voter.submit(
                request(entry, "a", VoteOp::Upvote, now + Duration::minutes(1)),
                now
            ),
            Err(VoteError::FutureDated)
        );
        let replay = request(entry, "a", VoteOp::Upvote, now);
        voter.submit(replay.clone(), now).unwrap();
        assert_eq!(voter.submit(replay, now), Err(VoteError::Replay));
        for weight in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            let invalid = SocialVoter::new(
                Box::new(Verifier),
                Box::new(Reputation(weight)),
                VotingConfig::default(),
            )
            .unwrap();
            assert!(matches!(
                invalid.submit(request(entry, "a", VoteOp::Upvote, now), now),
                Err(VoteError::InvalidWeight(_))
            ));
        }
    }
}
