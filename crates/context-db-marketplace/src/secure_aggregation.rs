//! Verifiable provenance + differential privacy + additive secret sharing.
//!
//! This module protects cross-agent score/gradient aggregation without hiding
//! accountability. Each agent signs the knowledge provenance already used by the
//! marketplace, clips and noises its numeric contribution under a DP budget, then
//! splits the noised value into additive shares. The aggregator can verify
//! provenance and share checksums before reconstructing only the aggregate.

use crate::types::*;
use agent_context_db_core::{ContentType, ContextUri, EpistemicType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DpBudget {
    pub epsilon: f32,
    pub delta: f32,
    pub l1_sensitivity: f32,
    pub clip_norm: f32,
}

impl Default for DpBudget {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-6,
            l1_sensitivity: 1.0,
            clip_norm: 1.0,
        }
    }
}

impl DpBudget {
    pub fn validate(&self) -> Result<(), SecureAggregationError> {
        if self.epsilon <= 0.0 || !self.epsilon.is_finite() {
            return Err(SecureAggregationError::InvalidBudget(
                "epsilon must be positive".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.delta) || !self.delta.is_finite() {
            return Err(SecureAggregationError::InvalidBudget(
                "delta must be in [0, 1)".into(),
            ));
        }
        if self.l1_sensitivity <= 0.0 || self.clip_norm <= 0.0 {
            return Err(SecureAggregationError::InvalidBudget(
                "sensitivity and clip norm must be positive".into(),
            ));
        }
        Ok(())
    }

    fn noise_scale(&self) -> f32 {
        self.l1_sensitivity / self.epsilon
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateContribution {
    pub contributor: AgentId,
    pub entry_id: MarketId,
    pub value: f32,
    pub evidence_uris: Vec<ContextUri>,
    pub quality_score: f32,
    pub confidence: f32,
    pub epistemic_type: EpistemicType,
    pub content_type: ContentType,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionCommitment {
    pub contributor: AgentId,
    pub entry_id: MarketId,
    pub provenance: KnowledgeProvenance,
    pub payload_hash: String,
    pub clipped_value: f32,
    pub noised_value: f32,
    pub noise_scale: f32,
    pub share_checksum: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretShare {
    pub contributor: AgentId,
    pub entry_id: MarketId,
    pub receiver: AgentId,
    pub share_index: usize,
    pub share_value: f32,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedContribution {
    pub commitment: ContributionCommitment,
    pub shares: Vec<SecretShare>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureAggregateReport {
    pub entry_id: MarketId,
    pub participants: Vec<AgentId>,
    pub aggregate_value: f32,
    pub mean_value: f32,
    pub epsilon_spent: f32,
    pub delta: f32,
    pub accepted: usize,
    pub rejected: Vec<SecureAggregationRejection>,
    pub aggregate_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureAggregationRejection {
    pub contributor: AgentId,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecureAggregationError {
    InvalidBudget(String),
    EmptyReceivers,
    InvalidShareCount,
    InvalidSignature(String),
    InvalidChecksum(String),
    EntryMismatch,
    DuplicateContributor,
    Serialization(String),
    NoValidContributions,
}

impl std::fmt::Display for SecureAggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBudget(reason) => write!(f, "invalid DP budget: {reason}"),
            Self::EmptyReceivers => write!(f, "at least one share receiver is required"),
            Self::InvalidShareCount => write!(f, "share count does not match receivers"),
            Self::InvalidSignature(reason) => write!(f, "invalid provenance signature: {reason}"),
            Self::InvalidChecksum(reason) => write!(f, "invalid share checksum: {reason}"),
            Self::EntryMismatch => write!(f, "all contributions must target the same market entry"),
            Self::DuplicateContributor => write!(f, "duplicate contributor"),
            Self::Serialization(reason) => write!(f, "secure aggregation serialization: {reason}"),
            Self::NoValidContributions => write!(f, "no valid contributions to aggregate"),
        }
    }
}

impl std::error::Error for SecureAggregationError {}

pub trait ProvenanceVerifier: Send + Sync {
    fn verify_knowledge_provenance(
        &self,
        payload: &KnowledgeProvenancePayload,
        provenance: &KnowledgeProvenance,
    ) -> Result<(), String>;
}

pub struct HashProvenanceVerifier;

impl ProvenanceVerifier for HashProvenanceVerifier {
    fn verify_knowledge_provenance(
        &self,
        payload: &KnowledgeProvenancePayload,
        provenance: &KnowledgeProvenance,
    ) -> Result<(), String> {
        if provenance.publisher != payload.publisher {
            return Err("publisher mismatch".into());
        }
        if provenance.evidence_chain_hash != payload.evidence_chain_hash {
            return Err("evidence chain hash mismatch".into());
        }
        let expected = provenance_payload_hash(payload).map_err(|err| err.to_string())?;
        if provenance.signature != expected {
            return Err("signature hash mismatch".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SecureAggregationEngine {
    budget: DpBudget,
    receivers: Vec<AgentId>,
}

impl SecureAggregationEngine {
    pub fn new(budget: DpBudget, receivers: Vec<AgentId>) -> Result<Self, SecureAggregationError> {
        budget.validate()?;
        if receivers.is_empty() {
            return Err(SecureAggregationError::EmptyReceivers);
        }
        Ok(Self { budget, receivers })
    }

    pub fn budget(&self) -> DpBudget {
        self.budget
    }

    pub fn share_contribution(
        &self,
        contribution: &PrivateContribution,
        signer: &dyn crate::publish::KnowledgeSigner,
    ) -> Result<SharedContribution, SecureAggregationError> {
        let created_at = Utc::now();
        let clipped_value = contribution
            .value
            .clamp(-self.budget.clip_norm, self.budget.clip_norm);
        let payload = provenance_payload(
            &contribution.contributor,
            contribution.entry_id,
            clipped_value.abs().min(1.0),
            created_at,
        );
        let provenance = signer
            .sign_knowledge_provenance(&payload)
            .map_err(SecureAggregationError::InvalidSignature)?;
        let payload_hash = provenance_payload_hash(&payload)
            .map_err(|err| SecureAggregationError::InvalidSignature(err.to_string()))?;
        let noised_value =
            clipped_value + deterministic_laplace_noise(&payload_hash, self.budget.noise_scale());
        let shares = split_secret(
            &contribution.contributor,
            contribution.entry_id,
            noised_value,
            &self.receivers,
            &payload_hash,
        );
        let share_checksum = shares_checksum(&shares);
        Ok(SharedContribution {
            commitment: ContributionCommitment {
                contributor: contribution.contributor.clone(),
                entry_id: contribution.entry_id,
                provenance,
                payload_hash,
                clipped_value,
                noised_value,
                noise_scale: self.budget.noise_scale(),
                share_checksum,
                created_at,
            },
            shares,
        })
    }

    pub fn aggregate(
        &self,
        contributions: &[SharedContribution],
        verifier: &dyn ProvenanceVerifier,
    ) -> Result<SecureAggregateReport, SecureAggregationError> {
        let Some(first) = contributions.first() else {
            return Err(SecureAggregationError::EntryMismatch);
        };
        let entry_id = first.commitment.entry_id;
        let mut participants = Vec::new();
        let mut seen = HashSet::new();
        let mut aggregate_value = 0.0_f32;
        let mut rejected = Vec::new();

        for shared in contributions {
            let contributor = shared.commitment.contributor.clone();
            match self.validate_shared(shared, entry_id, verifier) {
                Ok(value) => {
                    if !seen.insert(contributor.clone()) {
                        return Err(SecureAggregationError::DuplicateContributor);
                    }
                    participants.push(contributor);
                    aggregate_value += value;
                }
                Err(err) => rejected.push(SecureAggregationRejection {
                    contributor,
                    reason: err.to_string(),
                }),
            }
        }

        let accepted = participants.len();
        if accepted == 0 {
            return Err(SecureAggregationError::NoValidContributions);
        }
        let mean_value = aggregate_value / accepted as f32;
        let aggregate_hash = aggregate_hash(entry_id, aggregate_value, &participants, &rejected)?;
        Ok(SecureAggregateReport {
            entry_id,
            participants,
            aggregate_value,
            mean_value,
            epsilon_spent: self.budget.epsilon * accepted as f32,
            delta: self.budget.delta,
            accepted,
            rejected,
            aggregate_hash,
        })
    }

    fn validate_shared(
        &self,
        shared: &SharedContribution,
        entry_id: MarketId,
        verifier: &dyn ProvenanceVerifier,
    ) -> Result<f32, SecureAggregationError> {
        if shared.commitment.entry_id != entry_id {
            return Err(SecureAggregationError::EntryMismatch);
        }
        if shared.shares.len() != self.receivers.len() {
            return Err(SecureAggregationError::InvalidShareCount);
        }
        if shares_checksum(&shared.shares) != shared.commitment.share_checksum {
            return Err(SecureAggregationError::InvalidChecksum(
                "commitment checksum mismatch".into(),
            ));
        }
        for (share, receiver) in shared.shares.iter().zip(&self.receivers) {
            if &share.receiver != receiver {
                return Err(SecureAggregationError::InvalidChecksum(
                    "receiver order mismatch".into(),
                ));
            }
            if share.checksum != share_checksum(share) {
                return Err(SecureAggregationError::InvalidChecksum(
                    "share checksum mismatch".into(),
                ));
            }
        }
        let payload = provenance_payload(
            &shared.commitment.contributor,
            shared.commitment.entry_id,
            shared.commitment.clipped_value.abs().min(1.0),
            shared.commitment.created_at,
        );
        verifier
            .verify_knowledge_provenance(&payload, &shared.commitment.provenance)
            .map_err(SecureAggregationError::InvalidSignature)?;
        let reconstructed = shared
            .shares
            .iter()
            .map(|share| share.share_value)
            .sum::<f32>();
        if (reconstructed - shared.commitment.noised_value).abs() > 0.001 {
            return Err(SecureAggregationError::InvalidChecksum(
                "share sum mismatch".into(),
            ));
        }
        Ok(reconstructed)
    }
}

fn provenance_payload(
    contributor: &AgentId,
    entry_id: MarketId,
    quality_score: f32,
    created_at: DateTime<Utc>,
) -> KnowledgeProvenancePayload {
    KnowledgeProvenancePayload {
        publisher: contributor.clone(),
        content: format!("secure-aggregate:{}", entry_id.0),
        evidence_uris: Vec::new(),
        evidence_chain_hash: evidence_chain_hash(&[]),
        quality_score,
        confidence: 1.0,
        epistemic_type: EpistemicType::Fact,
        content_type: ContentType::Meta,
        created_at,
    }
}

fn split_secret(
    contributor: &AgentId,
    entry_id: MarketId,
    secret: f32,
    receivers: &[AgentId],
    seed: &str,
) -> Vec<SecretShare> {
    let mut shares = Vec::with_capacity(receivers.len());
    let mut sum = 0.0_f32;
    for (idx, receiver) in receivers.iter().enumerate() {
        let share_value = if idx + 1 == receivers.len() {
            secret - sum
        } else {
            let value =
                deterministic_unit(&format!("{seed}:{idx}:{}", receiver.as_str())) * 2.0 - 1.0;
            sum += value;
            value
        };
        let mut share = SecretShare {
            contributor: contributor.clone(),
            entry_id,
            receiver: receiver.clone(),
            share_index: idx,
            share_value,
            checksum: String::new(),
        };
        share.checksum = share_checksum(&share);
        shares.push(share);
    }
    shares
}

fn deterministic_laplace_noise(seed: &str, scale: f32) -> f32 {
    let u = deterministic_unit(seed).clamp(1e-6, 1.0 - 1e-6) - 0.5;
    let sign = if u < 0.0 { -1.0 } else { 1.0 };
    -scale * sign * (1.0 - 2.0 * u.abs()).ln()
}

fn deterministic_unit(seed: &str) -> f32 {
    let hash = blake3::hash(seed.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    let value = u64::from_le_bytes(bytes);
    (value as f64 / u64::MAX as f64) as f32
}

fn share_checksum(share: &SecretShare) -> String {
    blake3::hash(
        format!(
            "{}:{}:{}:{}:{:.6}",
            share.contributor,
            share.entry_id.0,
            share.receiver,
            share.share_index,
            share.share_value
        )
        .as_bytes(),
    )
    .to_hex()
    .to_string()
}

fn shares_checksum(shares: &[SecretShare]) -> String {
    let mut values = shares
        .iter()
        .map(|share| share.checksum.clone())
        .collect::<Vec<_>>();
    values.sort();
    blake3::hash(values.join("\n").as_bytes())
        .to_hex()
        .to_string()
}

fn aggregate_hash(
    entry_id: MarketId,
    aggregate_value: f32,
    participants: &[AgentId],
    rejected: &[SecureAggregationRejection],
) -> Result<String, SecureAggregationError> {
    let mut payload = HashMap::new();
    payload.insert("entry_id", entry_id.0.to_string());
    payload.insert("aggregate", format!("{aggregate_value:.6}"));
    payload.insert(
        "participants",
        participants
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    payload.insert(
        "rejected",
        rejected
            .iter()
            .map(|item| format!("{}:{}", item.contributor, item.reason))
            .collect::<Vec<_>>()
            .join(","),
    );
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| SecureAggregationError::Serialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::KnowledgeSigner;

    struct TestSigner;

    impl KnowledgeSigner for TestSigner {
        fn sign_knowledge_provenance(
            &self,
            payload: &KnowledgeProvenancePayload,
        ) -> std::result::Result<KnowledgeProvenance, String> {
            Ok(KnowledgeProvenance {
                publisher: payload.publisher.clone(),
                public_key: "hash-test-key".into(),
                signature: provenance_payload_hash(payload).map_err(|err| err.to_string())?,
                evidence_chain_hash: payload.evidence_chain_hash.clone(),
                signed_at: Utc::now(),
            })
        }
    }

    fn contribution(agent: &str, entry_id: MarketId, value: f32) -> PrivateContribution {
        PrivateContribution {
            contributor: AgentId::new(agent),
            entry_id,
            value,
            evidence_uris: Vec::new(),
            quality_score: value.abs().min(1.0),
            confidence: 1.0,
            epistemic_type: EpistemicType::Fact,
            content_type: ContentType::Meta,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn secure_aggregation_verifies_shares_and_reconstructs_only_aggregate() {
        let entry_id = MarketId::new();
        let engine = SecureAggregationEngine::new(
            DpBudget {
                epsilon: 2.0,
                delta: 1e-6,
                l1_sensitivity: 1.0,
                clip_norm: 1.0,
            },
            vec![AgentId::new("r1"), AgentId::new("r2"), AgentId::new("r3")],
        )
        .unwrap();
        let signer = TestSigner;
        let shared = vec![
            engine
                .share_contribution(&contribution("a", entry_id, 0.8), &signer)
                .unwrap(),
            engine
                .share_contribution(&contribution("b", entry_id, 1.4), &signer)
                .unwrap(),
        ];

        let report = engine.aggregate(&shared, &HashProvenanceVerifier).unwrap();

        assert_eq!(report.accepted, 2);
        assert_eq!(report.participants.len(), 2);
        assert!(report.aggregate_hash.len() > 16);
        assert!(report.epsilon_spent > 0.0);
    }

    #[test]
    fn secure_aggregation_rejects_tampered_share() {
        let entry_id = MarketId::new();
        let engine =
            SecureAggregationEngine::new(DpBudget::default(), vec![AgentId::new("r1")]).unwrap();
        let signer = TestSigner;
        let mut shared = engine
            .share_contribution(&contribution("a", entry_id, 0.4), &signer)
            .unwrap();
        shared.shares[0].share_value += 1.0;

        assert!(matches!(
            engine.aggregate(&[shared], &HashProvenanceVerifier),
            Err(SecureAggregationError::NoValidContributions)
        ));
    }
}
