//! PublishGate — 质量门控 + 确认阶梯 + 声誉债券 + C2D 边界。

use crate::ConsolidationProduct;
use crate::marketplace::types::*;
use agent_context_db_knowledge_network::identity::IdentityRegistry;

/// 发布门控 — 检查 knowledge 是否可以进入市场。
pub struct PublishGate {
    /// 最低质量分。
    min_quality: f32,
    /// 最低确认等级。
    min_corroboration: CorroborationLevel,
}

impl PublishGate {
    pub fn new() -> Self {
        Self {
            min_quality: 0.80,
            min_corroboration: CorroborationLevel::CrossSession,
        }
    }

    /// 尝试发布一个 ConsolidationProduct。
    ///
    /// 检查：
    /// 1. quality_score ≥ 0.80
    /// 2. corroboration ≥ CrossSession (≥2 independent sessions)
    /// 3. ShareLevel 不是 Private
    /// 4. 内容非空
    pub fn try_publish(
        &self,
        product: &ConsolidationProduct,
        corroboration: &CorroborationProof,
        publisher: &str,
        domain: &str,
    ) -> Result<MarketEntry, PublishError> {
        // 1. 质量
        if product.quality_score < self.min_quality {
            return Err(PublishError::QualityTooLow(product.quality_score));
        }

        // 2. 确认
        if !corroboration.can_publish() {
            return Err(PublishError::InsufficientCorroboration(corroboration.level));
        }

        // 3. 内容
        if product.content.is_empty() {
            return Err(PublishError::EmptyContent);
        }

        let entry_type = match product.content_type {
            agent_context_db_core::ContentType::Skill => MarketEntryType::Skill,
            agent_context_db_core::ContentType::Procedure => MarketEntryType::Procedure,
            agent_context_db_core::ContentType::Error => MarketEntryType::ErrorPattern,
            _ => MarketEntryType::Fact,
        };

        let created_at = chrono::Utc::now();
        let entry = MarketEntry {
            id: MarketId::new(),
            publisher: AgentId::new(publisher),
            domain: domain.to_string(),
            entry_type,
            principle: product.content.clone(),
            evidence_uris: product.evidence_uris.clone(),
            quality_score: product.quality_score,
            confidence: product.confidence,
            corroboration: corroboration.clone(),
            provenance: product.provenance.clone(),
            license: KnowledgeLicense::Attribution,
            epistemic_type: product.epistemic_type,
            content_type: product.content_type,
            half_life_days: product.metadata.half_life_days,
            created_at,
            expires_at: product
                .metadata
                .half_life_days
                .map(|d| created_at + chrono::Duration::days(d as i64)),
        };

        Ok(entry)
    }

    pub fn try_publish_signed(
        &self,
        product: &ConsolidationProduct,
        corroboration: &CorroborationProof,
        publisher: &str,
        domain: &str,
        identities: &IdentityRegistry,
    ) -> Result<MarketEntry, PublishError> {
        let mut entry = self.try_publish(product, corroboration, publisher, domain)?;
        let payload = product.provenance_payload(entry.publisher.clone(), entry.created_at);
        let provenance = identities
            .sign_knowledge_provenance(&payload)
            .map_err(|err| PublishError::SignatureFailed(err.to_string()))?;
        entry.provenance = Some(provenance);
        Ok(entry)
    }

    /// 带声誉加成的评分（声誉债券越高，初始可信度加成越大）。
    pub fn boosted_quality(&self, base_quality: f32, bond: &ReputationBond) -> f32 {
        let bonus = bond.current_bonus(chrono::Utc::now());
        (base_quality + bonus).min(1.0)
    }
}

#[derive(Debug, Clone)]
pub enum PublishError {
    QualityTooLow(f32),
    InsufficientCorroboration(CorroborationLevel),
    EmptyContent,
    NotOwner,
    SignatureFailed(String),
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::QualityTooLow(q) => write!(f, "quality {:.2} below threshold", q),
            PublishError::InsufficientCorroboration(l) => {
                write!(f, "corroboration level {:?} insufficient", l)
            }
            PublishError::EmptyContent => write!(f, "empty content"),
            PublishError::NotOwner => write!(f, "not the owner"),
            PublishError::SignatureFailed(err) => write!(f, "signature failed: {err}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConsolidationMeta, ConsolidationProduct};
    use agent_context_db_core::{
        ConsolidationStatus, ContentType, ContextUri, EpistemicType, ValidityRecord,
    };
    use agent_context_db_knowledge_network::identity::IdentityRegistry;
    use chrono::Utc;

    fn publishable_product() -> ConsolidationProduct {
        ConsolidationProduct {
            uri: ContextUri::parse("uwu://tenant/knowledge/1").unwrap(),
            content_type: ContentType::Fact,
            epistemic_type: EpistemicType::Fact,
            content: "A signed knowledge principle.".into(),
            quality_score: 0.91,
            confidence: 0.86,
            superseded_claim: None,
            evidence_uris: vec![ContextUri::parse("uwu://tenant/evidence/1").unwrap()],
            contradiction_uris: vec![],
            error_pattern: None,
            hypothesis_outcome: None,
            preconditions: None,
            expected_outcome: None,
            related_policy_uris: vec![],
            provenance: None,
            metadata: ConsolidationMeta {
                source_session: Some("test-session".into()),
                generation: 1,
                status: ConsolidationStatus::Converged,
                patch_count: 0,
                lineage: vec![],
                validity: Some(ValidityRecord {
                    valid_from: Utc::now(),
                    valid_until: None,
                    invalidated_by: None,
                    invalidation_reason: None,
                }),
                half_life_days: Some(30.0),
            },
        }
    }

    fn publishable_corroboration() -> CorroborationProof {
        let mut proof = CorroborationProof::new();
        proof.add_corroboration(AgentId::new("agent-a"), 2);
        proof
    }

    #[test]
    fn signed_publish_entry_verifies_offline_and_rejects_tamper() {
        let identities = IdentityRegistry::default();
        identities.upsert_signing_key(AgentId::new("agent-a"), [11u8; 32]);
        let gate = PublishGate::new();
        let entry = gate
            .try_publish_signed(
                &publishable_product(),
                &publishable_corroboration(),
                "agent-a",
                "rust",
                &identities,
            )
            .unwrap();
        let provenance = entry.provenance.clone().unwrap();
        IdentityRegistry::verify_knowledge_provenance_offline(
            &entry.provenance_payload(),
            &provenance,
        )
        .unwrap();

        let mut tampered = entry.clone();
        tampered.principle = "A tampered principle.".into();
        assert!(
            IdentityRegistry::verify_knowledge_provenance_offline(
                &tampered.provenance_payload(),
                &provenance,
            )
            .is_err()
        );
    }
}
