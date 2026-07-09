//! PublishGate — 质量门控 + 确认阶梯 + 声誉债券 + C2D 边界。

use crate::types::*;
use agent_context_db_core::{ContentType, ContextUri, EpistemicType};

/// 可发布的巩固产物端口。具体产物类型由 consolidation 或宿主实现。
pub trait PublishableProduct {
    fn quality_score(&self) -> f32;
    fn content(&self) -> &str;
    fn content_type(&self) -> ContentType;
    fn evidence_uris(&self) -> &[ContextUri];
    fn confidence(&self) -> f32;
    fn provenance(&self) -> Option<KnowledgeProvenance>;
    fn epistemic_type(&self) -> EpistemicType;
    fn half_life_days(&self) -> Option<f64>;
}

/// 知识来源签名端口。具体实现由 KnowledgeNetwork 或宿主注入。
pub trait KnowledgeSigner: Send + Sync {
    fn sign_knowledge_provenance(
        &self,
        payload: &KnowledgeProvenancePayload,
    ) -> std::result::Result<KnowledgeProvenance, String>;
}

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
        product: &dyn PublishableProduct,
        corroboration: &CorroborationProof,
        publisher: &str,
        domain: &str,
    ) -> Result<MarketEntry, PublishError> {
        // 1. 质量
        if product.quality_score() < self.min_quality {
            return Err(PublishError::QualityTooLow(product.quality_score()));
        }

        // 2. 确认
        if corroboration.level < self.min_corroboration {
            return Err(PublishError::InsufficientCorroboration(corroboration.level));
        }

        // 3. 内容
        if product.content().is_empty() {
            return Err(PublishError::EmptyContent);
        }

        let content_type = product.content_type();
        let entry_type = match content_type {
            ContentType::Skill => MarketEntryType::Skill,
            ContentType::Procedure => MarketEntryType::Procedure,
            ContentType::Error => MarketEntryType::ErrorPattern,
            _ => MarketEntryType::Fact,
        };

        let created_at = chrono::Utc::now();
        let entry = MarketEntry {
            id: MarketId::new(),
            publisher: AgentId::new(publisher),
            domain: domain.to_string(),
            entry_type,
            principle: product.content().to_string(),
            evidence_uris: product.evidence_uris().to_vec(),
            quality_score: product.quality_score(),
            confidence: product.confidence(),
            corroboration: corroboration.clone(),
            provenance: product.provenance(),
            license: KnowledgeLicense::Attribution,
            epistemic_type: product.epistemic_type(),
            content_type,
            half_life_days: product.half_life_days(),
            created_at,
            expires_at: product
                .half_life_days()
                .map(|d| created_at + chrono::Duration::days(d as i64)),
        };

        Ok(entry)
    }

    pub fn try_publish_signed(
        &self,
        product: &dyn PublishableProduct,
        corroboration: &CorroborationProof,
        publisher: &str,
        domain: &str,
        signer: &dyn KnowledgeSigner,
    ) -> Result<MarketEntry, PublishError> {
        let mut entry = self.try_publish(product, corroboration, publisher, domain)?;
        let payload = entry.provenance_payload();
        let provenance = signer
            .sign_knowledge_provenance(&payload)
            .map_err(PublishError::SignatureFailed)?;
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
    use chrono::Utc;

    struct TestProduct {
        evidence_uris: Vec<ContextUri>,
    }

    impl PublishableProduct for TestProduct {
        fn quality_score(&self) -> f32 {
            0.91
        }

        fn content(&self) -> &str {
            "A signed knowledge principle."
        }

        fn content_type(&self) -> ContentType {
            ContentType::Fact
        }

        fn evidence_uris(&self) -> &[ContextUri] {
            &self.evidence_uris
        }

        fn confidence(&self) -> f32 {
            0.86
        }

        fn provenance(&self) -> Option<KnowledgeProvenance> {
            None
        }

        fn epistemic_type(&self) -> EpistemicType {
            EpistemicType::Fact
        }

        fn half_life_days(&self) -> Option<f64> {
            Some(30.0)
        }
    }

    fn publishable_product() -> TestProduct {
        TestProduct {
            evidence_uris: vec![ContextUri::parse("uwu://tenant/evidence/1").unwrap()],
        }
    }

    fn publishable_corroboration() -> CorroborationProof {
        let mut proof = CorroborationProof::new();
        proof.add_corroboration(AgentId::new("agent-a"), 2);
        proof
    }

    struct TestSigner;

    impl KnowledgeSigner for TestSigner {
        fn sign_knowledge_provenance(
            &self,
            payload: &KnowledgeProvenancePayload,
        ) -> std::result::Result<KnowledgeProvenance, String> {
            Ok(KnowledgeProvenance {
                publisher: payload.publisher.clone(),
                public_key: "test-key".into(),
                signature: provenance_payload_hash(payload).map_err(|err| err.to_string())?,
                evidence_chain_hash: payload.evidence_chain_hash.clone(),
                signed_at: Utc::now(),
            })
        }
    }

    #[test]
    fn signed_publish_entry_attaches_signer_provenance() {
        let gate = PublishGate::new();
        let entry = gate
            .try_publish_signed(
                &publishable_product(),
                &publishable_corroboration(),
                "agent-a",
                "rust",
                &TestSigner,
            )
            .unwrap();
        let provenance = entry.provenance.clone().unwrap();
        assert_eq!(provenance.publisher, AgentId::new("agent-a"));
        assert_eq!(
            provenance.evidence_chain_hash,
            entry.provenance_payload().evidence_chain_hash
        );
        assert_eq!(
            provenance.signature,
            provenance_payload_hash(&entry.provenance_payload()).unwrap()
        );
    }
}
