//! CDT → Consolidation 适配层。
//!
//! CDT 训练会产生 gradient、reflection insight、agent 行为记忆和 synthesis 产物。
//! 这个模块把这些训练信号沉淀为 `ConsolidationProduct` / `ContextEntry`，让训练内循环能进入长期记忆巩固层。

use crate::reflection::SemanticGradient;
use crate::voting::EvolvableInsight;
use crate::{CognitiveGradient, GradientType, HypothesisOutcome};
use agent_context_db_consolidation::{
    ConsolidationMeta as ProductConsolidationMeta, ConsolidationProduct,
    HypothesisOutcome as ProductHypothesisOutcome,
};
use agent_context_db_core::{
    ConsolidationMeta as EntryConsolidationMeta, ConsolidationStatus, ContentType, ContextEntry,
    ContextUri, EpistemicType, LineageEntry, MvccVersion, StateScope, TenantId, ValidityRecord,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdtConsolidationSignal {
    pub uri: ContextUri,
    pub content_type: ContentType,
    pub epistemic_type: EpistemicType,
    pub content: String,
    pub quality_score: f32,
    pub confidence: f32,
    pub evidence_uris: Vec<ContextUri>,
    pub contradiction_uris: Vec<ContextUri>,
    pub source: CdtSignalSource,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdtSignalSource {
    Gradient,
    Reflexion,
    ExpeLInsight,
    GenAgent,
    StormSynthesis,
}

#[derive(Debug, Clone, Default)]
pub struct CdtConsolidationBatch {
    pub products: Vec<ConsolidationProduct>,
    pub entries: Vec<ContextEntry>,
}

#[derive(Debug, Clone)]
pub struct CdtConsolidationBridge {
    agent_scope: String,
    tenant: TenantId,
}

impl CdtConsolidationBridge {
    pub fn new(agent_scope: impl Into<String>, tenant: TenantId) -> Self {
        Self {
            agent_scope: agent_scope.into(),
            tenant,
        }
    }

    pub fn for_agent(agent_scope: impl Into<String>) -> Self {
        Self::new(agent_scope, TenantId(Uuid::new_v4()))
    }

    pub fn product_from_gradient(&self, gradient: &CognitiveGradient) -> ConsolidationProduct {
        let signal = self.signal_from_gradient(gradient);
        self.product_from_signal(&signal)
    }

    pub fn products_from_gradients(
        &self,
        gradients: &[CognitiveGradient],
    ) -> Vec<ConsolidationProduct> {
        gradients
            .iter()
            .map(|gradient| self.product_from_gradient(gradient))
            .collect()
    }

    pub fn signal_from_semantic_gradient(
        &self,
        index: usize,
        gradient: &SemanticGradient,
    ) -> CdtConsolidationSignal {
        CdtConsolidationSignal {
            uri: gradient
                .source_uri
                .clone()
                .unwrap_or_else(|| self.make_uri("reflection", index, &gradient.reflection_text)),
            content_type: ContentType::Reflection,
            epistemic_type: EpistemicType::Heuristic,
            content: format!(
                "REFLECTION: {}\nACTION: {}",
                gradient.reflection_text, gradient.action_improvement
            ),
            quality_score: gradient.priority.clamp(0.0, 1.0),
            confidence: gradient.priority.clamp(0.0, 1.0),
            evidence_uris: vec![],
            contradiction_uris: vec![],
            source: CdtSignalSource::Reflexion,
            tags: gradient.epistemic_tags.clone(),
        }
    }

    pub fn signal_from_insight(
        &self,
        index: usize,
        insight: &EvolvableInsight,
    ) -> CdtConsolidationSignal {
        CdtConsolidationSignal {
            uri: insight.uri.clone(),
            content_type: insight.epistemic_type,
            epistemic_type: epistemic_from_content_type(insight.epistemic_type),
            content: insight.content.clone(),
            quality_score: insight.votes.net_score.clamp(0.0, 1.0),
            confidence: insight.votes.net_score.clamp(0.0, 1.0),
            evidence_uris: vec![self.make_uri("reflection", index, &insight.content)],
            contradiction_uris: vec![],
            source: CdtSignalSource::ExpeLInsight,
            tags: insight.evidence.clone(),
        }
    }

    pub fn batch_from_signals(&self, signals: &[CdtConsolidationSignal]) -> CdtConsolidationBatch {
        CdtConsolidationBatch {
            products: signals
                .iter()
                .map(|signal| self.product_from_signal(signal))
                .collect(),
            entries: signals
                .iter()
                .map(|signal| self.entry_from_signal(signal))
                .collect(),
        }
    }

    pub fn product_from_signal(&self, signal: &CdtConsolidationSignal) -> ConsolidationProduct {
        let (
            superseded_claim,
            error_pattern,
            hypothesis_outcome,
            preconditions,
            expected_outcome,
            related_policy_uris,
        ) = product_details(signal);

        ConsolidationProduct {
            uri: signal.uri.clone(),
            content_type: signal.content_type,
            epistemic_type: signal.epistemic_type,
            content: signal.content.clone(),
            quality_score: signal.quality_score.clamp(0.0, 1.0),
            confidence: signal.confidence.clamp(0.0, 1.0),
            superseded_claim,
            evidence_uris: signal.evidence_uris.clone(),
            contradiction_uris: signal.contradiction_uris.clone(),
            error_pattern,
            hypothesis_outcome,
            preconditions,
            expected_outcome,
            related_policy_uris,
            provenance: None,
            metadata: ProductConsolidationMeta {
                source_session: Some(format!("cdt::{:?}", signal.source)),
                generation: 0,
                status: ConsolidationStatus::Converged,
                patch_count: 0,
                lineage: vec![LineageEntry {
                    version: MvccVersion(0),
                    timestamp: Utc::now(),
                    change_summary: format!("created from CDT {:?} signal", signal.source),
                }],
                validity: Some(ValidityRecord {
                    valid_from: Utc::now(),
                    valid_until: None,
                    invalidated_by: None,
                    invalidation_reason: None,
                }),
                half_life_days: Some(30.0 + signal.quality_score.clamp(0.0, 1.0) as f64 * 60.0),
            },
        }
    }

    pub fn entry_from_signal(&self, signal: &CdtConsolidationSignal) -> ContextEntry {
        let mut entry =
            ContextEntry::new_text(signal.uri.clone(), self.tenant, signal.content.clone());
        entry.metadata.content_type = Some(signal.content_type);
        entry.metadata.epistemic_type = Some(signal.epistemic_type);
        entry.metadata.quality_score = Some(signal.quality_score.clamp(0.0, 1.0));
        entry.metadata.state_scope = Some(StateScope::Long);
        entry.metadata.tags = signal.tags.clone();
        entry.metadata.validity = Some(ValidityRecord {
            valid_from: Utc::now(),
            valid_until: None,
            invalidated_by: None,
            invalidation_reason: None,
        });
        entry.metadata.consolidation = Some(EntryConsolidationMeta {
            source: format!("cdt::{:?}", signal.source),
            generation: 0,
            status: ConsolidationStatus::Converged,
            patch_count: 0,
            lineage: vec![LineageEntry {
                version: MvccVersion(0),
                timestamp: Utc::now(),
                change_summary: "materialized from CDT signal".into(),
            }],
            evidence_uris: signal.evidence_uris.clone(),
            corroboration: signal.evidence_uris.len(),
            half_life_days: Some(30.0 + signal.quality_score.clamp(0.0, 1.0) as f64 * 60.0),
            entangled_with: signal.contradiction_uris.clone(),
        });
        let _ = entry
            .metadata
            .set_custom_field("cdt_signal_source", &format!("{:?}", signal.source));
        entry
    }

    pub fn signal_from_gradient(&self, gradient: &CognitiveGradient) -> CdtConsolidationSignal {
        let (content_type, epistemic_type, content, tags) = match &gradient.gradient_type {
            GradientType::FactCorrection {
                old_claim,
                new_claim,
            } => (
                ContentType::Fact,
                EpistemicType::Fact,
                format!("Correct fact: replace `{old_claim}` with `{new_claim}`"),
                vec!["fact-correction".into()],
            ),
            GradientType::AvoidanceRule { pattern, reason } => (
                ContentType::Error,
                EpistemicType::Heuristic,
                format!("Avoid `{pattern}` because {reason}"),
                vec!["avoidance".into(), "error".into()],
            ),
            GradientType::ValidationRule {
                hypothesis,
                outcome,
            } => (
                ContentType::Hypothesis,
                EpistemicType::Hypothesis,
                format!("Hypothesis `{hypothesis}` was {}", outcome_label(*outcome)),
                vec!["validation".into()],
            ),
            GradientType::SkillExtraction {
                procedure,
                precondition,
                expected_outcome,
            } => (
                ContentType::Skill,
                EpistemicType::Procedure,
                format!(
                    "Skill: {procedure}\nPRECONDITION: {precondition}\nEXPECTED: {expected_outcome}"
                ),
                vec!["skill".into(), "procedure".into()],
            ),
            GradientType::PreferenceUpdate {
                key,
                old_value,
                new_value,
            } => (
                ContentType::Preference,
                EpistemicType::Belief,
                format!(
                    "Preference `{key}` changed from `{:?}` to `{new_value}`",
                    old_value
                ),
                vec!["preference".into()],
            ),
            GradientType::MetaCognitive {
                insight,
                applies_to,
            } => (
                ContentType::Reflection,
                EpistemicType::Heuristic,
                format!(
                    "Metacognitive insight: {insight}\nAPPLIES_TO: {}",
                    applies_to.len()
                ),
                vec!["reflection".into(), "metacognitive".into()],
            ),
        };

        CdtConsolidationSignal {
            uri: gradient.source_uri.clone(),
            content_type,
            epistemic_type,
            content,
            quality_score: gradient.weight.clamp(0.0, 1.0),
            confidence: gradient.confidence.clamp(0.0, 1.0),
            evidence_uris: gradient.evidence_uris.clone(),
            contradiction_uris: gradient.contradiction_uris.clone(),
            source: CdtSignalSource::Gradient,
            tags,
        }
    }

    fn make_uri(&self, content_type: &str, index: usize, content: &str) -> ContextUri {
        let hash = blake3::hash(content.as_bytes()).to_hex();
        let short = &hash[..8];
        ContextUri::parse(&format!(
            "uwu://{}/x/{}/{:02}-{}",
            self.agent_scope, content_type, index, short
        ))
        .unwrap_or_else(|_| ContextUri::parse("uwu://t/agent/cdt/meta/fallback").unwrap())
    }
}

fn product_details(
    signal: &CdtConsolidationSignal,
) -> (
    Option<String>,
    Option<String>,
    Option<ProductHypothesisOutcome>,
    Option<String>,
    Option<String>,
    Vec<ContextUri>,
) {
    match signal.content_type {
        ContentType::Error => (None, Some(signal.content.clone()), None, None, None, vec![]),
        ContentType::Hypothesis => (
            None,
            None,
            Some(ProductHypothesisOutcome::Inconclusive),
            None,
            None,
            vec![],
        ),
        ContentType::Skill | ContentType::Procedure => (
            None,
            None,
            None,
            Some("derived from CDT accepted trajectory".into()),
            Some(signal.content.clone()),
            vec![],
        ),
        ContentType::Reflection => (None, None, None, None, None, signal.evidence_uris.clone()),
        _ => (None, None, None, None, None, vec![]),
    }
}

fn epistemic_from_content_type(content_type: ContentType) -> EpistemicType {
    match content_type {
        ContentType::Fact => EpistemicType::Fact,
        ContentType::Belief
        | ContentType::Preference
        | ContentType::Profile
        | ContentType::Goal => EpistemicType::Belief,
        ContentType::Hypothesis => EpistemicType::Hypothesis,
        ContentType::Procedure | ContentType::Skill => EpistemicType::Procedure,
        ContentType::Heuristic
        | ContentType::Reflection
        | ContentType::Error
        | ContentType::Meta => EpistemicType::Heuristic,
        ContentType::Evidence => EpistemicType::Fact,
    }
}

fn outcome_label(outcome: HypothesisOutcome) -> &'static str {
    match outcome {
        HypothesisOutcome::Confirmed => "confirmed",
        HypothesisOutcome::Falsified => "falsified",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uri(s: &str) -> ContextUri {
        ContextUri::parse(s).unwrap()
    }

    #[test]
    fn gradient_becomes_consolidation_product_and_entry() {
        let bridge = CdtConsolidationBridge::for_agent("t/agent/cdt");
        let gradient = CognitiveGradient {
            source_uri: uri("uwu://t/agent/cdt/skill/test"),
            epistemic_type: ContentType::Skill,
            gradient_type: GradientType::SkillExtraction {
                procedure: "deploy safely".into(),
                precondition: "tests pass".into(),
                expected_outcome: "staging updated".into(),
            },
            confidence: 0.9,
            evidence_uris: vec![uri("uwu://t/agent/cdt/fact/evidence")],
            contradiction_uris: vec![],
            weight: 0.8,
        };

        let signal = bridge.signal_from_gradient(&gradient);
        let batch = bridge.batch_from_signals(&[signal]);
        assert_eq!(batch.products.len(), 1);
        assert_eq!(batch.entries.len(), 1);
        assert_eq!(batch.products[0].content_type, ContentType::Skill);
        assert_eq!(
            batch.entries[0].metadata.content_type,
            Some(ContentType::Skill)
        );
        assert!(batch.entries[0].metadata.consolidation.is_some());
    }
}
