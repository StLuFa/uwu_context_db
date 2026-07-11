//! Theory-of-Mind memory: structured models for interlocutors and peer agents.
//!
//! The retrieve layer owns this because persona/relationship intent routing needs a
//! compact, queryable model before the answer planner decides what to fetch.

use agent_context_db_core::{
    ContentPayload, ContentType, ContextEntry, ContextError, ContextMeta, ContextUri, MediaType,
    MvccVersion, Result, TenantId,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const TOM_KEY: &str = "theory_of_mind";
const TRUST_PRIOR: f32 = 0.5;
const CONFIDENCE_DECAY_PER_DAY: f32 = 0.015;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TomObservationKind {
    StatedPreference,
    DemonstratedPreference,
    KnowledgeEvidence,
    KnowledgeGap,
    TrustSignal,
    InteractionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TomObservation {
    pub subject_id: String,
    pub kind: TomObservationKind,
    pub key: String,
    pub value: String,
    pub confidence: f32,
    pub evidence_uri: Option<ContextUri>,
    pub observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefFacet {
    pub key: String,
    pub value: String,
    pub confidence: f32,
    pub evidence: Vec<ContextUri>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryOfMindModel {
    pub subject_id: String,
    pub known_topics: HashMap<String, f32>,
    pub unknown_topics: HashMap<String, f32>,
    pub preferences: HashMap<String, BeliefFacet>,
    pub relationship_strength: f32,
    pub trust: f32,
    pub last_interaction: Option<String>,
    pub updated_at: DateTime<Utc>,
}

impl TheoryOfMindModel {
    pub fn new(subject_id: impl Into<String>, now: DateTime<Utc>) -> Self {
        Self {
            subject_id: subject_id.into(),
            known_topics: HashMap::new(),
            unknown_topics: HashMap::new(),
            preferences: HashMap::new(),
            relationship_strength: 0.0,
            trust: TRUST_PRIOR,
            last_interaction: None,
            updated_at: now,
        }
    }

    pub fn apply_observation(&mut self, observation: TomObservation) {
        self.decay_to(observation.observed_at);
        let confidence = observation.confidence.clamp(0.0, 1.0);
        match observation.kind {
            TomObservationKind::StatedPreference | TomObservationKind::DemonstratedPreference => {
                let demonstrated_bonus =
                    if observation.kind == TomObservationKind::DemonstratedPreference {
                        0.10
                    } else {
                        0.0
                    };
                let facet = self
                    .preferences
                    .entry(observation.key.clone())
                    .or_insert_with(|| BeliefFacet {
                        key: observation.key.clone(),
                        value: observation.value.clone(),
                        confidence: 0.0,
                        evidence: Vec::new(),
                        updated_at: observation.observed_at,
                    });
                facet.value = observation.value;
                facet.confidence =
                    merge_confidence(facet.confidence, confidence + demonstrated_bonus);
                if let Some(uri) = observation.evidence_uri {
                    push_unique(&mut facet.evidence, uri);
                }
                facet.updated_at = observation.observed_at;
            }
            TomObservationKind::KnowledgeEvidence => {
                reinforce_score(&mut self.known_topics, &observation.key, confidence);
                self.unknown_topics.remove(&observation.key);
            }
            TomObservationKind::KnowledgeGap => {
                reinforce_score(&mut self.unknown_topics, &observation.key, confidence);
                soften_score(&mut self.known_topics, &observation.key, confidence * 0.5);
            }
            TomObservationKind::TrustSignal => {
                let signed = parse_signed_score(&observation.value).unwrap_or(confidence - 0.5);
                self.trust = (self.trust + signed * confidence * 0.25).clamp(0.0, 1.0);
                self.relationship_strength =
                    merge_confidence(self.relationship_strength, confidence * 0.8);
            }
            TomObservationKind::InteractionSummary => {
                self.last_interaction = Some(observation.value);
                self.relationship_strength =
                    merge_confidence(self.relationship_strength, confidence * 0.6);
            }
        }
        self.updated_at = observation.observed_at;
    }

    pub fn apply_observations(&mut self, observations: impl IntoIterator<Item = TomObservation>) {
        for observation in observations {
            self.apply_observation(observation);
        }
    }

    pub fn decay_to(&mut self, now: DateTime<Utc>) {
        if now <= self.updated_at {
            return;
        }
        let days = (now - self.updated_at).num_seconds() as f32 / 86_400.0;
        let retention = (1.0 - CONFIDENCE_DECAY_PER_DAY).powf(days.max(0.0));
        for score in self.known_topics.values_mut() {
            *score *= retention;
        }
        for score in self.unknown_topics.values_mut() {
            *score *= retention;
        }
        for facet in self.preferences.values_mut() {
            facet.confidence *= retention;
        }
        self.relationship_strength *= retention.sqrt();
        self.trust = TRUST_PRIOR + (self.trust - TRUST_PRIOR) * retention.sqrt();
        self.updated_at = now;
    }

    pub fn retrieval_hint(&self, max_terms: usize) -> TomRetrievalHint {
        let mut preference_terms: Vec<_> = self
            .preferences
            .values()
            .filter(|facet| facet.confidence >= 0.35)
            .map(|facet| (format!("{}:{}", facet.key, facet.value), facet.confidence))
            .collect();
        preference_terms.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut knowledge_terms = top_keys(&self.known_topics, max_terms);
        knowledge_terms.extend(
            top_keys(&self.unknown_topics, max_terms)
                .into_iter()
                .map(|term| format!("gap:{term}")),
        );
        knowledge_terms.truncate(max_terms);
        TomRetrievalHint {
            subject_id: self.subject_id.clone(),
            trust: self.trust,
            relationship_strength: self.relationship_strength,
            preference_terms: preference_terms
                .into_iter()
                .take(max_terms)
                .map(|(term, _)| term)
                .collect(),
            knowledge_terms,
        }
    }

    pub fn to_entry(&self, tenant: TenantId, owner_agent: &str) -> Result<ContextEntry> {
        let slug = sanitize_segment(&self.subject_id);
        let uri = ContextUri::parse(format!(
            "uwu://{}/agent/{}/persona/relations/{}",
            tenant.0, owner_agent, slug
        ))?;
        let mut metadata = ContextMeta {
            content_type: Some(ContentType::Profile),
            quality_score: Some(model_quality(self)),
            ..Default::default()
        };
        metadata.tags.push("theory-of-mind".into());
        metadata.tags.push(format!("subject:{}", self.subject_id));
        metadata.set_custom_field(TOM_KEY, self)?;
        let data = serde_json::to_value(self).map_err(ContextError::Serialization)?;
        Ok(ContextEntry {
            uri,
            tenant,
            payload: ContentPayload::Structured {
                summary: self.summary(),
                schema: None,
                data,
            },
            media_type: MediaType::Text,
            metadata,
            mvcc_version: MvccVersion(0),
            created_at: self.updated_at,
            updated_at: self.updated_at,
            derivation: None,
        })
    }

    fn summary(&self) -> String {
        let prefs = top_keys(
            &self
                .preferences
                .iter()
                .map(|(key, facet)| (key.clone(), facet.confidence))
                .collect(),
            3,
        );
        let known = top_keys(&self.known_topics, 3);
        format!(
            "Theory-of-Mind profile for {}. trust={:.2}, relationship={:.2}, preferences=[{}], known=[{}]",
            self.subject_id,
            self.trust,
            self.relationship_strength,
            prefs.join(", "),
            known.join(", ")
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TomRetrievalHint {
    pub subject_id: String,
    pub trust: f32,
    pub relationship_strength: f32,
    pub preference_terms: Vec<String>,
    pub knowledge_terms: Vec<String>,
}

pub fn model_from_entry(entry: &ContextEntry) -> Option<TheoryOfMindModel> {
    entry.metadata.custom_field(TOM_KEY)
}

fn merge_confidence(old: f32, incoming: f32) -> f32 {
    1.0 - (1.0 - old.clamp(0.0, 1.0)) * (1.0 - incoming.clamp(0.0, 1.0))
}

fn reinforce_score(map: &mut HashMap<String, f32>, key: &str, confidence: f32) {
    let entry = map.entry(key.to_string()).or_insert(0.0);
    *entry = merge_confidence(*entry, confidence);
}

fn soften_score(map: &mut HashMap<String, f32>, key: &str, penalty: f32) {
    if let Some(score) = map.get_mut(key) {
        *score = (*score * (1.0 - penalty.clamp(0.0, 0.8))).max(0.0);
    }
}

fn parse_signed_score(value: &str) -> Option<f32> {
    match value.trim().to_ascii_lowercase().as_str() {
        "positive" | "trust" | "trusted" => Some(1.0),
        "negative" | "distrust" | "failed" => Some(-1.0),
        other => other.parse::<f32>().ok().map(|v| v.clamp(-1.0, 1.0)),
    }
}

fn push_unique(values: &mut Vec<ContextUri>, uri: ContextUri) {
    if !values.iter().any(|existing| existing == &uri) {
        values.push(uri);
    }
}

fn top_keys(map: &HashMap<String, f32>, max_terms: usize) -> Vec<String> {
    let mut values: Vec<_> = map
        .iter()
        .filter(|(_, score)| **score >= 0.25)
        .map(|(key, score)| (key.clone(), *score))
        .collect();
    values.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    values
        .into_iter()
        .take(max_terms)
        .map(|(key, _)| key)
        .collect()
}

fn sanitize_segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    let sanitized = out.trim_matches('-').to_string();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

fn model_quality(model: &TheoryOfMindModel) -> f32 {
    let preference_signal = (model.preferences.len() as f32 / 8.0).min(1.0) * 0.25;
    let knowledge_signal =
        ((model.known_topics.len() + model.unknown_topics.len()) as f32 / 12.0).min(1.0) * 0.25;
    (0.25
        + preference_signal
        + knowledge_signal
        + model.relationship_strength * 0.15
        + model.trust * 0.10)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn observations_update_preferences_knowledge_and_trust() {
        let now = Utc::now();
        let mut model = TheoryOfMindModel::new("alice", now);
        model.apply_observations(vec![
            TomObservation {
                subject_id: "alice".into(),
                kind: TomObservationKind::StatedPreference,
                key: "tone".into(),
                value: "concise".into(),
                confidence: 0.7,
                evidence_uri: None,
                observed_at: now,
            },
            TomObservation {
                subject_id: "alice".into(),
                kind: TomObservationKind::KnowledgeEvidence,
                key: "rust".into(),
                value: "strong".into(),
                confidence: 0.8,
                evidence_uri: None,
                observed_at: now,
            },
            TomObservation {
                subject_id: "alice".into(),
                kind: TomObservationKind::TrustSignal,
                key: "review".into(),
                value: "positive".into(),
                confidence: 0.6,
                evidence_uri: None,
                observed_at: now,
            },
        ]);

        let hint = model.retrieval_hint(4);
        assert!(hint.preference_terms.contains(&"tone:concise".to_string()));
        assert!(hint.knowledge_terms.contains(&"rust".to_string()));
        assert!(hint.trust > 0.5);
    }

    #[test]
    fn model_writes_and_reads_structured_profile_entry() {
        let now = Utc::now();
        let mut model = TheoryOfMindModel::new("Peer A", now);
        model.apply_observation(TomObservation {
            subject_id: "Peer A".into(),
            kind: TomObservationKind::InteractionSummary,
            key: "last".into(),
            value: "prefers implementation details before summaries".into(),
            confidence: 0.9,
            evidence_uri: None,
            observed_at: now,
        });

        let entry = model
            .to_entry(TenantId(Uuid::nil()), "agent-1")
            .expect("test model should serialize");
        assert_eq!(entry.metadata.content_type, Some(ContentType::Profile));
        assert!(entry.uri.segments().ends_with(&[
            "agent".to_string(),
            "agent-1".to_string(),
            "persona".to_string(),
            "relations".to_string(),
            "peer-a".to_string(),
        ]));
        let restored = model_from_entry(&entry).unwrap();
        assert_eq!(restored.subject_id, "Peer A");
        assert_eq!(restored.last_interaction, model.last_interaction);
    }
}
