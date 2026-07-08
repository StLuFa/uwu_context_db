use crate::model::{Resolution, SemanticConflict};
use agent_context_db_core::{ContextEntry, ContextUri};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefPolarity {
    Affirmed,
    Negated,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BeliefPredicate {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefLiteral {
    pub predicate: BeliefPredicate,
    pub polarity: BeliefPolarity,
}

impl BeliefLiteral {
    pub fn contradicts(&self, other: &BeliefLiteral) -> bool {
        self.predicate == other.predicate && self.polarity != other.polarity
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefSentence {
    pub uri: ContextUri,
    pub source_text: String,
    pub literals: Vec<BeliefLiteral>,
    /// AGM epistemic entrenchment. Higher values are harder to contract.
    pub entrenchment: f32,
}

impl BeliefSentence {
    pub fn from_entry(entry: &ContextEntry) -> Self {
        Self::from_text(
            entry.uri.clone(),
            entry.payload.sparse_text(),
            entry.metadata.quality_score.unwrap_or(0.5),
        )
    }

    pub fn from_text(uri: ContextUri, text: impl Into<String>, entrenchment: f32) -> Self {
        let source_text = text.into();
        Self {
            uri,
            literals: parse_literals(&source_text),
            source_text,
            entrenchment: entrenchment.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefConflict {
    pub incoming_uri: ContextUri,
    pub existing_uri: ContextUri,
    pub incoming_literal: BeliefLiteral,
    pub existing_literal: BeliefLiteral,
    pub severity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefRevisionAction {
    Accept,
    ContractExisting,
    RejectIncoming,
    Defer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefRevisionDecision {
    pub action: BeliefRevisionAction,
    pub incoming: BeliefSentence,
    /// Minimal contraction set selected by AGM partial-meet revision.
    pub contractions: Vec<ContextUri>,
    pub conflicts: Vec<BeliefConflict>,
    pub retained: Vec<ContextUri>,
    pub reason: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct AgmRevisionConfig {
    /// If incoming is this much less entrenched than the conflicting base, reject it.
    pub reject_margin: f32,
    /// If incoming and base are too close to choose safely, defer.
    pub defer_margin: f32,
    /// Max contraction candidates considered per conflict component.
    pub max_component_width: usize,
}

impl Default for AgmRevisionConfig {
    fn default() -> Self {
        Self {
            reject_margin: 0.18,
            defer_margin: 0.06,
            max_component_width: 16,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgmBeliefReviser {
    config: AgmRevisionConfig,
}

impl AgmBeliefReviser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: AgmRevisionConfig) -> Self {
        Self { config }
    }

    pub fn resolve_conflict(&self, conflict: SemanticConflict) -> Resolution {
        match conflict {
            SemanticConflict::ContradictoryFact { uri, a, b } => {
                let base = BeliefSentence::from_text(uri.clone(), a, 0.55);
                let incoming = BeliefSentence::from_text(uri, b, 0.55);
                revision_to_resolution(self.revise(&[base], incoming))
            }
            SemanticConflict::ConflictingRelation { from, to, a, b } => Resolution::DeferToHuman {
                reason: format!(
                    "relation conflict between {from} and {to}: {:?} vs {:?}; AGM text revision only handles belief literals",
                    a, b
                ),
            },
            SemanticConflict::OverlappingEntity { a, b, similarity } => Resolution::DeferToHuman {
                reason: format!(
                    "entity overlap requires identity resolution before AGM revision: {a} vs {b} similarity {similarity:.3}"
                ),
            },
        }
    }

    /// AGM Levi revision: revise K by alpha as (K - not alpha) + alpha.
    ///
    /// The contraction part minimizes removal by preferring the least entrenched
    /// existing beliefs that restore logical consistency with the incoming sentence.
    pub fn revise(
        &self,
        base: &[BeliefSentence],
        incoming: BeliefSentence,
    ) -> BeliefRevisionDecision {
        let conflicts = detect_conflicts(base, &incoming);
        if conflicts.is_empty() {
            return BeliefRevisionDecision {
                action: BeliefRevisionAction::Accept,
                incoming,
                contractions: Vec::new(),
                conflicts,
                retained: base.iter().map(|b| b.uri.clone()).collect(),
                reason: "incoming belief is logically consistent with the belief base".into(),
                confidence: 1.0,
            };
        }

        let strongest_existing = conflicts
            .iter()
            .filter_map(|c| base.iter().find(|b| b.uri == c.existing_uri))
            .map(|b| b.entrenchment)
            .fold(0.0_f32, f32::max);
        let reject_confidence = (strongest_existing - incoming.entrenchment).clamp(0.0, 1.0);
        if strongest_existing - incoming.entrenchment >= self.config.reject_margin {
            return BeliefRevisionDecision {
                action: BeliefRevisionAction::RejectIncoming,
                incoming,
                contractions: Vec::new(),
                retained: base.iter().map(|b| b.uri.clone()).collect(),
                reason: "incoming belief contradicts more entrenched knowledge".into(),
                confidence: reject_confidence,
                conflicts,
            };
        }

        let candidates =
            minimal_contraction_set(base, &incoming, &conflicts, self.config.max_component_width);
        let contraction_cost = candidates
            .iter()
            .filter_map(|uri| base.iter().find(|b| &b.uri == uri))
            .map(|b| b.entrenchment)
            .sum::<f32>();
        let incoming_advantage = incoming.entrenchment - average_entrenchment(base, &candidates);

        if incoming_advantage.abs() <= self.config.defer_margin && !candidates.is_empty() {
            return BeliefRevisionDecision {
                action: BeliefRevisionAction::Defer,
                incoming,
                contractions: candidates,
                retained: retained_after(base, &[]),
                reason: "conflicting beliefs have similar entrenchment; defer revision".into(),
                confidence: 1.0 - incoming_advantage.abs().clamp(0.0, 1.0),
                conflicts,
            };
        }

        BeliefRevisionDecision {
            action: BeliefRevisionAction::ContractExisting,
            incoming,
            retained: retained_after(base, &candidates),
            contractions: candidates,
            reason: format!(
                "AGM contraction removes the least entrenched inconsistent subset with total cost {:.3}",
                contraction_cost
            ),
            confidence: incoming_advantage.max(0.0).clamp(0.0, 1.0),
            conflicts,
        }
    }
}

pub fn revision_to_resolution(decision: BeliefRevisionDecision) -> Resolution {
    match decision.action {
        BeliefRevisionAction::Accept => Resolution::PreferB {
            reason: decision.reason,
        },
        BeliefRevisionAction::ContractExisting => Resolution::PreferB {
            reason: format!(
                "{}; contracted {:?}",
                decision.reason, decision.contractions
            ),
        },
        BeliefRevisionAction::RejectIncoming => Resolution::PreferA {
            reason: decision.reason,
        },
        BeliefRevisionAction::Defer => Resolution::DeferToHuman {
            reason: decision.reason,
        },
    }
}

pub fn detect_conflicts(base: &[BeliefSentence], incoming: &BeliefSentence) -> Vec<BeliefConflict> {
    let mut conflicts = Vec::new();
    for existing in base {
        for incoming_literal in &incoming.literals {
            for existing_literal in &existing.literals {
                if incoming_literal.contradicts(existing_literal) {
                    conflicts.push(BeliefConflict {
                        incoming_uri: incoming.uri.clone(),
                        existing_uri: existing.uri.clone(),
                        incoming_literal: incoming_literal.clone(),
                        existing_literal: existing_literal.clone(),
                        severity: ((incoming.entrenchment + existing.entrenchment) / 2.0)
                            .clamp(0.0, 1.0),
                    });
                }
            }
        }
    }
    conflicts
}

fn minimal_contraction_set(
    base: &[BeliefSentence],
    incoming: &BeliefSentence,
    conflicts: &[BeliefConflict],
    max_component_width: usize,
) -> Vec<ContextUri> {
    let mut by_predicate: HashMap<BeliefPredicate, Vec<&BeliefConflict>> = HashMap::new();
    for conflict in conflicts {
        by_predicate
            .entry(conflict.incoming_literal.predicate.clone())
            .or_default()
            .push(conflict);
    }

    let entrenchment = base
        .iter()
        .map(|b| (b.uri.clone(), b.entrenchment))
        .collect::<HashMap<_, _>>();
    let mut contracted = HashSet::new();

    for component in by_predicate.values() {
        let mut candidates = component
            .iter()
            .map(|c| c.existing_uri.clone())
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            entrenchment
                .get(a)
                .copied()
                .unwrap_or(0.0)
                .partial_cmp(&entrenchment.get(b).copied().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.to_string().cmp(&b.to_string()))
        });
        candidates.dedup();
        candidates.truncate(max_component_width.max(1));

        let mut selected = Vec::new();
        for candidate in candidates {
            selected.push(candidate.clone());
            if component_conflicts_resolved(incoming, component, &selected) {
                break;
            }
        }
        contracted.extend(selected);
    }

    let mut contracted = contracted.into_iter().collect::<Vec<_>>();
    contracted.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
    contracted
}

fn component_conflicts_resolved(
    _incoming: &BeliefSentence,
    component: &[&BeliefConflict],
    selected: &[ContextUri],
) -> bool {
    component
        .iter()
        .all(|conflict| selected.contains(&conflict.existing_uri))
}

fn average_entrenchment(base: &[BeliefSentence], uris: &[ContextUri]) -> f32 {
    if uris.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut n = 0.0;
    for uri in uris {
        if let Some(belief) = base.iter().find(|b| &b.uri == uri) {
            sum += belief.entrenchment;
            n += 1.0;
        }
    }
    if n == 0.0 { 0.0 } else { sum / n }
}

fn retained_after(base: &[BeliefSentence], contractions: &[ContextUri]) -> Vec<ContextUri> {
    base.iter()
        .filter(|belief| !contractions.contains(&belief.uri))
        .map(|belief| belief.uri.clone())
        .collect()
}

fn parse_literals(text: &str) -> Vec<BeliefLiteral> {
    let mut literals = Vec::new();
    for sentence in split_sentences(text) {
        if let Some(literal) = parse_key_value_literal(sentence) {
            literals.push(literal);
            continue;
        }
        if let Some(literal) = parse_predicate_literal(sentence) {
            literals.push(literal);
        }
    }
    dedupe_literals(literals)
}

fn parse_key_value_literal(sentence: &str) -> Option<BeliefLiteral> {
    let normalized = normalize(sentence);
    for sep in ["!=", "=", ":"] {
        if let Some(idx) = normalized.find(sep) {
            let left = normalize(&normalized[..idx]);
            let right = normalize(&normalized[idx + sep.len()..]);
            if left.is_empty() || right.is_empty() {
                return None;
            }
            let polarity = if sep == "!=" {
                BeliefPolarity::Negated
            } else {
                BeliefPolarity::Affirmed
            };
            return Some(BeliefLiteral {
                predicate: BeliefPredicate {
                    subject: left,
                    relation: "is".into(),
                    object: right,
                },
                polarity,
            });
        }
    }
    None
}

fn parse_predicate_literal(sentence: &str) -> Option<BeliefLiteral> {
    let normalized = normalize(sentence);
    if normalized.is_empty() {
        return None;
    }
    let mut polarity = BeliefPolarity::Affirmed;
    let mut claim = normalized.as_str();
    for marker in NEGATION_MARKERS {
        if let Some(rest) = claim.strip_prefix(marker) {
            polarity = BeliefPolarity::Negated;
            claim = rest.trim();
            break;
        }
        if let Some(idx) = claim.find(&format!(" {marker} ")) {
            polarity = BeliefPolarity::Negated;
            let mut owned = claim.to_string();
            owned.replace_range(idx..idx + marker.len() + 2, " ");
            let cleaned = normalize(&owned);
            return literal_from_claim(&cleaned, polarity);
        }
    }
    literal_from_claim(claim, polarity)
}

fn literal_from_claim(claim: &str, polarity: BeliefPolarity) -> Option<BeliefLiteral> {
    let tokens = claim
        .split_whitespace()
        .filter(|t| !STOP_WORDS.contains(t))
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return None;
    }
    let subject = tokens.first().unwrap().to_string();
    let object = if tokens.len() == 1 {
        "true".into()
    } else {
        tokens[1..].join(" ")
    };
    Some(BeliefLiteral {
        predicate: BeliefPredicate {
            subject,
            relation: "asserts".into(),
            object,
        },
        polarity,
    })
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c| matches!(c, '.' | ';' | '\n' | '。' | '；'))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn normalize(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric()
                || c.is_whitespace()
                || c == '_'
                || c == '-'
                || c == '='
                || c == ':'
                || c == '!'
            {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn dedupe_literals(literals: Vec<BeliefLiteral>) -> Vec<BeliefLiteral> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for literal in literals {
        let key = (
            literal.predicate.clone(),
            matches!(literal.polarity, BeliefPolarity::Affirmed),
        );
        if seen.insert(key) {
            out.push(literal);
        }
    }
    out
}

const NEGATION_MARKERS: &[&str] = &[
    "not",
    "never",
    "no",
    "cannot",
    "can't",
    "does not",
    "do not",
    "without",
    "false",
    "禁用",
    "不能",
    "不会",
    "不是",
    "不支持",
    "无",
];

const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "to", "of", "and", "or", "for", "with", "in", "on", "this", "that", "is",
    "are", "be", "should", "must", "can", "will",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn uri(name: &str) -> ContextUri {
        ContextUri::parse(format!("uwu://t/agent/a/memories/fact/{name}")).unwrap()
    }

    #[test]
    fn detects_logical_negation_without_jaccard_threshold() {
        let old = BeliefSentence::from_text(uri("old"), "redis-cache enabled", 0.5);
        let incoming = BeliefSentence::from_text(uri("new"), "not redis-cache enabled", 0.8);

        let conflicts = detect_conflicts(&[old], &incoming);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0].incoming_literal.polarity,
            BeliefPolarity::Negated
        );
    }

    #[test]
    fn agm_revision_contracts_least_entrenched_conflicting_belief() {
        let weak = BeliefSentence::from_text(uri("weak"), "embedding-cache enabled", 0.25);
        let strong = BeliefSentence::from_text(uri("strong"), "graph-rag enabled", 0.9);
        let incoming =
            BeliefSentence::from_text(uri("incoming"), "embedding-cache is not enabled", 0.7);

        let decision = AgmBeliefReviser::new().revise(&[weak.clone(), strong.clone()], incoming);

        assert_eq!(decision.action, BeliefRevisionAction::ContractExisting);
        assert_eq!(decision.contractions, vec![weak.uri]);
        assert_eq!(decision.retained, vec![strong.uri]);
    }

    #[test]
    fn agm_revision_rejects_incoming_when_base_is_more_entrenched() {
        let base = BeliefSentence::from_text(uri("base"), "dp enabled", 0.95);
        let incoming = BeliefSentence::from_text(uri("incoming"), "dp is not enabled", 0.35);

        let decision = AgmBeliefReviser::new().revise(&[base.clone()], incoming);

        assert_eq!(decision.action, BeliefRevisionAction::RejectIncoming);
        assert!(decision.contractions.is_empty());
        assert_eq!(decision.retained, vec![base.uri]);
    }

    #[test]
    fn agm_reviser_resolves_existing_semantic_conflict_trait_shape() {
        let conflict = SemanticConflict::ContradictoryFact {
            uri: uri("same"),
            a: "cache enabled".into(),
            b: "cache is not enabled".into(),
        };

        let resolution = AgmBeliefReviser::new().resolve_conflict(conflict);
        assert!(matches!(
            resolution,
            Resolution::DeferToHuman { .. }
                | Resolution::PreferA { .. }
                | Resolution::PreferB { .. }
        ));
    }
}
