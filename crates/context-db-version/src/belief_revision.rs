use crate::model::{Resolution, SemanticConflict};
use agent_context_db_core::{
    ContextEntry, ContextUri, JsonSchema, LlmClient, LlmOpts, LlmTaskKind, PromptOptimization,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
    /// Neural extraction confidence blended with symbolic parse confidence.
    pub extraction_confidence: f32,
    /// Optional explanation for how the belief was extracted.
    pub rationale: Option<String>,
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
        let literals = parse_literals(&source_text);
        let extraction_confidence = if literals.is_empty() { 0.0 } else { 0.55 };
        Self {
            uri,
            literals,
            source_text,
            entrenchment: entrenchment.clamp(0.0, 1.0),
            extraction_confidence,
            rationale: None,
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
    pub contraction_cost: f32,
    pub revision_distance: f32,
}

#[derive(Debug, Clone)]
pub struct AgmRevisionConfig {
    /// If incoming is this much less entrenched than the conflicting base, reject it.
    pub reject_margin: f32,
    /// If incoming and base are too close to choose safely, defer.
    pub defer_margin: f32,
    /// Max contraction candidates considered per conflict component.
    pub max_component_width: usize,
    /// Below this extraction confidence, neural-symbolic revision refuses automatic contraction.
    pub min_extraction_confidence: f32,
    /// Penalizes contracting beliefs that are supported by many literals or high entrenchment.
    pub complexity_penalty: f32,
    /// Prior entrenchment used when an entry has no quality score.
    pub default_entrenchment: f32,
    /// Confidence assigned to a non-empty symbolic extraction.
    pub symbolic_extraction_confidence: f32,
    /// Weight of incoming advantage in the final confidence blend.
    pub advantage_weight: f32,
}

impl AgmRevisionConfig {
    pub fn validate(&self) -> crate::Result<()> {
        for (name, value) in [
            ("reject_margin", self.reject_margin),
            ("defer_margin", self.defer_margin),
            ("min_extraction_confidence", self.min_extraction_confidence),
            ("complexity_penalty", self.complexity_penalty),
            ("default_entrenchment", self.default_entrenchment),
            (
                "symbolic_extraction_confidence",
                self.symbolic_extraction_confidence,
            ),
            ("advantage_weight", self.advantage_weight),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(crate::VersionError::InvalidConfig(format!(
                    "{name} must be finite and in 0..=1"
                )));
            }
        }
        if self.max_component_width == 0 {
            return Err(crate::VersionError::InvalidConfig(
                "max_component_width must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

impl Default for AgmRevisionConfig {
    fn default() -> Self {
        Self {
            reject_margin: 0.18,
            defer_margin: 0.06,
            max_component_width: 16,
            min_extraction_confidence: 0.45,
            complexity_penalty: 0.12,
            default_entrenchment: 0.5,
            symbolic_extraction_confidence: 0.55,
            advantage_weight: 0.7,
        }
    }
}

#[derive(Clone)]
pub struct AgmBeliefReviser {
    config: AgmRevisionConfig,
    llm: Option<Arc<dyn LlmClient>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBeliefExtraction {
    pub literals: Vec<BeliefLiteral>,
    pub confidence: f32,
    pub rationale: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawNeuralBeliefExtraction {
    literals: Vec<RawBeliefLiteral>,
    confidence: f32,
    rationale: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawBeliefLiteral {
    subject: String,
    relation: Option<String>,
    object: String,
    polarity: Option<String>,
}

impl AgmBeliefReviser {
    pub fn new(config: AgmRevisionConfig) -> crate::Result<Self> {
        config.validate()?;
        Ok(Self { config, llm: None })
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn sentence_from_entry(&self, entry: &ContextEntry) -> BeliefSentence {
        self.sentence_from_text(
            entry.uri.clone(),
            entry.payload.sparse_text(),
            entry
                .metadata
                .quality_score
                .unwrap_or(self.config.default_entrenchment),
        )
        .await
    }

    pub async fn sentence_from_text(
        &self,
        uri: ContextUri,
        text: impl Into<String>,
        entrenchment: f32,
    ) -> BeliefSentence {
        let source_text = text.into();
        let mut symbolic =
            BeliefSentence::from_text(uri.clone(), source_text.clone(), entrenchment);
        if !symbolic.literals.is_empty() {
            symbolic.extraction_confidence = self.config.symbolic_extraction_confidence;
        }
        let Some(llm) = &self.llm else {
            return symbolic;
        };
        match extract_neural_beliefs(llm, &source_text).await {
            Some(extraction) if extraction.confidence >= self.config.min_extraction_confidence => {
                let mut literals = symbolic.literals.clone();
                literals.extend(extraction.literals);
                BeliefSentence {
                    uri,
                    source_text,
                    literals: dedupe_literals(literals),
                    entrenchment: entrenchment.clamp(0.0, 1.0),
                    extraction_confidence: extraction.confidence,
                    rationale: Some(extraction.rationale),
                }
            }
            _ => symbolic,
        }
    }

    pub async fn revise_entries(
        &self,
        base: &[ContextEntry],
        incoming: &ContextEntry,
    ) -> BeliefRevisionDecision {
        let mut sentences = Vec::with_capacity(base.len());
        for entry in base {
            sentences.push(self.sentence_from_entry(entry).await);
        }
        let incoming = self.sentence_from_entry(incoming).await;
        self.revise(&sentences, incoming)
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
                contraction_cost: 0.0,
                revision_distance: 0.0,
            };
        }

        let strongest_existing = conflicts
            .iter()
            .filter_map(|c| base.iter().find(|b| b.uri == c.existing_uri))
            .map(|b| b.entrenchment)
            .fold(0.0_f32, f32::max);
        let reject_confidence = (strongest_existing - incoming.entrenchment).clamp(0.0, 1.0);
        if incoming.extraction_confidence < self.config.min_extraction_confidence {
            return BeliefRevisionDecision {
                action: BeliefRevisionAction::Defer,
                incoming,
                contractions: Vec::new(),
                retained: base.iter().map(|b| b.uri.clone()).collect(),
                reason:
                    "incoming belief extraction confidence is too low for automatic AGM revision"
                        .into(),
                confidence: 0.0,
                contraction_cost: 0.0,
                revision_distance: 0.0,
                conflicts,
            };
        }
        if strongest_existing - incoming.entrenchment >= self.config.reject_margin {
            return BeliefRevisionDecision {
                action: BeliefRevisionAction::RejectIncoming,
                incoming,
                contractions: Vec::new(),
                retained: base.iter().map(|b| b.uri.clone()).collect(),
                reason: "incoming belief contradicts more entrenched knowledge".into(),
                confidence: reject_confidence,
                contraction_cost: 0.0,
                revision_distance: 0.0,
                conflicts,
            };
        }

        let candidates = minimal_contraction_set(
            base,
            &incoming,
            &conflicts,
            self.config.max_component_width,
            self.config.complexity_penalty,
        );
        let contraction_cost = contraction_cost(base, &candidates, self.config.complexity_penalty);
        let revision_distance = revision_distance(base, &candidates);
        let incoming_advantage = incoming.entrenchment - average_entrenchment(base, &candidates);

        if incoming_advantage.abs() <= self.config.defer_margin && !candidates.is_empty() {
            return BeliefRevisionDecision {
                action: BeliefRevisionAction::Defer,
                incoming,
                contractions: candidates,
                retained: retained_after(base, &[]),
                reason: "conflicting beliefs have similar entrenchment; defer revision".into(),
                confidence: 1.0 - incoming_advantage.abs().clamp(0.0, 1.0),
                contraction_cost,
                revision_distance,
                conflicts,
            };
        }

        BeliefRevisionDecision {
            action: BeliefRevisionAction::ContractExisting,
            incoming,
            retained: retained_after(base, &candidates),
            contractions: candidates,
            reason: format!(
                "AGM partial-meet contraction removes the least entrenched inconsistent subset with weighted cost {:.3}",
                contraction_cost
            ),
            confidence: (incoming_advantage.max(0.0) * self.config.advantage_weight
                + (1.0 - revision_distance) * (1.0 - self.config.advantage_weight))
                .clamp(0.0, 1.0),
            contraction_cost,
            revision_distance,
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
    complexity_penalty: f32,
) -> Vec<ContextUri> {
    let mut by_predicate: HashMap<BeliefPredicate, Vec<&BeliefConflict>> = HashMap::new();
    for conflict in conflicts {
        by_predicate
            .entry(conflict.incoming_literal.predicate.clone())
            .or_default()
            .push(conflict);
    }

    let cost = base
        .iter()
        .map(|b| (b.uri.clone(), sentence_cost(b, complexity_penalty)))
        .collect::<HashMap<_, _>>();
    let mut contracted = HashSet::new();

    for component in by_predicate.values() {
        let mut candidates = component
            .iter()
            .map(|c| c.existing_uri.clone())
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            cost.get(a)
                .copied()
                .unwrap_or(0.0)
                .partial_cmp(&cost.get(b).copied().unwrap_or(0.0))
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
    contracted.sort_by_key(|a| a.to_string());
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

fn sentence_cost(sentence: &BeliefSentence, complexity_penalty: f32) -> f32 {
    let literal_complexity = (sentence.literals.len() as f32).sqrt() * complexity_penalty;
    (sentence.entrenchment + literal_complexity).clamp(0.0, 2.0)
}

fn contraction_cost(
    base: &[BeliefSentence],
    contractions: &[ContextUri],
    complexity_penalty: f32,
) -> f32 {
    contractions
        .iter()
        .filter_map(|uri| base.iter().find(|b| &b.uri == uri))
        .map(|belief| sentence_cost(belief, complexity_penalty))
        .sum()
}

fn revision_distance(base: &[BeliefSentence], contractions: &[ContextUri]) -> f32 {
    if base.is_empty() {
        return 0.0;
    }
    (contractions.len() as f32 / base.len() as f32).clamp(0.0, 1.0)
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

async fn extract_neural_beliefs(
    llm: &Arc<dyn LlmClient>,
    text: &str,
) -> Option<NeuralBeliefExtraction> {
    let prompt = format!(
        r#"Extract compact logical beliefs from this text.

Text:
{text}

Return JSON only:
{{"literals":[{{"subject":"...","relation":"...","object":"...","polarity":"affirmed|negated"}}],"confidence":0.0,"rationale":"..."}}

Use stable lowercase identifiers. Extract only claims that can participate in contradiction checks."#
    );
    let schema = JsonSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "literals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type":"string"},
                        "relation": {"type":"string"},
                        "object": {"type":"string"},
                        "polarity": {"type":"string"}
                    },
                    "required": ["subject", "object"]
                }
            },
            "confidence": {"type":"number"},
            "rationale": {"type":"string"}
        },
        "required": ["literals", "confidence"]
    }));
    let opts = LlmOpts {
        max_tokens: Some(768),
        temperature: Some(0.0),
        task: LlmTaskKind::Extraction,
        prompt: PromptOptimization::default()
            .force_cache()
            .target_tokens(1_500),
        ..Default::default()
    };
    let raw = match llm.complete_json(&prompt, &schema, &opts).await {
        Ok(value) => value,
        Err(_) => llm.complete(&prompt, &opts).await.ok()?,
    };
    parse_neural_extraction(&raw)
}

fn parse_neural_extraction(raw: &str) -> Option<NeuralBeliefExtraction> {
    let json = serde_json::from_str::<RawNeuralBeliefExtraction>(raw)
        .or_else(|_| serde_json::from_str(&extract_json_object(raw)))
        .ok()?;
    let literals = json
        .literals
        .into_iter()
        .filter_map(|raw| {
            let subject = normalize(&raw.subject);
            let relation = normalize(raw.relation.as_deref().unwrap_or("asserts"));
            let object = normalize(&raw.object);
            if subject.is_empty() || object.is_empty() {
                return None;
            }
            let polarity = match raw.polarity.as_deref().unwrap_or("affirmed") {
                "negated" | "negative" | "false" | "not" => BeliefPolarity::Negated,
                _ => BeliefPolarity::Affirmed,
            };
            Some(BeliefLiteral {
                predicate: BeliefPredicate {
                    subject,
                    relation,
                    object,
                },
                polarity,
            })
        })
        .collect::<Vec<_>>();
    if literals.is_empty() {
        return None;
    }
    Some(NeuralBeliefExtraction {
        literals: dedupe_literals(literals),
        confidence: json.confidence.clamp(0.0, 1.0),
        rationale: json.rationale.unwrap_or_default(),
    })
}

fn extract_json_object(text: &str) -> String {
    let text = text.trim();
    if let Some(start) = text.find('{')
        && let Some(end) = text.rfind('}')
    {
        return text[start..=end].to_string();
    }
    text.to_string()
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
    let subject = tokens.first()?.to_string();
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
    text.split(['.', ';', '\n', '。', '；'])
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
    use agent_context_db_core::LlmError;
    use async_trait::async_trait;

    struct ExtractionLlm;

    #[async_trait]
    impl LlmClient for ExtractionLlm {
        async fn complete(&self, _prompt: &str, _opts: &LlmOpts) -> Result<String, LlmError> {
            Ok(r#"{"literals":[{"subject":"cache","relation":"enabled","object":"redis","polarity":"negated"}],"confidence":0.91,"rationale":"explicit disable statement"}"#.into())
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> Result<String, LlmError> {
            self.complete(_prompt, _opts).await
        }
    }

    fn uri(name: &str) -> ContextUri {
        ContextUri::parse(format!("uwu://t/agent/a/memory/fact/{name}")).unwrap()
    }

    #[test]
    fn detects_logical_negation_without_jaccard_threshold() {
        let mut old = BeliefSentence::from_text(uri("old"), "redis-cache enabled", 0.5);
        old.extraction_confidence = 0.8;
        let mut incoming = BeliefSentence::from_text(uri("new"), "not redis-cache enabled", 0.8);
        incoming.extraction_confidence = 0.8;

        let conflicts = detect_conflicts(&[old], &incoming);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0].incoming_literal.polarity,
            BeliefPolarity::Negated
        );
    }

    #[test]
    fn agm_revision_contracts_least_entrenched_conflicting_belief() {
        let mut weak = BeliefSentence::from_text(uri("weak"), "embedding-cache enabled", 0.25);
        weak.extraction_confidence = 0.8;
        let mut strong = BeliefSentence::from_text(uri("strong"), "graph-rag enabled", 0.9);
        strong.extraction_confidence = 0.8;
        let mut incoming =
            BeliefSentence::from_text(uri("incoming"), "embedding-cache is not enabled", 0.7);
        incoming.extraction_confidence = 0.8;

        let decision = AgmBeliefReviser::new(AgmRevisionConfig::default())
            .unwrap()
            .revise(&[weak.clone(), strong.clone()], incoming);

        assert_eq!(decision.action, BeliefRevisionAction::ContractExisting);
        assert_eq!(decision.contractions, vec![weak.uri]);
        assert_eq!(decision.retained, vec![strong.uri]);
    }

    #[test]
    fn agm_revision_rejects_incoming_when_base_is_more_entrenched() {
        let mut base = BeliefSentence::from_text(uri("base"), "dp enabled", 0.95);
        base.extraction_confidence = 0.8;
        let mut incoming = BeliefSentence::from_text(uri("incoming"), "dp is not enabled", 0.35);
        incoming.extraction_confidence = 0.8;

        let decision = AgmBeliefReviser::new(AgmRevisionConfig::default())
            .unwrap()
            .revise(&[base.clone()], incoming);

        assert_eq!(decision.action, BeliefRevisionAction::RejectIncoming);
        assert!(decision.contractions.is_empty());
        assert_eq!(decision.retained, vec![base.uri]);
    }

    #[tokio::test]
    async fn neural_symbolic_extraction_augments_symbolic_literals() {
        let reviser = AgmBeliefReviser::new(AgmRevisionConfig::default())
            .unwrap()
            .with_llm(Arc::new(ExtractionLlm));
        let sentence = reviser
            .sentence_from_text(uri("incoming"), "redis cache should be disabled", 0.7)
            .await;

        assert!(sentence.extraction_confidence > 0.9);
        assert!(sentence.literals.iter().any(|literal| {
            literal.predicate.subject == "cache"
                && literal.predicate.relation == "enabled"
                && literal.predicate.object == "redis"
                && literal.polarity == BeliefPolarity::Negated
        }));
    }

    #[test]
    fn agm_reviser_resolves_existing_semantic_conflict_trait_shape() {
        let conflict = SemanticConflict::ContradictoryFact {
            uri: uri("same"),
            a: "cache enabled".into(),
            b: "cache is not enabled".into(),
        };

        let resolution = AgmBeliefReviser::new(AgmRevisionConfig::default())
            .unwrap()
            .resolve_conflict(conflict);
        assert!(matches!(
            resolution,
            Resolution::DeferToHuman { .. }
                | Resolution::PreferA { .. }
                | Resolution::PreferB { .. }
        ));
    }
}
