//! Horizon-aware memory quality scoring.
//!
//! Short/Mid/Long memory horizons use different scoring weights, but share the
//! same belief-state shape so Sleeptime can promote, demote, rehearse, archive,
//! or keep memories using one pipeline.

use std::collections::HashMap;

use agent_context_db_core::{ContentPayload, ContentType, ContextEntry, StateScope};
use agent_context_db_knowledge_network::types::{
    FederatedQueryIntent, FederationReturnMode, MeshDiscoveryOpts,
};
use agent_context_db_marketplace_types::{CorroborationLevel, DiscoveryQuery, MarketEntryType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDim {
    Adoption,
    Consistency,
    Recall,
    Freshness,
    Downstream,
    InfoGain,
    Consensus,
}

pub struct QualityScorer {
    weights: HashMap<QualityDim, f32>,
}

impl QualityScorer {
    pub fn new() -> Self {
        let mut w = HashMap::new();
        w.insert(QualityDim::Adoption, 0.20);
        w.insert(QualityDim::Consistency, 0.15);
        w.insert(QualityDim::Recall, 0.10);
        w.insert(QualityDim::Freshness, 0.10);
        w.insert(QualityDim::Downstream, 0.15);
        w.insert(QualityDim::InfoGain, 0.10);
        w.insert(QualityDim::Consensus, 0.20);
        Self { weights: w }
    }

    pub fn score(&self, entry: &ContextEntry, fb: &QualityFeedback) -> f32 {
        let w = &self.weights;
        let base = entry.metadata.quality_score.unwrap_or(0.5);
        (w[&QualityDim::Adoption] * fb.adoption_rate
            + w[&QualityDim::Consistency] * if fb.contradictions == 0 { 1.0 } else { 0.5 }
            + w[&QualityDim::Recall] * fb.recall_rate
            + w[&QualityDim::Freshness] * base
            + w[&QualityDim::Downstream] * if fb.downstream_positive { 1.0 } else { 0.3 }
            + w[&QualityDim::InfoGain] * fb.info_gain.clamp(0.0, 1.0)
            + w[&QualityDim::Consensus] * (fb.corroboration.min(3) as f32 / 3.0))
            .clamp(0.0, 1.0)
    }
}

impl Default for QualityScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct QualityFeedback {
    pub adopted: bool,
    pub adoption_rate: f32,
    pub contradictions: usize,
    pub recall_rate: f32,
    pub downstream_positive: bool,
    pub info_gain: f32,
    pub corroboration: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryHorizon {
    ShortTerm,
    MidTerm,
    LongTerm,
}

impl MemoryHorizon {
    pub fn from_scope(scope: Option<StateScope>) -> Self {
        match scope {
            Some(StateScope::Short) => Self::ShortTerm,
            Some(StateScope::Mid) => Self::MidTerm,
            Some(StateScope::Long) => Self::LongTerm,
            None => Self::MidTerm,
        }
    }

    pub fn as_scope(self) -> StateScope {
        match self {
            Self::ShortTerm => StateScope::Short,
            Self::MidTerm => StateScope::Mid,
            Self::LongTerm => StateScope::Long,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DimensionBelief {
    pub mean: f32,
    pub uncertainty: f32,
    pub alpha: f32,
    pub beta: f32,
}

impl DimensionBelief {
    pub fn from_mean(mean: f32, evidence_mass: f32) -> Self {
        let mean = mean.clamp(0.0, 1.0);
        let mass = evidence_mass.max(2.0);
        let alpha = (mean * mass).max(0.01);
        let beta = ((1.0 - mean) * mass).max(0.01);
        let uncertainty = (1.0 / mass.sqrt()).clamp(0.0, 1.0);
        Self {
            mean,
            uncertainty,
            alpha,
            beta,
        }
    }

    pub fn update(self, observation: f32, reliability: f32, weight: f32) -> Self {
        let obs = observation.clamp(0.0, 1.0);
        let mass = (reliability * weight).clamp(0.0, 8.0);
        let alpha = self.alpha + obs * mass;
        let beta = self.beta + (1.0 - obs) * mass;
        let total = (alpha + beta).max(0.01);
        Self {
            mean: alpha / total,
            uncertainty: (1.0 / total.sqrt()).clamp(0.0, 1.0),
            alpha,
            beta,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBeliefState {
    pub horizon: MemoryHorizon,
    pub overall: f32,
    pub confidence: f32,
    pub uncertainty: f32,
    pub intrinsic: DimensionBelief,
    pub relevance: DimensionBelief,
    pub utility: DimensionBelief,
    pub factuality: DimensionBelief,
    pub consistency: DimensionBelief,
    pub freshness: DimensionBelief,
    pub stability: DimensionBelief,
    pub evidence: DimensionBelief,
    pub trainability: DimensionBelief,
    pub promotion_readiness: f32,
    pub decay_pressure: f32,
    pub contamination_risk: f32,
    pub last_scored_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityBeliefDimension {
    Intrinsic,
    Relevance,
    Utility,
    Factuality,
    Consistency,
    Freshness,
    Stability,
    Evidence,
    Trainability,
}

impl QualityBeliefDimension {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Intrinsic => "intrinsic",
            Self::Relevance => "relevance",
            Self::Utility => "utility",
            Self::Factuality => "factuality",
            Self::Consistency => "consistency",
            Self::Freshness => "freshness",
            Self::Stability => "stability",
            Self::Evidence => "evidence",
            Self::Trainability => "trainability",
        }
    }
}

impl QualityBeliefState {
    pub fn dimension(&self, dim: QualityBeliefDimension) -> DimensionBelief {
        match dim {
            QualityBeliefDimension::Intrinsic => self.intrinsic,
            QualityBeliefDimension::Relevance => self.relevance,
            QualityBeliefDimension::Utility => self.utility,
            QualityBeliefDimension::Factuality => self.factuality,
            QualityBeliefDimension::Consistency => self.consistency,
            QualityBeliefDimension::Freshness => self.freshness,
            QualityBeliefDimension::Stability => self.stability,
            QualityBeliefDimension::Evidence => self.evidence,
            QualityBeliefDimension::Trainability => self.trainability,
        }
    }

    pub fn ranked_uncertainties(&self) -> Vec<(QualityBeliefDimension, DimensionBelief)> {
        let mut values = [
            QualityBeliefDimension::Intrinsic,
            QualityBeliefDimension::Relevance,
            QualityBeliefDimension::Utility,
            QualityBeliefDimension::Factuality,
            QualityBeliefDimension::Consistency,
            QualityBeliefDimension::Freshness,
            QualityBeliefDimension::Stability,
            QualityBeliefDimension::Evidence,
            QualityBeliefDimension::Trainability,
        ]
        .into_iter()
        .map(|dim| (dim, self.dimension(dim)))
        .collect::<Vec<_>>();
        values.sort_by(|(a_dim, a), (b_dim, b)| {
            active_learning_score(*b_dim, *b)
                .partial_cmp(&active_learning_score(*a_dim, *a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        values
    }
}

#[derive(Debug, Clone)]
pub struct HorizonQualitySignals {
    pub adoption_rate: f32,
    pub recall_rate: f32,
    pub downstream_success_rate: f32,
    pub contradiction_count: usize,
    pub corroboration_count: usize,
    pub repeated_observations: usize,
    pub user_confirmed: bool,
    pub user_corrected: bool,
    pub retrieval_ignored_rate: f32,
    pub info_gain: f32,
    pub now: DateTime<Utc>,
}

impl Default for HorizonQualitySignals {
    fn default() -> Self {
        Self {
            adoption_rate: 0.0,
            recall_rate: 0.0,
            downstream_success_rate: 0.5,
            contradiction_count: 0,
            corroboration_count: 0,
            repeated_observations: 1,
            user_confirmed: false,
            user_corrected: false,
            retrieval_ignored_rate: 0.0,
            info_gain: 0.0,
            now: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityRoute {
    KeepInShortTerm,
    BoostForCurrentTask,
    CompressToMidTerm,
    PromoteToLongTerm,
    KeepLongTerm,
    Rehearse,
    Revalidate,
    DemoteRetrievalRank,
    MarkSuperseded,
    Archive,
    ForgetShortTerm,
    ForgetCandidate,
    IncludeInTraining,
    ExcludeFromTraining,
}

#[derive(Debug, Clone)]
pub struct QualityReassessmentOutcome {
    pub horizon: MemoryHorizon,
    pub prior_quality: f32,
    pub posterior: QualityBeliefState,
    pub route: QualityRoute,
    pub should_writeback: bool,
}

impl QualityReassessmentOutcome {
    pub fn active_learning_tasks(
        &self,
        entry: &ContextEntry,
        planner: &ActiveLearningPlanner,
    ) -> Vec<ActiveLearningTask> {
        planner.plan(entry, &self.posterior)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveLearningAction {
    LocalQuery,
    FederatedDiscovery,
    UserVerification,
    RehearsalProbe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningTask {
    pub target_uri: agent_context_db_core::ContextUri,
    pub dimension: QualityBeliefDimension,
    pub action: ActiveLearningAction,
    pub priority: f32,
    pub question: String,
    pub discovery_query: Option<DiscoveryQuery>,
    pub mesh_opts: Option<MeshDiscoveryOpts>,
}

#[derive(Debug, Clone)]
pub struct ActiveLearningPlanner {
    pub uncertainty_threshold: f32,
    pub max_tasks: usize,
    pub min_priority: f32,
}

impl Default for ActiveLearningPlanner {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.42,
            max_tasks: 3,
            min_priority: 0.25,
        }
    }
}

impl ActiveLearningPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn plan(
        &self,
        entry: &ContextEntry,
        state: &QualityBeliefState,
    ) -> Vec<ActiveLearningTask> {
        let text = entry.payload.sparse_text();
        let mut tasks = state
            .ranked_uncertainties()
            .into_iter()
            .filter(|(_, belief)| belief.uncertainty >= self.uncertainty_threshold)
            .filter_map(|(dimension, belief)| {
                let priority = active_learning_score(dimension, belief);
                if priority < self.min_priority {
                    return None;
                }

                Some(self.task_for_dimension(entry, state, dimension, belief, text, priority))
            })
            .collect::<Vec<_>>();

        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tasks.dedup_by(|a, b| a.dimension == b.dimension && a.action == b.action);
        tasks.truncate(self.max_tasks);
        tasks
    }

    fn task_for_dimension(
        &self,
        entry: &ContextEntry,
        state: &QualityBeliefState,
        dimension: QualityBeliefDimension,
        belief: DimensionBelief,
        text: &str,
        priority: f32,
    ) -> ActiveLearningTask {
        let action = action_for_dimension(dimension, state);
        let question = active_learning_question(dimension, entry, belief, text);
        let needs_federation = matches!(action, ActiveLearningAction::FederatedDiscovery);
        ActiveLearningTask {
            target_uri: entry.uri.clone(),
            dimension,
            action,
            priority,
            question: question.clone(),
            discovery_query: needs_federation.then(|| discovery_query_for(entry, dimension, text)),
            mesh_opts: needs_federation.then(|| mesh_opts_for(dimension, priority)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HorizonAwareQualityScorer {
    pub writeback_threshold: f32,
    pub max_delta_per_run: f32,
}

impl Default for HorizonAwareQualityScorer {
    fn default() -> Self {
        Self {
            writeback_threshold: 0.03,
            max_delta_per_run: 0.15,
        }
    }
}

fn active_learning_score(dim: QualityBeliefDimension, belief: DimensionBelief) -> f32 {
    let epistemic_weight = match dim {
        QualityBeliefDimension::Factuality => 1.15,
        QualityBeliefDimension::Consistency => 1.10,
        QualityBeliefDimension::Evidence => 1.12,
        QualityBeliefDimension::Stability => 0.92,
        QualityBeliefDimension::Trainability => 0.88,
        QualityBeliefDimension::Utility => 0.82,
        QualityBeliefDimension::Relevance => 0.74,
        QualityBeliefDimension::Freshness => 0.68,
        QualityBeliefDimension::Intrinsic => 0.55,
    };
    let low_confidence_bonus = (1.0 - belief.mean).max(0.0) * 0.20;
    (belief.uncertainty * epistemic_weight + low_confidence_bonus).clamp(0.0, 1.0)
}

fn action_for_dimension(
    dimension: QualityBeliefDimension,
    state: &QualityBeliefState,
) -> ActiveLearningAction {
    match dimension {
        QualityBeliefDimension::Factuality
        | QualityBeliefDimension::Evidence
        | QualityBeliefDimension::Consistency => ActiveLearningAction::FederatedDiscovery,
        QualityBeliefDimension::Utility | QualityBeliefDimension::Relevance => {
            ActiveLearningAction::LocalQuery
        }
        QualityBeliefDimension::Trainability if state.contamination_risk > 0.35 => {
            ActiveLearningAction::FederatedDiscovery
        }
        QualityBeliefDimension::Trainability => ActiveLearningAction::RehearsalProbe,
        QualityBeliefDimension::Freshness | QualityBeliefDimension::Stability => {
            ActiveLearningAction::RehearsalProbe
        }
        QualityBeliefDimension::Intrinsic => ActiveLearningAction::UserVerification,
    }
}

fn active_learning_question(
    dimension: QualityBeliefDimension,
    entry: &ContextEntry,
    belief: DimensionBelief,
    text: &str,
) -> String {
    let claim = summarize_claim(text);
    match dimension {
        QualityBeliefDimension::Factuality => format!(
            "Find independent evidence that confirms or contradicts `{claim}` (mean {:.2}, uncertainty {:.2}).",
            belief.mean, belief.uncertainty
        ),
        QualityBeliefDimension::Evidence => format!(
            "Find stronger corroborating sources or provenance for `{claim}` (uncertainty {:.2}).",
            belief.uncertainty
        ),
        QualityBeliefDimension::Consistency => format!(
            "Search for knowledge that conflicts with or resolves contradictions around `{claim}`."
        ),
        QualityBeliefDimension::Utility => format!(
            "Probe whether `{claim}` improves downstream task outcomes for {}.",
            entry.uri
        ),
        QualityBeliefDimension::Relevance => format!(
            "Check whether `{claim}` is still relevant to current retrieval and task contexts."
        ),
        QualityBeliefDimension::Freshness => {
            format!("Refresh stale evidence around `{claim}` and compare with newer knowledge.")
        }
        QualityBeliefDimension::Stability => {
            format!("Rehearse `{claim}` across repeated contexts to measure stability and recall.")
        }
        QualityBeliefDimension::Trainability => {
            format!("Validate whether `{claim}` is safe and useful as a training candidate.")
        }
        QualityBeliefDimension::Intrinsic => format!(
            "Ask for clarification or richer source content for underspecified memory `{}`.",
            entry.uri
        ),
    }
}

fn discovery_query_for(
    entry: &ContextEntry,
    dimension: QualityBeliefDimension,
    text: &str,
) -> DiscoveryQuery {
    DiscoveryQuery {
        query_embedding: lightweight_query_embedding(&format!(
            "{} {} {}",
            dimension.as_str(),
            entry
                .content_type()
                .map(|ty| ty.as_path_segment())
                .unwrap_or("memory"),
            text
        )),
        domains: discovery_domains(entry, text),
        entry_types: market_entry_types_for(entry.content_type()),
        min_quality: match dimension {
            QualityBeliefDimension::Factuality | QualityBeliefDimension::Evidence => 0.62,
            QualityBeliefDimension::Consistency => 0.55,
            _ => 0.45,
        },
        min_corroboration_level: match dimension {
            QualityBeliefDimension::Factuality | QualityBeliefDimension::Evidence => {
                CorroborationLevel::CrossAgent
            }
            QualityBeliefDimension::Consistency | QualityBeliefDimension::Trainability => {
                CorroborationLevel::CrossSession
            }
            _ => CorroborationLevel::SingleSession,
        },
        license_compatible: true,
    }
}

fn mesh_opts_for(dimension: QualityBeliefDimension, priority: f32) -> MeshDiscoveryOpts {
    let intent = match dimension {
        QualityBeliefDimension::Factuality | QualityBeliefDimension::Evidence => {
            FederatedQueryIntent::CorroborationCheck
        }
        QualityBeliefDimension::Consistency => FederatedQueryIntent::HighPrecision,
        QualityBeliefDimension::Trainability => FederatedQueryIntent::TrainingCandidate,
        _ => FederatedQueryIntent::HighRecall,
    };
    MeshDiscoveryOpts {
        intent,
        return_mode: if priority > 0.68 {
            FederationReturnMode::Exhaustive
        } else {
            FederationReturnMode::Balanced
        },
        max_peers: if priority > 0.68 { 32 } else { 16 },
        probe_peers: if priority > 0.68 { 24 } else { 12 },
        fetch_peers: if priority > 0.68 { 10 } else { 6 },
        final_top_k: if priority > 0.68 { 32 } else { 16 },
        deadline_ms: if priority > 0.68 { 1800 } else { 1000 },
    }
}

fn discovery_domains(entry: &ContextEntry, text: &str) -> Vec<String> {
    let mut domains = Vec::new();
    if let Some(content_type) = entry.content_type() {
        domains.push(content_type.as_path_segment().to_string());
    }
    for token in text
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|token| token.len() >= 4)
        .take(4)
    {
        domains.push(token.to_ascii_lowercase());
    }
    domains.sort();
    domains.dedup();
    domains.truncate(6);
    domains
}

fn market_entry_types_for(content_type: Option<ContentType>) -> Vec<MarketEntryType> {
    match content_type {
        Some(ContentType::Fact) | Some(ContentType::Belief) | Some(ContentType::Hypothesis) => {
            vec![MarketEntryType::Fact]
        }
        Some(ContentType::Skill) => vec![MarketEntryType::Skill],
        Some(ContentType::Procedure) => vec![MarketEntryType::Procedure],
        Some(ContentType::Error) => vec![MarketEntryType::ErrorPattern],
        Some(ContentType::Heuristic) => vec![MarketEntryType::Procedure, MarketEntryType::Skill],
        _ => vec![MarketEntryType::Fact, MarketEntryType::Procedure],
    }
}

fn lightweight_query_embedding(text: &str) -> Vec<f32> {
    let mut values = vec![0.0; 16];
    for token in text
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|token| !token.is_empty())
    {
        let hash = blake3::hash(token.to_ascii_lowercase().as_bytes());
        let bytes = hash.as_bytes();
        let idx = bytes[0] as usize % values.len();
        let sign = if bytes[1] & 1 == 0 { 1.0 } else { -1.0 };
        values[idx] += sign * (1.0 + (token.len().min(12) as f32 / 12.0));
    }
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut values {
            *v /= norm;
        }
    }
    values
}

fn summarize_claim(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= 96 {
        return trimmed.to_string();
    }
    trimmed.chars().take(96).collect::<String>()
}

impl HorizonAwareQualityScorer {
    pub fn reassess(
        &self,
        entry: &ContextEntry,
        signals: HorizonQualitySignals,
    ) -> QualityReassessmentOutcome {
        let horizon = MemoryHorizon::from_scope(entry.metadata.state_scope);
        let prior_quality = entry.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
        let prior = self.initial_state(horizon, prior_quality, signals.now);
        let intrinsic = self.intrinsic(entry);
        let freshness = self.freshness(entry, horizon, signals.now);
        let relevance = self.relevance(horizon, &signals);
        let utility = self.utility(&signals);
        let factuality = self.factuality(&signals);
        let consistency = self.consistency(&signals);
        let stability = self.stability(horizon, &signals);
        let evidence = self.evidence(&signals);
        let trainability = self.trainability(horizon, factuality, consistency, utility, &signals);

        let mut posterior = prior;
        posterior.intrinsic = posterior.intrinsic.update(intrinsic, 0.9, 2.0);
        posterior.freshness = posterior.freshness.update(freshness, 0.8, 1.6);
        posterior.relevance = posterior.relevance.update(relevance, 0.8, 1.5);
        posterior.utility = posterior.utility.update(utility, 0.8, 1.8);
        posterior.factuality = posterior.factuality.update(factuality, 0.85, 2.0);
        posterior.consistency = posterior.consistency.update(consistency, 0.9, 2.2);
        posterior.stability = posterior.stability.update(stability, 0.75, 1.6);
        posterior.evidence = posterior.evidence.update(evidence, 0.85, 1.8);
        posterior.trainability = posterior.trainability.update(trainability, 0.75, 1.4);

        posterior.promotion_readiness = self.promotion_readiness(horizon, &posterior, &signals);
        posterior.decay_pressure =
            self.decay_pressure(horizon, freshness, utility, posterior.overall);
        posterior.contamination_risk = self.contamination_risk(&posterior, &signals);
        let raw_overall = self.overall(horizon, &posterior);
        posterior.overall = self.clamped_delta(prior_quality, raw_overall);
        posterior.confidence = self.confidence(&posterior);
        posterior.uncertainty = self.uncertainty(&posterior);
        posterior.last_scored_at = signals.now;

        let route = self.route(horizon, &posterior, &signals);
        let should_writeback = (posterior.overall - prior_quality).abs()
            >= self.writeback_threshold
            || entry.metadata.state_scope != Some(posterior.horizon.as_scope());

        QualityReassessmentOutcome {
            horizon,
            prior_quality,
            posterior,
            route,
            should_writeback,
        }
    }

    fn initial_state(
        &self,
        horizon: MemoryHorizon,
        quality: f32,
        now: DateTime<Utc>,
    ) -> QualityBeliefState {
        let belief = DimensionBelief::from_mean(quality, 4.0);
        QualityBeliefState {
            horizon,
            overall: quality,
            confidence: quality,
            uncertainty: 0.5,
            intrinsic: belief,
            relevance: belief,
            utility: belief,
            factuality: belief,
            consistency: belief,
            freshness: belief,
            stability: belief,
            evidence: belief,
            trainability: belief,
            promotion_readiness: 0.0,
            decay_pressure: 0.0,
            contamination_risk: 0.0,
            last_scored_at: now,
        }
    }

    fn intrinsic(&self, entry: &ContextEntry) -> f32 {
        let text = entry.payload.sparse_text();
        let len_score: f32 = match text.len() {
            0 => 0.15,
            1..=24 => 0.45,
            25..=2000 => 0.9,
            2001..=8000 => 0.75,
            _ => 0.55,
        };
        let payload_score: f32 = match &entry.payload {
            ContentPayload::Text { dense, .. } if !dense.is_empty() => 0.95,
            ContentPayload::Text { .. } => 0.75,
            _ => 0.65,
        };
        (len_score * 0.6 + payload_score * 0.4).clamp(0.0, 1.0)
    }

    fn freshness(&self, entry: &ContextEntry, horizon: MemoryHorizon, now: DateTime<Utc>) -> f32 {
        let age_days = (now - entry.updated_at).num_hours().max(0) as f32 / 24.0;
        let half_life_days = match horizon {
            MemoryHorizon::ShortTerm => 0.5,
            MemoryHorizon::MidTerm => 14.0,
            MemoryHorizon::LongTerm => 90.0,
        };
        (-age_days / half_life_days).exp().clamp(0.02, 1.0)
    }

    fn relevance(&self, horizon: MemoryHorizon, signals: &HorizonQualitySignals) -> f32 {
        let base = match horizon {
            MemoryHorizon::ShortTerm => 0.75,
            MemoryHorizon::MidTerm => 0.55,
            MemoryHorizon::LongTerm => 0.45,
        };
        (base + signals.adoption_rate * 0.25 - signals.retrieval_ignored_rate * 0.25)
            .clamp(0.0, 1.0)
    }

    fn utility(&self, signals: &HorizonQualitySignals) -> f32 {
        (signals.adoption_rate * 0.35
            + signals.recall_rate * 0.20
            + signals.downstream_success_rate * 0.30
            + signals.info_gain.clamp(0.0, 1.0) * 0.15
            - signals.retrieval_ignored_rate * 0.20)
            .clamp(0.0, 1.0)
    }

    fn factuality(&self, signals: &HorizonQualitySignals) -> f32 {
        let correction_penalty = if signals.user_corrected { 0.45 } else { 0.0 };
        let contradiction_penalty = (signals.contradiction_count as f32 * 0.18).min(0.55);
        let confirmation_bonus = if signals.user_confirmed { 0.20 } else { 0.0 };
        (0.65 + confirmation_bonus - correction_penalty - contradiction_penalty).clamp(0.0, 1.0)
    }

    fn consistency(&self, signals: &HorizonQualitySignals) -> f32 {
        (1.0 - (signals.contradiction_count as f32 * 0.22).min(0.75)
            + signals.corroboration_count.min(4) as f32 * 0.04)
            .clamp(0.0, 1.0)
    }

    fn stability(&self, horizon: MemoryHorizon, signals: &HorizonQualitySignals) -> f32 {
        let repeat = (signals.repeated_observations.min(6) as f32 / 6.0).clamp(0.0, 1.0);
        let base = match horizon {
            MemoryHorizon::ShortTerm => 0.25,
            MemoryHorizon::MidTerm => 0.45,
            MemoryHorizon::LongTerm => 0.70,
        };
        (base * 0.5 + repeat * 0.5).clamp(0.0, 1.0)
    }

    fn evidence(&self, signals: &HorizonQualitySignals) -> f32 {
        let support = signals.corroboration_count.min(5) as f32 / 5.0;
        let confirmation = if signals.user_confirmed { 0.35 } else { 0.0 };
        let correction = if signals.user_corrected { 0.45 } else { 0.0 };
        (0.35 + support * 0.35 + confirmation - correction).clamp(0.0, 1.0)
    }

    fn trainability(
        &self,
        horizon: MemoryHorizon,
        factuality: f32,
        consistency: f32,
        utility: f32,
        signals: &HorizonQualitySignals,
    ) -> f32 {
        if matches!(horizon, MemoryHorizon::ShortTerm) {
            return 0.0;
        }
        let horizon_bonus = if matches!(horizon, MemoryHorizon::LongTerm) {
            0.15
        } else {
            0.0
        };
        let correction_penalty = if signals.user_corrected { 0.45 } else { 0.0 };
        (factuality * 0.35 + consistency * 0.30 + utility * 0.20 + horizon_bonus
            - correction_penalty)
            .clamp(0.0, 1.0)
    }

    fn promotion_readiness(
        &self,
        horizon: MemoryHorizon,
        state: &QualityBeliefState,
        signals: &HorizonQualitySignals,
    ) -> f32 {
        match horizon {
            MemoryHorizon::ShortTerm => (state.relevance.mean * 0.25
                + state.utility.mean * 0.25
                + state.consistency.mean * 0.20
                + state.stability.mean * 0.20
                + if signals.user_confirmed { 0.10 } else { 0.0 }
                - self.contamination_risk(state, signals) * 0.35)
                .clamp(0.0, 1.0),
            MemoryHorizon::MidTerm => (state.evidence.mean * 0.25
                + state.stability.mean * 0.25
                + state.utility.mean * 0.20
                + state.factuality.mean * 0.20
                + if signals.user_confirmed { 0.10 } else { 0.0 }
                - self.contamination_risk(state, signals) * 0.40)
                .clamp(0.0, 1.0),
            MemoryHorizon::LongTerm => 1.0,
        }
    }

    fn decay_pressure(
        &self,
        horizon: MemoryHorizon,
        freshness: f32,
        utility: f32,
        overall: f32,
    ) -> f32 {
        let horizon_factor = match horizon {
            MemoryHorizon::ShortTerm => 1.0,
            MemoryHorizon::MidTerm => 0.65,
            MemoryHorizon::LongTerm => 0.35,
        };
        ((1.0 - freshness) * 0.45 + (1.0 - utility) * 0.35 + (1.0 - overall) * 0.20)
            * horizon_factor
    }

    fn contamination_risk(
        &self,
        state: &QualityBeliefState,
        signals: &HorizonQualitySignals,
    ) -> f32 {
        (state.uncertainty * 0.30
            + (1.0 - state.factuality.mean) * 0.30
            + (1.0 - state.consistency.mean) * 0.25
            + if signals.user_corrected { 0.25 } else { 0.0 })
        .clamp(0.0, 1.0)
    }

    fn overall(&self, horizon: MemoryHorizon, state: &QualityBeliefState) -> f32 {
        match horizon {
            MemoryHorizon::ShortTerm => {
                state.relevance.mean * 0.30
                    + state.freshness.mean * 0.25
                    + state.utility.mean * 0.20
                    + state.intrinsic.mean * 0.10
                    + state.consistency.mean * 0.10
                    + state.evidence.mean * 0.05
            }
            MemoryHorizon::MidTerm => {
                state.stability.mean * 0.25
                    + state.utility.mean * 0.20
                    + state.consistency.mean * 0.15
                    + state.evidence.mean * 0.15
                    + state.factuality.mean * 0.15
                    + state.freshness.mean * 0.10
            }
            MemoryHorizon::LongTerm => {
                state.factuality.mean * 0.25
                    + state.evidence.mean * 0.20
                    + state.stability.mean * 0.20
                    + state.consistency.mean * 0.15
                    + state.utility.mean * 0.10
                    + state.freshness.mean * 0.05
                    + state.trainability.mean * 0.05
            }
        }
        .clamp(0.0, 1.0)
    }

    fn confidence(&self, state: &QualityBeliefState) -> f32 {
        (state.factuality.mean * 0.30
            + state.consistency.mean * 0.25
            + state.evidence.mean * 0.20
            + state.stability.mean * 0.15
            + (1.0 - state.contamination_risk) * 0.10)
            .clamp(0.0, 1.0)
    }

    fn uncertainty(&self, state: &QualityBeliefState) -> f32 {
        ((state.intrinsic.uncertainty
            + state.utility.uncertainty
            + state.factuality.uncertainty
            + state.consistency.uncertainty
            + state.evidence.uncertainty
            + state.stability.uncertainty)
            / 6.0)
            .clamp(0.0, 1.0)
    }

    fn route(
        &self,
        horizon: MemoryHorizon,
        state: &QualityBeliefState,
        signals: &HorizonQualitySignals,
    ) -> QualityRoute {
        if signals.user_corrected && !matches!(horizon, MemoryHorizon::ShortTerm) {
            return QualityRoute::ExcludeFromTraining;
        }
        if state.consistency.mean < 0.45 || state.factuality.mean < 0.45 {
            return QualityRoute::Revalidate;
        }
        match horizon {
            MemoryHorizon::ShortTerm
                if state.decay_pressure > 0.70 && state.promotion_readiness < 0.35 =>
            {
                QualityRoute::ForgetShortTerm
            }
            MemoryHorizon::ShortTerm if state.promotion_readiness >= 0.70 => {
                QualityRoute::CompressToMidTerm
            }
            MemoryHorizon::ShortTerm if state.relevance.mean >= 0.70 => {
                QualityRoute::BoostForCurrentTask
            }
            MemoryHorizon::ShortTerm => QualityRoute::KeepInShortTerm,
            MemoryHorizon::MidTerm
                if state.promotion_readiness >= 0.78 && state.contamination_risk <= 0.35 =>
            {
                QualityRoute::PromoteToLongTerm
            }
            MemoryHorizon::MidTerm if state.decay_pressure > 0.65 && state.utility.mean < 0.35 => {
                QualityRoute::Archive
            }
            MemoryHorizon::MidTerm if state.utility.mean >= 0.65 && state.uncertainty >= 0.45 => {
                QualityRoute::Rehearse
            }
            MemoryHorizon::MidTerm => QualityRoute::Rehearse,
            MemoryHorizon::LongTerm
                if state.overall >= 0.78
                    && state.confidence >= 0.70
                    && state.uncertainty <= 0.45 =>
            {
                QualityRoute::IncludeInTraining
            }
            MemoryHorizon::LongTerm if state.decay_pressure > 0.70 && state.utility.mean < 0.25 => {
                QualityRoute::ForgetCandidate
            }
            MemoryHorizon::LongTerm if state.uncertainty > 0.55 => QualityRoute::Revalidate,
            MemoryHorizon::LongTerm => QualityRoute::KeepLongTerm,
        }
    }

    fn clamped_delta(&self, prior: f32, raw: f32) -> f32 {
        let delta = (raw - prior).clamp(-self.max_delta_per_run, self.max_delta_per_run);
        (prior + delta).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContextEntry, ContextMeta, ContextUri, TenantId};
    use uuid::Uuid;

    fn entry(content_type: ContentType, text: &str) -> ContextEntry {
        let mut e = ContextEntry::new_text(
            ContextUri::parse("uwu://t/agent/a/memories/fact/high-uncertainty").unwrap(),
            TenantId(Uuid::nil()),
            text,
        );
        e.metadata = ContextMeta {
            content_type: Some(content_type),
            quality_score: Some(0.5),
            ..Default::default()
        };
        e
    }

    fn belief(mean: f32, uncertainty: f32) -> DimensionBelief {
        DimensionBelief {
            mean,
            uncertainty,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    fn state_with_uncertainty(dim: QualityBeliefDimension, uncertainty: f32) -> QualityBeliefState {
        let base = belief(0.65, 0.15);
        let high = belief(0.42, uncertainty);
        let mut state = QualityBeliefState {
            horizon: MemoryHorizon::MidTerm,
            overall: 0.5,
            confidence: 0.5,
            uncertainty,
            intrinsic: base,
            relevance: base,
            utility: base,
            factuality: base,
            consistency: base,
            freshness: base,
            stability: base,
            evidence: base,
            trainability: base,
            promotion_readiness: 0.0,
            decay_pressure: 0.0,
            contamination_risk: 0.2,
            last_scored_at: Utc::now(),
        };
        match dim {
            QualityBeliefDimension::Intrinsic => state.intrinsic = high,
            QualityBeliefDimension::Relevance => state.relevance = high,
            QualityBeliefDimension::Utility => state.utility = high,
            QualityBeliefDimension::Factuality => state.factuality = high,
            QualityBeliefDimension::Consistency => state.consistency = high,
            QualityBeliefDimension::Freshness => state.freshness = high,
            QualityBeliefDimension::Stability => state.stability = high,
            QualityBeliefDimension::Evidence => state.evidence = high,
            QualityBeliefDimension::Trainability => state.trainability = high,
        }
        state
    }

    #[test]
    fn active_learning_routes_epistemic_uncertainty_to_federated_discovery() {
        let entry = entry(
            ContentType::Fact,
            "redis embedding cache prevents duplicate API calls",
        );
        let state = state_with_uncertainty(QualityBeliefDimension::Evidence, 0.82);
        let planner = ActiveLearningPlanner::default();

        let tasks = planner.plan(&entry, &state);
        assert!(!tasks.is_empty());
        let task = &tasks[0];
        assert_eq!(task.dimension, QualityBeliefDimension::Evidence);
        assert_eq!(task.action, ActiveLearningAction::FederatedDiscovery);
        assert!(task.question.contains("corroborating"));

        let query = task.discovery_query.as_ref().unwrap();
        assert!(!query.query_embedding.is_empty());
        assert!(query.domains.contains(&"fact".to_string()));
        assert_eq!(
            query.min_corroboration_level,
            CorroborationLevel::CrossAgent
        );

        let opts = task.mesh_opts.as_ref().unwrap();
        assert_eq!(opts.intent, FederatedQueryIntent::CorroborationCheck);
        assert_eq!(opts.return_mode, FederationReturnMode::Exhaustive);
    }

    #[test]
    fn active_learning_uses_local_query_for_utility_uncertainty() {
        let entry = entry(
            ContentType::Procedure,
            "run cargo check before promoting memory",
        );
        let state = state_with_uncertainty(QualityBeliefDimension::Utility, 0.70);
        let planner = ActiveLearningPlanner::default();

        let tasks = planner.plan(&entry, &state);
        assert!(!tasks.is_empty());
        assert_eq!(tasks[0].dimension, QualityBeliefDimension::Utility);
        assert_eq!(tasks[0].action, ActiveLearningAction::LocalQuery);
        assert!(tasks[0].discovery_query.is_none());
        assert!(tasks[0].mesh_opts.is_none());
    }
}
