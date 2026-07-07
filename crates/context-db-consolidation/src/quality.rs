//! Horizon-aware memory quality scoring.
//!
//! Short/Mid/Long memory horizons use different scoring weights, but share the
//! same belief-state shape so Sleeptime can promote, demote, rehearse, archive,
//! or keep memories using one pipeline.

use std::collections::HashMap;

use agent_context_db_core::{ContentPayload, ContextEntry, StateScope};
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
