use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Narrow runtime port used by producers that only need to publish outcomes.
pub trait ReactionSink: Send + Sync {
    fn emit(&self, reaction: Reaction);
}

/// Configured boundary for allowing calibrated outcomes to affect online state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineUpdateConfig {
    pub enabled: bool,
    pub min_samples: usize,
    pub max_ece: f32,
    pub learning_rate: f32,
}

impl Default for OnlineUpdateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_samples: 20,
            max_ece: 0.15,
            learning_rate: 0.1,
        }
    }
}

impl OnlineUpdateConfig {
    pub fn allows(&self, calibration: &CalibrationMetrics) -> bool {
        self.enabled && calibration.count >= self.min_samples && calibration.ece <= self.max_ece
    }
}

/// Narrow online-learning port. Callers cannot mutate policy state directly.
pub trait OnlinePolicyUpdater: Send + Sync {
    fn update_from_reaction(
        &self,
        reaction: &Reaction,
        calibration: &CalibrationMetrics,
        config: OnlineUpdateConfig,
    );
}

pub fn emit_and_update(
    sink: &Arc<dyn ReactionSink>,
    updater: Option<&Arc<dyn OnlinePolicyUpdater>>,
    reaction: Reaction,
    calibration: CalibrationMetrics,
    config: OnlineUpdateConfig,
) {
    sink.emit(reaction.clone());
    if let Some(updater) = updater {
        updater.update_from_reaction(&reaction, &calibration, config);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalAttribution {
    pub cause_id: String,
    pub credit: f32,
    pub confidence: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reaction {
    pub id: String,
    pub subject_id: String,
    pub execution_id: String,
    pub outcome: f32,
    pub predicted_outcome: Option<f32>,
    pub observed_at: DateTime<Utc>,
    pub attributions: Vec<CausalAttribution>,
    #[serde(default)]
    pub traits: HashMap<String, f32>,
}
impl Reaction {
    pub fn normalize(&mut self) {
        self.outcome = self.outcome.clamp(0.0, 1.0);
        self.predicted_outcome = self.predicted_outcome.map(|v| v.clamp(0.0, 1.0));
        let total: f32 = self
            .attributions
            .iter()
            .map(|a| a.credit.max(0.0) * a.confidence.clamp(0.0, 1.0))
            .sum();
        if total > 0.0 {
            for a in &mut self.attributions {
                a.credit = a.credit.max(0.0) * a.confidence.clamp(0.0, 1.0) / total;
            }
        }
    }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TraitEstimate {
    pub value: f32,
    pub observations: u64,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub count: usize,
    pub brier: f32,
    pub ece: f32,
}
#[derive(Debug, Default)]
pub struct MemoryReactionStore {
    reactions: RwLock<Vec<Reaction>>,
    traits: RwLock<HashMap<(String, String), TraitEstimate>>,
}
impl ReactionSink for MemoryReactionStore {
    fn emit(&self, reaction: Reaction) {
        self.record(reaction);
    }
}

impl OnlinePolicyUpdater for MemoryReactionStore {
    fn update_from_reaction(
        &self,
        reaction: &Reaction,
        calibration: &CalibrationMetrics,
        config: OnlineUpdateConfig,
    ) {
        if !config.allows(calibration) {
            return;
        }
        let mut traits = self.traits.write();
        for (name, signal) in &reaction.traits {
            let estimate = traits
                .entry((reaction.subject_id.clone(), name.clone()))
                .or_default();
            estimate.value +=
                config.learning_rate.clamp(0.0, 1.0) * (signal.clamp(0.0, 1.0) - estimate.value);
        }
    }
}

impl MemoryReactionStore {
    pub fn record(&self, mut reaction: Reaction) {
        reaction.normalize();
        {
            let mut traits = self.traits.write();
            for (name, signal) in &reaction.traits {
                let e = traits
                    .entry((reaction.subject_id.clone(), name.clone()))
                    .or_default();
                e.observations += 1;
                let rate = 1.0 / (e.observations as f32).sqrt();
                e.value += rate * (signal.clamp(0.0, 1.0) - e.value);
            }
        }
        self.reactions.write().push(reaction);
    }
    pub fn trait_estimate(&self, subject: &str, name: &str) -> Option<TraitEstimate> {
        self.traits
            .read()
            .get(&(subject.into(), name.into()))
            .cloned()
    }
    pub fn reactions(&self) -> Vec<Reaction> {
        self.reactions.read().clone()
    }
    pub fn calibration(&self, bins: usize) -> CalibrationMetrics {
        let samples: Vec<_> = self
            .reactions
            .read()
            .iter()
            .filter_map(|r| r.predicted_outcome.map(|p| (p, r.outcome)))
            .collect();
        if samples.is_empty() {
            return CalibrationMetrics::default();
        }
        let n = samples.len() as f32;
        let brier = samples.iter().map(|(p, y)| (p - y).powi(2)).sum::<f32>() / n;
        let bins = bins.max(1);
        let mut ece = 0.0;
        for b in 0..bins {
            let lo = b as f32 / bins as f32;
            let hi = (b + 1) as f32 / bins as f32;
            let bucket: Vec<_> = samples
                .iter()
                .filter(|(p, _)| *p >= lo && (*p < hi || b + 1 == bins))
                .collect();
            if !bucket.is_empty() {
                let m = bucket.len() as f32;
                let ap = bucket.iter().map(|(p, _)| *p).sum::<f32>() / m;
                let ay = bucket.iter().map(|(_, y)| *y).sum::<f32>() / m;
                ece += m / n * (ap - ay).abs();
            }
        }
        CalibrationMetrics {
            count: samples.len(),
            brier,
            ece,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_calibration_and_traits() {
        let store = MemoryReactionStore::default();
        store.record(Reaction {
            id: "1".into(),
            subject_id: "u".into(),
            execution_id: "e".into(),
            outcome: 1.0,
            predicted_outcome: Some(0.8),
            observed_at: Utc::now(),
            attributions: vec![],
            traits: HashMap::from([("trust".into(), 0.9)]),
        });
        let metrics = store.calibration(10);
        assert!((metrics.brier - 0.04).abs() < 1e-5);
        assert_eq!(
            store
                .trait_estimate("u", "trust")
                .expect("trait estimate")
                .observations,
            1
        );
    }
}
