use agent_context_db_marketplace::AgentId;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteOutcome {
    pub peer: AgentId,
    pub success: bool,
    pub latency_ms: u64,
    pub hit_count: usize,
    pub avg_score: f32,
    pub observed_at: DateTime<Utc>,
}

impl RouteOutcome {
    pub fn into_reaction(self, execution_id: impl Into<String>) -> agent_context_db_core::Reaction {
        let latency_score = (1.0 - self.latency_ms as f32 / 2_000.0).clamp(0.0, 1.0);
        let outcome = (self.success as u8 as f32 * 0.5
            + self.avg_score.clamp(0.0, 1.0) * 0.3
            + latency_score * 0.2)
            .clamp(0.0, 1.0);
        agent_context_db_core::Reaction {
            id: format!(
                "route:{}:{}",
                self.peer,
                self.observed_at.timestamp_micros()
            ),
            subject_id: self.peer.to_string(),
            execution_id: execution_id.into(),
            outcome,
            predicted_outcome: None,
            observed_at: self.observed_at,
            attributions: vec![agent_context_db_core::CausalAttribution {
                cause_id: "route_peer".into(),
                credit: 1.0,
                confidence: self.avg_score.clamp(0.0, 1.0),
            }],
            traits: HashMap::from([
                ("route_reliability".into(), self.success as u8 as f32),
                ("route_latency".into(), latency_score),
            ]),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouteLearningState {
    pub attempts: u64,
    pub successes: u64,
    pub avg_latency_ms: f32,
    pub avg_hit_count: f32,
    pub avg_score: f32,
    pub last_seen_at: Option<DateTime<Utc>>,
}

impl RouteLearningState {
    pub fn reliability(&self) -> f32 {
        if self.attempts == 0 {
            return 0.5;
        }
        (self.successes as f32 / self.attempts as f32).clamp(0.0, 1.0)
    }

    pub fn utility(&self) -> f32 {
        let latency = (1.0 - (self.avg_latency_ms / 2_000.0)).clamp(0.0, 1.0);
        let yield_score = (self.avg_hit_count / 10.0).clamp(0.0, 1.0);
        (self.reliability() * 0.35 + latency * 0.25 + yield_score * 0.20 + self.avg_score * 0.20)
            .clamp(0.0, 1.0)
    }
}

#[derive(Debug, Default)]
pub struct RouteOutcomeLearning {
    states: RwLock<HashMap<AgentId, RouteLearningState>>,
}

impl RouteOutcomeLearning {
    pub fn record(&self, outcome: RouteOutcome) {
        let mut states = self.states.write();
        let state = states.entry(outcome.peer).or_default();
        state.attempts = state.attempts.saturating_add(1);
        if outcome.success {
            state.successes = state.successes.saturating_add(1);
        }
        let n = state.attempts as f32;
        state.avg_latency_ms += (outcome.latency_ms as f32 - state.avg_latency_ms) / n;
        state.avg_hit_count += (outcome.hit_count as f32 - state.avg_hit_count) / n;
        state.avg_score += (outcome.avg_score - state.avg_score) / n;
        state.last_seen_at = Some(outcome.observed_at);
    }

    pub fn route_bonus(&self, peer: &AgentId) -> f32 {
        self.states
            .read()
            .get(peer)
            .map(|s| (s.utility() - 0.5) * 0.18)
            .unwrap_or(0.0)
    }

    pub fn snapshot(&self) -> HashMap<AgentId, RouteLearningState> {
        self.states.read().clone()
    }
}
