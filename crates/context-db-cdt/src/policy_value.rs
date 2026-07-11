//! Policy/Value 分离 — 策略分离 PolicyModule + ValueModule。

use agent_context_db_core::ContextUri;

/// 认知状态快照。该类型表示已观测状态，不允许搜索模拟直接改写。
#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveState {
    pub graph_density: f32,
    pub recent_errors: Vec<ContextUri>,
    pub active_hypotheses: Vec<ContextUri>,
    pub avg_confidence: f32,
}

/// 动作候选。
#[derive(Debug, Clone, PartialEq)]
pub struct ActionCandidate {
    pub description: String,
    pub confidence: f32,
    /// Policy prior used by MCTS/PUCT. It is normalized by [`PolicyModule::prioritize`].
    pub prior: f32,
    pub expected_effects: Vec<String>,
}

impl ActionCandidate {
    pub fn new(
        description: impl Into<String>,
        confidence: f32,
        prior: f32,
        expected_effects: Vec<String>,
    ) -> Self {
        Self {
            description: description.into(),
            confidence: confidence.clamp(0.0, 1.0),
            prior: prior.max(0.0),
            expected_effects,
        }
    }
}

/// 认知价值 — 多维健康度评估。
#[derive(Debug, Clone)]
pub struct CognitiveValue {
    pub knowledge_consistency: f32,
    pub epistemic_confidence: f32,
    pub evidence_coverage: f32,
    pub composite: f32,
}

impl CognitiveValue {
    pub fn compute(state: &CognitiveState) -> Self {
        let consistency = state.graph_density.clamp(0.0, 1.0);
        let confidence = state.avg_confidence.clamp(0.0, 1.0);
        let coverage = if state.active_hypotheses.is_empty() {
            1.0
        } else {
            (0.55 + state.graph_density * 0.25 - state.active_hypotheses.len() as f32 * 0.02)
                .clamp(0.0, 1.0)
        };
        let error_penalty = (state.recent_errors.len() as f32 * 0.03).min(0.25);
        let composite = (consistency * 0.4 + confidence * 0.35 + coverage * 0.25 - error_penalty)
            .clamp(0.0, 1.0);
        Self {
            knowledge_consistency: consistency,
            epistemic_confidence: confidence,
            evidence_coverage: coverage,
            composite,
        }
    }
}

/// Policy Module — 生成"该怎么做"的候选动作。
pub struct PolicyModule;

impl CognitiveState {
    pub fn transition_delta(&self, next: &CognitiveState) -> CognitiveStateDelta {
        CognitiveStateDelta {
            error_delta: self.recent_errors.len() as i32 - next.recent_errors.len() as i32,
            hypothesis_delta: self.active_hypotheses.len() as i32
                - next.active_hypotheses.len() as i32,
            confidence_delta: next.avg_confidence - self.avg_confidence,
            graph_density_delta: next.graph_density - self.graph_density,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CognitiveStateDelta {
    pub error_delta: i32,
    pub hypothesis_delta: i32,
    pub confidence_delta: f32,
    pub graph_density_delta: f32,
}

impl Default for PolicyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyModule {
    pub fn new() -> Self {
        Self
    }

    pub fn generate(&self, state: &CognitiveState) -> Vec<ActionCandidate> {
        let mut candidates = Vec::new();
        if !state.recent_errors.is_empty() {
            candidates.push(ActionCandidate::new(
                "analyze and fix recent errors",
                0.72,
                0.42 + (state.recent_errors.len() as f32 * 0.04).min(0.18),
                vec!["reduce errors".into(), "increase confidence".into()],
            ));
        }
        if !state.active_hypotheses.is_empty() {
            candidates.push(ActionCandidate::new(
                "validate active hypotheses",
                0.64,
                0.34 + (state.active_hypotheses.len() as f32 * 0.03).min(0.16),
                vec!["confirm or falsify".into(), "increase evidence".into()],
            ));
        }
        if state.graph_density < 0.55 {
            candidates.push(ActionCandidate::new(
                "link related knowledge nodes",
                0.58,
                0.28 + (0.55 - state.graph_density).max(0.0),
                vec!["improve graph density".into()],
            ));
        }
        candidates.push(ActionCandidate::new(
            "consolidate high-confidence knowledge",
            0.52,
            0.25 + state.avg_confidence * 0.2,
            vec!["improve coherence".into()],
        ));
        self.prioritize(candidates)
    }

    /// Normalize and sort policy priors so MCTS can use them directly.
    pub fn prioritize(&self, mut candidates: Vec<ActionCandidate>) -> Vec<ActionCandidate> {
        let total: f32 = candidates.iter().map(|c| c.prior.max(0.0)).sum();
        if total > f32::EPSILON {
            for c in &mut candidates {
                c.prior = (c.prior / total).clamp(0.0, 1.0);
            }
        } else if !candidates.is_empty() {
            let p = 1.0 / candidates.len() as f32;
            for c in &mut candidates {
                c.prior = p;
            }
        }
        candidates.sort_by(|a, b| {
            b.prior
                .partial_cmp(&a.prior)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Conservative transition projection used only for MCTS ranking.
    ///
    /// Candidate effects are proposals, not observations. The projection therefore keeps errors,
    /// hypotheses and epistemic confidence unchanged. It only applies a bounded topology estimate
    /// so search can distinguish graph-oriented actions without manufacturing task success.
    pub fn predict(&self, state: &CognitiveState, action: &ActionCandidate) -> CognitiveState {
        let mut predicted = state.clone();
        if action
            .expected_effects
            .iter()
            .any(|effect| effect == "improve graph density" || effect == "improve coherence")
        {
            predicted.graph_density =
                (predicted.graph_density + 0.02 * action.confidence).clamp(0.0, 1.0);
        }
        predicted
    }
}

/// Value Module — 评估"当前认知状态有多健康"。
pub struct ValueModule;

impl Default for ValueModule {
    fn default() -> Self {
        Self::new()
    }
}

impl ValueModule {
    pub fn new() -> Self {
        Self
    }
    pub fn evaluate(&self, state: &CognitiveState) -> CognitiveValue {
        CognitiveValue::compute(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uri(s: &str) -> ContextUri {
        ContextUri::parse(s).unwrap()
    }

    #[test]
    fn policy_priors_are_normalized() {
        let state = CognitiveState {
            graph_density: 0.2,
            recent_errors: vec![uri("uwu://t/agent/a/memories/error/e1")],
            active_hypotheses: vec![uri("uwu://t/agent/a/memories/hypothesis/h1")],
            avg_confidence: 0.4,
        };
        let policy = PolicyModule::new();
        let actions = policy.generate(&state);
        let total: f32 = actions.iter().map(|a| a.prior).sum();
        assert!((total - 1.0).abs() < 0.001);
        assert!(actions.len() >= 3);
    }

    #[test]
    fn action_application_improves_state() {
        let state = CognitiveState {
            graph_density: 0.2,
            recent_errors: vec![uri("uwu://t/agent/a/memories/error/e1")],
            active_hypotheses: vec![],
            avg_confidence: 0.3,
        };
        let policy = PolicyModule::new();
        let action = policy.generate(&state).remove(0);
        let next = policy.predict(&state, &action);
        assert_eq!(next.avg_confidence, state.avg_confidence);
        assert_eq!(next.recent_errors, state.recent_errors);
    }
}
