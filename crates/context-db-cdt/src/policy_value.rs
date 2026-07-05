//! Policy/Value 分离 — 策略分离 PolicyModule + ValueModule。

use agent_context_db_core::{ContentType, ContextUri, Result};

/// 认知状态快照。
#[derive(Debug, Clone)]
pub struct CognitiveState {
    pub graph_density: f32,
    pub recent_errors: Vec<ContextUri>,
    pub active_hypotheses: Vec<ContextUri>,
    pub avg_confidence: f32,
}

/// 动作候选。
#[derive(Debug, Clone)]
pub struct ActionCandidate {
    pub description: String,
    pub confidence: f32,
    pub expected_effects: Vec<String>,
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
        let consistency = state.graph_density;
        let confidence = state.avg_confidence;
        let coverage = if state.active_hypotheses.is_empty() { 1.0 } else { 0.5 };
        let composite = consistency * 0.4 + confidence * 0.35 + coverage * 0.25;
        Self { knowledge_consistency: consistency, epistemic_confidence: confidence, evidence_coverage: coverage, composite }
    }
}

/// Policy Module — 生成"该怎么做"的候选动作。
pub struct PolicyModule;

impl PolicyModule {
    pub fn new() -> Self { Self }
    pub fn generate(&self, state: &CognitiveState) -> Vec<ActionCandidate> {
        // 基于认知状态生成改进候选
        let mut candidates = Vec::new();
        if !state.recent_errors.is_empty() {
            candidates.push(ActionCandidate { description: "analyze and fix recent errors".into(), confidence: 0.7, expected_effects: vec!["reduce errors".into()] });
        }
        if !state.active_hypotheses.is_empty() {
            candidates.push(ActionCandidate { description: "validate active hypotheses".into(), confidence: 0.6, expected_effects: vec!["confirm or falsify".into()] });
        }
        candidates.push(ActionCandidate { description: "consolidate high-confidence knowledge".into(), confidence: 0.5, expected_effects: vec!["improve coherence".into()] });
        candidates
    }
}

/// Value Module — 评估"当前认知状态有多健康"。
pub struct ValueModule;

impl ValueModule {
    pub fn new() -> Self { Self }
    pub fn evaluate(&self, state: &CognitiveState) -> CognitiveValue {
        CognitiveValue::compute(state)
    }
}
