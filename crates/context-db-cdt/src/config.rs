//! Validated tuning configuration for CDT production subsystems.

use agent_context_db_core::{ContextError, Result};

fn finite(name: &str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ContextError::Unsupported(format!("{name} must be finite")))
    }
}
fn unit(name: &str, value: f32) -> Result<()> {
    finite(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ContextError::Unsupported(format!(
            "{name} must be within 0..=1 (got {value})"
        )))
    }
}
fn positive(name: &str, value: usize) -> Result<()> {
    if value > 0 {
        Ok(())
    } else {
        Err(ContextError::Unsupported(format!(
            "{name} must be greater than zero"
        )))
    }
}

#[derive(Debug, Clone)]
pub struct StormConfig {
    pub max_questions_per_perspective: usize,
    pub fallback_evidence_limit: usize,
    pub claims_per_section: usize,
    pub open_questions_limit: usize,
    pub synthesis_confidence_weight: f32,
    pub question_base_priority: f32,
    pub question_index_weight: f32,
    pub evidence_scarcity_weight: f32,
    pub empty_section_confidence: f32,
    pub section_base_confidence: f32,
    pub claim_confidence_weight: f32,
}
impl Default for StormConfig {
    fn default() -> Self {
        Self {
            max_questions_per_perspective: 2,
            fallback_evidence_limit: 3,
            claims_per_section: 5,
            open_questions_limit: 5,
            synthesis_confidence_weight: 0.8,
            question_base_priority: 0.7,
            question_index_weight: 0.05,
            evidence_scarcity_weight: 0.1,
            empty_section_confidence: 0.1,
            section_base_confidence: 0.35,
            claim_confidence_weight: 0.12,
        }
    }
}
impl StormConfig {
    pub fn validate(&self) -> Result<()> {
        positive(
            "storm.max_questions_per_perspective",
            self.max_questions_per_perspective,
        )?;
        positive(
            "storm.fallback_evidence_limit",
            self.fallback_evidence_limit,
        )?;
        positive("storm.claims_per_section", self.claims_per_section)?;
        positive("storm.open_questions_limit", self.open_questions_limit)?;
        for (n, v) in [
            (
                "synthesis_confidence_weight",
                self.synthesis_confidence_weight,
            ),
            ("question_base_priority", self.question_base_priority),
            ("question_index_weight", self.question_index_weight),
            ("evidence_scarcity_weight", self.evidence_scarcity_weight),
            ("empty_section_confidence", self.empty_section_confidence),
            ("section_base_confidence", self.section_base_confidence),
            ("claim_confidence_weight", self.claim_confidence_weight),
        ] {
            unit(&format!("storm.{n}"), v)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VotingConfig {
    pub deprecate_threshold: f32,
    pub merge_threshold: f32,
    pub similarity_prefix_tokens: usize,
}
impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            deprecate_threshold: 0.0,
            merge_threshold: 0.72,
            similarity_prefix_tokens: 8,
        }
    }
}
impl VotingConfig {
    pub fn validate(&self) -> Result<()> {
        finite("voting.deprecate_threshold", self.deprecate_threshold)?;
        unit("voting.merge_threshold", self.merge_threshold)?;
        positive(
            "voting.similarity_prefix_tokens",
            self.similarity_prefix_tokens,
        )
    }
}

#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    pub exploration_ratio: f32,
    pub zpd_difficulty: f32,
    pub known_confidence_threshold: f32,
    pub bootstrap_known_confidence: f32,
    pub traversal_hops: usize,
}
impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            exploration_ratio: 0.2,
            zpd_difficulty: 0.6,
            known_confidence_threshold: 0.7,
            bootstrap_known_confidence: 0.85,
            traversal_hops: 1,
        }
    }
}
impl CurriculumConfig {
    pub fn validate(&self) -> Result<()> {
        for (n, v) in [
            ("exploration_ratio", self.exploration_ratio),
            ("zpd_difficulty", self.zpd_difficulty),
            (
                "known_confidence_threshold",
                self.known_confidence_threshold,
            ),
            (
                "bootstrap_known_confidence",
                self.bootstrap_known_confidence,
            ),
        ] {
            unit(&format!("curriculum.{n}"), v)?;
        }
        positive("curriculum.traversal_hops", self.traversal_hops)
    }
}

#[derive(Debug, Clone)]
pub struct SelfVerifyConfig {
    pub success_threshold: f32,
}
impl Default for SelfVerifyConfig {
    fn default() -> Self {
        Self {
            success_threshold: 0.0,
        }
    }
}
impl SelfVerifyConfig {
    pub fn validate(&self) -> Result<()> {
        unit("self_verify.success_threshold", self.success_threshold)
    }
}

#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    pub metric_threshold: f32,
    pub max_demos: usize,
}
impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            metric_threshold: 0.5,
            max_demos: 5,
        }
    }
}
impl BootstrapConfig {
    pub fn validate(&self) -> Result<()> {
        unit("bootstrap.metric_threshold", self.metric_threshold)?;
        positive("bootstrap.max_demos", self.max_demos)
    }
}

#[derive(Debug, Clone)]
pub struct PolicyGateConfig {
    pub threshold: f32,
}
impl Default for PolicyGateConfig {
    fn default() -> Self {
        Self { threshold: 0.55 }
    }
}
impl PolicyGateConfig {
    pub fn validate(&self) -> Result<()> {
        unit("policy_gate.threshold", self.threshold)
    }
}

#[derive(Debug, Clone)]
pub struct HybridRetrievalConfig {
    pub relevance_weight: f32,
    pub recency_weight: f32,
    pub confidence_weight: f32,
}
impl Default for HybridRetrievalConfig {
    fn default() -> Self {
        Self {
            relevance_weight: 0.4,
            recency_weight: 0.3,
            confidence_weight: 0.3,
        }
    }
}
impl HybridRetrievalConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("relevance_weight", self.relevance_weight),
            ("recency_weight", self.recency_weight),
            ("confidence_weight", self.confidence_weight),
        ] {
            unit(&format!("hybrid_retrieval.{name}"), value)?;
        }
        let sum = self.relevance_weight + self.recency_weight + self.confidence_weight;
        if (sum - 1.0).abs() > f32::EPSILON * 4.0 {
            return Err(ContextError::Unsupported(format!(
                "hybrid_retrieval weights must sum to 1 (got {sum})"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub dpo_beta: f64,
    pub dpo_constraint_weight: f64,
    pub policy_gate_threshold: f32,
    pub bootstrap_demo_limit: usize,
    pub skill_retrieval_limit: usize,
    pub baseline_accuracy: f32,
}
impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            dpo_beta: 0.1,
            dpo_constraint_weight: 0.5,
            policy_gate_threshold: 0.55,
            bootstrap_demo_limit: 5,
            skill_retrieval_limit: 5,
            baseline_accuracy: 0.5,
        }
    }
}
impl PipelineConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.dpo_beta.is_finite() || self.dpo_beta <= 0.0 {
            return Err(ContextError::Unsupported(
                "pipeline.dpo_beta must be finite and positive".into(),
            ));
        }
        if !self.dpo_constraint_weight.is_finite() || self.dpo_constraint_weight < 0.0 {
            return Err(ContextError::Unsupported(
                "pipeline.dpo_constraint_weight must be finite and non-negative".into(),
            ));
        }
        unit("pipeline.policy_gate_threshold", self.policy_gate_threshold)?;
        unit("pipeline.baseline_accuracy", self.baseline_accuracy)?;
        positive("pipeline.bootstrap_demo_limit", self.bootstrap_demo_limit)?;
        positive("pipeline.skill_retrieval_limit", self.skill_retrieval_limit)
    }
}

#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    pub writeback_min_priority: f32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub recall_max_depth: usize,
    pub half_life_days: f32,
}
impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            writeback_min_priority: 0.35,
            max_tokens: 512,
            temperature: 0.1,
            recall_max_depth: 3,
            half_life_days: 120.0,
        }
    }
}
impl ReflectionConfig {
    pub fn validate(&self) -> Result<()> {
        unit(
            "reflection.writeback_min_priority",
            self.writeback_min_priority,
        )?;
        unit("reflection.temperature", self.temperature)?;
        if self.max_tokens == 0
            || self.recall_max_depth == 0
            || !self.half_life_days.is_finite()
            || self.half_life_days <= 0.0
        {
            return Err(ContextError::Unsupported(
                "reflection limits and half life must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TrajectoryEncodingConfig {
    pub procedure_min_steps: usize,
    pub uri_hash_chars: usize,
}
impl Default for TrajectoryEncodingConfig {
    fn default() -> Self {
        Self {
            procedure_min_steps: 2,
            uri_hash_chars: 8,
        }
    }
}
impl TrajectoryEncodingConfig {
    pub fn validate(&self) -> Result<()> {
        positive("trajectory.procedure_min_steps", self.procedure_min_steps)?;
        if !(1..=64).contains(&self.uri_hash_chars) {
            return Err(ContextError::Unsupported(
                "trajectory.uri_hash_chars must be within 1..=64".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SkillLibraryConfig {
    pub default_retrieval_limit: usize,
    pub collection_hash_chars: usize,
    pub min_writeback_success_rate: f32,
}
impl Default for SkillLibraryConfig {
    fn default() -> Self {
        Self {
            default_retrieval_limit: 5,
            collection_hash_chars: 24,
            min_writeback_success_rate: 0.0,
        }
    }
}
impl SkillLibraryConfig {
    pub fn validate(&self) -> Result<()> {
        positive(
            "skill.default_retrieval_limit",
            self.default_retrieval_limit,
        )?;
        if !(1..=64).contains(&self.collection_hash_chars) {
            return Err(ContextError::Unsupported(
                "skill.collection_hash_chars must be within 1..=64".into(),
            ));
        }
        unit(
            "skill.min_writeback_success_rate",
            self.min_writeback_success_rate,
        )
    }
}
