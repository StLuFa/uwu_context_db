//! # context-db-cdt
//!
//! 认知驱动训练（Cognition-Driven Training）：
//! 认知偏好优化 + 主动课程 + Policy/Value 分离 + 指标驱动。
//!
//! ## 核心闭环
//! 轨迹采集 → 认知编码 → 巩固精炼 → 认知梯度提取 → 策略优化 → Skill写入 → 反馈回流
//!
//! ## 模块
//! - `pipeline` — CognitiveTrainingPipeline (真实 Epoch/Trial 循环)
//! - `preference` — CognitivePreferenceExtractor (三层偏好信号)
//! - `dpo` — KnowledgeConstrained偏好loss
//! - `curriculum` — CurriculumGenerator (主动课程)
//! - `skill_library` — SkillLibrary + embedding 检索
//! - `voting` — Insight 投票演化
//! - `policy_value` — Policy/Value 分离 + 门控
//! - `metric` — 认知 Metric + 可组合优化器
//! - `self_play` — 认知自我对弈
//! - `tree_search` — 树搜索
//! - `multi_perspective` — 多视角巩固
//! - `hybrid_retrieval` — 三维混合检索

pub mod config;
pub mod consolidation;
pub mod curriculum;
pub mod dpo;
pub mod gen_agents;
pub mod hybrid_retrieval;
pub mod lats;
pub mod metric;
pub mod multi_perspective;
pub mod pipeline;
pub mod policy_value;
pub mod preference;
pub mod reflection;
pub mod self_play;
pub mod self_verify;
pub mod skill_library;
pub mod trajectory_encoder;
pub mod tree_search;
pub mod voting;

use agent_context_db_consolidation::{
    ConsolidationProduct, HypothesisOutcome as ProductHypothesisOutcome,
};
use agent_context_db_core::{ConsolidationStatus, ContentType, ContextEntry, ContextUri, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ===========================================================================
// 认知梯度
// ===========================================================================

/// 认知梯度 — 从巩固产物提取的策略改进信号。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveGradient {
    pub source_uri: ContextUri,
    pub epistemic_type: ContentType,
    pub gradient_type: GradientType,
    pub confidence: f32,
    pub evidence_uris: Vec<ContextUri>,
    pub contradiction_uris: Vec<ContextUri>,
    pub weight: f32,
}

/// 梯度类型。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientType {
    FactCorrection {
        old_claim: String,
        new_claim: String,
    },
    AvoidanceRule {
        pattern: String,
        reason: String,
    },
    ValidationRule {
        hypothesis: String,
        outcome: HypothesisOutcome,
    },
    SkillExtraction {
        procedure: String,
        precondition: String,
        expected_outcome: String,
    },
    PreferenceUpdate {
        key: String,
        old_value: Option<String>,
        new_value: String,
    },
    MetaCognitive {
        insight: String,
        applies_to: Vec<ContextUri>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisOutcome {
    Confirmed,
    Falsified,
}

impl CognitiveGradient {
    /// 认识论类型 → 梯度权重映射。
    pub fn compute_weight(ct: ContentType, confidence: f32) -> f32 {
        let base = match ct {
            ContentType::Fact => 1.0,
            ContentType::Error => 0.9,
            ContentType::Skill => 0.85,
            ContentType::Procedure => 0.7,
            ContentType::Preference => 0.6,
            ContentType::Heuristic => 0.5,
            ContentType::Reflection => 0.4,
            ContentType::Belief => 0.3,
            ContentType::Hypothesis => 0.15,
            _ => 0.0,
        };
        base * confidence
    }
}

// ===========================================================================
// 认知偏好对（DPO 融合）
// ===========================================================================

/// 认知偏好对 — 从 trajectory 对比中提取。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePreferencePair {
    pub chosen: TrajectorySummary,
    pub rejected: TrajectorySummary,
    pub preference_source: PreferenceSource,
    pub confidence: f32,
    pub cognitive_delta: CognitiveDelta,
}

/// 轨迹摘要。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectorySummary {
    pub task_id: String,
    pub task_description: String,
    pub success: bool,
    pub steps: usize,
    pub contradictions: usize,
    pub avg_confidence: f32,
}

/// 偏好信号来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreferenceSource {
    TaskOutcome,
    /// Counterfactual ranking from tree search; never an observed success label.
    Simulation,
}

/// 认知差异。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDelta {
    pub contradiction_diff: i32,
    pub confidence_diff: f32,
    pub evidence_diff: i32,
    pub knowledge_graph_growth: i32,
}

// ===========================================================================
// 主动课程（Voyager 融合）
// ===========================================================================

/// 训练目标。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingGoal {
    pub target_node: ContextUri,
    pub difficulty: f32,
    pub prerequisite_skills: Vec<ContextUri>,
    pub expected_new_knowledge: String,
}

pub use curriculum::{CurriculumGenerator, FrontierNode};

// ===========================================================================
// 训练管线
// ===========================================================================

/// 训练配置。
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub trials_per_epoch: usize,
    pub gradient_batch_size: usize,
    pub min_confidence: f32,
    pub forgetting_aware: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 4,
            trials_per_epoch: 10,
            gradient_batch_size: 50,
            min_confidence: 0.3,
            forgetting_aware: true,
        }
    }
}

/// 训练报告。
#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub epochs: Vec<EpochResult>,
    pub accuracy_delta: f32,
}

/// Epoch 结果。
#[derive(Debug, Clone)]
pub struct EpochResult {
    pub epoch: usize,
    pub memories_encoded: usize,
    pub gradients_extracted: usize,
    pub gradients_applied: usize,
    pub gradients_rejected: usize,
    pub accuracy: f32,
}

// ===========================================================================
// 认知梯度提取（ConsolidationProduct → CognitiveGradient）———— P1-1
// ===========================================================================

/// 从巩固产物批量提取认知梯度。
///
/// 根据认识论类型差异化提取：Fact → FactCorrection, Error → AvoidanceRule, etc.
pub fn extract_gradients_from_products(
    products: &[ConsolidationProduct],
    min_confidence: f32,
) -> Vec<CognitiveGradient> {
    let mut gradients = Vec::new();

    let now = Utc::now();
    for product in products {
        // Training consumes only completed, currently valid products with a traceable source.
        // Facts and hypotheses additionally require evidence because they can alter the agent's
        // epistemic state; locally generated procedural signals may use source_session lineage.
        let valid = product.metadata.validity.as_ref().is_none_or(|validity| {
            validity.valid_from <= now
                && validity.valid_until.is_none_or(|until| until > now)
                && validity.invalidated_by.is_none()
                && validity.invalidation_reason.is_none()
        });
        let traceable = product.provenance.is_some()
            || (product.metadata.source_session.is_some() && !product.metadata.lineage.is_empty());
        let evidence_required = matches!(
            product.content_type,
            ContentType::Fact | ContentType::Hypothesis
        );
        if product.metadata.status != ConsolidationStatus::Converged
            || !valid
            || !traceable
            || product.content.trim().is_empty()
            || !product.quality_score.is_finite()
            || !product.confidence.is_finite()
            || product.confidence < min_confidence
            || (evidence_required && product.evidence_uris.is_empty())
        {
            continue;
        }

        let effective_confidence = product
            .quality_score
            .min(product.confidence)
            .clamp(0.0, 1.0);
        let weight = CognitiveGradient::compute_weight(product.content_type, effective_confidence);
        if weight < min_confidence {
            continue;
        }

        let gradient = match product.content_type {
            ContentType::Fact => CognitiveGradient {
                source_uri: product.uri.clone(),
                epistemic_type: ContentType::Fact,
                gradient_type: GradientType::FactCorrection {
                    old_claim: product.superseded_claim.clone().unwrap_or_default(),
                    new_claim: product.content.clone(),
                },
                confidence: effective_confidence,
                evidence_uris: product.evidence_uris.clone(),
                contradiction_uris: product.contradiction_uris.clone(),
                weight,
            },
            ContentType::Error => CognitiveGradient {
                source_uri: product.uri.clone(),
                epistemic_type: ContentType::Error,
                gradient_type: GradientType::AvoidanceRule {
                    pattern: product.error_pattern.clone().unwrap_or_default(),
                    reason: product.content.clone(),
                },
                confidence: effective_confidence,
                evidence_uris: vec![],
                contradiction_uris: vec![],
                weight,
            },
            ContentType::Hypothesis => {
                let outcome = match product.hypothesis_outcome {
                    Some(ProductHypothesisOutcome::Confirmed) => HypothesisOutcome::Confirmed,
                    Some(ProductHypothesisOutcome::Falsified) => HypothesisOutcome::Falsified,
                    Some(ProductHypothesisOutcome::Inconclusive) | None => continue,
                };
                CognitiveGradient {
                    source_uri: product.uri.clone(),
                    epistemic_type: ContentType::Hypothesis,
                    gradient_type: GradientType::ValidationRule {
                        hypothesis: product.content.clone(),
                        outcome,
                    },
                    confidence: effective_confidence,
                    evidence_uris: product.evidence_uris.clone(),
                    contradiction_uris: vec![],
                    weight,
                }
            }
            ContentType::Procedure | ContentType::Skill => CognitiveGradient {
                source_uri: product.uri.clone(),
                epistemic_type: product.content_type,
                gradient_type: GradientType::SkillExtraction {
                    procedure: product.content.clone(),
                    precondition: product.preconditions.clone().unwrap_or_default(),
                    expected_outcome: product.expected_outcome.clone().unwrap_or_default(),
                },
                confidence: effective_confidence,
                evidence_uris: product.evidence_uris.clone(),
                contradiction_uris: vec![],
                weight,
            },
            ContentType::Preference => CognitiveGradient {
                source_uri: product.uri.clone(),
                epistemic_type: ContentType::Preference,
                gradient_type: GradientType::PreferenceUpdate {
                    key: product.uri.to_string(),
                    old_value: product.superseded_claim.clone(),
                    new_value: product.content.clone(),
                },
                confidence: effective_confidence,
                evidence_uris: vec![],
                contradiction_uris: vec![],
                weight,
            },
            ContentType::Reflection => CognitiveGradient {
                source_uri: product.uri.clone(),
                epistemic_type: ContentType::Reflection,
                gradient_type: GradientType::MetaCognitive {
                    insight: product.content.clone(),
                    applies_to: product.related_policy_uris.clone(),
                },
                confidence: effective_confidence,
                evidence_uris: vec![],
                contradiction_uris: vec![],
                weight,
            },
            _ => continue, // Evidence/Meta/Profile/Goal 不产生梯度
        };

        gradients.push(gradient);
    }

    // 按权重降序
    gradients.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    gradients
}

#[cfg(test)]
mod gradient_gate_tests {
    use super::*;
    use agent_context_db_consolidation::{
        ConsolidationProductMeta, HypothesisOutcome as ProductOutcome,
    };
    use agent_context_db_core::{
        ConsolidationStatus, EpistemicType, LineageEntry, MvccVersion, ValidityRecord,
    };

    fn product(content_type: ContentType) -> ConsolidationProduct {
        let now = Utc::now();
        ConsolidationProduct {
            uri: ContextUri::parse(format!(
                "uwu://tenant/agent/a/memory/{}/gate/test",
                content_type.as_path_segment()
            ))
            .unwrap(),
            content_type,
            epistemic_type: EpistemicType::Fact,
            content: "tested claim".into(),
            quality_score: 0.9,
            confidence: 0.8,
            evidence_required: false,
            superseded_claim: None,
            evidence_uris: vec![
                ContextUri::parse("uwu://tenant/agent/a/memory/evidence/e1").unwrap(),
            ],
            contradiction_uris: vec![],
            error_pattern: Some("failure".into()),
            hypothesis_outcome: None,
            preconditions: Some("precondition".into()),
            expected_outcome: Some("outcome".into()),
            related_policy_uris: vec![],
            provenance: None,
            metadata: ConsolidationProductMeta {
                source_session: Some("verified-session".into()),
                generation: 1,
                status: ConsolidationStatus::Converged,
                patch_count: 1,
                lineage: vec![LineageEntry {
                    version: MvccVersion(1),
                    timestamp: now,
                    change_summary: "verified".into(),
                }],
                validity: Some(ValidityRecord {
                    valid_from: now - chrono::Duration::minutes(1),
                    valid_until: None,
                    invalidated_by: None,
                    invalidation_reason: None,
                }),
                half_life: Some(agent_context_db_core::HalfLife::Finite { days: 30.0 }),
            },
        }
    }

    #[test]
    fn hypothesis_outcomes_preserve_confirmed_and_falsified() {
        let mut confirmed = product(ContentType::Hypothesis);
        confirmed.hypothesis_outcome = Some(ProductOutcome::Confirmed);
        let mut falsified = confirmed.clone();
        falsified.uri =
            ContextUri::parse("uwu://tenant/agent/a/memory/hypothesis/gate/falsified").unwrap();
        falsified.hypothesis_outcome = Some(ProductOutcome::Falsified);
        let gradients = extract_gradients_from_products(&[confirmed, falsified], 0.1);
        assert!(gradients.iter().any(|gradient| matches!(
            gradient.gradient_type,
            GradientType::ValidationRule {
                outcome: HypothesisOutcome::Confirmed,
                ..
            }
        )));
        assert!(gradients.iter().any(|gradient| matches!(
            gradient.gradient_type,
            GradientType::ValidationRule {
                outcome: HypothesisOutcome::Falsified,
                ..
            }
        )));
    }

    #[test]
    fn inconclusive_unknown_and_unsafe_products_are_rejected() {
        let mut inconclusive = product(ContentType::Hypothesis);
        inconclusive.hypothesis_outcome = Some(ProductOutcome::Inconclusive);
        let mut unknown = product(ContentType::Hypothesis);
        unknown.hypothesis_outcome = None;
        let mut pending = product(ContentType::Fact);
        pending.metadata.status = ConsolidationStatus::Pending;
        let mut unsupported = product(ContentType::Fact);
        unsupported.evidence_uris.clear();
        let mut untraceable = product(ContentType::Skill);
        untraceable.metadata.source_session = None;
        untraceable.metadata.lineage.clear();
        assert!(
            extract_gradients_from_products(
                &[inconclusive, unknown, pending, unsupported, untraceable],
                0.1
            )
            .is_empty()
        );
    }

    #[test]
    fn confidence_uses_lower_of_quality_and_declared_confidence() {
        let product = product(ContentType::Fact);
        let gradient = extract_gradients_from_products(&[product], 0.1)
            .pop()
            .unwrap();
        assert_eq!(gradient.confidence, 0.8);
        assert_eq!(gradient.weight, 0.8);
    }
}

// ===========================================================================
// Skill 生命周期状态机 ———— P1-2
// ===========================================================================

/// Skill — Procedure 经训练验证后的升级态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub uri: ContextUri,
    pub procedure: String,
    pub precondition: String,
    pub expected_outcome: String,
    pub validation_status: SkillValidationStatus,
    pub success_count: u32,
    pub failure_count: u32,
    pub success_rate: f32,
    pub last_validated: DateTime<Utc>,
    pub source_gradient: Option<ContextUri>,
    pub related_facts: Vec<ContextUri>,
    pub avoidance_rules: Vec<ContextUri>,
}

/// Skill 验证状态机。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SkillValidationStatus {
    /// 假设态：从轨迹提取但未验证。
    Hypothesized,
    /// 验证中：正在训练评估。
    Validating { epoch: usize, trials_done: u32 },
    /// 已验证：训练评估通过。
    Validated {
        success_rate: f32,
        benchmark: String,
    },
    /// 已证伪：训练评估失败。
    Falsified { reason: String },
    /// 已废弃：被更好的 Skill 取代。
    Deprecated { replaced_by: ContextUri },
}

impl Skill {
    /// 创建新的假设态 Skill。
    pub fn new_hypothesis(
        uri: ContextUri,
        procedure: String,
        precondition: String,
        expected_outcome: String,
    ) -> Self {
        Self {
            uri,
            procedure,
            precondition,
            expected_outcome,
            validation_status: SkillValidationStatus::Hypothesized,
            success_count: 0,
            failure_count: 0,
            success_rate: 0.0,
            last_validated: Utc::now(),
            source_gradient: None,
            related_facts: vec![],
            avoidance_rules: vec![],
        }
    }

    /// 开始验证。
    pub fn start_validating(&mut self, epoch: usize) {
        self.validation_status = SkillValidationStatus::Validating {
            epoch,
            trials_done: 0,
        };
    }

    /// 记录一次试验结果。
    pub fn record_trial(&mut self, success: bool) {
        if success {
            self.success_count = self.success_count.saturating_add(1);
        } else {
            self.failure_count = self.failure_count.saturating_add(1);
        }
        let total = self.success_count + self.failure_count;
        self.success_rate = if total > 0 {
            self.success_count as f32 / total as f32
        } else {
            0.0
        };
        self.last_validated = Utc::now();

        if let SkillValidationStatus::Validating { trials_done, .. } = &mut self.validation_status {
            *trials_done = trials_done.saturating_add(1);
        }
    }

    /// 评估并推进状态机。
    pub fn evaluate(
        &mut self,
        success_threshold: f32,
        failure_threshold: f32,
    ) -> SkillValidationStatus {
        let total = self.success_count + self.failure_count;
        if total < 3 {
            return self.validation_status.clone(); // 需要更多数据
        }

        if self.success_rate >= success_threshold {
            self.validation_status = SkillValidationStatus::Validated {
                success_rate: self.success_rate,
                benchmark: "internal".to_string(),
            };
        } else if self.success_rate <= failure_threshold {
            self.validation_status = SkillValidationStatus::Falsified {
                reason: format!(
                    "success_rate {:.2} below threshold {:.2}",
                    self.success_rate, failure_threshold
                ),
            };
        }
        // 否则保持 Validating
        self.validation_status.clone()
    }

    /// 废弃此 Skill，指向更好的替代。
    pub fn deprecate(&mut self, replaced_by: ContextUri) {
        self.validation_status = SkillValidationStatus::Deprecated { replaced_by };
    }

    /// 是否已经验证。
    pub fn is_validated(&self) -> bool {
        matches!(
            self.validation_status,
            SkillValidationStatus::Validated { .. }
        )
    }

    /// 是否已证伪。
    pub fn is_falsified(&self) -> bool {
        matches!(
            self.validation_status,
            SkillValidationStatus::Falsified { .. }
        )
    }
}

// ===========================================================================
// 反馈回流（评估结果 → 认知记忆）———— P1-3
// ===========================================================================

/// 评估结果。
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub epoch: usize,
    pub accuracy: f32,
    pub successes: Vec<SuccessCase>,
    pub failures: Vec<FailureCase>,
}

/// 成功 case。
#[derive(Debug, Clone)]
pub struct SuccessCase {
    pub skill_extracted: String,
    pub procedure: String,
    pub precondition: String,
    pub expected_outcome: String,
}

/// 失败 case。
#[derive(Debug, Clone)]
pub struct FailureCase {
    pub description: String,
    pub analysis: String,
    pub trace: String,
    pub root_cause_contradiction: Option<String>,
}

/// 将评估结果反馈为认知记忆。
///
/// - 成功 case → Skill 记忆（可指导下轮训练）
/// - 失败 case → Error 记忆（避免重复踩坑）
pub fn feedback_evaluation_to_memories(
    eval: &EvalResult,
    agent_scope: &str,
    tenant: agent_context_db_core::TenantId,
) -> agent_context_db_core::Result<Vec<ContextEntry>> {
    let mut entries = Vec::new();

    // 成功 case → Skill 记忆
    for success in &eval.successes {
        let uri = ContextUri::parse(format!(
            "uwu://{}/memory/skill/epoch-{}/{:x}",
            agent_scope,
            eval.epoch,
            success.skill_extracted.len()
        ))?;

        let entry = ContextEntry::new_text(uri, tenant, &success.skill_extracted);
        entries.push(entry);
    }

    // 失败 case → Error 记忆
    for failure in &eval.failures {
        let uri = ContextUri::parse(format!(
            "uwu://{}/memory/error/epoch-{}/{:x}",
            agent_scope,
            eval.epoch,
            failure.description.len()
        ))?;

        let entry = ContextEntry::new_text(uri, tenant, &failure.analysis);
        entries.push(entry);
    }

    Ok(entries)
}

#[cfg(test)]
mod feedback_memory_tests {
    use super::*;
    use agent_context_db_core::TenantId;

    #[test]
    fn feedback_memories_preserve_caller_tenant() {
        let tenant = TenantId(uuid::Uuid::new_v4());
        let eval = EvalResult {
            epoch: 2,
            accuracy: 0.8,
            successes: vec![SuccessCase {
                skill_extracted: "safe deploy".into(),
                procedure: "deploy".into(),
                precondition: "tests pass".into(),
                expected_outcome: "healthy service".into(),
            }],
            failures: vec![FailureCase {
                description: "failed rollout".into(),
                analysis: "rollback required".into(),
                trace: "deploy -> fail".into(),
                root_cause_contradiction: None,
            }],
        };
        let entries = feedback_evaluation_to_memories(&eval, "tenant/agent/a", tenant).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|entry| entry.tenant == tenant));
    }
}

/// 认知 Metric — 多维评估。
#[derive(Debug, Clone)]
pub struct CognitiveMetric {
    pub consistency: f32,
    pub confidence: f32,
    pub completion: f32,
    pub efficiency: f32,
    pub composite: f32,
}

impl CognitiveMetric {
    pub fn compute(
        contradictions: usize,
        avg_confidence: f32,
        task_succeeded: bool,
        steps: usize,
    ) -> Self {
        let consistency = 1.0 / (1.0 + contradictions as f32);
        let confidence = avg_confidence;
        let completion = if task_succeeded { 1.0 } else { 0.0 };
        let efficiency = if steps > 0 {
            1.0 / (steps as f32).ln().max(1.0)
        } else {
            0.0
        };
        let composite =
            consistency * 0.35 + confidence * 0.25 + completion * 0.25 + efficiency * 0.15;
        Self {
            consistency,
            confidence,
            completion,
            efficiency,
            composite,
        }
    }
}

// ===========================================================================
// Policy/Value 门控（AlphaZero 融合）
// ===========================================================================

/// 策略门控决策。
#[derive(Debug, Clone)]
pub enum GateDecision {
    Replace {
        win_rate: f32,
        avg_delta: f32,
    },
    Keep {
        win_rate: f32,
        avg_delta: f32,
        reason: String,
    },
}

/// 策略门控 — 新策略必须胜率 > 阈值才能替换。
pub struct PolicyGate {
    pub threshold: f32,
}

impl PolicyGate {
    pub fn new(config: config::PolicyGateConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            threshold: config.threshold,
        })
    }

    pub fn should_replace(&self, new_wins: usize, total: usize, avg_delta: f32) -> GateDecision {
        let win_rate = if total > 0 {
            new_wins as f32 / total as f32
        } else {
            0.0
        };
        if win_rate >= self.threshold && avg_delta > 0.0 {
            GateDecision::Replace {
                win_rate,
                avg_delta,
            }
        } else {
            GateDecision::Keep {
                win_rate,
                avg_delta,
                reason: format!("win_rate {win_rate:.2} < threshold {:.2}", self.threshold),
            }
        }
    }
}
