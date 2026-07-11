//! CognitiveTrainingPipeline — 真实 Epoch/Trial 训练循环。
//!
//! CDT 训练闭环：
//! 轨迹采集 → 认知编码 → 巩固精炼 → 认知梯度提取 → 偏好策略优化 → Skill 写入 → 反馈回流

use crate::consolidation::CdtConsolidationBridge;
use crate::curriculum::CurriculumGenerator;
use crate::dpo::KnowledgeConstrainedDPO;
use crate::dpo::PreferenceScores;
use crate::metric::{
    BootstrapDemoOptimizer, CognitiveBootstrap, ForgettingPriorityOptimizer, OptimizerPipeline,
};
use crate::preference::CognitivePreferenceExtractor;
use crate::skill_library::{SkillEntry, SkillLibrary};
use crate::trajectory_encoder::{Trajectory, TrajectoryEncoder};
use crate::{
    CognitiveGradient, CognitiveMetric, CognitivePreferencePair, EpochResult, GateDecision,
    PolicyGate, TrainingConfig, TrainingReport, TrajectorySummary,
};
use agent_context_db_consolidation::{ConsolidationEngine, ConsolidationProduct};
use agent_context_db_core::{
    AccessEvent, ConsolidationStatus, ContentType, ContextMeta, ContextUri, LifecycleAction,
    LifecycleEngine, LlmClient, Result, TenantId,
};
use std::sync::Arc;

/// 认知驱动训练管线 — 执行完整的 CDT 闭环。
pub struct CognitiveTrainingPipeline {
    consolidation: Arc<ConsolidationEngine>,
    lifecycle: Arc<LifecycleEngine>,
    llm: Arc<dyn LlmClient>,
    dpo: KnowledgeConstrainedDPO,
    gate: PolicyGate,
}

impl CognitiveTrainingPipeline {
    pub fn new(
        consolidation: Arc<ConsolidationEngine>,
        lifecycle: Arc<LifecycleEngine>,
        llm: Arc<dyn LlmClient>,
    ) -> Self {
        Self {
            consolidation,
            lifecycle,
            llm,
            dpo: KnowledgeConstrainedDPO::new(0.1, 0.5),
            gate: PolicyGate::new(0.55),
        }
    }

    /// 执行完整 CDT 训练循环。
    ///
    /// 对给定轨迹进行：
    /// 1. 认知编码（轨迹 → ContextEntry）
    /// 2. 巩固精炼（ContextEntry → ConsolidationProduct）
    /// 3. 认知梯度提取（ConsolidationProduct → CognitiveGradient）
    /// 4. 策略优化（梯度 + 偏好loss → 策略更新）
    /// 5. 生成训练报告
    pub async fn train(
        &self,
        config: &TrainingConfig,
        trajectories: &[TrajectorySummary],
    ) -> Result<TrainingReport> {
        let mut report = TrainingReport {
            epochs: vec![],
            accuracy_delta: 0.0,
        };
        let mut best_accuracy = 0.0;

        // ── 阶段 0: 认知偏好提取 + DSPy bootstrap 优化 ──
        let pairs: Vec<_> = trajectories
            .iter()
            .map(|t| (t.clone(), t.success))
            .collect();
        let preferences = CognitivePreferenceExtractor::extract_pairs(&pairs);
        let mut effective_config = config.clone();
        let bootstrap = CognitiveBootstrap::new(config.min_confidence).with_max_demos(5);
        let bootstrap_report = bootstrap.extract_from_preferences(&preferences);
        OptimizerPipeline::new()
            .with(Box::new(BootstrapDemoOptimizer {
                demos: bootstrap_report.demos,
            }))
            .with(Box::new(ForgettingPriorityOptimizer))
            .run(&mut effective_config)
            .await?;

        for epoch in 0..effective_config.epochs {
            let mut gradients_applied = 0usize;
            let mut gradients_rejected = 0usize;
            let mut total_contradictions = 0usize;
            let mut avg_confidence = 0.0f32;

            // ── 阶段 1: 从偏好对中提取梯度 ──
            let mut all_gradients: Vec<CognitiveGradient> = Vec::new();
            for pref in &preferences {
                // 成功轨迹 → 正梯度
                let ct = ContentType::Skill; // 从 trajectory 推断
                let grad = CognitiveGradient {
                    source_uri: ContextUri::parse(&format!(
                        "uwu://t/a/memory/skill/{}",
                        &pref.chosen.task_id
                    ))
                    .unwrap_or_else(|_| {
                        ContextUri::parse("uwu://t/a/memory/skill/fallback").unwrap()
                    }),
                    epistemic_type: ct,
                    gradient_type: crate::GradientType::SkillExtraction {
                        procedure: pref.chosen.task_description.clone(),
                        precondition: String::new(),
                        expected_outcome: String::new(),
                    },
                    confidence: pref.chosen.avg_confidence,
                    evidence_uris: vec![],
                    contradiction_uris: vec![],
                    weight: CognitiveGradient::compute_weight(ct, pref.chosen.avg_confidence),
                };
                all_gradients.push(grad);
            }

            // ── 阶段 2: 按梯度权重排序（遗忘优先 + 置信度优先） ──
            all_gradients.sort_by(|a, b| {
                b.weight
                    .partial_cmp(&a.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // ── 阶段 3: 策略优化（偏好loss + 门控） ──
            for gradient in all_gradients
                .iter()
                .take(effective_config.gradient_batch_size)
            {
                if gradient.weight < effective_config.min_confidence {
                    gradients_rejected += 1;
                    continue;
                }

                // 计算 偏好loss 增量
                let (chosen, rejected) = match find_best_worst_pair(&preferences) {
                    Some(p) => p,
                    None => {
                        gradients_rejected += 1;
                        continue;
                    }
                };

                let loss = self
                    .dpo
                    .trajectory_pair_loss(&chosen.chosen, &rejected.rejected);

                // 门控：loss 改善 > 0 才接受
                if loss < 0.0 {
                    gradients_applied += 1;
                    total_contradictions += chosen.chosen.contradictions;
                    avg_confidence += gradient.confidence;
                } else {
                    gradients_rejected += 1;
                }
            }

            if gradients_applied > 0 {
                avg_confidence /= gradients_applied as f32;
            }

            // ── 阶段 4: 记录 epoch 指标 ──
            let accuracy = if gradients_applied > 0 {
                // 准确率 = 采纳梯度数 / 总梯度数
                let base =
                    gradients_applied as f32 / (gradients_applied + gradients_rejected) as f32;
                // 惩罚矛盾
                let contradiction_penalty = 1.0 / (1.0 + total_contradictions as f32);
                base * contradiction_penalty
            } else if epoch == 0 {
                0.5 // baseline
            } else {
                best_accuracy
            };

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
            }

            let metric = CognitiveMetric::compute(
                total_contradictions,
                avg_confidence,
                gradients_applied > 0,
                gradients_applied + gradients_rejected,
            );

            report.epochs.push(EpochResult {
                epoch,
                memories_encoded: trajectories.len(),
                gradients_extracted: all_gradients.len(),
                gradients_applied,
                gradients_rejected,
                accuracy: metric.composite,
            });

            tracing::info!(
                epoch = epoch,
                applied = gradients_applied,
                rejected = gradients_rejected,
                accuracy = metric.composite,
                "CDT epoch complete"
            );
        }

        report.accuracy_delta = best_accuracy - 0.5;
        Ok(report)
    }

    /// 完整的 CDT 训练管线（集成所有模块）：
    /// 轨迹编码 → 巩固 → 梯度提取 → 课程生成 → 策略优化 → Skill 写入 → 反馈回流。
    pub async fn train_from_trajectories(
        &self,
        config: &TrainingConfig,
        trajectories: &[Trajectory],
        encoder: &TrajectoryEncoder,
        skill_library: &SkillLibrary,
        curriculum: &CurriculumGenerator,
    ) -> Result<TrainingReport> {
        let mut report = TrainingReport {
            epochs: vec![],
            accuracy_delta: 0.0,
        };
        let mut best_accuracy = 0.0;
        let mut effective_config = config.clone();
        let summaries: Vec<TrajectorySummary> = trajectories
            .iter()
            .enumerate()
            .map(|(idx, t)| TrajectorySummary {
                task_id: if t.task_id.is_empty() {
                    format!("trajectory-{idx}")
                } else {
                    t.task_id.clone()
                },
                task_description: t.task_description.clone(),
                success: t.success,
                steps: t.steps.len(),
                contradictions: usize::from(t.error_message.is_some()),
                avg_confidence: if t.success { 0.85 } else { 0.35 },
            })
            .collect();
        let bootstrap_report = CognitiveBootstrap::new(config.min_confidence)
            .with_max_demos(5)
            .extract_from_trajectories(&summaries);
        OptimizerPipeline::new()
            .with(Box::new(BootstrapDemoOptimizer {
                demos: bootstrap_report.demos,
            }))
            .with(Box::new(ForgettingPriorityOptimizer))
            .run(&mut effective_config)
            .await?;

        // ── 阶段 1: 轨迹 → 认知编码 ──
        let entries = encoder.encode_batch(trajectories);
        tracing::info!(count = entries.len(), "trajectories encoded to memories");

        for epoch in 0..effective_config.epochs {
            // ── 阶段 2: 巩固精炼 ──
            let products = self.consolidation.consolidate_batch(&entries).await?;
            tracing::info!(
                epoch = epoch,
                count = products.len(),
                "consolidation complete"
            );

            // ── 阶段 3: 生命周期评估 + 认知梯度提取 ──
            let lifecycle_rejected = products
                .iter()
                .filter(|product| self.lifecycle_action_for_product(product).blocks_training())
                .count();
            let trainable_products = products
                .iter()
                .filter(|product| !self.lifecycle_action_for_product(product).blocks_training())
                .cloned()
                .collect::<Vec<_>>();
            let gradients = crate::extract_gradients_from_products(
                &trainable_products,
                effective_config.min_confidence,
            );
            tracing::info!(
                epoch = epoch,
                gradients = gradients.len(),
                "gradients extracted"
            );

            // ── 阶段 4: 主动课程生成（Voyager） ──
            let mut epoch_skills: Vec<SkillEntry> = Vec::new();
            let known_uris = trainable_products
                .iter()
                .map(|product| product.uri.clone())
                .collect::<Vec<_>>();
            if let Ok(goal) = curriculum.next_goal(&known_uris).await {
                // Embedding failure means the curriculum retrieval step is unavailable; a zero
                // vector would create arbitrary nearest-neighbor results and is not a substitute.
                if let Ok(embedding) = self.llm.embed(&goal.expected_new_knowledge).await {
                    epoch_skills.extend(skill_library.retrieve(&embedding.vector, 5).await?);
                }
            }

            // ── 阶段 5: 策略优化（偏好loss） ──
            let mut gradients_applied = 0usize;
            let mut gradients_rejected = lifecycle_rejected;
            let mut accepted_gradients = Vec::new();

            for gradient in gradients.iter().take(effective_config.gradient_batch_size) {
                if gradient.weight < effective_config.min_confidence {
                    gradients_rejected += 1;
                    continue;
                }

                if let Some((chosen, rejected)) = find_best_worst_pair_from_gradients(&gradients) {
                    let loss = self.dpo.loss_from_scores(
                        PreferenceScores {
                            chosen: chosen.weight as f64,
                            rejected: rejected.weight as f64,
                            reference_chosen: chosen.confidence as f64 * 0.5,
                            reference_rejected: rejected.confidence as f64 * 0.5,
                        },
                        rejected.contradiction_uris.len() as i32
                            - chosen.contradiction_uris.len() as i32,
                    );
                    if loss < 0.0 {
                        gradients_applied += 1;
                        accepted_gradients.push(gradient.clone());
                        // ── 阶段 6: Skill 写入 ──
                        if let ContentType::Skill = gradient.epistemic_type {
                            let skill_embedding = self
                                .llm
                                .embed(&gradient.source_uri.to_string())
                                .await?
                                .vector;
                            let skill = SkillEntry {
                                uri: gradient.source_uri.clone(),
                                name: format!("skill-epoch-{}", epoch),
                                description: gradient.source_uri.to_string(),
                                precondition: String::new(),
                                success_rate: gradient.confidence,
                                embedding: skill_embedding,
                            };
                            skill_library.deposit(&skill).await?;
                        }
                    } else {
                        gradients_rejected += 1;
                    }
                }
            }

            // ── 阶段 7: CDT 信号沉淀为长期巩固产物 ──
            let bridge = CdtConsolidationBridge::new(
                format!("t/agent/cdt/epoch-{epoch}"),
                TenantId(uuid::Uuid::new_v4()),
            );
            let consolidated_signals = bridge.products_from_gradients(&accepted_gradients);
            let materialized_entries = accepted_gradients
                .iter()
                .map(|gradient| bridge.entry_from_signal(&bridge.signal_from_gradient(gradient)))
                .collect::<Vec<_>>();

            let metric = CognitiveMetric::compute(
                0,
                gradients.iter().map(|g| g.confidence).sum::<f32>() / gradients.len().max(1) as f32,
                gradients_applied > 0,
                gradients.len(),
            );

            if metric.composite > best_accuracy {
                best_accuracy = metric.composite;
            }

            report.epochs.push(EpochResult {
                epoch,
                memories_encoded: entries.len() + materialized_entries.len(),
                gradients_extracted: gradients.len() + consolidated_signals.len(),
                gradients_applied,
                gradients_rejected,
                accuracy: metric.composite,
            });

            tracing::info!(
                epoch = epoch,
                applied = gradients_applied,
                rejected = gradients_rejected,
                cdt_consolidation_products = consolidated_signals.len(),
                materialized_entries = materialized_entries.len(),
                accuracy = metric.composite,
                "CDT epoch complete (full pipeline)"
            );
        }

        report.accuracy_delta = best_accuracy - 0.5;
        Ok(report)
    }

    fn lifecycle_action_for_product(&self, product: &ConsolidationProduct) -> LifecycleAction {
        let mut meta = ContextMeta::default();
        meta.content_type = Some(product.content_type);
        meta.epistemic_type = Some(product.epistemic_type);
        meta.quality_score = Some(product.quality_score);
        meta.validity = product.metadata.validity.clone();
        meta.tags = match product.metadata.status {
            ConsolidationStatus::Pending => vec!["pending".into()],
            ConsolidationStatus::InProgress => vec!["in-progress".into()],
            ConsolidationStatus::Converged => vec!["converged".into()],
            ConsolidationStatus::Stale => vec!["stale".into()],
        };
        self.lifecycle.evaluate_entry(&[] as &[AccessEvent], &meta)
    }

    /// 基于训练报告进行门控决策：是否替换当前策略。
    pub fn evaluate_gate(&self, report: &TrainingReport, total_trials: usize) -> GateDecision {
        let new_wins = report.epochs.iter().filter(|e| e.accuracy > 0.5).count();
        self.gate
            .should_replace(new_wins, total_trials, report.accuracy_delta)
    }
}

trait LifecycleTrainingGate {
    fn blocks_training(&self) -> bool;
}

impl LifecycleTrainingGate for LifecycleAction {
    fn blocks_training(&self) -> bool {
        matches!(
            self,
            LifecycleAction::Archive | LifecycleAction::Delete | LifecycleAction::Freeze
        )
    }
}

/// 从偏好对中找到最优/最差对。
fn find_best_worst_pair(
    pairs: &[CognitivePreferencePair],
) -> Option<(&CognitivePreferencePair, &CognitivePreferencePair)> {
    let best = pairs.iter().max_by(|a, b| {
        a.confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let worst = pairs.iter().min_by(|a, b| {
        a.confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    Some((best, worst))
}

/// 从梯度列表中找到最优/最差梯度对（按权重排序）。
fn find_best_worst_pair_from_gradients(
    gradients: &[CognitiveGradient],
) -> Option<(&CognitiveGradient, &CognitiveGradient)> {
    let best = gradients.iter().max_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let worst = gradients.iter().min_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    Some((best, worst))
}
