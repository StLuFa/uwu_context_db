//! CognitiveTrainingPipeline — 真实 Epoch/Trial 训练循环。
//!
//! CDT 训练闭环：
//! 轨迹采集 → 认知编码 → 巩固精炼 → 认知梯度提取 → 偏好策略优化 → Skill 写入 → 反馈回流

use crate::config::{BootstrapConfig, PipelineConfig, PolicyGateConfig};
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
use agent_context_db_consolidation::{ConsolidationEngine, ConsolidationProduct, SignalProvider};
use agent_context_db_core::{
    AccessEvent, ConsolidationStatus, ContentType, ContextError, ContextMeta, ContextUri,
    GraphStore, LifecycleAction, LifecycleEngine, LlmClient, Result, TenantId,
};
use std::sync::Arc;

/// 认知驱动训练管线 — 执行完整的 CDT 闭环。
pub struct CognitiveTrainingPipeline {
    consolidation: Arc<ConsolidationEngine>,
    lifecycle: Arc<LifecycleEngine>,
    llm: Arc<dyn LlmClient>,
    graph: Arc<dyn GraphStore>,
    signals: Arc<dyn SignalProvider>,
    dpo: KnowledgeConstrainedDPO,
    gate: PolicyGate,
    config: PipelineConfig,
}

impl CognitiveTrainingPipeline {
    pub fn new(
        consolidation: Arc<ConsolidationEngine>,
        lifecycle: Arc<LifecycleEngine>,
        llm: Arc<dyn LlmClient>,
        graph: Arc<dyn GraphStore>,
        signals: Arc<dyn SignalProvider>,
        config: PipelineConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            consolidation,
            lifecycle,
            llm,
            graph,
            signals,
            dpo: KnowledgeConstrainedDPO::new(config.dpo_beta, config.dpo_constraint_weight),
            gate: PolicyGate::new(PolicyGateConfig {
                threshold: config.policy_gate_threshold,
            })?,
            config,
        })
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
        let bootstrap = CognitiveBootstrap::new(BootstrapConfig {
            metric_threshold: config.min_confidence,
            max_demos: self.config.bootstrap_demo_limit,
        })?;
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
                    source_uri: ContextUri::parse(format!(
                        "uwu://t/a/memory/skill/{}",
                        &pref.chosen.task_id
                    ))?,
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
                self.config.baseline_accuracy
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

        report.accuracy_delta = best_accuracy - self.config.baseline_accuracy;
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
        let bootstrap_report = CognitiveBootstrap::new(crate::config::BootstrapConfig {
            metric_threshold: config.min_confidence,
            max_demos: self.config.bootstrap_demo_limit,
        })?
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
            let mut lifecycle_rejected = 0usize;
            let mut trainable_products = Vec::with_capacity(products.len());
            for product in &products {
                let action = self.lifecycle_action_for_product(product).await?;
                if action.blocks_training() {
                    lifecycle_rejected += 1;
                } else {
                    trainable_products.push(product.clone());
                }
            }
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
                    epoch_skills.extend(skill_library.retrieve(&embedding.vector).await?);
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
                .collect::<Result<Vec<_>>>()?;

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

        report.accuracy_delta = best_accuracy - self.config.baseline_accuracy;
        Ok(report)
    }

    async fn lifecycle_action_for_product(
        &self,
        product: &ConsolidationProduct,
    ) -> Result<LifecycleAction> {
        let centrality = self.graph.centrality(&product.uri).await?;
        validate_lifecycle_signal("centrality", centrality, &product.uri)?;
        let signals = self.signals.signals(&product.uri).await?;
        let tenant_priority = signals.tenant_priority.ok_or_else(|| {
            ContextError::TrustPolicy(format!(
                "tenant priority is required for training lifecycle gate: {}",
                product.uri
            ))
        })?;
        validate_lifecycle_signal("tenant priority", tenant_priority, &product.uri)?;

        let meta = ContextMeta {
            content_type: Some(product.content_type),
            epistemic_type: Some(product.epistemic_type),
            quality_score: Some(product.quality_score),
            validity: product.metadata.validity.clone(),
            tags: match product.metadata.status {
                ConsolidationStatus::Pending => vec!["pending".into()],
                ConsolidationStatus::InProgress => vec!["in-progress".into()],
                ConsolidationStatus::Converged => vec!["converged".into()],
                ConsolidationStatus::Stale => vec!["stale".into()],
            },
            ..Default::default()
        };
        Ok(self.lifecycle.evaluate_entry(
            &[] as &[AccessEvent],
            &meta,
            Some(centrality),
            Some(tenant_priority),
        ))
    }

    /// 基于训练报告进行门控决策：是否替换当前策略。
    pub fn evaluate_gate(&self, report: &TrainingReport, total_trials: usize) -> GateDecision {
        let new_wins = report
            .epochs
            .iter()
            .filter(|e| e.accuracy > self.config.baseline_accuracy)
            .count();
        self.gate
            .should_replace(new_wins, total_trials, report.accuracy_delta)
    }
}

fn validate_lifecycle_signal(name: &str, value: f32, uri: &ContextUri) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ContextError::TrustPolicy(format!(
            "{name} must be finite and within 0..=1 for training lifecycle gate: {uri} (got {value})"
        )))
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_consolidation::{ConsolidationProductMeta, EntrySignals};
    use agent_context_db_core::{
        EpistemicType, GraphRelation, ImportanceWeights, JsonSchema, LlmError, LlmOpts,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct NoopLlm;

    #[async_trait]
    impl LlmClient for NoopLlm {
        async fn complete(&self, _: &str, _: &LlmOpts) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }

        async fn complete_json(
            &self,
            _: &str,
            _: &JsonSchema,
            _: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }
    }

    struct MockGraph {
        value: Result<f32>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl GraphStore for MockGraph {
        async fn add_edge(&self, _: &ContextUri, _: &ContextUri, _: GraphRelation) -> Result<()> {
            Ok(())
        }
        async fn remove_edge(&self, _: &ContextUri, _: &ContextUri) -> Result<()> {
            Ok(())
        }
        async fn outgoing_neighbors(
            &self,
            _: &ContextUri,
            _: Option<GraphRelation>,
        ) -> Result<Vec<ContextUri>> {
            Ok(vec![])
        }
        async fn batch_traverse(
            &self,
            _: &[ContextUri],
            _: &[GraphRelation],
            _: usize,
        ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
            Ok(vec![])
        }
        async fn centrality(&self, _: &ContextUri) -> Result<f32> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.value
                .as_ref()
                .copied()
                .map_err(|error| ContextError::Storage(error.to_string()))
        }
    }

    struct MockSignals {
        value: Result<Option<f32>>,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl SignalProvider for MockSignals {
        async fn signals(&self, _: &ContextUri) -> Result<EntrySignals> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(EntrySignals {
                tenant_priority: self
                    .value
                    .as_ref()
                    .copied()
                    .map_err(|error| ContextError::Storage(error.to_string()))?,
                ..Default::default()
            })
        }
    }

    fn product() -> ConsolidationProduct {
        ConsolidationProduct {
            uri: ContextUri::parse("uwu://tenant/agent/memory/fact/test").unwrap(),
            content_type: ContentType::Fact,
            epistemic_type: EpistemicType::Fact,
            content: "test".into(),
            quality_score: 0.8,
            confidence: 0.8,
            evidence_required: false,
            superseded_claim: None,
            evidence_uris: vec![],
            contradiction_uris: vec![],
            error_pattern: None,
            hypothesis_outcome: None,
            preconditions: None,
            expected_outcome: None,
            related_policy_uris: vec![],
            provenance: None,
            metadata: ConsolidationProductMeta {
                source_session: None,
                generation: 0,
                status: ConsolidationStatus::Converged,
                patch_count: 0,
                lineage: vec![],
                validity: None,
                half_life: None,
            },
        }
    }

    fn pipeline(
        graph: Arc<dyn GraphStore>,
        signals: Arc<dyn SignalProvider>,
    ) -> CognitiveTrainingPipeline {
        let llm: Arc<dyn LlmClient> = Arc::new(NoopLlm);
        CognitiveTrainingPipeline::new(
            Arc::new(
                ConsolidationEngine::new(
                    agent_context_db_consolidation::ConsolidationConfig::default(),
                    llm.clone(),
                )
                .unwrap(),
            ),
            Arc::new(LifecycleEngine::new(
                LifecycleEngine::default_rules(),
                ImportanceWeights::default(),
            )),
            llm,
            graph,
            signals,
            PipelineConfig::default(),
        )
        .unwrap()
    }

    fn graph(value: Result<f32>) -> Arc<MockGraph> {
        Arc::new(MockGraph {
            value,
            calls: AtomicUsize::new(0),
        })
    }

    fn signals(value: Result<Option<f32>>) -> Arc<MockSignals> {
        Arc::new(MockSignals {
            value,
            calls: AtomicUsize::new(0),
        })
    }

    #[tokio::test]
    async fn real_lifecycle_values_trigger_freeze_and_are_queried_once() {
        let graph = graph(Ok(0.4));
        let signals = signals(Ok(Some(0.95)));
        let action = pipeline(graph.clone(), signals.clone())
            .lifecycle_action_for_product(&product())
            .await
            .unwrap();
        assert!(matches!(action, LifecycleAction::Freeze));
        assert_eq!(graph.calls.load(Ordering::SeqCst), 1);
        assert_eq!(signals.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn missing_tenant_priority_is_an_error() {
        let error = pipeline(graph(Ok(0.4)), signals(Ok(None)))
            .lifecycle_action_for_product(&product())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("tenant priority is required"));
    }

    #[tokio::test]
    async fn graph_and_signal_provider_errors_propagate() {
        let error = pipeline(
            graph(Err(ContextError::Storage("graph failed".into()))),
            signals(Ok(Some(0.5))),
        )
        .lifecycle_action_for_product(&product())
        .await
        .unwrap_err();
        assert!(error.to_string().contains("graph failed"));

        let error = pipeline(
            graph(Ok(0.5)),
            signals(Err(ContextError::Storage("signals failed".into()))),
        )
        .lifecycle_action_for_product(&product())
        .await
        .unwrap_err();
        assert!(error.to_string().contains("signals failed"));
    }

    #[tokio::test]
    async fn rejects_non_finite_and_out_of_range_values() {
        for centrality in [f32::NAN, -0.1, 1.1] {
            assert!(
                pipeline(graph(Ok(centrality)), signals(Ok(Some(0.5))))
                    .lifecycle_action_for_product(&product())
                    .await
                    .is_err()
            );
        }
        for priority in [f32::INFINITY, -0.1, 1.1] {
            assert!(
                pipeline(graph(Ok(0.5)), signals(Ok(Some(priority))))
                    .lifecycle_action_for_product(&product())
                    .await
                    .is_err()
            );
        }
    }
}
