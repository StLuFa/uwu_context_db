//! DSPy 风格优化 — CognitiveBootstrap + 可组合 OptimizerPipeline。
//!
//! 这里负责把 CDT 训练中的成功轨迹、偏好对和认知 metric 转成可复用 demo，
//! 再通过优化器流水线调整训练配置，形成“评估 → bootstrap → optimize”的闭环。

use crate::config::BootstrapConfig;
use crate::{CognitiveMetric, CognitivePreferencePair, TrainingConfig, TrajectorySummary};
use agent_context_db_core::Result;

/// Bootstrap demo — 从成功执行或高分偏好中自动提取 few-shot demo。
pub struct CognitiveBootstrap {
    pub metric_threshold: f32,
    pub max_demos: usize,
}

#[derive(Debug, Clone)]
pub struct Demo {
    pub input: String,
    pub output: Vec<String>,
    pub metric: f32,
    pub rationale: String,
    pub source_task_id: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct BootstrapReport {
    pub demos: Vec<Demo>,
    pub candidates_seen: usize,
    pub rejected: usize,
    pub avg_metric: f32,
}

#[derive(Debug, Clone)]
pub struct OptimizerRunReport {
    pub stages_run: Vec<&'static str>,
    pub demos_used: usize,
    pub before: TrainingConfig,
    pub after: TrainingConfig,
}

impl CognitiveBootstrap {
    pub fn new(config: BootstrapConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            metric_threshold: config.metric_threshold,
            max_demos: config.max_demos,
        })
    }

    pub fn extract_demos(&self, metrics: &[(String, CognitiveMetric)]) -> Vec<Demo> {
        let mut demos: Vec<_> = metrics
            .iter()
            .filter(|(_, m)| m.composite >= self.metric_threshold)
            .map(|(input, m)| Demo {
                input: input.clone(),
                output: vec![format!(
                    "consistency={:.3}; confidence={:.3}; completion={:.3}; efficiency={:.3}",
                    m.consistency, m.confidence, m.completion, m.efficiency
                )],
                metric: m.composite,
                rationale: "selected by cognitive metric threshold".into(),
                source_task_id: None,
            })
            .collect();
        self.rank_and_trim(&mut demos);
        demos
    }

    pub fn extract_from_preferences(&self, pairs: &[CognitivePreferencePair]) -> BootstrapReport {
        let mut demos: Vec<_> = pairs
            .iter()
            .filter(|pair| pair.chosen.success && pair.confidence >= self.metric_threshold)
            .map(|pair| Demo {
                input: pair.chosen.task_description.clone(),
                output: vec![format!(
                    "prefer trajectory with confidence {:.3}, contradictions {}, steps {}",
                    pair.chosen.avg_confidence, pair.chosen.contradictions, pair.chosen.steps
                )],
                metric: pair
                    .confidence
                    .max(pair.chosen.avg_confidence)
                    .clamp(0.0, 1.0),
                rationale: format!("preference source: {:?}", pair.preference_source),
                source_task_id: Some(pair.chosen.task_id.clone()),
            })
            .collect();
        self.report_from_demos(pairs.len(), &mut demos)
    }

    pub fn extract_from_trajectories(&self, trajectories: &[TrajectorySummary]) -> BootstrapReport {
        let mut demos: Vec<_> = trajectories
            .iter()
            .filter(|t| t.success && t.avg_confidence >= self.metric_threshold)
            .map(|t| Demo {
                input: t.task_description.clone(),
                output: vec![format!(
                    "success=true; confidence={:.3}; contradictions={}; steps={}",
                    t.avg_confidence, t.contradictions, t.steps
                )],
                metric: CognitiveMetric::compute(
                    t.contradictions,
                    t.avg_confidence,
                    t.success,
                    t.steps,
                )
                .composite,
                rationale: "selected from successful trajectory".into(),
                source_task_id: Some(t.task_id.clone()),
            })
            .collect();
        self.report_from_demos(trajectories.len(), &mut demos)
    }

    fn report_from_demos(&self, candidates_seen: usize, demos: &mut Vec<Demo>) -> BootstrapReport {
        self.rank_and_trim(demos);
        let avg_metric = if demos.is_empty() {
            0.0
        } else {
            demos.iter().map(|d| d.metric).sum::<f32>() / demos.len() as f32
        };
        BootstrapReport {
            demos: demos.clone(),
            candidates_seen,
            rejected: candidates_seen.saturating_sub(demos.len()),
            avg_metric,
        }
    }

    fn rank_and_trim(&self, demos: &mut Vec<Demo>) {
        demos.sort_by(|a, b| {
            b.metric
                .partial_cmp(&a.metric)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        demos.truncate(self.max_demos);
    }
}

/// 训练优化器 trait。
#[async_trait::async_trait]
pub trait TrainingOptimizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn demos_used(&self) -> usize {
        0
    }
    async fn optimize(&self, config: &mut TrainingConfig) -> Result<()>;
}

/// 可组合优化器流水线 — 多阶段串联。
pub struct OptimizerPipeline {
    stages: Vec<Box<dyn TrainingOptimizer>>,
}

impl OptimizerPipeline {
    pub fn new() -> Self {
        Self { stages: vec![] }
    }

    pub fn with(mut self, stage: Box<dyn TrainingOptimizer>) -> Self {
        self.stages.push(stage);
        self
    }

    pub async fn run(&self, config: &mut TrainingConfig) -> Result<()> {
        self.run_with_report(config).await.map(|_| ())
    }

    pub async fn run_with_report(&self, config: &mut TrainingConfig) -> Result<OptimizerRunReport> {
        let before = config.clone();
        let mut stages_run = Vec::new();
        let mut demos_used = 0usize;
        for stage in &self.stages {
            stage.optimize(config).await?;
            stages_run.push(stage.name());
            demos_used += stage.demos_used();
        }
        Ok(OptimizerRunReport {
            stages_run,
            demos_used,
            before,
            after: config.clone(),
        })
    }
}

impl Default for OptimizerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// 遗忘优先调度优化器 — 按半衰期紧急度调整 batch_size。
pub struct ForgettingPriorityOptimizer;

#[async_trait::async_trait]
impl TrainingOptimizer for ForgettingPriorityOptimizer {
    fn name(&self) -> &'static str {
        "forgetting-priority"
    }

    async fn optimize(&self, config: &mut TrainingConfig) -> Result<()> {
        if config.forgetting_aware {
            if config.min_confidence > 0.5 {
                config.gradient_batch_size = (config.gradient_batch_size as f32 * 1.5) as usize;
            } else {
                config.gradient_batch_size = config.gradient_batch_size.max(20);
            }
        }
        Ok(())
    }
}

/// Bootstrap demo 注入优化器 — 将成功 demo 注入到训练配置中。
pub struct BootstrapDemoOptimizer {
    pub demos: Vec<Demo>,
}

#[async_trait::async_trait]
impl TrainingOptimizer for BootstrapDemoOptimizer {
    fn name(&self) -> &'static str {
        "bootstrap-demo"
    }

    fn demos_used(&self) -> usize {
        self.demos.len()
    }

    async fn optimize(&self, config: &mut TrainingConfig) -> Result<()> {
        if self.demos.is_empty() {
            return Ok(());
        }
        let avg_metric: f32 =
            self.demos.iter().map(|d| d.metric).sum::<f32>() / self.demos.len() as f32;
        if avg_metric > 0.7 {
            config.min_confidence = (config.min_confidence * 0.8).max(0.15);
        }
        config.trials_per_epoch = ((config.trials_per_epoch as f32 * 1.2) as usize).max(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CognitiveDelta, PreferenceSource};

    fn trajectory(task_id: &str, success: bool, confidence: f32) -> TrajectorySummary {
        TrajectorySummary {
            task_id: task_id.into(),
            task_description: format!("task {task_id}"),
            success,
            steps: 3,
            contradictions: if success { 0 } else { 2 },
            avg_confidence: confidence,
        }
    }

    #[test]
    fn bootstrap_extracts_ranked_demos_from_preferences() {
        let pairs = vec![CognitivePreferencePair {
            chosen: trajectory("good", true, 0.9),
            rejected: trajectory("bad", false, 0.2),
            preference_source: PreferenceSource::TaskOutcome,
            confidence: 0.82,
            cognitive_delta: CognitiveDelta {
                contradiction_diff: 2,
                confidence_diff: 0.7,
                evidence_diff: 1,
                knowledge_graph_growth: 0,
            },
        }];

        let report = CognitiveBootstrap::new(BootstrapConfig::default())
            .unwrap()
            .extract_from_preferences(&pairs);
        assert_eq!(report.candidates_seen, 1);
        assert_eq!(report.demos.len(), 1);
        assert_eq!(report.demos[0].source_task_id.as_deref(), Some("good"));
        assert!(report.avg_metric > 0.8);
    }

    #[tokio::test]
    async fn optimizer_pipeline_reports_config_changes() {
        let mut config = TrainingConfig {
            min_confidence: 0.8,
            trials_per_epoch: 10,
            gradient_batch_size: 10,
            ..Default::default()
        };
        let demos = CognitiveBootstrap::new(BootstrapConfig::default())
            .unwrap()
            .extract_from_trajectories(&[trajectory("demo", true, 0.95)])
            .demos;
        let pipeline = OptimizerPipeline::new()
            .with(Box::new(BootstrapDemoOptimizer { demos }))
            .with(Box::new(ForgettingPriorityOptimizer));

        let report = pipeline.run_with_report(&mut config).await.unwrap();
        assert_eq!(report.stages_run.len(), 2);
        assert_eq!(report.demos_used, 1);
        assert!(report.after.trials_per_epoch > report.before.trials_per_epoch);
        assert!(report.after.gradient_batch_size > report.before.gradient_batch_size);
    }
}
