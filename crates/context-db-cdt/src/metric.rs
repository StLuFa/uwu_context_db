//! 可组合优化 — Bootstrap demo 生成 + 可组合 OptimizerPipeline。

use crate::{CognitiveMetric, TrainingConfig};
use agent_context_db_core::Result;

/// Bootstrap demo — 从成功执行中自动提取 few-shot demo。
pub struct CognitiveBootstrap {
    pub metric_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct Demo {
    pub input: String,
    pub output: Vec<String>,
    pub metric: f32,
}

impl CognitiveBootstrap {
    pub fn new(threshold: f32) -> Self {
        Self {
            metric_threshold: threshold,
        }
    }

    pub fn extract_demos(&self, metrics: &[(String, CognitiveMetric)]) -> Vec<Demo> {
        let mut demos: Vec<_> = metrics
            .iter()
            .filter(|(_, m)| m.composite >= self.metric_threshold)
            .map(|(input, m)| Demo {
                input: input.clone(),
                output: vec![],
                metric: m.composite,
            })
            .collect();
        demos.sort_by(|a, b| {
            b.metric
                .partial_cmp(&a.metric)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        demos.truncate(5); // top-5
        demos
    }
}

/// 训练优化器 trait。
#[async_trait::async_trait]
pub trait TrainingOptimizer: Send + Sync {
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
        for stage in &self.stages {
            stage.optimize(config).await?;
        }
        Ok(())
    }
}

/// 遗忘优先调度优化器 — 按半衰期紧急度调整 batch_size。
pub struct ForgettingPriorityOptimizer;

#[async_trait::async_trait]
impl TrainingOptimizer for ForgettingPriorityOptimizer {
    async fn optimize(&self, config: &mut TrainingConfig) -> Result<()> {
        // 遗忘感知：batch_size 与当前 min_confidence 负相关
        // min_confidence 越高 → 需要更多筛选 → batch_size 调大
        if config.min_confidence > 0.5 {
            config.gradient_batch_size = (config.gradient_batch_size as f32 * 1.5) as usize;
        } else {
            config.gradient_batch_size = config.gradient_batch_size.max(20);
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
    async fn optimize(&self, config: &mut TrainingConfig) -> Result<()> {
        if self.demos.is_empty() {
            return Ok(());
        }
        // 有优质 demo → 放宽最低置信度，让更多梯度通过
        let avg_metric: f32 =
            self.demos.iter().map(|d| d.metric).sum::<f32>() / self.demos.len() as f32;
        if avg_metric > 0.7 {
            config.min_confidence = (config.min_confidence * 0.8).max(0.15);
        }
        // 增加 trials 以利用 demo 引导
        config.trials_per_epoch = (config.trials_per_epoch as f32 * 1.2) as usize;
        Ok(())
    }
}
