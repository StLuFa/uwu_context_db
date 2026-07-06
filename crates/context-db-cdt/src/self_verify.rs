//! CognitiveSelfVerifier — Skill 执行后检查认知健康度（课程驱动）。
//!
//! 验证 Skill 执行是否引入知识图谱矛盾，认知置信度是否提升。

use agent_context_db_core::{ContextUri, GraphStore, Result};

/// 执行记录。
pub struct Execution {
    pub task_succeeded: bool,
    pub new_contradictions: usize,
    pub confidence_delta: f32,
}

/// 验证结果。
#[derive(Debug, Clone)]
pub enum VerificationResult {
    Passed,
    Failed(String),
    Partial(String),
}

/// 认知自检器 — Skill 执行后检查认知健康度。
pub struct CognitiveSelfVerifier {
    graph: Option<Box<dyn GraphStore>>,
    success_threshold: f32,
}

impl CognitiveSelfVerifier {
    pub fn new(success_threshold: f32) -> Self {
        Self { graph: None, success_threshold }
    }

    pub fn with_graph(mut self, graph: Box<dyn GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// 验证 Skill 执行后的认知健康度。
    ///
    /// 检查：
    /// 1. 任务完成度
    /// 2. 执行后是否新增知识图谱矛盾
    /// 3. 认知增益（认识论置信度是否提升）
    pub async fn verify(&self, _skill_uri: &ContextUri, execution: &Execution) -> VerificationResult {
        // 1. 任务完成度
        if !execution.task_succeeded {
            return VerificationResult::Failed("task not completed".into());
        }

        // 2. 认知自检：执行后是否有新增矛盾
        if execution.new_contradictions > 0 {
            if execution.new_contradictions > 1 {
                return VerificationResult::Failed(format!(
                    "task succeeded but {} new contradictions introduced",
                    execution.new_contradictions
                ));
            }
            return VerificationResult::Partial(format!(
                "task succeeded but {} new contradiction found — review recommended",
                execution.new_contradictions
            ));
        }

        // 3. 认知增益：置信度是否提升
        if execution.confidence_delta < 0.0 {
            return VerificationResult::Partial(format!(
                "task succeeded but confidence decreased by {:.3}",
                -execution.confidence_delta
            ));
        }

        if execution.confidence_delta > 0.1 {
            return VerificationResult::Passed;
        }

        // 微小提升 — 要求足够 trial 数据
        VerificationResult::Partial(
            "task succeeded with marginal confidence gain — more trials needed".into(),
        )
    }

    /// 批量验证（用于多个 Skill 的并行验证）。
    pub async fn verify_batch<'a>(
        &self,
        verifications: &[(&'a ContextUri, &Execution)],
    ) -> Vec<(&'a ContextUri, VerificationResult)> {
        let mut results = Vec::new();
        for (uri, exec) in verifications {
            results.push((*uri, self.verify(uri, exec).await));
        }
        results
    }
}
