//! 语义梯度反馈 — 失败时生成可操作的改进建议（反思改进，P2）。
//!
//! 理论根基：反思的语义梯度 —— 不只是记录"失败了"，
//! 而是用 LLM 分析"为什么失败"、"应该怎么做"、"涉及哪些认识论类型"。

use agent_context_db_core::{ContentType, ContextUri, EpistemicType, LlmClient, LlmOpts, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 语义梯度 — 失败轨迹的可操作改进建议。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGradient {
    /// 失败关联的认识论类型。
    pub error_type: ContentType,
    /// 反思文本："我不应该在第 i 步做 a，应该做 a'，因为..."
    pub reflection_text: String,
    /// 可执行的改进建议。
    pub action_improvement: String,
    /// 认识论标签（fact / heuristic / procedure / ...）。
    pub epistemic_tags: Vec<String>,
    /// 失败轨迹的 URI（用于追溯）。
    pub source_uri: Option<ContextUri>,
    /// 改进优先级（0-1，越高越紧急）。
    pub priority: f32,
}

/// LLM 驱动的反思生成器。
pub struct ReflectionGenerator {
    llm: Arc<dyn LlmClient>,
}

/// LLM 返回的 JSON 解析结构。
#[derive(Debug, Deserialize)]
struct LlmReflectionResponse {
    why_failed: String,
    action_improvement: String,
    epistemic_tags: Vec<String>,
    priority: f32,
}

impl ReflectionGenerator {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// 分析失败轨迹，生成语义梯度。
    ///
    /// Prompt 策略：
    /// 1. 描述任务和失败上下文
    /// 2. 要求结构化输出（why / action / tags / priority）
    /// 3. LLM 不可用时回退到启发式规则
    pub async fn reflect(
        &self,
        task_description: &str,
        failed_step: &str,
        error_message: &str,
        relevant_knowledge: &[String],
        trace: &[String],
    ) -> SemanticGradient {
        let prompt = format!(
            r#"Analyze the following failed trajectory and generate an actionable improvement suggestion.

Task: {task_description}
Failed step: {failed_step}
Error: {error_message}
Relevant knowledge: {knowledge}
Execution trace:
{trace_formatted}

Output a JSON object with:
- "why_failed": one sentence explaining the root cause
- "action_improvement": specific, executable steps to avoid this failure
- "epistemic_tags": list of relevant epistemology types (fact, heuristic, procedure, hypothesis, belief)
- "priority": 0.0-1.0 urgency score"#,
            knowledge = relevant_knowledge.join(", "),
            trace_formatted = trace
                .iter()
                .enumerate()
                .map(|(i, s)| format!("  {}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let opts = LlmOpts {
            max_tokens: Some(512),
            temperature: Some(0.1),
            ..Default::default()
        };

        // 尝试 LLM 反思
        if let Ok(response) = self.llm.complete(&prompt, &opts).await {
            if let Ok(parsed) = serde_json::from_str::<LlmReflectionResponse>(&response) {
                return SemanticGradient {
                    error_type: ContentType::Error,
                    reflection_text: parsed.why_failed,
                    action_improvement: parsed.action_improvement,
                    epistemic_tags: parsed.epistemic_tags,
                    source_uri: None,
                    priority: parsed.priority.clamp(0.0, 1.0),
                };
            }
        }

        // 启发式回退
        Self::heuristic_reflect(task_description, failed_step, error_message, trace)
    }

    /// 启发式反思（无需 LLM）。
    fn heuristic_reflect(
        task_description: &str,
        _failed_step: &str,
        error_message: &str,
        trace: &[String],
    ) -> SemanticGradient {
        let step_count = trace.len();
        let reflection = format!(
            "Task '{}' failed after {} steps. Error: {}",
            task_description, step_count, error_message
        );

        let improvement = if error_message.contains("timeout") {
            "Add retry logic with exponential backoff".to_string()
        } else if error_message.contains("not found") {
            "Add pre-condition check before executing".to_string()
        } else if error_message.contains("permission") {
            "Verify access permissions before operation".to_string()
        } else {
            format!("Review step {} and add error handling", step_count)
        };

        SemanticGradient {
            error_type: ContentType::Error,
            reflection_text: reflection,
            action_improvement: improvement,
            epistemic_tags: vec!["heuristic".to_string(), "procedure".to_string()],
            source_uri: None,
            priority: 0.7,
        }
    }

    /// 批量反思 — 对多个失败轨迹并行生成改进建议。
    pub async fn reflect_batch(
        &self,
        failures: &[(String, String, String, Vec<String>, Vec<String>)],
    ) -> Vec<SemanticGradient> {
        let mut results = Vec::new();
        for (task, step, err, knowledge, trace) in failures {
            results.push(self.reflect(task, step, err, knowledge, trace).await);
        }
        results
    }

    /// 将反思结果编码为改进后的训练轨迹（用于下轮 CDT）。
    pub fn to_training_guidance(gradients: &[SemanticGradient]) -> Vec<String> {
        gradients
            .iter()
            .map(|g| {
                format!(
                    "REFLECTION: {}\nACTION: {}\nTAGS: [{}]",
                    g.reflection_text,
                    g.action_improvement,
                    g.epistemic_tags.join(", ")
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_produces_valid_gradient() {
        let g = ReflectionGenerator::heuristic_reflect(
            "deploy app",
            "kubectl apply",
            "connection timeout",
            &[
                "build docker image".to_string(),
                "kubectl apply".to_string(),
            ],
        );
        assert!(!g.reflection_text.is_empty());
        assert!(!g.action_improvement.is_empty());
        assert!(g.epistemic_tags.contains(&"heuristic".to_string()));
    }

    #[test]
    fn to_training_guidance_formats_correctly() {
        let g = SemanticGradient {
            error_type: ContentType::Error,
            reflection_text: "timeout on deploy".into(),
            action_improvement: "add retry".into(),
            epistemic_tags: vec!["procedure".into()],
            source_uri: None,
            priority: 0.8,
        };
        let guidance = ReflectionGenerator::to_training_guidance(&[g]);
        assert!(guidance[0].contains("REFLECTION"));
        assert!(guidance[0].contains("ACTION"));
    }
}
