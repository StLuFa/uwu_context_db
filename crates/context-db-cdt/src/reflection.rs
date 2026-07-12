//! 语义梯度反馈 — 失败时生成可操作的改进建议（反思改进，P2）。
//!
//! 理论根基：反思的语义梯度 —— 不只是记录"失败了"，
//! 而是用 LLM 分析"为什么失败"、"应该怎么做"、"涉及哪些认识论类型"。

use crate::config::ReflectionConfig;
use crate::voting::{EvolutionReport, EvolvableInsight, InsightEvolutionEngine};
use agent_context_db_core::{
    ConsolidationMeta, ConsolidationStatus, ContentPayload, ContentType, ContextEntry, ContextUri,
    EpistemicType, FindPattern, LlmClient, LlmOpts, MvccVersion, StateScope, TenantId,
};
use chrono::Utc;
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

/// 失败轨迹样本，用于 Reflexion loop。
#[derive(Debug, Clone)]
pub struct FailureTrace {
    pub task_description: String,
    pub failed_step: String,
    pub error_message: String,
    pub relevant_knowledge: Vec<String>,
    pub trace: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReflectionFailure {
    pub task: String,
    pub step: String,
    pub error: String,
    pub knowledge: Vec<String>,
    pub trace: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReflectionWritebackConfig {
    pub agent_scope: String,
    pub tenant: TenantId,
    pub min_priority: f32,
}

impl ReflectionWritebackConfig {
    pub fn for_agent(agent_scope: impl Into<String>, tenant: TenantId) -> Self {
        Self {
            agent_scope: agent_scope.into(),
            tenant,
            min_priority: 0.35,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReflectionRecallHint {
    pub pattern: FindPattern,
    pub query_text: String,
    pub min_priority: f32,
}

/// Reflexion + ExpeL 的一次闭环结果。
#[derive(Debug, Clone)]
pub struct ReflexionEvolutionResult {
    pub gradients: Vec<SemanticGradient>,
    pub insights: Vec<EvolvableInsight>,
    pub report: EvolutionReport,
    pub training_guidance: Vec<String>,
}

/// LLM 驱动的反思生成器。
pub struct ReflectionGenerator {
    llm: Arc<dyn LlmClient>,
    config: ReflectionConfig,
}

/// LLM 返回的 JSON 解析结构。
#[derive(Debug, Deserialize)]
struct LlmReflectionResponse {
    why_failed: String,
    action_improvement: String,
    epistemic_tags: Vec<String>,
    priority: f32,
}

impl SemanticGradient {
    pub fn to_memory_entries(
        &self,
        config: &ReflectionWritebackConfig,
        index: usize,
    ) -> agent_context_db_core::Result<Vec<ContextEntry>> {
        if self.priority < config.min_priority {
            return Ok(Vec::new());
        }
        let hash = blake3::hash(
            format!(
                "{}:{}:{}",
                self.reflection_text, self.action_improvement, index
            )
            .as_bytes(),
        );
        let short = hash.to_hex().chars().take(8).collect::<String>();
        let error_uri = ContextUri::parse(format!(
            "uwu://{}/memory/error/reflection/{:02}-{}",
            config.agent_scope, index, short
        ))?;
        let heuristic_uri = ContextUri::parse(format!(
            "uwu://{}/memory/heuristic/reflection/{:02}-{}",
            config.agent_scope, index, short
        ))?;

        let mut error_entry = ContextEntry::new_text(
            error_uri,
            config.tenant,
            format!(
                "FAILURE: {}\nACTION: {}",
                self.reflection_text, self.action_improvement
            ),
        );
        error_entry.metadata.content_type = Some(ContentType::Error);
        error_entry.metadata.epistemic_type = Some(EpistemicType::Heuristic);
        error_entry.metadata.quality_score = Some(self.priority.clamp(0.0, 1.0));
        error_entry.metadata.state_scope = Some(StateScope::Long);
        error_entry.metadata.tags = self.prefixed_tags("reflection:error");
        error_entry.metadata.consolidation = Some(reflection_meta(self, self.source_uri.clone()));

        let mut heuristic_entry = ContextEntry::new_text(
            heuristic_uri,
            config.tenant,
            format!("WHEN SIMILAR FAILURE: {}", self.action_improvement),
        );
        heuristic_entry.payload = ContentPayload::Text {
            sparse: format!("Avoid repeated failure: {}", self.action_improvement),
            dense: format!(
                "Reflection lesson\nFailure: {}\nImprovement: {}\nTags: {}",
                self.reflection_text,
                self.action_improvement,
                self.epistemic_tags.join(", ")
            ),
            full: format!(
                "Reflection lesson\nFailure: {}\nImprovement: {}\nSource: {:?}\nTags: {}",
                self.reflection_text,
                self.action_improvement,
                self.source_uri,
                self.epistemic_tags.join(", ")
            ),
        };
        heuristic_entry.metadata.content_type = Some(ContentType::Heuristic);
        heuristic_entry.metadata.epistemic_type = Some(EpistemicType::Heuristic);
        heuristic_entry.metadata.quality_score = Some(self.priority.clamp(0.0, 1.0));
        heuristic_entry.metadata.state_scope = Some(StateScope::Long);
        heuristic_entry.metadata.tags = self.prefixed_tags("reflection:heuristic");
        heuristic_entry.metadata.consolidation =
            Some(reflection_meta(self, Some(error_entry.uri.clone())));

        Ok(vec![error_entry, heuristic_entry])
    }

    pub fn recall_hint(
        &self,
        agent_scope: &str,
    ) -> agent_context_db_core::Result<ReflectionRecallHint> {
        let scope = ContextUri::parse(format!("uwu://{agent_scope}/memory/error/reflection"))?;
        Ok(ReflectionRecallHint {
            pattern: FindPattern {
                scope: Some(scope),
                name_glob: Some(format!(
                    "*{}*",
                    recall_token(&self.reflection_text, &self.action_improvement)
                )),
                content_type: Some(ContentType::Error),
                max_depth: Some(3),
            },
            query_text: format!("{} {}", self.reflection_text, self.action_improvement),
            min_priority: self.priority,
        })
    }

    fn prefixed_tags(&self, prefix: &str) -> Vec<String> {
        let mut tags = vec![prefix.to_string()];
        tags.extend(
            self.epistemic_tags
                .iter()
                .map(|tag| format!("epistemic:{tag}")),
        );
        tags.sort();
        tags.dedup();
        tags
    }
}

fn reflection_meta(
    gradient: &SemanticGradient,
    evidence_uri: Option<ContextUri>,
) -> ConsolidationMeta {
    let corroboration = usize::from(evidence_uri.is_some());
    ConsolidationMeta {
        source: "reflection-writeback".to_string(),
        generation: 1,
        status: ConsolidationStatus::Pending,
        patch_count: 0,
        lineage: vec![agent_context_db_core::LineageEntry {
            version: MvccVersion(0),
            timestamp: Utc::now(),
            change_summary: format!("semantic gradient priority {:.2}", gradient.priority),
        }],
        evidence_uris: evidence_uri.into_iter().collect(),
        corroboration,
        half_life: Some(agent_context_db_core::HalfLife::Finite { days: 120.0 }),
        entangled_with: gradient.source_uri.iter().cloned().collect(),
    }
}

fn recall_token(reflection: &str, action: &str) -> String {
    reflection
        .split_whitespace()
        .chain(action.split_whitespace())
        .find(|token| token.chars().filter(|c| c.is_alphanumeric()).count() >= 5)
        .unwrap_or("reflection")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect::<String>()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    Timeout,
    MissingPrecondition,
    Permission,
    Validation,
    ResourceExhaustion,
    Dependency,
    Unknown,
}

impl FailureClass {
    fn classify(failed_step: &str, error_message: &str, trace: &[String]) -> Self {
        let haystack = std::iter::once(failed_step)
            .chain(std::iter::once(error_message))
            .chain(trace.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join("\n")
            .to_ascii_lowercase();
        if contains_any(&haystack, &["timeout", "timed out", "deadline", "elapsed"]) {
            Self::Timeout
        } else if contains_any(
            &haystack,
            &["not found", "missing", "no such", "404", "absent"],
        ) {
            Self::MissingPrecondition
        } else if contains_any(
            &haystack,
            &[
                "permission",
                "forbidden",
                "unauthorized",
                "denied",
                "401",
                "403",
            ],
        ) {
            Self::Permission
        } else if contains_any(
            &haystack,
            &["invalid", "schema", "parse", "validation", "deserialize"],
        ) {
            Self::Validation
        } else if contains_any(
            &haystack,
            &[
                "quota",
                "rate limit",
                "oom",
                "memory",
                "disk full",
                "capacity",
            ],
        ) {
            Self::ResourceExhaustion
        } else if contains_any(
            &haystack,
            &[
                "connection",
                "unavailable",
                "dns",
                "refused",
                "upstream",
                "dependency",
            ],
        ) {
            Self::Dependency
        } else {
            Self::Unknown
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::MissingPrecondition => "missing-precondition",
            Self::Permission => "permission",
            Self::Validation => "validation",
            Self::ResourceExhaustion => "resource-exhaustion",
            Self::Dependency => "dependency",
            Self::Unknown => "unknown",
        }
    }

    fn action(self, failed_step: &str, knowledge_signal: usize) -> String {
        let evidence_clause = if knowledge_signal > 0 {
            "reuse the retrieved knowledge before retrying"
        } else {
            "capture the missing evidence before retrying"
        };
        match self {
            Self::Timeout => format!(
                "Wrap '{failed_step}' with bounded retries, exponential backoff, and an idempotency check; {evidence_clause}."
            ),
            Self::MissingPrecondition => format!(
                "Before '{failed_step}', verify required resources and create or fetch missing prerequisites; {evidence_clause}."
            ),
            Self::Permission => format!(
                "Before '{failed_step}', validate credentials, tenant scope, and capability grants; fail closed when access is ambiguous."
            ),
            Self::Validation => format!(
                "Add schema validation around '{failed_step}', preserve the rejected payload, and repair only fields with explicit evidence."
            ),
            Self::ResourceExhaustion => format!(
                "Throttle '{failed_step}', reduce batch size, and record resource pressure before rescheduling the task."
            ),
            Self::Dependency => format!(
                "Probe upstream dependency health before '{failed_step}', use a bounded fallback path, and record dependency evidence."
            ),
            Self::Unknown => format!(
                "Instrument '{failed_step}' with structured error context and postpone writeback until a repeatable failure class is observed."
            ),
        }
    }

    fn tags(self) -> Vec<String> {
        let mut tags = vec!["procedure".to_string(), format!("failure:{}", self.label())];
        if matches!(self, Self::Unknown) {
            tags.push("hypothesis".to_string());
        } else {
            tags.push("heuristic".to_string());
        }
        tags
    }

    fn priority(self, step_count: usize, knowledge_signal: usize) -> f32 {
        let base: f32 = match self {
            Self::Permission | Self::Validation => 0.72,
            Self::Dependency | Self::ResourceExhaustion => 0.66,
            Self::Timeout | Self::MissingPrecondition => 0.60,
            Self::Unknown => 0.32,
        };
        let trace_weight = (step_count as f32 / 20.0).min(0.12);
        let knowledge_weight = (knowledge_signal as f32 * 0.03).min(0.09);
        (base + trace_weight + knowledge_weight).clamp(0.0, 0.88)
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

impl ReflectionGenerator {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        config: ReflectionConfig,
    ) -> agent_context_db_core::Result<Self> {
        config.validate()?;
        Ok(Self { llm, config })
    }

    /// 分析失败轨迹，生成语义梯度。
    ///
    /// Prompt 策略：
    /// 1. 描述任务和失败上下文
    /// 2. 要求结构化输出（why / action / tags / priority）
    /// 3. LLM 不可用时使用本地失败分类器生成低污染度反思
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
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            ..Default::default()
        };

        // 尝试 LLM 反思
        if let Ok(response) = self.llm.complete(&prompt, &opts).await
            && let Ok(parsed) = serde_json::from_str::<LlmReflectionResponse>(&response)
        {
            return SemanticGradient {
                error_type: ContentType::Error,
                reflection_text: parsed.why_failed,
                action_improvement: parsed.action_improvement,
                epistemic_tags: parsed.epistemic_tags,
                source_uri: None,
                priority: parsed.priority.clamp(0.0, 1.0),
            };
        }

        Self::local_reflection(
            task_description,
            failed_step,
            error_message,
            relevant_knowledge,
            trace,
        )
    }

    fn local_reflection(
        task_description: &str,
        failed_step: &str,
        error_message: &str,
        relevant_knowledge: &[String],
        trace: &[String],
    ) -> SemanticGradient {
        let class = FailureClass::classify(failed_step, error_message, trace);
        let step_count = trace.len();
        let last_step = trace.last().map(String::as_str).unwrap_or(failed_step);
        let knowledge_signal = relevant_knowledge
            .iter()
            .filter(|item| !item.trim().is_empty())
            .count();
        let priority = class.priority(step_count, knowledge_signal);
        let reflection = format!(
            "Task '{}' failed at '{}' after {} steps; classified as {} from error '{}' and last trace '{}'.",
            task_description,
            failed_step,
            step_count,
            class.label(),
            error_message,
            last_step
        );

        SemanticGradient {
            error_type: ContentType::Error,
            reflection_text: reflection,
            action_improvement: class.action(failed_step, knowledge_signal),
            epistemic_tags: class.tags(),
            source_uri: None,
            priority,
        }
    }

    /// 批量反思 — 对多个失败轨迹并行生成改进建议。
    pub async fn reflect_batch(&self, failures: &[ReflectionFailure]) -> Vec<SemanticGradient> {
        let traces: Vec<FailureTrace> = failures
            .iter()
            .map(|failure| FailureTrace {
                task_description: failure.task.clone(),
                failed_step: failure.step.clone(),
                error_message: failure.error.clone(),
                relevant_knowledge: failure.knowledge.clone(),
                trace: failure.trace.clone(),
            })
            .collect();
        self.reflect_failures(&traces).await
    }

    /// Reflexion：失败轨迹 → 语义梯度。
    pub async fn reflect_failures(&self, failures: &[FailureTrace]) -> Vec<SemanticGradient> {
        let mut results = Vec::new();
        for failure in failures {
            results.push(
                self.reflect(
                    &failure.task_description,
                    &failure.failed_step,
                    &failure.error_message,
                    &failure.relevant_knowledge,
                    &failure.trace,
                )
                .await,
            );
        }
        results
    }

    /// Reflexion + ExpeL：失败轨迹 → 语义梯度 → 可投票演化 insight。
    pub async fn evolve_failures(
        &self,
        failures: &[FailureTrace],
        existing: &mut Vec<EvolvableInsight>,
        evolution: &InsightEvolutionEngine,
    ) -> agent_context_db_core::Result<ReflexionEvolutionResult> {
        let gradients = self.reflect_failures(failures).await;
        let report = evolution.evolve_from_gradients(existing, &gradients)?;
        let training_guidance = Self::to_training_guidance(&gradients);
        Ok(ReflexionEvolutionResult {
            gradients,
            insights: existing.clone(),
            report,
            training_guidance,
        })
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
    use agent_context_db_core::{JsonSchema, LlmError};
    use async_trait::async_trait;

    struct FailingLlm;

    #[async_trait]
    impl LlmClient for FailingLlm {
        async fn complete(
            &self,
            _prompt: &str,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Err(LlmError::Provider("fail".into()))
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Err(LlmError::Provider("fail".into()))
        }

        async fn embed(
            &self,
            _text: &str,
        ) -> std::result::Result<agent_context_db_core::EmbeddingVector, LlmError> {
            Ok(agent_context_db_core::EmbeddingVector::new(
                vec![0.0; 8],
                "test",
                1,
            ))
        }
    }

    #[test]
    fn local_reflection_produces_valid_gradient() {
        let g = ReflectionGenerator::local_reflection(
            "deploy app",
            "kubectl apply",
            "connection timeout",
            &["retry policy".to_string()],
            &[
                "build docker image".to_string(),
                "kubectl apply".to_string(),
            ],
        );
        assert!(g.reflection_text.contains("timeout"));
        assert!(g.action_improvement.contains("bounded retries"));
        assert!(g.epistemic_tags.contains(&"failure:timeout".to_string()));
        assert!(g.priority > 0.6);
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

    #[tokio::test]
    async fn evolve_failures_creates_insights_and_guidance() {
        let generator =
            ReflectionGenerator::new(Arc::new(FailingLlm), ReflectionConfig::default()).unwrap();
        let failures = vec![FailureTrace {
            task_description: "deploy app".into(),
            failed_step: "kubectl apply".into(),
            error_message: "connection timeout".into(),
            relevant_knowledge: vec!["k8s deploy".into()],
            trace: vec!["build".into(), "apply".into()],
        }];
        let mut insights = Vec::new();
        let evolution =
            InsightEvolutionEngine::new(crate::config::VotingConfig::default()).unwrap();
        let result = generator
            .evolve_failures(&failures, &mut insights, &evolution)
            .await
            .unwrap();
        assert_eq!(result.gradients.len(), 1);
        assert_eq!(result.report.added, 1);
        assert_eq!(result.insights.len(), 1);
        assert_eq!(result.training_guidance.len(), 1);
    }
}
