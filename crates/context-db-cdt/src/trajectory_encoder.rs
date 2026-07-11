//! 轨迹 → 记忆编码器 — 将执行轨迹分类编码为 ContextEntry。
//!
//! CDT Pipeline 阶段 1：从 trajectory 中提取 fact/error/hypothesis/skill/procedure/reflection 六类记忆。

use agent_context_db_core::{
    ContentPayload, ContentType, ContextEntry, ContextMeta, ContextUri, EpistemicType, MediaType,
    TenantId,
};
use chrono::Utc;
use uuid::Uuid;

/// 轨迹摘要 — 被编码器消费的输入。
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub task_id: String,
    pub task_description: String,
    pub steps: Vec<String>,
    pub error_message: Option<String>,
    pub success: bool,
    pub relevant_knowledge: Vec<String>,
}

/// 轨迹编码结果。
#[derive(Debug, Clone)]
pub struct TrajectoryEncoding {
    pub facts: Vec<ContextEntry>,
    pub errors: Vec<ContextEntry>,
    pub hypotheses: Vec<ContextEntry>,
    pub skills: Vec<ContextEntry>,
    pub procedures: Vec<ContextEntry>,
    pub reflections: Vec<ContextEntry>,
}

/// 轨迹编码器 — 将原始 trajectory 编码为分类的 ContextEntry。
pub struct TrajectoryEncoder {
    agent_scope: String,
}

impl TrajectoryEncoder {
    pub fn new(agent_scope: impl Into<String>) -> Self {
        Self {
            agent_scope: agent_scope.into(),
        }
    }

    /// 编码一个轨迹为所有类型的记忆。
    pub fn encode(&self, traj: &Trajectory) -> TrajectoryEncoding {
        TrajectoryEncoding {
            facts: self.extract_facts(traj),
            errors: self.analyze_errors(traj),
            hypotheses: self.generate_hypotheses(traj),
            skills: self.discover_skills(traj),
            procedures: self.extract_procedures(traj),
            reflections: self.reflect(traj),
        }
    }

    /// 获取所有编码记忆的扁平列表。
    pub fn encode_all(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let encoding = self.encode(traj);
        let mut all = Vec::new();
        all.extend(encoding.facts);
        all.extend(encoding.errors);
        all.extend(encoding.hypotheses);
        all.extend(encoding.skills);
        all.extend(encoding.procedures);
        all.extend(encoding.reflections);
        all
    }

    /// 批量编码。
    pub fn encode_batch(&self, trajectories: &[Trajectory]) -> Vec<ContextEntry> {
        trajectories
            .iter()
            .flat_map(|t| self.encode_all(t))
            .collect()
    }

    // ── 内部提取方法 ──────────────────────────────

    /// 事实提取 → Fact 记忆。
    fn extract_facts(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let mut facts = Vec::new();

        // 从 relevant_knowledge 中提取陈述性事实
        for (i, knowledge) in traj.relevant_knowledge.iter().enumerate() {
            let uri = self.make_uri("fact", i, knowledge);
            facts.push(self.text_entry(uri, knowledge, ContentType::Fact, EpistemicType::Fact));
        }

        // 任务结果作为 fact
        let outcome = if traj.success { "succeeded" } else { "failed" };
        let outcome_fact = format!("Task `{}` {}", traj.task_description, outcome);
        let uri = self.make_uri("fact", traj.relevant_knowledge.len(), &outcome_fact);
        facts.push(self.text_entry(uri, &outcome_fact, ContentType::Fact, EpistemicType::Fact));

        facts
    }

    /// 错误分析 → Error 记忆。
    fn analyze_errors(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let mut errors = Vec::new();

        if let Some(ref err_msg) = traj.error_message {
            let content = format!("Task `{}` failed: {}", traj.task_description, err_msg);
            let uri = self.make_uri("error", 0, &content);
            errors.push(self.text_entry(uri, &content, ContentType::Error, EpistemicType::Fact));
        }

        if !traj.success {
            // 从 failed steps 中提取错误模式
            for (i, step) in traj.steps.iter().enumerate() {
                if step.contains("error") || step.contains("fail") || step.contains("panic") {
                    let content = format!("Step {} in `{}`: {}", i, traj.task_description, step);
                    let uri = self.make_uri("error", i + 1, &content);
                    errors.push(self.text_entry(
                        uri,
                        &content,
                        ContentType::Error,
                        EpistemicType::Fact,
                    ));
                }
            }
        }

        errors
    }

    /// 假设生成 → Hypothesis 记忆。
    fn generate_hypotheses(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let mut hyps = Vec::new();

        // 成功→假设"这个做法是正确的"；失败→假设"这个做法需要修正"
        let hyp = if traj.success {
            format!(
                "Hypothesis: approach used in `{}` is effective",
                traj.task_description
            )
        } else if let Some(ref err) = traj.error_message {
            format!(
                "Hypothesis: error `{}` in `{}` can be avoided by modifying approach",
                err, traj.task_description
            )
        } else {
            return hyps; // 无法生成假设
        };

        let uri = self.make_uri("hypothesis", 0, &hyp);
        hyps.push(self.text_entry(
            uri,
            &hyp,
            ContentType::Hypothesis,
            EpistemicType::Hypothesis,
        ));
        hyps
    }

    /// 技能发现 → Skill 记忆（已验证）/ Procedure 记忆（未验证）。
    fn discover_skills(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let mut skills = Vec::new();

        // 从步骤序列中提取 procedure
        if !traj.steps.is_empty() {
            let procedure = traj.steps.join(" → ");
            let content = format!("How to {}: {}", traj.task_description, procedure);
            let ct = if traj.success {
                ContentType::Skill // 成功→可能是已验证的 skill
            } else {
                ContentType::Procedure // 失败→只是 procedure
            };
            let uri = self.make_uri("skill", 0, &content);
            skills.push(self.text_entry(uri, &content, ct, EpistemicType::Procedure));
        }

        // 如果成功且有 relevant_knowledge，提取为 skill
        if traj.success {
            for (i, knowledge) in traj.relevant_knowledge.iter().enumerate() {
                let uri = self.make_uri("skill", i + 1, knowledge);
                skills.push(self.text_entry(
                    uri,
                    knowledge,
                    ContentType::Skill,
                    EpistemicType::Procedure,
                ));
            }
        }

        skills
    }

    /// 过程提取 → Procedure 记忆。
    fn extract_procedures(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let mut procs = Vec::new();

        // 多步骤 → Procedure
        if traj.steps.len() >= 2 {
            let steps_text = traj
                .steps
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n");
            let content = format!("Procedure for `{}`:\n{}", traj.task_description, steps_text);
            let uri = self.make_uri("procedure", 0, &content);
            procs.push(self.text_entry(
                uri,
                &content,
                ContentType::Procedure,
                EpistemicType::Procedure,
            ));
        }

        procs
    }

    /// 反思 → Reflection 记忆。
    fn reflect(&self, traj: &Trajectory) -> Vec<ContextEntry> {
        let mut reflections = Vec::new();

        let insight = if traj.success {
            format!(
                "Reflection: `{}` succeeded with {} steps. Strategy can be reused.",
                traj.task_description,
                traj.steps.len()
            )
        } else {
            format!(
                "Reflection: `{}` failed after {} steps. Error: {}. Should modify approach next time.",
                traj.task_description,
                traj.steps.len(),
                traj.error_message.as_deref().unwrap_or("unknown")
            )
        };

        let uri = self.make_uri("reflection", 0, &insight);
        reflections.push(self.text_entry(
            uri,
            &insight,
            ContentType::Reflection,
            EpistemicType::Heuristic,
        ));

        reflections
    }

    // ── 辅助 ──────────────────────────────────────

    fn make_uri(&self, ctype: &str, index: usize, content: &str) -> ContextUri {
        let hash = blake3::hash(content.as_bytes()).to_hex();
        let short = &hash[..8];
        ContextUri::parse(format!(
            "uwu://{}/memory/{}/{}/{:02}-{}",
            self.agent_scope,
            ctype,
            Utc::now().format("%Y%m%d"),
            index,
            short
        ))
        .unwrap_or_else(|_| {
            ContextUri::parse(format!(
                "uwu://{}/memory/{}/fallback-{}",
                self.agent_scope,
                ctype,
                Uuid::new_v4()
            ))
            .unwrap()
        })
    }

    fn text_entry(
        &self,
        uri: ContextUri,
        text: &str,
        ct: ContentType,
        et: EpistemicType,
    ) -> ContextEntry {
        let now = Utc::now();
        ContextEntry {
            uri,
            tenant: TenantId(Uuid::new_v4()),
            payload: ContentPayload::Text {
                sparse: text.to_string(),
                dense: text.to_string(),
                full: String::new(),
            },
            media_type: MediaType::Text,
            metadata: ContextMeta {
                content_type: Some(ct),
                epistemic_type: Some(et),
                ..Default::default()
            },
            mvcc_version: agent_context_db_core::MvccVersion(0),
            created_at: now,
            updated_at: now,
            derivation: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(success: bool) -> Trajectory {
        Trajectory {
            task_id: "task-1".into(),
            task_description: "deploy application to staging".into(),
            steps: vec![
                "build docker image".into(),
                "push to registry".into(),
                "kubectl apply".into(),
            ],
            error_message: if success {
                None
            } else {
                Some("connection timeout to registry".into())
            },
            success,
            relevant_knowledge: vec![
                "Docker images must be tagged with registry prefix".into(),
                "kubectl requires valid kubeconfig".into(),
            ],
        }
    }

    #[test]
    fn success_trajectory_produces_skills() {
        let encoder = TrajectoryEncoder::new("t/agent-a");
        let traj = make_trajectory(true);
        let encoding = encoder.encode(&traj);

        assert!(!encoding.facts.is_empty());
        assert!(!encoding.skills.is_empty());
        assert!(encoding.errors.is_empty()); // 成功 = 无错误
        assert_eq!(
            encoding.facts[0].metadata.content_type,
            Some(ContentType::Fact)
        );
    }

    #[test]
    fn failure_trajectory_produces_errors() {
        let encoder = TrajectoryEncoder::new("t/agent-a");
        let traj = make_trajectory(false);
        let encoding = encoder.encode(&traj);

        assert!(!encoding.errors.is_empty());
        assert!(
            encoding
                .hypotheses
                .iter()
                .any(|h| { h.payload.sparse_text().contains("Hypothesis") })
        );
    }

    #[test]
    fn encode_all_returns_flat_list() {
        let encoder = TrajectoryEncoder::new("t/agent-a");
        let traj = make_trajectory(true);
        let all = encoder.encode_all(&traj);
        assert!(all.len() >= 5); // facts + skills + procedures + hypotheses + reflections
    }
}
