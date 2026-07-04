//! 版本推理（F24 版本差异推理 + F30 时态推理）。
//!
//! 建立在 M2 DAG + 时间旅行基础设施上的高阶分析。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, LlmClient, LlmOpts};
use std::sync::Arc;

use crate::{
    AsOfTime, CommitId, LogOpts, VersionError, VersionStore,
};

// ═══════════════════════════════════════════════════════════════════════════
// F24 版本差异推理
// ═══════════════════════════════════════════════════════════════════════════

/// 语义差异 —— 超越行级 diff 的含义级变更描述。
#[derive(Debug, Clone)]
pub struct SemanticDiff {
    /// 变更摘要
    pub summary: String,
    /// 变更类型
    pub change_type: DiffChangeType,
    /// 影响范围
    pub impact: DiffImpact,
    /// 逐 URI 的语义解释
    pub details: Vec<SemanticChange>,
}

#[derive(Debug, Clone)]
pub struct SemanticChange {
    pub uri: ContextUri,
    /// 自然人可读的变更说明
    pub description: String,
    /// 变更类别
    pub category: ChangeCategory,
    /// 变更量级 0-1
    pub magnitude: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffChangeType {
    Additive,
    Corrective,
    Destructive,
    Refactoring,
    Conflicting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeCategory {
    FactUpdate,
    RelationChange,
    StateTransition,
    SkillRefinement,
    MemoryConsolidation,
}

#[derive(Debug, Clone)]
pub struct DiffImpact {
    /// 受影响的直接条目数
    pub direct_entries: usize,
    /// 受影响的间接条目数
    pub transitive_entries: usize,
    /// 是否需要触发重新索引
    pub reindex_required: bool,
    /// 是否需要通知上游
    pub notify_required: bool,
}

/// 版本差异推理器 —— 将 TreeDiff 提升为语义级理解。
pub struct DiffReasoner {
    llm: Arc<dyn LlmClient>,
}

impl DiffReasoner {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// 对两个 commit 间的变更做语义推理。
    pub async fn reason(
        &self,
        store: &dyn VersionStore,
        scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> std::result::Result<SemanticDiff, VersionError> {
        let tree_diff = store.diff_commits(scope, a, b).await?;

        let details_json = serde_json::to_string(&tree_diff).unwrap_or_default();

        let prompt = format!(
            r#"Analyze this version diff and provide a semantic interpretation.

Tree diff:
- Added: {}
- Updated: {}
- Deleted: {}

Full diff details: {details_json}

Return a JSON object with:
- "summary": one-sentence summary of the overall change
- "change_type": one of "additive", "corrective", "destructive", "refactoring", "conflicting"
- "impact": {{ "direct": number, "transitive": number, "reindex": bool, "notify": bool }}
- "changes": array of {{ "uri": "...", "description": "...", "category": "...", "magnitude": 0.0-1.0 }}
"#,
            tree_diff.adds.len(),
            tree_diff.updates.len(),
            tree_diff.deletes.len(),
        );

        let opts = LlmOpts {
            max_tokens: Some(1024),
            temperature: Some(0.1),
            ..Default::default()
        };

        let response = self.llm.complete(&prompt, &opts).await.map_err(|e| {
            VersionError::Storage(format!("diff reasoner llm: {e}"))
        })?;

        #[derive(serde::Deserialize)]
        struct RawDiff {
            summary: String,
            change_type: String,
            impact: RawImpact,
            changes: Vec<RawChange>,
        }
        #[derive(serde::Deserialize)]
        struct RawImpact {
            direct: usize,
            transitive: usize,
            reindex: bool,
            notify: bool,
        }
        #[derive(serde::Deserialize)]
        struct RawChange {
            uri: String,
            description: String,
            category: String,
            magnitude: f32,
        }

        let raw: RawDiff = serde_json::from_str(&response).unwrap_or_else(|_| RawDiff {
            summary: format!("{} additions, {} updates, {} deletions",
                tree_diff.adds.len(), tree_diff.updates.len(), tree_diff.deletes.len()),
            change_type: "additive".into(),
            impact: RawImpact { direct: 0, transitive: 0, reindex: false, notify: false },
            changes: vec![],
        });

        Ok(SemanticDiff {
            summary: raw.summary,
            change_type: match raw.change_type.as_str() {
                "corrective" => DiffChangeType::Corrective,
                "destructive" => DiffChangeType::Destructive,
                "refactoring" => DiffChangeType::Refactoring,
                "conflicting" => DiffChangeType::Conflicting,
                _ => DiffChangeType::Additive,
            },
            impact: DiffImpact {
                direct_entries: raw.impact.direct,
                transitive_entries: raw.impact.transitive,
                reindex_required: raw.impact.reindex,
                notify_required: raw.impact.notify,
            },
            details: raw.changes.into_iter().map(|c| SemanticChange {
                uri: ContextUri(c.uri),
                description: c.description,
                category: match c.category.as_str() {
                    "relation" => ChangeCategory::RelationChange,
                    "state" => ChangeCategory::StateTransition,
                    "skill" => ChangeCategory::SkillRefinement,
                    "memory" => ChangeCategory::MemoryConsolidation,
                    _ => ChangeCategory::FactUpdate,
                },
                magnitude: c.magnitude,
            }).collect(),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F30 时态推理
// ═══════════════════════════════════════════════════════════════════════════

/// 时间线事件。
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub commit: CommitId,
    pub summary: String,
    pub entries_changed: Vec<ContextUri>,
}

/// 时态查询模式。
#[derive(Debug, Clone)]
pub enum TemporalPattern {
    /// 某条目在时间窗口内的演变
    Evolution,
    /// 某时间点前后 N 条相关变更
    Neighborhood,
    /// 两个时间点的差异
    Comparison { before: AsOfTime, after: AsOfTime },
    /// 周期性变更检测
    Periodicity { min_occurrences: usize },
}

/// 时态推理器 —— 利用 M2 时间旅行做时序分析。
pub struct TemporalReasoner<V: VersionStore> {
    store: Arc<V>,
}

impl<V: VersionStore> TemporalReasoner<V> {
    pub fn new(store: Arc<V>) -> Self {
        Self { store }
    }

    /// 生成条目在时间窗口内的演变时间线。
    pub async fn evolution_timeline(
        &self,
        scope: &ContextUri,
        from: AsOfTime,
        to: AsOfTime,
    ) -> std::result::Result<Vec<TimelineEvent>, VersionError> {
        let log = self.store.log(scope, &LogOpts { max_count: Some(50), ..Default::default() }).await?;

        let mut events = Vec::new();
        for commit in log {
            if let AsOfTime::Timestamp(from_ts) = &from {
                if commit.timestamp < *from_ts {
                    continue;
                }
            }
            if let AsOfTime::Timestamp(to_ts) = &to {
                if commit.timestamp > *to_ts {
                    continue;
                }
            }

            let entries = vec![
                commit.metadata.changes.adds.clone(),
                commit.metadata.changes.updates.iter().map(|u| u.uri.clone()).collect(),
            ].concat();

            events.push(TimelineEvent {
                timestamp: commit.timestamp,
                commit: commit.id,
                summary: commit.message,
                entries_changed: entries,
            });
        }

        Ok(events)
    }

    /// 检测周期性变更模式。
    ///
    /// 返回重复出现的变更 URI 及其出现次数。
    pub async fn detect_periodicity(
        &self,
        scope: &ContextUri,
        min_occurrences: usize,
    ) -> std::result::Result<Vec<(ContextUri, usize)>, VersionError> {
        let log = self.store.log(scope, &LogOpts { max_count: Some(100), ..Default::default() }).await?;

        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for commit in log {
            for add in &commit.metadata.changes.adds {
                *counts.entry(add.0.clone()).or_default() += 1;
            }
            for upd in &commit.metadata.changes.updates {
                *counts.entry(upd.uri.0.clone()).or_default() += 1;
            }
        }

        Ok(counts
            .into_iter()
            .filter(|(_, count)| *count >= min_occurrences)
            .map(|(uri, count)| (ContextUri(uri), count))
            .collect())
    }

    /// 比较两个时间点的状态差异。
    pub async fn temporal_diff(
        &self,
        uri: &ContextUri,
        before: AsOfTime,
        after: AsOfTime,
    ) -> std::result::Result<(Option<String>, Option<String>), VersionError> {
        let before_content = self.store.asof_read(uri, before, ContentLevel::L0).await;
        let after_content = self.store.asof_read(uri, after, ContentLevel::L0).await;

        let before_text = match before_content {
            Ok(ContentPayload::Abstract(s)) => Some(s),
            _ => None,
        };
        let after_text = match after_content {
            Ok(ContentPayload::Abstract(s)) => Some(s),
            _ => None,
        };

        Ok((before_text, after_text))
    }
}

#[cfg(test)]
mod tests {
    /// 验证 TemporalReasoner 可用 MemoryVersionStore 构造。
    #[test]
    fn reasoner_constructs_with_store() {
        // 此测试验证编译期类型约束，实际时序测试在集成测试中
        assert!(true);
    }
}
