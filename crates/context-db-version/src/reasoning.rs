//! 版本推理（F24 版本差异推理 + F30 时态推理）。
//!
//! 建立在 M2 DAG + 时间旅行基础设施上的高阶分析。

use agent_context_db_core::{ContentLevel, ContextUri, LlmClient, LlmOpts};
use std::sync::Arc;

use crate::{AsOfTime, CommitId, LogOpts, VersionError, VersionStore};

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

#[derive(Debug, Clone)]
pub struct ReasoningConfig {
    pub diff_max_tokens: usize,
    pub diff_temperature: f32,
    pub timeline_log_limit: usize,
    pub periodicity_log_limit: usize,
}

impl ReasoningConfig {
    pub fn validate(&self) -> crate::Result<()> {
        if self.diff_max_tokens == 0
            || self.diff_max_tokens > u32::MAX as usize
            || self.timeline_log_limit == 0
            || self.periodicity_log_limit == 0
        {
            return Err(VersionError::InvalidConfig(
                "reasoning token and log limits must be greater than zero".into(),
            ));
        }
        if !self.diff_temperature.is_finite() || !(0.0..=1.0).contains(&self.diff_temperature) {
            return Err(VersionError::InvalidConfig(
                "diff_temperature must be finite and in 0..=1".into(),
            ));
        }
        Ok(())
    }
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            diff_max_tokens: 1024,
            diff_temperature: 0.1,
            timeline_log_limit: 50,
            periodicity_log_limit: 100,
        }
    }
}

/// 版本差异推理器 —— 将 TreeDiff 提升为语义级理解。
pub struct DiffReasoner {
    llm: Arc<dyn LlmClient>,
    config: ReasoningConfig,
}

impl DiffReasoner {
    pub fn new(llm: Arc<dyn LlmClient>, config: ReasoningConfig) -> crate::Result<Self> {
        config.validate()?;
        Ok(Self { llm, config })
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
            max_tokens: Some(self.config.diff_max_tokens as u32),
            temperature: Some(self.config.diff_temperature),
            ..Default::default()
        };

        let response = self
            .llm
            .complete(&prompt, &opts)
            .await
            .map_err(|e| VersionError::Storage(format!("diff reasoner llm: {e}")))?;

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
            summary: format!(
                "{} additions, {} updates, {} deletions",
                tree_diff.adds.len(),
                tree_diff.updates.len(),
                tree_diff.deletes.len()
            ),
            change_type: "additive".into(),
            impact: RawImpact {
                direct: 0,
                transitive: 0,
                reindex: false,
                notify: false,
            },
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
            details: raw
                .changes
                .into_iter()
                .filter_map(|c| {
                    Some(SemanticChange {
                        uri: ContextUri::parse(c.uri).ok()?,
                        description: c.description,
                        category: match c.category.as_str() {
                            "relation" => ChangeCategory::RelationChange,
                            "state" => ChangeCategory::StateTransition,
                            "skill" => ChangeCategory::SkillRefinement,
                            "memory" => ChangeCategory::MemoryConsolidation,
                            _ => ChangeCategory::FactUpdate,
                        },
                        magnitude: c.magnitude,
                    })
                })
                .collect(),
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
    config: ReasoningConfig,
}

impl<V: VersionStore> TemporalReasoner<V> {
    pub fn new(store: Arc<V>, config: ReasoningConfig) -> crate::Result<Self> {
        config.validate()?;
        Ok(Self { store, config })
    }

    /// 生成条目在时间窗口内的演变时间线。
    pub async fn evolution_timeline(
        &self,
        scope: &ContextUri,
        from: AsOfTime,
        to: AsOfTime,
    ) -> std::result::Result<Vec<TimelineEvent>, VersionError> {
        let log = self
            .store
            .log(
                scope,
                &LogOpts {
                    max_count: Some(self.config.timeline_log_limit),
                    ..Default::default()
                },
            )
            .await?;

        let mut events = Vec::new();
        for commit in log {
            if let AsOfTime::Timestamp(from_ts) = &from
                && commit.timestamp < *from_ts
            {
                continue;
            }
            if let AsOfTime::Timestamp(to_ts) = &to
                && commit.timestamp > *to_ts
            {
                continue;
            }

            let entries = [
                commit
                    .metadata
                    .changes
                    .adds
                    .iter()
                    .map(|entry| entry.uri.clone())
                    .collect::<Vec<_>>(),
                commit
                    .metadata
                    .changes
                    .updates
                    .iter()
                    .map(|u| u.uri.clone())
                    .collect(),
            ]
            .concat();

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
        let log = self
            .store
            .log(
                scope,
                &LogOpts {
                    max_count: Some(self.config.periodicity_log_limit),
                    ..Default::default()
                },
            )
            .await?;

        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for commit in log {
            for add in &commit.metadata.changes.adds {
                *counts.entry(add.uri.to_string()).or_default() += 1;
            }
            for upd in &commit.metadata.changes.updates {
                *counts.entry(upd.uri.to_string()).or_default() += 1;
            }
        }

        Ok(counts
            .into_iter()
            .filter(|(_, count)| *count >= min_occurrences)
            .filter_map(|(uri, count)| ContextUri::parse(uri).ok().map(|u| (u, count)))
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
            Ok(ref p) => Some(p.sparse_text().to_string()),
            _ => None,
        };
        let after_text = match after_content {
            Ok(ref p) => Some(p.sparse_text().to_string()),
            _ => None,
        };

        Ok((before_text, after_text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Author, Branch, BranchName, BranchType, ChangeSet, Commit, CommitMeta, ContentHash,
        GcPolicy, GcReport, ImpactAnalysis, KnowledgeMergeStrategy, MergeResult, MergeStrategy,
        ProvenanceGraph, StructuredDiff, Tag, TreeDiff, VersionRef,
    };
    use agent_context_db_core::{ContentPayload, ContextUri};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;

    struct TemporalTestStore {
        commits: Vec<Commit>,
        snapshots: HashMap<CommitId, HashMap<String, ContentPayload>>,
    }

    #[async_trait]
    impl VersionStore for TemporalTestStore {
        async fn commit(
            &self,
            _scope: &ContextUri,
            _changes: ChangeSet,
            _meta: CommitMeta,
        ) -> crate::Result<CommitId> {
            unsupported()
        }

        async fn create_branch(
            &self,
            _scope: &ContextUri,
            _name: BranchName,
            _from: CommitId,
            _bt: BranchType,
        ) -> crate::Result<Branch> {
            unsupported()
        }

        async fn list_branches(&self, _scope: &ContextUri) -> crate::Result<Vec<Branch>> {
            unsupported()
        }

        async fn delete_branch(
            &self,
            _scope: &ContextUri,
            _name: &BranchName,
        ) -> crate::Result<()> {
            unsupported()
        }

        async fn create_tag(&self, _scope: &ContextUri, _tag: Tag) -> crate::Result<()> {
            unsupported()
        }

        async fn list_tags(&self, _scope: &ContextUri) -> crate::Result<Vec<Tag>> {
            unsupported()
        }

        async fn log(&self, _scope: &ContextUri, _opts: &LogOpts) -> crate::Result<Vec<Commit>> {
            Ok(self.commits.clone())
        }

        async fn read_at(
            &self,
            uri: &ContextUri,
            ref_: VersionRef,
            _level: ContentLevel,
        ) -> crate::Result<ContentPayload> {
            match ref_ {
                VersionRef::Commit(id) => {
                    self.asof_read(uri, AsOfTime::Commit(id), ContentLevel::L0)
                        .await
                }
                _ => unsupported(),
            }
        }

        async fn asof_read(
            &self,
            uri: &ContextUri,
            when: AsOfTime,
            _level: ContentLevel,
        ) -> crate::Result<ContentPayload> {
            let AsOfTime::Commit(commit) = when else {
                return unsupported();
            };
            self.snapshots
                .get(&commit)
                .and_then(|snapshot| snapshot.get(uri.as_str()))
                .cloned()
                .ok_or_else(|| VersionError::NotFound(uri.to_string()))
        }

        async fn merge(
            &self,
            _scope: &ContextUri,
            _from: &BranchName,
            _into: &BranchName,
            _strategy: MergeStrategy,
        ) -> crate::Result<MergeResult> {
            unsupported()
        }

        async fn diff_commits(
            &self,
            _scope: &ContextUri,
            _a: &CommitId,
            _b: &CommitId,
        ) -> crate::Result<TreeDiff> {
            unsupported()
        }

        async fn switch_head(
            &self,
            _scope: &ContextUri,
            _branch: &BranchName,
        ) -> crate::Result<()> {
            unsupported()
        }

        async fn cherry_pick(
            &self,
            _scope: &ContextUri,
            _commit: &CommitId,
            _onto: &BranchName,
            _strategy: crate::ConflictStrategy,
        ) -> crate::Result<CommitId> {
            unsupported()
        }

        async fn rebase(
            &self,
            _scope: &ContextUri,
            _branch: &BranchName,
            _onto: &BranchName,
            _strategy: crate::ConflictStrategy,
        ) -> crate::Result<Vec<CommitId>> {
            unsupported()
        }

        async fn squash(
            &self,
            _scope: &ContextUri,
            _commits: Vec<CommitId>,
            _message: &str,
        ) -> crate::Result<crate::SquashResult> {
            unsupported()
        }

        async fn gc(&self, _scope: &ContextUri, _policy: &GcPolicy) -> crate::Result<GcReport> {
            unsupported()
        }

        async fn evaluate_semantic_tags(
            &self,
            _scope: &ContextUri,
        ) -> crate::Result<Vec<(crate::model::TagName, CommitId)>> {
            unsupported()
        }

        async fn provenance(&self, _uri: &ContextUri) -> crate::Result<ProvenanceGraph> {
            unsupported()
        }

        async fn impact_analysis(&self, _commit: &CommitId) -> crate::Result<ImpactAnalysis> {
            unsupported()
        }

        async fn semantic_diff(
            &self,
            _scope: &ContextUri,
            _a: &CommitId,
            _b: &CommitId,
        ) -> crate::Result<StructuredDiff> {
            unsupported()
        }

        async fn evolution(&self, _uri: &ContextUri) -> crate::Result<Vec<crate::TemporalVersion>> {
            unsupported()
        }

        async fn knowledge_merge(
            &self,
            _scope: &ContextUri,
            _from: &BranchName,
            _into: &BranchName,
            _strategy: KnowledgeMergeStrategy,
        ) -> crate::Result<MergeResult> {
            unsupported()
        }
    }

    fn unsupported<T>() -> crate::Result<T> {
        Err(VersionError::Storage(
            "unsupported in TemporalTestStore".into(),
        ))
    }

    fn text_payload(text: &str) -> ContentPayload {
        ContentPayload::Text {
            sparse: text.into(),
            dense: String::new(),
            full: text.into(),
        }
    }

    fn test_commit(
        id: CommitId,
        changes: ChangeSet,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Commit {
        Commit {
            id,
            parents: Vec::new(),
            tree_hash: ContentHash("tree".into()),
            author: Author {
                agent_id: None,
                user_id: None,
                system: true,
            },
            message: "temporal test".into(),
            timestamp,
            metadata: CommitMeta {
                changes,
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn temporal_reasoner_reports_periodicity_timeline_and_diff() {
        let scope = ContextUri::parse("uwu://tenant/agent/a/memory/fact/scope").unwrap();
        let uri = ContextUri::parse("uwu://tenant/agent/a/memory/fact/topic/item").unwrap();
        let first = CommitId::new();
        let second = CommitId::new();
        let t0 = chrono::Utc::now();
        let commits = vec![
            test_commit(
                first.clone(),
                ChangeSet {
                    adds: vec![agent_context_db_core::ContextEntry::new_text(
                        uri.clone(),
                        agent_context_db_core::TenantId(uuid::Uuid::nil()),
                        "before",
                    )],
                    ..Default::default()
                },
                t0,
            ),
            test_commit(
                second.clone(),
                ChangeSet {
                    updates: vec![crate::UriChange {
                        uri: uri.clone(),
                        old_hash: None,
                        new_hash: ContentHash("v2".into()),
                        diff_summary: "updated fact text".into(),
                        entry: agent_context_db_core::ContextEntry::new_text(
                            uri.clone(),
                            agent_context_db_core::TenantId(uuid::Uuid::nil()),
                            "after",
                        ),
                    }],
                    ..Default::default()
                },
                t0 + chrono::Duration::seconds(1),
            ),
        ];
        let snapshots = HashMap::from([
            (
                first.clone(),
                HashMap::from([(uri.to_string(), text_payload("before"))]),
            ),
            (
                second.clone(),
                HashMap::from([(uri.to_string(), text_payload("after"))]),
            ),
        ]);
        let reasoner = TemporalReasoner::new(
            Arc::new(TemporalTestStore { commits, snapshots }),
            ReasoningConfig::default(),
        )
        .unwrap();

        let periodic = reasoner.detect_periodicity(&scope, 2).await.unwrap();
        assert_eq!(periodic, vec![(uri.clone(), 2)]);

        let timeline = reasoner
            .evolution_timeline(
                &scope,
                AsOfTime::Timestamp(t0 - chrono::Duration::seconds(1)),
                AsOfTime::Timestamp(t0 + chrono::Duration::seconds(2)),
            )
            .await
            .unwrap();
        assert_eq!(timeline.len(), 2);
        assert!(
            timeline
                .iter()
                .any(|event| event.entries_changed.contains(&uri))
        );

        let (before, after) = reasoner
            .temporal_diff(&uri, AsOfTime::Commit(first), AsOfTime::Commit(second))
            .await
            .unwrap();
        assert_eq!(before.as_deref(), Some("before"));
        assert_eq!(after.as_deref(), Some("after"));
    }
}
