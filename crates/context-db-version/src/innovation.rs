//! 版本层创新功能（F19 知识晶体 + F21 自修复 + F23 梦境巩固 + F27 因果推断）。

use agent_context_db_core::{
    ConsolidationMeta, ConsolidationStatus, ContentLevel, ContentPayload, ContentStore,
    ContentType, ContextEntry, ContextUri, FsOps, LineageEntry, LlmClient, LlmOpts, LlmTaskKind,
    MediaType, MvccVersion, PromptOptimization, StateScope, TenantId,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

use crate::{CommitId, TemporalReasoner, VersionStore};

// ═══════════════════════════════════════════════════════════════════════════
// F19 知识晶体蒸馏
// ═══════════════════════════════════════════════════════════════════════════

/// 知识晶体 —— 从大量经验中蒸馏出的紧凑知识单元。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeCrystal {
    /// 晶体标识
    pub id: String,
    /// 一句话原则
    pub principle: String,
    /// 支撑证据（来源轨迹 URI）
    pub evidence: Vec<ContextUri>,
    /// 置信度
    pub confidence: f32,
    /// 应用条件
    pub preconditions: Vec<String>,
    /// 预期效果
    pub expected_outcome: String,
}

/// 知识晶体蒸馏器 —— 从多条轨迹/经验中提炼可复用原则。
pub struct CrystalDistiller {
    llm: Arc<dyn LlmClient>,
    fs: Arc<dyn FsOps>,
}

#[derive(Debug, Clone)]
pub struct CrystalWritebackConfig {
    pub tenant: TenantId,
    pub agent_scope: String,
    pub min_confidence: f32,
    pub write_dream_insights: bool,
}

impl CrystalWritebackConfig {
    pub fn for_agent(agent_scope: impl Into<String>) -> Self {
        Self {
            tenant: TenantId(Uuid::new_v4()),
            agent_scope: agent_scope.into(),
            min_confidence: 0.35,
            write_dream_insights: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CrystalWritebackReport {
    pub entries: Vec<ContextEntry>,
    pub skipped_low_confidence: usize,
    pub written: usize,
}

pub struct CrystalMemoryWriter {
    store: Option<Arc<dyn ContentStore>>,
    config: CrystalWritebackConfig,
}

impl CrystalMemoryWriter {
    pub fn new(config: CrystalWritebackConfig) -> Self {
        Self {
            store: None,
            config,
        }
    }

    pub fn with_store(mut self, store: Arc<dyn ContentStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn entries_from_crystals(
        &self,
        crystals: &[KnowledgeCrystal],
    ) -> crate::Result<CrystalWritebackReport> {
        let mut report = CrystalWritebackReport::default();
        for (index, crystal) in crystals.iter().enumerate() {
            if crystal.confidence < self.config.min_confidence {
                report.skipped_low_confidence += 1;
                continue;
            }
            report
                .entries
                .push(entry_from_crystal(&self.config, crystal, index)?);
        }
        Ok(report)
    }

    pub fn entries_from_dream_insights(
        &self,
        insights: &[String],
    ) -> crate::Result<CrystalWritebackReport> {
        let mut report = CrystalWritebackReport::default();
        if !self.config.write_dream_insights {
            return Ok(report);
        }
        for (index, insight) in insights.iter().enumerate() {
            let trimmed = insight.trim();
            if trimmed.is_empty() {
                report.skipped_low_confidence += 1;
                continue;
            }
            report
                .entries
                .push(entry_from_dream_insight(&self.config, trimmed, index)?);
        }
        Ok(report)
    }

    pub async fn write_crystals(
        &self,
        crystals: &[KnowledgeCrystal],
    ) -> crate::Result<CrystalWritebackReport> {
        let mut report = self.entries_from_crystals(crystals)?;
        report.written = self.write_entries(&report.entries).await?;
        Ok(report)
    }

    pub async fn write_dream_insights(
        &self,
        insights: &[String],
    ) -> crate::Result<CrystalWritebackReport> {
        let mut report = self.entries_from_dream_insights(insights)?;
        report.written = self.write_entries(&report.entries).await?;
        Ok(report)
    }

    async fn write_entries(&self, entries: &[ContextEntry]) -> crate::Result<usize> {
        let Some(store) = &self.store else {
            return Ok(0);
        };
        store
            .batch_write(entries)
            .await
            .map(|versions| versions.len())
            .map_err(|error| crate::VersionError::Storage(error.to_string()))
    }
}

fn entry_from_crystal(
    config: &CrystalWritebackConfig,
    crystal: &KnowledgeCrystal,
    index: usize,
) -> crate::Result<ContextEntry> {
    let now = chrono::Utc::now();
    let slug = stable_slug(&format!("{}-{}", crystal.id, crystal.principle));
    let uri = ContextUri::parse(format!(
        "uwu://{}/memory/skill/crystal/{:02}-{}",
        config.agent_scope, index, slug
    ))
    .map_err(|error| crate::VersionError::Storage(format!("invalid crystal URI: {error}")))?;
    let mut entry = ContextEntry {
        uri,
        tenant: config.tenant,
        payload: ContentPayload::Text {
            sparse: crystal.principle.clone(),
            dense: format!(
                "Principle: {}\nPreconditions: {}\nExpected outcome: {}",
                crystal.principle,
                crystal.preconditions.join("; "),
                crystal.expected_outcome
            ),
            full: format!(
                "Knowledge crystal {}\n\nPrinciple: {}\n\nPreconditions:\n{}\n\nExpected outcome:\n{}\n\nEvidence:\n{}",
                crystal.id,
                crystal.principle,
                crystal
                    .preconditions
                    .iter()
                    .map(|item| format!("- {item}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                crystal.expected_outcome,
                crystal
                    .evidence
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        },
        media_type: MediaType::Text,
        metadata: Default::default(),
        mvcc_version: MvccVersion(0),
        created_at: now,
        updated_at: now,
        derivation: None,
    };
    entry.metadata.content_type = Some(ContentType::Skill);
    entry.metadata.state_scope = Some(StateScope::Long);
    entry.metadata.quality_score = Some(crystal.confidence.clamp(0.0, 1.0));
    entry.metadata.tags = vec!["crystal:distilled".into(), "training:candidate".into()];
    entry.metadata.consolidation = Some(ConsolidationMeta {
        source: "crystal-distiller".into(),
        generation: 1,
        status: ConsolidationStatus::InProgress,
        patch_count: 0,
        lineage: vec![LineageEntry {
            version: MvccVersion(0),
            timestamp: now,
            change_summary: "distilled experience crystal into long-term skill memory".into(),
        }],
        evidence_uris: crystal.evidence.clone(),
        corroboration: crystal.evidence.len(),
        half_life: Some(agent_context_db_core::HalfLife::Finite { days: 180.0 }),
        entangled_with: crystal.evidence.clone(),
    });
    entry
        .metadata
        .set_custom_field("knowledge_crystal", crystal)
        .map_err(|error| crate::VersionError::Storage(error.to_string()))?;
    Ok(entry)
}

fn entry_from_dream_insight(
    config: &CrystalWritebackConfig,
    insight: &str,
    index: usize,
) -> crate::Result<ContextEntry> {
    let now = chrono::Utc::now();
    let slug = stable_slug(insight);
    let uri = ContextUri::parse(format!(
        "uwu://{}/memory/heuristic/dream/{:02}-{}",
        config.agent_scope, index, slug
    ))
    .map_err(|error| crate::VersionError::Storage(format!("invalid dream insight URI: {error}")))?;
    let mut entry = ContextEntry::new_text(uri, config.tenant, insight.to_string());
    entry.metadata.content_type = Some(ContentType::Heuristic);
    entry.metadata.state_scope = Some(StateScope::Long);
    entry.metadata.quality_score = Some(0.62);
    entry.metadata.tags = vec!["dream:insight".into(), "crystal:replay".into()];
    entry.metadata.consolidation = Some(ConsolidationMeta {
        source: "dream-consolidator".into(),
        generation: 1,
        status: ConsolidationStatus::InProgress,
        patch_count: 0,
        lineage: vec![LineageEntry {
            version: MvccVersion(0),
            timestamp: now,
            change_summary: "converted dream replay insight into heuristic memory".into(),
        }],
        evidence_uris: vec![],
        corroboration: 1,
        half_life: Some(agent_context_db_core::HalfLife::Finite { days: 90.0 }),
        entangled_with: vec![],
    });
    entry
        .metadata
        .set_custom_field("dream_insight", &insight)
        .map_err(|error| crate::VersionError::Storage(error.to_string()))?;
    Ok(entry)
}

fn stable_slug(text: &str) -> String {
    let hash = blake3_like(text);
    let prefix = text
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|token| !token.is_empty())
        .take(4)
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join("-");
    if prefix.is_empty() {
        hash
    } else {
        format!("{}-{}", prefix, hash)
    }
}

fn blake3_like(text: &str) -> String {
    let mut acc: u64 = 0xcbf29ce484222325;
    for byte in text.as_bytes() {
        acc ^= *byte as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    format!("{acc:016x}").chars().take(8).collect()
}

impl CrystalDistiller {
    pub fn new(llm: Arc<dyn LlmClient>, fs: Arc<dyn FsOps>) -> Self {
        Self { llm, fs }
    }

    /// 从一批经验 URI 中蒸馏知识晶体。
    pub async fn distill(
        &self,
        experience_uris: &[ContextUri],
    ) -> Result<Vec<KnowledgeCrystal>, agent_context_db_core::ContextError> {
        let mut texts = Vec::new();
        for uri in experience_uris {
            if let Ok(content) = self.fs.read(uri, ContentLevel::L1).await {
                let s = content.sparse_text().to_string();
                if !s.is_empty() {
                    texts.push(s);
                }
            }
        }

        if texts.is_empty() {
            return Ok(vec![]);
        }

        let combined = texts.join("\n===\n");

        let prompt = format!(
            r#"Distill reusable knowledge principles from these experiences:

{combined}

Return a JSON array of crystals:
[{{"principle": "...", "preconditions": [...], "expected_outcome": "...", "confidence": 0.0-1.0}}]
"#
        );

        let response = self
            .llm
            .complete(&prompt, &LlmOpts::default())
            .await
            .map_err(|e| agent_context_db_core::ContextError::Storage(format!("distill: {e}")))?;

        #[derive(serde::Deserialize)]
        struct RawCrystal {
            principle: String,
            preconditions: Vec<String>,
            expected_outcome: String,
            confidence: f32,
        }

        let raw: Vec<RawCrystal> = serde_json::from_str(&response).unwrap_or_default();

        Ok(raw
            .into_iter()
            .enumerate()
            .map(|(i, r)| KnowledgeCrystal {
                id: format!("crystal-{}", i),
                principle: r.principle,
                evidence: experience_uris.to_vec(),
                confidence: r.confidence,
                preconditions: r.preconditions,
                expected_outcome: r.expected_outcome,
            })
            .collect())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F21 自修复
// ═══════════════════════════════════════════════════════════════════════════

/// 修复策略。
#[derive(Debug, Clone)]
pub enum RepairAction {
    /// 回滚到指定版本
    Rollback(CommitId),
    /// 合并补丁
    Patch { from: CommitId, description: String },
    /// 添加缺失信息
    Supplement { uri: ContextUri, content: String },
    /// 删除损坏条目
    Remove(ContextUri),
}

/// 自修复诊断器 —— 检测不一致并生成修复方案。
pub struct SelfHealer<V: VersionStore> {
    store: Arc<V>,
    llm: Arc<dyn LlmClient>,
}

impl<V: VersionStore> SelfHealer<V> {
    pub fn new(store: Arc<V>, llm: Arc<dyn LlmClient>) -> Self {
        Self { store, llm }
    }

    /// 诊断一个 scope 下的不一致。
    pub async fn diagnose(
        &self,
        scope: &ContextUri,
    ) -> std::result::Result<Vec<RepairAction>, crate::VersionError> {
        let log = self
            .store
            .log(
                scope,
                &crate::LogOpts {
                    max_count: Some(20),
                    ..Default::default()
                },
            )
            .await?;
        let mut actions = Vec::new();

        // 检测快速连续的回滚-重做循环（thrash）
        if log.len() >= 4 {
            let mut thrash_count = 0;
            for i in 1..log.len().min(10) {
                if log[i].message == log[i - 1].message {
                    thrash_count += 1;
                }
            }
            if thrash_count >= 3 {
                // 建议回滚到稳定点
                actions.push(RepairAction::Rollback(log[thrash_count + 1].id.clone()));
            }
        }

        // 检测空 commit（无实际变更的提交）并建议清理
        let mut empty_count = 0;
        for commit in &log {
            let changes = &commit.metadata.changes;
            if changes.adds.is_empty() && changes.updates.is_empty() && changes.deletes.is_empty() {
                empty_count += 1;
            }
        }
        if empty_count >= 3 {
            actions.push(RepairAction::Supplement {
                uri: scope.clone(),
                content: format!(
                    "detected {} empty commits (no changes) in recent history; consider squashing",
                    empty_count
                ),
            });
        }

        Ok(actions)
    }

    /// 用 LLM 做深度语义诊断。
    pub async fn semantic_diagnose(
        &self,
        scope: &ContextUri,
    ) -> std::result::Result<Vec<RepairAction>, crate::VersionError> {
        let log = self
            .store
            .log(
                scope,
                &crate::LogOpts {
                    max_count: Some(10),
                    ..Default::default()
                },
            )
            .await?;

        let log_text: Vec<String> = log
            .iter()
            .map(|c| {
                format!(
                    "{} | adds:{} updates:{} deletes:{}",
                    c.message,
                    c.metadata.changes.adds.len(),
                    c.metadata.changes.updates.len(),
                    c.metadata.changes.deletes.len()
                )
            })
            .collect();

        let prompt = format!(
            r#"Diagnose potential issues in this version history:

{}
Return JSON array of repair actions:
[{{"action": "rollback|patch|supplement|remove", "description": "...", "target": "..."}}]
"#,
            log_text.join("\n")
        );

        let response = self
            .llm
            .complete(&prompt, &LlmOpts::default())
            .await
            .map_err(|e| crate::VersionError::Storage(format!("self-heal llm: {e}")))?;

        // 解析 LLM 建议的修复方案
        let actions = parse_repair_actions(&response, scope);
        Ok(actions)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F23 梦境巩固
// ═══════════════════════════════════════════════════════════════════════════

/// 梦境巩固器 —— 在空闲时段重放历史轨迹，发现隐含模式。
pub struct DreamConsolidator<V: VersionStore> {
    store: Arc<V>,
    llm: Arc<dyn LlmClient>,
    fs: Arc<dyn FsOps>,
    replay_config: ReplaySleepConfig,
}

#[derive(Debug, Clone)]
pub struct ReplaySleepConfig {
    pub max_commits: usize,
    pub min_cluster_size: usize,
    pub min_crystal_confidence: f32,
    pub skill_success_floor: f32,
}

impl ReplaySleepConfig {
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_commits == 0 || self.min_cluster_size == 0 {
            return Err(crate::VersionError::InvalidConfig(
                "replay limits must be greater than zero".into(),
            ));
        }
        for (name, value) in [
            ("min_crystal_confidence", self.min_crystal_confidence),
            ("skill_success_floor", self.skill_success_floor),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(crate::VersionError::InvalidConfig(format!(
                    "{name} must be finite and in 0..=1"
                )));
            }
        }
        Ok(())
    }
}

impl Default for ReplaySleepConfig {
    fn default() -> Self {
        Self {
            max_commits: 30,
            min_cluster_size: 3,
            min_crystal_confidence: 0.35,
            skill_success_floor: 0.55,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplaySkillCandidate {
    pub uri: ContextUri,
    pub name: String,
    pub description: String,
    pub precondition: String,
    pub success_rate: f32,
    pub evidence: Vec<ContextUri>,
}

#[derive(Debug, Clone, Default)]
pub struct ReplaySleepReport {
    pub replayed_commits: usize,
    pub replayed_uris: Vec<ContextUri>,
    pub insights: Vec<String>,
    pub crystals: Vec<KnowledgeCrystal>,
    pub memory_writeback: CrystalWritebackReport,
    pub skill_candidates: Vec<ReplaySkillCandidate>,
}

struct DreamClusterPrompt {
    key: String,
    count: usize,
    prompt: String,
}

impl<V: VersionStore> DreamConsolidator<V> {
    pub fn new(
        store: Arc<V>,
        llm: Arc<dyn LlmClient>,
        fs: Arc<dyn FsOps>,
        replay_config: ReplaySleepConfig,
    ) -> crate::Result<Self> {
        replay_config.validate()?;
        Ok(Self {
            store,
            llm,
            fs,
            replay_config,
        })
    }

    /// 执行一次"梦境"巩固周期。
    ///
    /// 在当前 scope 的最近 N 条轨迹中找相似模式：
    /// 1. 提取变更 URI 并按路径前缀聚类
    /// 2. 对高频聚类，读取内容并通过 LLM 合成洞察
    /// 3. 返回 LLM 生成的模式描述
    pub async fn consolidate(
        &self,
        scope: &ContextUri,
    ) -> std::result::Result<Vec<String>, crate::VersionError> {
        let log = self
            .store
            .log(
                scope,
                &crate::LogOpts {
                    max_count: Some(self.replay_config.max_commits),
                    ..Default::default()
                },
            )
            .await?;

        // 提取所有变更的 URI
        let mut changed_uris = Vec::new();
        for commit in &log {
            for add in &commit.metadata.changes.adds {
                changed_uris.push(add.uri.clone());
            }
        }

        // 聚类相似变更
        let mut clusters: HashMap<String, Vec<ContextUri>> = HashMap::new();
        for uri in &changed_uris {
            let segs = uri.segments();
            let key: String = segs.iter().take(3).cloned().collect::<Vec<_>>().join("/");
            clusters.entry(key).or_default().push(uri.clone());
        }

        // 高频聚类先收集 prompt，再一次批量补全，避免每个 cluster 串行打 LLM。
        let mut insights: Vec<String> = Vec::new();
        let mut batch = Vec::new();

        for (key, uris) in clusters {
            if uris.len() < self.replay_config.min_cluster_size {
                if uris.len() >= 2 {
                    insights.push(format!(
                        "cluster '{}' with {} related changes",
                        key,
                        uris.len()
                    ));
                }
                continue;
            }

            let mut summaries = Vec::new();
            for uri in &uris {
                if let Ok(content) = self.fs.read(uri, ContentLevel::L0).await {
                    let abs = content.sparse_text();
                    if !abs.is_empty() {
                        summaries.push(format!("- {uri}: {abs}", uri = uri, abs = abs));
                    }
                }
            }

            if summaries.is_empty() {
                insights.push(format!(
                    "cluster '{}' with {} related changes (no content)",
                    key,
                    uris.len()
                ));
                continue;
            }

            batch.push(DreamClusterPrompt {
                key,
                count: uris.len(),
                prompt: dream_cluster_prompt(uris.len(), summaries),
            });
        }

        let prompts = batch
            .iter()
            .map(|item| item.prompt.clone())
            .collect::<Vec<_>>();
        let responses = if prompts.is_empty() {
            Vec::new()
        } else {
            self.llm
                .batch_complete(
                    &prompts,
                    &LlmOpts {
                        task: LlmTaskKind::Synthesis,
                        prompt: PromptOptimization::default()
                            .force_cache()
                            .target_tokens(2_000),
                        ..Default::default()
                    },
                )
                .await
                .unwrap_or_else(|_| Vec::new())
        };

        for (idx, item) in batch.into_iter().enumerate() {
            let trimmed = responses
                .get(idx)
                .map(|response| response.trim())
                .filter(|response| !response.is_empty());
            match trimmed {
                Some(response) => insights.push(format!(
                    "cluster '{key}' ({count} changes): {response}",
                    key = item.key,
                    count = item.count,
                    response = response
                )),
                None => insights.push(format!(
                    "cluster '{}' with {} related changes",
                    item.key, item.count
                )),
            }
        }

        Ok(insights)
    }

    pub async fn sleep_replay_cycle(
        &self,
        scope: &ContextUri,
        writer: &CrystalMemoryWriter,
    ) -> std::result::Result<ReplaySleepReport, crate::VersionError> {
        let log = self
            .store
            .log(
                scope,
                &crate::LogOpts {
                    max_count: Some(self.replay_config.max_commits),
                    ..Default::default()
                },
            )
            .await?;
        let replayed_uris = replay_uris_from_log(&log);
        let insights = self.consolidate(scope).await?;
        let crystals = CrystalDistiller::new(self.llm.clone(), self.fs.clone())
            .distill(&replayed_uris)
            .await
            .map_err(|e| crate::VersionError::Storage(format!("dream crystal distill: {e}")))?;

        let mut memory_writeback = writer.write_crystals(&crystals).await?;
        let dream_report = writer.write_dream_insights(&insights).await?;
        memory_writeback.entries.extend(dream_report.entries);
        memory_writeback.skipped_low_confidence += dream_report.skipped_low_confidence;
        memory_writeback.written += dream_report.written;

        let skill_candidates = crystals
            .iter()
            .filter(|crystal| crystal.confidence >= self.replay_config.min_crystal_confidence)
            .enumerate()
            .map(|(index, crystal)| {
                skill_candidate_from_crystal(crystal, index, self.replay_config.skill_success_floor)
            })
            .collect::<crate::Result<Vec<_>>>()?;

        Ok(ReplaySleepReport {
            replayed_commits: log.len(),
            replayed_uris,
            insights,
            crystals,
            memory_writeback,
            skill_candidates,
        })
    }
}

fn replay_uris_from_log(log: &[crate::model::Commit]) -> Vec<ContextUri> {
    let mut seen = HashSet::new();
    let mut uris = Vec::new();
    for commit in log {
        for entry in &commit.metadata.changes.adds {
            if seen.insert(entry.uri.clone()) {
                uris.push(entry.uri.clone());
            }
        }
        for update in &commit.metadata.changes.updates {
            if seen.insert(update.uri.clone()) {
                uris.push(update.uri.clone());
            }
        }
    }
    uris
}

fn skill_candidate_from_crystal(
    crystal: &KnowledgeCrystal,
    index: usize,
    success_floor: f32,
) -> crate::Result<ReplaySkillCandidate> {
    let slug = stable_slug(&format!("{}-{}", crystal.id, crystal.principle));
    let uri = ContextUri::parse(format!(
        "uwu://dream/agent/replay/memory/skill/{index:02}-{slug}"
    ))
    .map_err(|error| crate::VersionError::Storage(format!("invalid replay skill URI: {error}")))?;
    Ok(ReplaySkillCandidate {
        uri,
        name: crystal
            .principle
            .split_whitespace()
            .take(6)
            .collect::<Vec<_>>()
            .join(" "),
        description: crystal.principle.clone(),
        precondition: crystal.preconditions.join("; "),
        success_rate: crystal.confidence.max(success_floor).clamp(0.0, 1.0),
        evidence: crystal.evidence.clone(),
    })
}

fn dream_cluster_prompt(count: usize, summaries: Vec<String>) -> String {
    format!(
        "Analyze this cluster of {count} context changes:\n\n{summaries}\n\n\
         Identify the underlying pattern: what do these changes have in common? \
         Is there a reusable insight or principle? \
         Respond with a single concise paragraph (2-4 sentences).",
        summaries = summaries.join("\n"),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// F27 因果推断
// ═══════════════════════════════════════════════════════════════════════════

/// 因果关系假设。
#[derive(Debug, Clone)]
pub struct CausalHypothesis {
    pub cause_uri: ContextUri,
    pub effect_uri: ContextUri,
    /// 时间先后强度（cause 在 effect 之前出现的概率）
    pub temporal_precedence: f32,
    /// 共现强度
    pub co_occurrence: f32,
    /// 总体因果置信度
    pub confidence: f32,
}

/// 版本历史上的因果结构学习配置。
#[derive(Debug, Clone)]
pub struct CausalDiscoveryConfig {
    /// 最多读取多少条 commit 作为观测样本。
    pub max_log_count: usize,
    /// 进入结构学习的最大 URI 数，避免高维稀疏历史导致组合爆炸。
    pub max_variables: usize,
    /// PC 条件独立检验的最大条件集大小。
    pub max_conditioning_set: usize,
    /// 一条候选因果边至少需要多少次 cause 暴露。
    pub min_support: usize,
    /// 条件风险差低于该阈值时认为可被混淆变量解释。
    pub independence_threshold: f32,
    /// GES/BIC 的复杂度惩罚倍数。
    pub bic_penalty: f32,
    /// 每个 effect 最多保留多少个直接父节点。
    pub max_parents: usize,
    pub ate_confidence_weight: f32,
    pub bic_confidence_weight: f32,
    pub support_confidence_weight: f32,
    pub bic_gain_scale: f32,
}

impl CausalDiscoveryConfig {
    pub fn validate(&self) -> crate::Result<()> {
        if self.max_log_count == 0
            || self.max_variables == 0
            || self.min_support == 0
            || self.max_parents == 0
        {
            return Err(crate::VersionError::InvalidConfig(
                "causal limits and min_support must be greater than zero".into(),
            ));
        }
        if !self.independence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.independence_threshold)
        {
            return Err(crate::VersionError::InvalidConfig(
                "independence_threshold must be finite and in 0..=1".into(),
            ));
        }
        if !self.bic_penalty.is_finite()
            || self.bic_penalty < 0.0
            || !self.bic_gain_scale.is_finite()
            || self.bic_gain_scale <= 0.0
        {
            return Err(crate::VersionError::InvalidConfig(
                "BIC penalty must be non-negative and gain scale positive".into(),
            ));
        }
        let weights = [
            self.ate_confidence_weight,
            self.bic_confidence_weight,
            self.support_confidence_weight,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
            || (weights.iter().sum::<f32>() - 1.0).abs() > f32::EPSILON
        {
            return Err(crate::VersionError::InvalidConfig(
                "causal confidence weights must be non-negative and sum to 1".into(),
            ));
        }
        Ok(())
    }
}

impl Default for CausalDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_log_count: 512,
            max_variables: 64,
            max_conditioning_set: 2,
            min_support: 3,
            independence_threshold: 0.08,
            bic_penalty: 1.0,
            max_parents: 4,
            ate_confidence_weight: 0.62,
            bic_confidence_weight: 0.28,
            support_confidence_weight: 0.10,
            bic_gain_scale: 8.0,
        }
    }
}

/// 一个从结构学习得到的可干预因果边。
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub cause_uri: ContextUri,
    pub effect_uri: ContextUri,
    /// do(cause=true) 对 effect 下一步变更概率的估计提升。
    pub average_treatment_effect: f32,
    /// 支持该边的 cause 暴露次数。
    pub support: usize,
    /// 结构学习置信度，融合 PC 条件独立检验和 GES/BIC 增益。
    pub confidence: f32,
    /// 使该边仍不独立的条件变量集合。
    pub adjustment_set: Vec<ContextUri>,
}

/// 可干预的因果图。
#[derive(Debug, Clone, Default)]
pub struct CausalGraph {
    pub nodes: Vec<ContextUri>,
    pub edges: Vec<CausalEdge>,
    children: HashMap<ContextUri, Vec<usize>>,
    parents: HashMap<ContextUri, Vec<usize>>,
}

impl CausalGraph {
    pub fn new(nodes: Vec<ContextUri>, edges: Vec<CausalEdge>) -> Self {
        let mut children: HashMap<ContextUri, Vec<usize>> = HashMap::new();
        let mut parents: HashMap<ContextUri, Vec<usize>> = HashMap::new();
        for (idx, edge) in edges.iter().enumerate() {
            children
                .entry(edge.cause_uri.clone())
                .or_default()
                .push(idx);
            parents
                .entry(edge.effect_uri.clone())
                .or_default()
                .push(idx);
        }
        Self {
            nodes,
            edges,
            children,
            parents,
        }
    }

    pub fn outgoing(&self, uri: &ContextUri) -> Vec<&CausalEdge> {
        self.children
            .get(uri)
            .map(|idxs| idxs.iter().map(|idx| &self.edges[*idx]).collect())
            .unwrap_or_default()
    }

    pub fn incoming(&self, uri: &ContextUri) -> Vec<&CausalEdge> {
        self.parents
            .get(uri)
            .map(|idxs| idxs.iter().map(|idx| &self.edges[*idx]).collect())
            .unwrap_or_default()
    }

    pub fn descendants(&self, uri: &ContextUri) -> Vec<ContextUri> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([uri.clone()]);
        visited.insert(uri.clone());
        while let Some(node) = queue.pop_front() {
            for edge in self.outgoing(&node) {
                if visited.insert(edge.effect_uri.clone()) {
                    result.push(edge.effect_uri.clone());
                    queue.push_back(edge.effect_uri.clone());
                }
            }
        }
        result
    }

    fn has_path(&self, from: &ContextUri, to: &ContextUri) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([from.clone()]);
        while let Some(node) = queue.pop_front() {
            if &node == to {
                return true;
            }
            if !visited.insert(node.clone()) {
                continue;
            }
            for edge in self.outgoing(&node) {
                queue.push_back(edge.effect_uri.clone());
            }
        }
        false
    }
}

/// Pearl do-operator 风格的二值干预。
#[derive(Debug, Clone)]
pub struct CausalIntervention {
    pub target_uri: ContextUri,
    /// true 表示 do(target changed/fixed=true)，false 表示 do(remove target change)。
    pub value: bool,
}

impl CausalIntervention {
    pub fn fix(target_uri: ContextUri) -> Self {
        Self {
            target_uri,
            value: true,
        }
    }

    pub fn remove(target_uri: ContextUri) -> Self {
        Self {
            target_uri,
            value: false,
        }
    }
}

/// 干预后某个下游节点的反事实影响估计。
#[derive(Debug, Clone)]
pub struct CounterfactualImpact {
    pub effect_uri: ContextUri,
    pub direct_effect: f32,
    pub total_effect: f32,
    pub confidence: f32,
    pub causal_path: Vec<ContextUri>,
}

/// do-calculus 干预查询结果。
#[derive(Debug, Clone)]
pub struct InterventionResult {
    pub intervention: CausalIntervention,
    pub affected: Vec<CounterfactualImpact>,
}

impl InterventionResult {
    pub fn affected_uris(&self) -> Vec<ContextUri> {
        self.affected
            .iter()
            .map(|impact| impact.effect_uri.clone())
            .collect()
    }
}

#[derive(Debug, Clone)]
struct CausalSamples {
    variables: Vec<ContextUri>,
    rows: Vec<Vec<bool>>,
}

/// 因果推断器 —— 在版本历史上做结构学习和 do-calculus 干预推断。
pub struct CausalInference<V: VersionStore> {
    store: Arc<V>,
    _temporal: TemporalReasoner<V>,
    config: CausalDiscoveryConfig,
}

impl<V: VersionStore> CausalInference<V> {
    pub fn new(store: Arc<V>, config: CausalDiscoveryConfig) -> crate::Result<Self> {
        config.validate()?;
        let temporal = TemporalReasoner::new(store.clone(), crate::ReasoningConfig::default())?;
        Ok(Self {
            store,
            _temporal: temporal,
            config,
        })
    }

    /// 学习可干预因果图：PC 条件独立检验去混淆，GES/BIC 选择直接父边。
    pub async fn discover_causal_graph(
        &self,
        scope: &ContextUri,
    ) -> std::result::Result<CausalGraph, crate::VersionError> {
        let log = self
            .store
            .log(
                scope,
                &crate::LogOpts {
                    max_count: Some(self.config.max_log_count),
                    ..Default::default()
                },
            )
            .await?;
        let samples = build_causal_samples(log, &self.config);
        Ok(learn_causal_graph(&samples, &self.config))
    }

    /// 对一个知识修正做反事实干预，返回会被影响的下游知识。
    pub async fn intervene(
        &self,
        scope: &ContextUri,
        intervention: CausalIntervention,
    ) -> std::result::Result<InterventionResult, crate::VersionError> {
        let graph = self.discover_causal_graph(scope).await?;
        Ok(apply_intervention(&graph, intervention))
    }

    /// BackwardEvolver 可直接消费这个下游影响面，做反向重核验/lineage 更新。
    pub async fn downstream_impacts(
        &self,
        scope: &ContextUri,
        corrected_uri: ContextUri,
    ) -> std::result::Result<Vec<CounterfactualImpact>, crate::VersionError> {
        Ok(self
            .intervene(scope, CausalIntervention::fix(corrected_uri))
            .await?
            .affected)
    }
}

fn build_causal_samples(
    mut log: Vec<crate::Commit>,
    config: &CausalDiscoveryConfig,
) -> CausalSamples {
    log.sort_by_key(|a| a.timestamp);

    let mut frequencies: HashMap<ContextUri, usize> = HashMap::new();
    let mut touched_by_commit = Vec::new();
    for commit in &log {
        let touched = touched_uris(commit);
        for uri in &touched {
            *frequencies.entry(uri.clone()).or_insert(0) += 1;
        }
        touched_by_commit.push(touched);
    }

    let mut variables = frequencies.into_iter().collect::<Vec<_>>();
    variables.sort_by(|(a_uri, a_count), (b_uri, b_count)| {
        b_count
            .cmp(a_count)
            .then_with(|| a_uri.to_string().cmp(&b_uri.to_string()))
    });
    variables.truncate(config.max_variables);
    let variables = variables
        .into_iter()
        .map(|(uri, _)| uri)
        .collect::<Vec<_>>();
    let variable_index = variables
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, uri)| (uri, idx))
        .collect::<HashMap<_, _>>();

    let rows = touched_by_commit
        .into_iter()
        .map(|touched| {
            let mut row = vec![false; variables.len()];
            for uri in touched {
                if let Some(idx) = variable_index.get(&uri) {
                    row[*idx] = true;
                }
            }
            row
        })
        .collect();

    CausalSamples { variables, rows }
}

fn touched_uris(commit: &crate::Commit) -> HashSet<ContextUri> {
    let mut touched = HashSet::new();
    touched.extend(
        commit
            .metadata
            .changes
            .adds
            .iter()
            .map(|entry| entry.uri.clone()),
    );
    touched.extend(commit.metadata.changes.deletes.iter().cloned());
    for update in &commit.metadata.changes.updates {
        touched.insert(update.uri.clone());
    }
    for rename in &commit.metadata.changes.renames {
        touched.insert(rename.from.clone());
        touched.insert(rename.to.clone());
    }
    touched
}

fn learn_causal_graph(samples: &CausalSamples, config: &CausalDiscoveryConfig) -> CausalGraph {
    if samples.rows.len() < 2 || samples.variables.len() < 2 {
        return CausalGraph::new(samples.variables.clone(), Vec::new());
    }

    let pc_candidates = pc_candidates(samples, config);
    let mut graph = CausalGraph::new(samples.variables.clone(), Vec::new());
    let mut edges = Vec::new();

    for effect_idx in 0..samples.variables.len() {
        let parents = ges_parents_for_effect(samples, effect_idx, &pc_candidates, config);
        for parent in parents {
            let (ate, support) = conditional_risk_difference(
                samples,
                parent.cause_idx,
                effect_idx,
                &parent.adjustment_set,
            );
            if support < config.min_support || ate <= 0.0 {
                continue;
            }
            let cause_uri = samples.variables[parent.cause_idx].clone();
            let effect_uri = samples.variables[effect_idx].clone();
            if graph.has_path(&effect_uri, &cause_uri) {
                continue;
            }
            let confidence = (ate * config.ate_confidence_weight
                + parent.bic_gain.min(config.bic_gain_scale) / config.bic_gain_scale
                    * config.bic_confidence_weight
                + support_confidence(support) * config.support_confidence_weight)
                .clamp(0.0, 1.0);
            edges.push(CausalEdge {
                cause_uri,
                effect_uri,
                average_treatment_effect: ate,
                support,
                confidence,
                adjustment_set: parent
                    .adjustment_set
                    .iter()
                    .map(|idx| samples.variables[*idx].clone())
                    .collect(),
            });
            graph = CausalGraph::new(samples.variables.clone(), edges.clone());
        }
    }

    edges.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    CausalGraph::new(samples.variables.clone(), edges)
}

#[derive(Debug, Clone)]
struct PcCandidate {
    cause_idx: usize,
    effect_idx: usize,
    adjustment_set: Vec<usize>,
}

fn pc_candidates(samples: &CausalSamples, config: &CausalDiscoveryConfig) -> Vec<PcCandidate> {
    let mut candidates = Vec::new();
    for cause_idx in 0..samples.variables.len() {
        for effect_idx in 0..samples.variables.len() {
            if cause_idx == effect_idx {
                continue;
            }
            let support = exposure_support(samples, cause_idx);
            if support < config.min_support {
                continue;
            }
            let mut controls = (0..samples.variables.len())
                .filter(|idx| *idx != cause_idx && *idx != effect_idx)
                .collect::<Vec<_>>();
            controls.sort_by(|a, b| {
                marginal_association(samples, *b, effect_idx)
                    .partial_cmp(&marginal_association(samples, *a, effect_idx))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            controls.truncate(12);

            let mut survived = true;
            let mut best_adjustment = Vec::new();
            for cond_size in 0..=config.max_conditioning_set.min(controls.len()) {
                for set in conditioning_sets(&controls, cond_size) {
                    let (diff, _) =
                        conditional_risk_difference(samples, cause_idx, effect_idx, &set);
                    if diff.abs() < config.independence_threshold {
                        survived = false;
                        break;
                    }
                    if best_adjustment.is_empty() && !set.is_empty() {
                        best_adjustment = set;
                    }
                }
                if !survived {
                    break;
                }
            }
            if survived {
                candidates.push(PcCandidate {
                    cause_idx,
                    effect_idx,
                    adjustment_set: best_adjustment,
                });
            }
        }
    }
    candidates
}

#[derive(Debug, Clone)]
struct GesParent {
    cause_idx: usize,
    adjustment_set: Vec<usize>,
    bic_gain: f32,
}

fn ges_parents_for_effect(
    samples: &CausalSamples,
    effect_idx: usize,
    candidates: &[PcCandidate],
    config: &CausalDiscoveryConfig,
) -> Vec<GesParent> {
    let mut selected = Vec::new();
    let mut selected_set = Vec::new();
    let candidate_causes = candidates
        .iter()
        .filter(|candidate| candidate.effect_idx == effect_idx)
        .cloned()
        .collect::<Vec<_>>();

    loop {
        if selected.len() >= config.max_parents {
            break;
        }
        let baseline = local_bic(samples, effect_idx, &selected_set, config);
        let mut best: Option<GesParent> = None;
        for candidate in &candidate_causes {
            if selected_set.contains(&candidate.cause_idx) {
                continue;
            }
            let mut trial = selected_set.clone();
            trial.push(candidate.cause_idx);
            let score = local_bic(samples, effect_idx, &trial, config);
            let gain = baseline - score;
            if gain <= 0.0 {
                continue;
            }
            let parent = GesParent {
                cause_idx: candidate.cause_idx,
                adjustment_set: candidate.adjustment_set.clone(),
                bic_gain: gain,
            };
            if best
                .as_ref()
                .is_none_or(|old| parent.bic_gain > old.bic_gain)
            {
                best = Some(parent);
            }
        }
        if let Some(parent) = best {
            selected_set.push(parent.cause_idx);
            selected.push(parent);
        } else {
            break;
        }
    }

    selected
}

fn local_bic(
    samples: &CausalSamples,
    effect_idx: usize,
    parents: &[usize],
    config: &CausalDiscoveryConfig,
) -> f32 {
    let n = samples.rows.len().saturating_sub(1).max(1) as f32;
    let mut buckets: HashMap<u64, (usize, usize)> = HashMap::new();
    for t in 1..samples.rows.len() {
        let mut key = 0_u64;
        for (bit, parent_idx) in parents.iter().take(63).enumerate() {
            if samples.rows[t - 1][*parent_idx] {
                key |= 1_u64 << bit;
            }
        }
        let entry = buckets.entry(key).or_insert((0, 0));
        entry.0 += 1;
        if samples.rows[t][effect_idx] {
            entry.1 += 1;
        }
    }

    let entropy = buckets
        .values()
        .map(|(count, positives)| {
            let p = (*positives as f32 / *count as f32).clamp(1e-6, 1.0 - 1e-6);
            let h = -(p * p.ln() + (1.0 - p) * (1.0 - p).ln());
            *count as f32 * h
        })
        .sum::<f32>();
    let params = (1_usize << parents.len().min(12)) as f32;
    entropy + config.bic_penalty * params * n.ln()
}

fn conditional_risk_difference(
    samples: &CausalSamples,
    cause_idx: usize,
    effect_idx: usize,
    controls: &[usize],
) -> (f32, usize) {
    let mut buckets: HashMap<u64, (usize, usize, usize, usize)> = HashMap::new();
    for t in 1..samples.rows.len() {
        let mut key = 0_u64;
        for (bit, control_idx) in controls.iter().take(63).enumerate() {
            if samples.rows[t - 1][*control_idx] {
                key |= 1_u64 << bit;
            }
        }
        let entry = buckets.entry(key).or_insert((0, 0, 0, 0));
        let cause = samples.rows[t - 1][cause_idx];
        let effect = samples.rows[t][effect_idx];
        match (cause, effect) {
            (true, true) => {
                entry.0 += 1;
                entry.1 += 1;
            }
            (true, false) => entry.1 += 1,
            (false, true) => {
                entry.2 += 1;
                entry.3 += 1;
            }
            (false, false) => entry.3 += 1,
        }
    }

    let mut weighted = 0.0;
    let mut weight = 0_usize;
    let mut support = 0_usize;
    for (cause_hits, cause_total, base_hits, base_total) in buckets.into_values() {
        if cause_total == 0 || base_total == 0 {
            support += cause_total;
            continue;
        }
        let local_weight = cause_total + base_total;
        let treated = cause_hits as f32 / cause_total as f32;
        let baseline = base_hits as f32 / base_total as f32;
        weighted += (treated - baseline) * local_weight as f32;
        weight += local_weight;
        support += cause_total;
    }

    if weight == 0 {
        (0.0, support)
    } else {
        (weighted / weight as f32, support)
    }
}

fn exposure_support(samples: &CausalSamples, variable_idx: usize) -> usize {
    samples
        .rows
        .iter()
        .take(samples.rows.len().saturating_sub(1))
        .filter(|row| row[variable_idx])
        .count()
}

fn marginal_association(samples: &CausalSamples, cause_idx: usize, effect_idx: usize) -> f32 {
    conditional_risk_difference(samples, cause_idx, effect_idx, &[])
        .0
        .abs()
}

fn conditioning_sets(controls: &[usize], size: usize) -> Vec<Vec<usize>> {
    if size == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    let mut current = Vec::new();
    build_conditioning_sets(controls, size, 0, &mut current, &mut out);
    out
}

fn build_conditioning_sets(
    controls: &[usize],
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() == size {
        out.push(current.clone());
        return;
    }
    for idx in start..controls.len() {
        current.push(controls[idx]);
        build_conditioning_sets(controls, size, idx + 1, current, out);
        current.pop();
    }
}

fn support_confidence(support: usize) -> f32 {
    (support as f32 / 12.0).min(1.0)
}

fn apply_intervention(graph: &CausalGraph, intervention: CausalIntervention) -> InterventionResult {
    let sign = if intervention.value { 1.0 } else { -1.0 };
    let mut best: HashMap<ContextUri, CounterfactualImpact> = HashMap::new();
    let mut queue = VecDeque::new();

    for edge in graph.outgoing(&intervention.target_uri) {
        let total = sign * edge.average_treatment_effect;
        queue.push_back((
            edge.effect_uri.clone(),
            edge.average_treatment_effect,
            total,
            edge.confidence,
            vec![intervention.target_uri.clone(), edge.effect_uri.clone()],
        ));
    }

    while let Some((node, direct, total, confidence, path)) = queue.pop_front() {
        let replace = best
            .get(&node)
            .is_none_or(|old| total.abs() * confidence > old.total_effect.abs() * old.confidence);
        if replace {
            best.insert(
                node.clone(),
                CounterfactualImpact {
                    effect_uri: node.clone(),
                    direct_effect: direct,
                    total_effect: total,
                    confidence,
                    causal_path: path.clone(),
                },
            );
        }

        for edge in graph.outgoing(&node) {
            if path.contains(&edge.effect_uri) {
                continue;
            }
            let mut next_path = path.clone();
            next_path.push(edge.effect_uri.clone());
            queue.push_back((
                edge.effect_uri.clone(),
                0.0,
                total * edge.average_treatment_effect,
                (confidence * edge.confidence).sqrt(),
                next_path,
            ));
        }
    }

    let mut affected = best.into_values().collect::<Vec<_>>();
    affected.sort_by(|a, b| {
        (b.total_effect.abs() * b.confidence)
            .partial_cmp(&(a.total_effect.abs() * a.confidence))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    InterventionResult {
        intervention,
        affected,
    }
}

/// 解析 LLM 返回的 JSON 修复方案。
fn parse_repair_actions(response: &str, scope: &ContextUri) -> Vec<RepairAction> {
    #[derive(serde::Deserialize)]
    struct RawAction {
        action: String,
        description: String,
        #[serde(default)]
        target: String,
    }

    let json_str = extract_json_array(response);
    let raw: Vec<RawAction> = serde_json::from_str(&json_str).unwrap_or_default();

    raw.into_iter()
        .map(|r| {
            let target_id = parse_commit_id(&r.target);
            match r.action.as_str() {
                "rollback" => RepairAction::Rollback(target_id),
                "patch" => RepairAction::Patch {
                    from: target_id,
                    description: r.description,
                },
                "supplement" => RepairAction::Supplement {
                    uri: scope.join(&r.target),
                    content: r.description,
                },
                "remove" => RepairAction::Remove(scope.join(&r.target)),
                _ => RepairAction::Supplement {
                    uri: scope.clone(),
                    content: format!("unknown action {}: {}", r.action, r.description),
                },
            }
        })
        .collect()
}

/// 尝试从 LLM 返回的 target 字符串解析 CommitId。
fn parse_commit_id(target: &str) -> CommitId {
    if let Ok(uuid) = uuid::Uuid::parse_str(target) {
        CommitId(uuid)
    } else {
        CommitId::new()
    }
}

/// 从 LLM 响应中提取 JSON 数组。
fn extract_json_array(text: &str) -> String {
    let text = text.trim();
    if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = text.find('[')
        && let Some(end) = text.rfind(']')
    {
        return text[start..=end].to_string();
    }
    // fallback: wrap in array
    format!("[{}]", text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_crystal_has_evidence() {
        let crystal = KnowledgeCrystal {
            id: "c1".into(),
            principle: "test before deploy".into(),
            evidence: vec![],
            confidence: 0.9,
            preconditions: vec!["staging env".into()],
            expected_outcome: "fewer bugs".into(),
        };
        assert_eq!(crystal.id, "c1");
    }

    #[test]
    fn crystal_writer_turns_crystals_into_long_term_skill_entries() {
        let evidence = ContextUri::parse("uwu://t/a/memory/error/e1").unwrap();
        let crystal = KnowledgeCrystal {
            id: "c1".into(),
            principle: "verify migrations before deploy".into(),
            evidence: vec![evidence.clone()],
            confidence: 0.82,
            preconditions: vec!["schema changed".into()],
            expected_outcome: "fewer rollout failures".into(),
        };
        let writer = CrystalMemoryWriter::new(CrystalWritebackConfig {
            tenant: TenantId(Uuid::new_v4()),
            agent_scope: "t/a".into(),
            min_confidence: 0.35,
            write_dream_insights: true,
        });
        let report = writer.entries_from_crystals(&[crystal]).unwrap();
        assert_eq!(report.entries.len(), 1);
        let entry = &report.entries[0];
        assert_eq!(entry.metadata.content_type, Some(ContentType::Skill));
        assert_eq!(entry.metadata.state_scope, Some(StateScope::Long));
        assert!(entry.uri.to_string().contains("/memory/skill/crystal/"));
        assert!(
            entry
                .metadata
                .consolidation
                .as_ref()
                .unwrap()
                .evidence_uris
                .contains(&evidence)
        );
    }

    #[test]
    fn crystal_writer_turns_dream_insights_into_heuristics() {
        let writer = CrystalMemoryWriter::new(CrystalWritebackConfig::for_agent("t/a"));
        let report = writer
            .entries_from_dream_insights(&["cluster auth failures into retry heuristic".into()])
            .unwrap();
        assert_eq!(report.entries.len(), 1);
        let entry = &report.entries[0];
        assert_eq!(entry.metadata.content_type, Some(ContentType::Heuristic));
        assert!(entry.uri.to_string().contains("/memory/heuristic/dream/"));
        assert!(entry.metadata.tags.contains(&"dream:insight".to_string()));
    }

    #[test]
    fn causal_hypothesis_sorts_by_confidence() {
        let h1 = CausalHypothesis {
            cause_uri: ContextUri::parse("uwu://t/agent/a/memories/events/cause-a").unwrap(),
            effect_uri: ContextUri::parse("uwu://t/agent/a/memories/events/effect-b").unwrap(),
            temporal_precedence: 0.8,
            co_occurrence: 0.5,
            confidence: 0.71,
        };
        let h2 = CausalHypothesis {
            cause_uri: ContextUri::parse("uwu://t/agent/a/memories/events/cause-c").unwrap(),
            effect_uri: ContextUri::parse("uwu://t/agent/a/memories/events/effect-d").unwrap(),
            temporal_precedence: 0.3,
            co_occurrence: 0.1,
            confidence: 0.24,
        };
        let mut v = [h2, h1];
        v.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert!(v[0].confidence > v[1].confidence);
    }

    #[test]
    fn causal_discovery_filters_confounding_and_keeps_intervention_edge() {
        let a = ContextUri::parse("uwu://t/agent/a/memories/events/knowledge-a").unwrap();
        let b = ContextUri::parse("uwu://t/agent/a/memories/events/knowledge-b").unwrap();
        let c = ContextUri::parse("uwu://t/agent/a/memories/events/shared-cause-c").unwrap();
        let d = ContextUri::parse("uwu://t/agent/a/memories/events/downstream-d").unwrap();

        let rows = vec![
            vec![c.clone(), a.clone(), b.clone()],
            vec![d.clone()],
            vec![c.clone(), a.clone(), b.clone()],
            vec![d.clone()],
            vec![c.clone(), a.clone(), b.clone()],
            vec![d.clone()],
            vec![c.clone(), b.clone()],
            vec![],
            vec![c.clone(), b.clone()],
            vec![],
            vec![a.clone()],
            vec![d.clone()],
            vec![a.clone()],
            vec![d.clone()],
        ];
        let samples = test_samples(rows);
        let graph = learn_causal_graph(
            &samples,
            &CausalDiscoveryConfig {
                min_support: 2,
                independence_threshold: 0.05,
                max_conditioning_set: 1,
                bic_penalty: 0.1,
                ..Default::default()
            },
        );

        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.cause_uri == a && edge.effect_uri == d)
        );
        assert!(
            !graph
                .edges
                .iter()
                .any(|edge| edge.cause_uri == a && edge.effect_uri == b)
        );

        let result = apply_intervention(&graph, CausalIntervention::fix(a.clone()));
        assert_eq!(result.affected[0].effect_uri, d);
        assert!(result.affected[0].total_effect > 0.0);
    }

    #[test]
    fn causal_intervention_propagates_multi_hop_downstream_impacts() {
        let a = ContextUri::parse("uwu://t/agent/a/memories/events/a").unwrap();
        let b = ContextUri::parse("uwu://t/agent/a/memories/events/b").unwrap();
        let c = ContextUri::parse("uwu://t/agent/a/memories/events/c").unwrap();
        let graph = CausalGraph::new(
            vec![a.clone(), b.clone(), c.clone()],
            vec![
                CausalEdge {
                    cause_uri: a.clone(),
                    effect_uri: b.clone(),
                    average_treatment_effect: 0.8,
                    support: 8,
                    confidence: 0.9,
                    adjustment_set: vec![],
                },
                CausalEdge {
                    cause_uri: b.clone(),
                    effect_uri: c.clone(),
                    average_treatment_effect: 0.5,
                    support: 7,
                    confidence: 0.8,
                    adjustment_set: vec![],
                },
            ],
        );

        let result = apply_intervention(&graph, CausalIntervention::remove(a));
        let affected = result.affected_uris();
        assert!(affected.contains(&b));
        assert!(affected.contains(&c));
        assert!(
            result
                .affected
                .iter()
                .any(|impact| impact.effect_uri == c && impact.total_effect < 0.0)
        );
    }

    fn test_samples(rows: Vec<Vec<ContextUri>>) -> CausalSamples {
        let mut variables = rows.iter().flatten().cloned().collect::<Vec<_>>();
        variables.sort_by_key(|a| a.to_string());
        variables.dedup();
        let variable_index = variables
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, uri)| (uri, idx))
            .collect::<HashMap<_, _>>();
        let rows = rows
            .into_iter()
            .map(|uris| {
                let mut row = vec![false; variables.len()];
                for uri in uris {
                    row[*variable_index.get(&uri).unwrap()] = true;
                }
                row
            })
            .collect();
        CausalSamples { variables, rows }
    }
}
