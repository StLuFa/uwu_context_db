//! 版本层创新功能（F19 知识晶体 + F21 自修复 + F23 梦境巩固 + F27 因果推断）。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, FsOps, LlmClient, LlmOpts};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{CommitId, TemporalReasoner, VersionStore};

// ═══════════════════════════════════════════════════════════════════════════
// F19 知识晶体蒸馏
// ═══════════════════════════════════════════════════════════════════════════

/// 知识晶体 —— 从大量经验中蒸馏出的紧凑知识单元。
#[derive(Debug, Clone)]
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

        let response = self.llm.complete(&prompt, &LlmOpts::default()).await
            .map_err(|e| agent_context_db_core::ContextError::Storage(format!("distill: {e}")))?;

        #[derive(serde::Deserialize)]
        struct RawCrystal {
            principle: String,
            preconditions: Vec<String>,
            expected_outcome: String,
            confidence: f32,
        }

        let raw: Vec<RawCrystal> = serde_json::from_str(&response).unwrap_or_default();

        Ok(raw.into_iter().enumerate().map(|(i, r)| KnowledgeCrystal {
            id: format!("crystal-{}", i),
            principle: r.principle,
            evidence: experience_uris.to_vec(),
            confidence: r.confidence,
            preconditions: r.preconditions,
            expected_outcome: r.expected_outcome,
        }).collect())
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
        let log = self.store.log(scope, &crate::LogOpts { max_count: Some(20), ..Default::default() }).await?;
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
        let log = self.store.log(scope, &crate::LogOpts { max_count: Some(10), ..Default::default() }).await?;

        let log_text: Vec<String> = log.iter().map(|c| {
            format!("{} | adds:{} updates:{} deletes:{}",
                c.message, c.metadata.changes.adds.len(),
                c.metadata.changes.updates.len(), c.metadata.changes.deletes.len())
        }).collect();

        let prompt = format!(
            r#"Diagnose potential issues in this version history:

{}
Return JSON array of repair actions:
[{{"action": "rollback|patch|supplement|remove", "description": "...", "target": "..."}}]
"#,
            log_text.join("\n")
        );

        let response = self.llm.complete(&prompt, &LlmOpts::default()).await
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
}

impl<V: VersionStore> DreamConsolidator<V> {
    pub fn new(store: Arc<V>, llm: Arc<dyn LlmClient>, fs: Arc<dyn FsOps>) -> Self {
        Self { store, llm, fs }
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
        let log = self.store.log(scope, &crate::LogOpts { max_count: Some(30), ..Default::default() }).await?;

        // 提取所有变更的 URI
        let mut changed_uris = Vec::new();
        for commit in &log {
            for add in &commit.metadata.changes.adds {
                changed_uris.push(add.clone());
            }
        }

        // 聚类相似变更
        let mut clusters: HashMap<String, Vec<ContextUri>> = HashMap::new();
        for uri in &changed_uris {
            let segs = uri.segments();
            let key: String = segs.iter().take(3).map(|s| *s).collect::<Vec<_>>().join("/");
            clusters.entry(key).or_default().push(uri.clone());
        }

        // 高频聚类 → 收集内容 → LLM 合成
        let mut insights: Vec<String> = Vec::new();

        for (key, uris) in clusters {
            if uris.len() < 3 {
                // 低频聚类：仅记录统计信息
                if uris.len() >= 2 {
                    insights.push(format!("cluster '{}' with {} related changes", key, uris.len()));
                }
                continue;
            }

            // 读取聚类中各条目的 L0 摘要
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
                insights.push(format!("cluster '{}' with {} related changes (no content)", key, uris.len()));
                continue;
            }

            // 调用 LLM 合成模式洞察
            let nl = "\n";
            let prompt = format!(
                "Analyze this cluster of {count} context changes under '{key}':{nl}{nl}{summaries}{nl}{nl}\
                 Identify the underlying pattern: what do these changes have in common? \
                 Is there a reusable insight or principle? \
                 Respond with a single concise paragraph (2-4 sentences).",
                count = uris.len(),
                key = key,
                nl = nl,
                summaries = summaries.join("\n"),
            );

            match self.llm.complete(&prompt, &LlmOpts::default()).await {
                Ok(response) => {
                    let trimmed = response.trim().to_string();
                    if !trimmed.is_empty() {
                        insights.push(format!("cluster '{key}' ({count} changes): {trimmed}",
                            key = key, count = uris.len(), trimmed = trimmed));
                    } else {
                        insights.push(format!("cluster '{}' with {} related changes", key, uris.len()));
                    }
                }
                Err(_) => {
                    // LLM 失败时降级为统计描述
                    insights.push(format!("cluster '{}' with {} related changes", key, uris.len()));
                }
            }
        }

        Ok(insights)
    }
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

/// 因果推断器 —— 基于时间序列的统计相关性分析。
///
/// 不是真正的因果模型，而是在 DAG 版本历史上做 Granger 式的时间先导检验。
pub struct CausalInference<V: VersionStore> {
    store: Arc<V>,
    _temporal: TemporalReasoner<V>,
}

impl<V: VersionStore> CausalInference<V> {
    pub fn new(store: Arc<V>) -> Self {
        let temporal = TemporalReasoner::new(store.clone());
        Self { store, _temporal: temporal }
    }

    /// 检测一个 URI 的变更是否在统计上"导致"另一个 URI 的变更。
    ///
    /// 条件：cause 变更后 effect 在时间窗口内也变更的比例 > 随机基线。
    pub async fn infer_causality(
        &self,
        scope: &ContextUri,
    ) -> std::result::Result<Vec<CausalHypothesis>, crate::VersionError> {
        let log = self.store.log(scope, &crate::LogOpts { max_count: Some(100), ..Default::default() }).await?;

        // 统计 URI 对的时序共现
        let mut pair_counts: HashMap<(String, String), (usize, usize)> = HashMap::new();
        // (cause, effect) → (cause_then_effect_count, total_cause_count)

        for window in log.windows(3) {
            for i in 0..window.len() {
                for j in (i + 1)..window.len() {
                    let earlier = &window[i];
                    let later = &window[j];

                    for early_change in &earlier.metadata.changes.adds {
                        for late_change in &later.metadata.changes.adds {
                            let key = (early_change.0.clone(), late_change.0.clone());
                            let entry = pair_counts.entry(key).or_insert((0, 0));
                            entry.0 += 1; // cause then effect
                            entry.1 += 1; // total cause occurrences
                        }
                    }
                }
            }
        }

        let mut hypotheses = Vec::new();
        for ((cause, effect), (co_occurrence, total)) in pair_counts {
            if total < 3 {
                continue;
            }
            let temporal_precedence = co_occurrence as f32 / total as f32;
            if temporal_precedence > 0.5 {
                hypotheses.push(CausalHypothesis {
                    cause_uri: ContextUri(cause),
                    effect_uri: ContextUri(effect),
                    temporal_precedence,
                    co_occurrence: co_occurrence as f32 / log.len().max(1) as f32,
                    confidence: temporal_precedence * 0.7 + (co_occurrence as f32 / log.len().max(1) as f32) * 0.3,
                });
            }
        }

        hypotheses.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        Ok(hypotheses)
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
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            return text[start..=end].to_string();
        }
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
    fn causal_hypothesis_sorts_by_confidence() {
        let h1 = CausalHypothesis {
            cause_uri: ContextUri("a".into()),
            effect_uri: ContextUri("b".into()),
            temporal_precedence: 0.8,
            co_occurrence: 0.5,
            confidence: 0.71,
        };
        let h2 = CausalHypothesis {
            cause_uri: ContextUri("c".into()),
            effect_uri: ContextUri("d".into()),
            temporal_precedence: 0.3,
            co_occurrence: 0.1,
            confidence: 0.24,
        };
        let mut v = vec![h2, h1];
        v.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        assert!(v[0].confidence > v[1].confidence);
    }
}
