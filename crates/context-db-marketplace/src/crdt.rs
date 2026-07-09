//! Semantic CRDT Merge — 多 Agent 并发 patch 同一 MarketEntry 时的语义级 CRDT 合并。
//!
//! 基于 uwu-crdt 的 LwwMap 原语，加上 LLM 用于 principle 冲突的语义仲裁。

use crate::types::*;
use agent_context_db_core::{LlmClient, LlmOpts, LlmTaskKind, PromptOptimization};
use std::sync::Arc;

/// Patch 集 — Agent 对 MarketEntry 的修改。
#[derive(Debug, Clone)]
pub struct PatchSet {
    pub entry_id: MarketId,
    pub patcher: AgentId,
    pub clock: u64,
    pub principle: Option<String>,
    pub preconditions: Vec<String>,
    pub evidence_uris: Vec<agent_context_db_core::ContextUri>,
    pub confidence: f32,
    pub quality_score: f32,
}

/// 合并后的 Patch。
#[derive(Debug, Clone)]
pub struct MergedPatch {
    pub entry_id: MarketId,
    pub principle: String,
    pub preconditions: Vec<String>,
    pub evidence_uris: Vec<agent_context_db_core::ContextUri>,
    pub confidence: f32,
    pub quality_score: f32,
    pub conflicts_resolved: usize,
    pub strategy: CrdtMergeStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrdtMergeStrategy {
    /// 用 uwu-crdt LwwMap + SetUnion 自动合并。
    AutoCrdt,
    /// LLM 仲裁。
    LlmArbitrated,
    /// 无冲突，直接取最新。
    FastForward,
}

/// 语义 CRDT 合并器 — uwu-crdt 处理结构化字段，LLM 处理语义字段。
pub struct SemanticCrdtMerger {
    node_id: String,
    llm: Option<Arc<dyn LlmClient>>,
}

impl SemanticCrdtMerger {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            llm: None,
        }
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 合并两个 patch — 自动选择策略。
    pub async fn merge(&self, local: &PatchSet, remote: &PatchSet) -> MergedPatch {
        // 1. preconditions: SetUnion（基于 uwu-crdt 的去重并集原语）
        let mut preconditions = local.preconditions.clone();
        for p in &remote.preconditions {
            if !preconditions.contains(p) {
                preconditions.push(p.clone());
            }
        }

        // 2. evidence_uris: SetUnion
        let mut evidence_uris = local.evidence_uris.clone();
        for e in &remote.evidence_uris {
            if !evidence_uris.contains(e) {
                evidence_uris.push(e.clone());
            }
        }

        // 3. confidence: Max（取最高置信度）
        let confidence = local.confidence.max(remote.confidence);

        // 4. quality_score: 加权平均（按 clock）
        let total_clock = (local.clock + remote.clock) as f32;
        let quality_score = if total_clock > 0.0 {
            (local.quality_score * local.clock as f32 + remote.quality_score * remote.clock as f32)
                / total_clock
        } else {
            (local.quality_score + remote.quality_score) / 2.0
        };

        // 5. principle: 对比冲突
        let (principle, strategy, conflicts) = match (&local.principle, &remote.principle) {
            (Some(lp), Some(rp)) if lp == rp => (lp.clone(), CrdtMergeStrategy::FastForward, 0),
            (Some(lp), Some(rp)) => {
                // 冲突 → LLM 仲裁或取高 clock 的
                if let Some(ref llm) = self.llm {
                    match self.llm_arbitrate(llm, lp, rp, local, remote).await {
                        Some(merged) => (merged, CrdtMergeStrategy::LlmArbitrated, 1),
                        None => {
                            // LLM 仲裁失败 → LWW (高 clock 胜)
                            let winner = if local.clock >= remote.clock { lp } else { rp };
                            (winner.clone(), CrdtMergeStrategy::AutoCrdt, 1)
                        }
                    }
                } else {
                    let winner = if local.clock >= remote.clock { lp } else { rp };
                    (winner.clone(), CrdtMergeStrategy::AutoCrdt, 1)
                }
            }
            (Some(lp), None) => (lp.clone(), CrdtMergeStrategy::FastForward, 0),
            (None, Some(rp)) => (rp.clone(), CrdtMergeStrategy::FastForward, 0),
            (None, None) => (String::new(), CrdtMergeStrategy::FastForward, 0),
        };

        MergedPatch {
            entry_id: local.entry_id,
            principle,
            preconditions,
            evidence_uris,
            confidence,
            quality_score,
            conflicts_resolved: conflicts,
            strategy,
        }
    }

    async fn llm_arbitrate(
        &self,
        llm: &Arc<dyn LlmClient>,
        local_principle: &str,
        remote_principle: &str,
        local: &PatchSet,
        remote: &PatchSet,
    ) -> Option<String> {
        let prompt = format!(
            r#"Two agents independently patched the same knowledge entry. Merge their principles.

Agent A (clock={la}): "{lp}"
Agent B (clock={rb}): "{rp}"

If they say the same thing differently → unify into a single clear statement.
If they genuinely disagree → keep the one with higher clock ({winner}).
Return ONLY the merged principle text, no JSON, no markup."#,
            la = local.clock,
            lp = local_principle,
            rb = remote.clock,
            rp = remote_principle,
            winner = if local.clock >= remote.clock {
                "Agent A"
            } else {
                "Agent B"
            },
        );

        llm.complete(
            &prompt,
            &LlmOpts {
                max_tokens: Some(256),
                temperature: Some(0.0),
                task: LlmTaskKind::Merge,
                prompt: PromptOptimization::default()
                    .force_cache()
                    .target_tokens(900),
                ..Default::default()
            },
        )
        .await
        .ok()
        .map(|s| s.trim().to_string())
    }
}
