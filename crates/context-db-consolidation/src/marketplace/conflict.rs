//! ConflictResolver — 多维证据对比 + LLM 仲裁 + Resolution 类型。

use crate::marketplace::types::*;
use agent_context_db_core::{ContextUri, LlmClient, LlmOpts};
use std::sync::Arc;

/// 冲突 — 两个 MarketEntry 对同一事实给出矛盾结论。
#[derive(Debug, Clone)]
pub struct MarketConflict {
    pub entry_a: MarketEntry,
    pub entry_b: MarketEntry,
    pub conflict_type: ConflictType,
    pub similarity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// 直接语义矛盾（A说X对，B说X错）
    DirectContradiction,
    /// 部分重叠但结论不同
    Overlapping,
    /// 同一证据得出不同结论
    SameEvidenceDifferentConclusion,
}

/// 仲裁结果。
#[derive(Debug, Clone)]
pub enum Resolution {
    /// 保留 A，让 B 过期。
    KeepA { reason: String },
    /// 保留 B，让 A 过期。
    KeepB { reason: String },
    /// 两者合并。
    Fuse { merged_content: String, reason: String },
    /// 交由人工审议。
    DeferToHuman { reason: String },
    /// 两者都保留（不同上下文适用）。
    KeepBoth { reason: String },
}

/// 冲突仲裁器。
pub struct ConflictResolver {
    llm: Option<Arc<dyn LlmClient>>,
}

impl ConflictResolver {
    pub fn new() -> Self {
        Self { llm: None }
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 检测两个条目是否存在冲突。
    pub fn detect(&self, a: &MarketEntry, b: &MarketEntry) -> Option<MarketConflict> {
        // 只检测同领域、同类型的条目
        if a.domain != b.domain { return None; }
        if a.entry_type != b.entry_type { return None; }

        // Jaccard 重叠度
        let words_a: std::collections::HashSet<&str> = a.principle.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b.principle.split_whitespace().collect();
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        let jaccard = if union > 0 { intersection as f32 / union as f32 } else { 0.0 };

        // 高重叠度 + 低置信度 → 可能是重复
        if jaccard > 0.8 {
            return None; // 大概率是同一知识，不需要仲裁
        }

        // 中高重叠度 + 否定词 → 可能矛盾
        if jaccard > 0.4 && has_contradiction_marker(&a.principle, &b.principle) {
            // 确定冲突类型
            let common_evidence = a.evidence_uris.iter().any(|u| b.evidence_uris.contains(u));
            let ct = if common_evidence {
                ConflictType::SameEvidenceDifferentConclusion
            } else if jaccard > 0.6 {
                ConflictType::DirectContradiction
            } else {
                ConflictType::Overlapping
            };

            return Some(MarketConflict {
                entry_a: a.clone(),
                entry_b: b.clone(),
                conflict_type: ct,
                similarity: jaccard,
            });
        }

        None
    }

    /// 批量检测。
    pub fn detect_all(&self, entries: &[MarketEntry]) -> Vec<MarketConflict> {
        let mut conflicts = Vec::new();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                if let Some(c) = self.detect(&entries[i], &entries[j]) {
                    conflicts.push(c);
                }
            }
        }
        conflicts
    }

    /// 仲裁冲突 — 基于多维证据对比。
    pub async fn arbitrate(&self, conflict: &MarketConflict) -> Resolution {
        // 1. 多维对比
        let a = &conflict.entry_a;
        let b = &conflict.entry_b;

        // 证据数量
        let a_evidence = a.evidence_uris.len();
        let b_evidence = b.evidence_uris.len();

        // 确认度
        let a_corrob = a.corroboration.independent_sources;
        let b_corrob = b.corroboration.independent_sources;

        // 质量分
        let a_quality = a.quality_score;
        let b_quality = b.quality_score;

        // 发布者声誉（在完整实现中查询 ReputationEngine）

        // 2. 如果一方显著优势 → 直接裁决
        if a_corrob > b_corrob * 2 && a_quality > b_quality + 0.2 {
            return Resolution::KeepA {
                reason: format!("{} corroborators vs {}, quality {:.2} vs {:.2}", a_corrob, b_corrob, a_quality, b_quality),
            };
        }
        if b_corrob > a_corrob * 2 && b_quality > a_quality + 0.2 {
            return Resolution::KeepB {
                reason: format!("{} corroborators vs {}, quality {:.2} vs {:.2}", b_corrob, a_corrob, b_quality, a_quality),
            };
        }

        // 3. LLM 仲裁
        if let Some(ref llm) = self.llm {
            let prompt = format!(
                r#"Two agents have published contradictory knowledge. As an impartial arbiter, determine the resolution.

Entry A (by {}): "{}"
  Evidence: {} URIs, {} corroborators, quality={:.2}

Entry B (by {}): "{}"
  Evidence: {} URIs, {} corroborators, quality={:.2}

Conflict type: {:?}

Respond with JSON:
{{"resolution": "keep_a"|"keep_b"|"fuse"|"defer"|"keep_both", "reason": "...", "merged": "..."}}"#,
                a.publisher, a.principle, a_evidence, a_corrob, a_quality,
                b.publisher, b.principle, b_evidence, b_corrob, b_quality,
                conflict.conflict_type,
            );

            if let Ok(response) = llm.complete(&prompt, &LlmOpts { max_tokens: Some(512), temperature: Some(0.0), ..Default::default() }).await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response) {
                    return match json["resolution"].as_str() {
                        Some("keep_a") => Resolution::KeepA { reason: json["reason"].as_str().unwrap_or("LLM arbitrated").into() },
                        Some("keep_b") => Resolution::KeepB { reason: json["reason"].as_str().unwrap_or("LLM arbitrated").into() },
                        Some("fuse") => Resolution::Fuse { merged_content: json["merged"].as_str().unwrap_or("").into(), reason: json["reason"].as_str().unwrap_or("").into() },
                        Some("keep_both") => Resolution::KeepBoth { reason: json["reason"].as_str().unwrap_or("").into() },
                        _ => Resolution::DeferToHuman { reason: "LLM response unclear".into() },
                    };
                }
            }
        }

        // 4. 无法自动裁决 → 人工
        Resolution::DeferToHuman {
            reason: format!("Evidence tie: A({}ev,{}cor,{:.2}q) vs B({}ev,{}cor,{:.2}q)", a_evidence, a_corrob, a_quality, b_evidence, b_corrob, b_quality),
        }
    }
}

fn has_contradiction_marker(a: &str, b: &str) -> bool {
    let negation_words = ["not", "no", "never", "impossible", "cannot", "can't", "don't", "false", "wrong", "incorrect", "不", "没有", "错误"];
    let a_neg = negation_words.iter().any(|n| a.to_lowercase().contains(n));
    let b_neg = negation_words.iter().any(|n| b.to_lowercase().contains(n));
    a_neg != b_neg
}
