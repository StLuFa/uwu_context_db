//! 可解释性血统 — 证据树追溯到原始 session。

use agent_context_db_core::{ContextEntry, ContextUri, Result};
use chrono::{DateTime, Utc};

/// 证据树 — 从产物追溯到原始证据的完整链路。
#[derive(Debug, Clone)]
pub struct EvidenceTree {
    pub root_uri: ContextUri,
    pub root_principle: String,
    pub evidence_chain: Vec<EvidenceNode>,
    pub evolution: Vec<EvolutionStep>,
}

/// 证据节点。
#[derive(Debug, Clone)]
pub struct EvidenceNode {
    pub uri: ContextUri,
    pub content_summary: String,
    pub source_session: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub children: Vec<EvidenceNode>,
}

/// 演化步骤。
#[derive(Debug, Clone)]
pub struct EvolutionStep {
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub change_summary: String,
}

/// 可解释性血统追踪器。
pub struct ExplainableLineage;

impl ExplainableLineage {
    pub fn new() -> Self { Self }

    /// 构建证据树 — 从产物沿 evidence_uris 递归追溯。
    pub fn explain(
        &self,
        product_uri: &ContextUri,
        content: &str,
        evidence_uris: &[ContextUri],
    ) -> EvidenceTree {
        let children: Vec<EvidenceNode> = evidence_uris
            .iter()
            .map(|uri| EvidenceNode {
                uri: uri.clone(),
                content_summary: String::new(),
                source_session: None,
                timestamp: None,
                children: vec![],
            })
            .collect();

        EvidenceTree {
            root_uri: product_uri.clone(),
            root_principle: content.to_string(),
            evidence_chain: children,
            evolution: vec![],
        }
    }

    /// 添加演化步骤到证据树。
    pub fn with_evolution(
        &self,
        mut tree: EvidenceTree,
        steps: Vec<(u64, DateTime<Utc>, String)>,
    ) -> EvidenceTree {
        tree.evolution = steps
            .into_iter()
            .map(|(version, timestamp, change_summary)| EvolutionStep {
                version,
                timestamp,
                change_summary,
            })
            .collect();
        tree
    }

    /// 格式化为人类可读的解释文本。
    pub fn format(&self, tree: &EvidenceTree) -> String {
        let mut out = format!(
            "Evidence trace for: {}\nPrinciple: {}\n\n",
            tree.root_uri, tree.root_principle
        );
        out.push_str("Evidence chain:\n");
        for (i, node) in tree.evidence_chain.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} — {}\n",
                i + 1,
                node.uri,
                node.content_summary
            ));
        }
        if !tree.evolution.is_empty() {
            out.push_str("\nEvolution:\n");
            for step in &tree.evolution {
                out.push_str(&format!(
                    "  v{} ({}) — {}\n",
                    step.version, step.timestamp, step.change_summary
                ));
            }
        }
        out
    }
}
