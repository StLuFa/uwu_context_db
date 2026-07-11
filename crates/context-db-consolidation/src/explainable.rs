//! 可解释性血统 — 证据树追溯到原始 session。

use agent_context_db_core::{ContentLevel, ContextUri, FsOps, GraphRelation, GraphStore, Result};
use chrono::{DateTime, Utc};
use std::{collections::HashSet, sync::Arc};

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
///
/// 无 `fs`：仅生成占位证据链（URI-only）。
/// 注入 `fs`：`explain_async` 会加载 L0 摘要 + 递归展开子证据。
pub struct ExplainableLineage {
    fs: Arc<dyn FsOps>,
    graph: Arc<dyn GraphStore>,
    max_depth: usize,
}

impl ExplainableLineage {
    pub fn new(
        fs: Arc<dyn FsOps>,
        graph: Arc<dyn GraphStore>,
        max_depth: usize,
    ) -> std::result::Result<Self, crate::ConfigError> {
        if max_depth == 0 {
            return Err(crate::ConfigError("max_depth must be non-zero".into()));
        }
        Ok(Self {
            fs,
            graph,
            max_depth,
        })
    }

    /// 同步版：仅构造占位节点（URI-only），不加载内容。
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

    /// 异步版：注入 `fs` 后，从存储加载每条证据的 L0 摘要。
    ///
    /// - 加载 evidence L0 payload
    /// - 从 URI 中提取 session_id（`uwu://.../s/{session}/memory/...`）
    /// - 从 payload 中提取 sparse text 作为 content_summary
    /// - 若 fs 未注入 → 退化为 `explain()`
    pub async fn explain_async(
        &self,
        product_uri: &ContextUri,
        content: &str,
        evidence_uris: &[ContextUri],
    ) -> Result<EvidenceTree> {
        let mut roots = evidence_uris.to_vec();
        roots.sort_by_key(ToString::to_string);
        roots.dedup();
        let mut visited = HashSet::new();
        let mut children = Vec::with_capacity(roots.len());
        for uri in roots {
            if visited.insert(uri.clone()) {
                children.push(self.build_node(&uri, 0, &mut visited).await?);
            }
        }

        Ok(EvidenceTree {
            root_uri: product_uri.clone(),
            root_principle: content.to_string(),
            evidence_chain: children,
            evolution: vec![],
        })
    }

    fn build_node<'a>(
        &'a self,
        uri: &'a ContextUri,
        depth: usize,
        visited: &'a mut HashSet<ContextUri>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<EvidenceNode>> + Send + 'a>>
    {
        Box::pin(async move {
            let payload = self.fs.read(uri, ContentLevel::L0).await?;
            let summary: String = payload.sparse_text().chars().take(240).collect();
            let summary = if payload.sparse_text().chars().count() > 240 {
                format!("{summary}…")
            } else {
                summary
            };
            let mut children = Vec::new();
            if depth < self.max_depth {
                let mut candidates = self
                    .graph
                    .outgoing_neighbors(uri, Some(GraphRelation::DerivedFrom))
                    .await?;
                candidates.extend(
                    self.graph
                        .incoming_neighbors(uri, Some(GraphRelation::EvidenceOf))
                        .await?,
                );
                candidates.sort_by_key(ToString::to_string);
                candidates.dedup();
                for child in candidates {
                    if visited.insert(child.clone()) {
                        children.push(self.build_node(&child, depth + 1, visited).await?);
                    }
                }
            }
            Ok(EvidenceNode {
                uri: uri.clone(),
                content_summary: summary,
                source_session: extract_session_id(uri),
                timestamp: None,
                children,
            })
        })
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
        fn format_node(out: &mut String, node: &EvidenceNode, depth: usize, ordinal: usize) {
            let indent = "  ".repeat(depth + 1);
            let src = node.source_session.as_deref().unwrap_or("-");
            out.push_str(&format!(
                "{indent}{ordinal}. {} [session={src}] — {}\n",
                node.uri, node.content_summary
            ));
            for (index, child) in node.children.iter().enumerate() {
                format_node(out, child, depth + 1, index + 1);
            }
        }
        for (index, node) in tree.evidence_chain.iter().enumerate() {
            format_node(&mut out, node, 0, index + 1);
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

/// 从 URI 路径中提取 session id。约定：`uwu://t/{tenant}/a/{agent}/s/{session}/...`。
fn extract_session_id(uri: &ContextUri) -> Option<String> {
    let s = uri.to_string();
    let mut it = s.split('/');
    while let Some(seg) = it.next() {
        if seg == "s" {
            return it.next().map(|x| x.to_string());
        }
    }
    None
}
