//! 可解释性血统 — 证据树追溯到原始 session。

use agent_context_db_core::{ContentLevel, ContextUri, FsOps, Result};
use chrono::{DateTime, Utc};
use std::sync::Arc;

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
    fs: Option<Arc<dyn FsOps>>,
    max_depth: usize,
}

impl ExplainableLineage {
    pub fn new() -> Self {
        Self { fs: None, max_depth: 3 }
    }

    pub fn with_fs(fs: Arc<dyn FsOps>) -> Self {
        Self { fs: Some(fs), max_depth: 3 }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
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
    /// - 从 URI 中提取 session_id（`uwu://.../s/{session}/x/...`）
    /// - 从 payload 中提取 sparse text 作为 content_summary
    /// - 若 fs 未注入 → 退化为 `explain()`
    pub async fn explain_async(
        &self,
        product_uri: &ContextUri,
        content: &str,
        evidence_uris: &[ContextUri],
    ) -> Result<EvidenceTree> {
        let fs = match &self.fs {
            Some(f) => f,
            None => return Ok(self.explain(product_uri, content, evidence_uris)),
        };

        let mut children = Vec::with_capacity(evidence_uris.len());
        for uri in evidence_uris {
            let node = self.build_node(fs.as_ref(), uri, 0).await;
            children.push(node);
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
        fs: &'a dyn FsOps,
        uri: &'a ContextUri,
        depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = EvidenceNode> + Send + 'a>> {
        Box::pin(async move {
            let (summary, timestamp) = match fs.read(uri, ContentLevel::L0).await {
                Ok(payload) => {
                    let text = payload.sparse_text().to_string();
                    let trimmed = if text.len() > 240 {
                        format!("{}…", &text[..240])
                    } else {
                        text
                    };
                    (trimmed, None)
                }
                Err(_) => (String::new(), None),
            };
            let session = extract_session_id(uri);
            // 递归 —— DerivedFrom / EvidenceOf 子引用暂不透过 FsOps 拿到，
            // 后续可接入 GraphStore 遍历。此处仅在 depth < max_depth 时占位。
            let children = if depth + 1 < self.max_depth {
                vec![]
            } else {
                vec![]
            };
            EvidenceNode {
                uri: uri.clone(),
                content_summary: summary,
                source_session: session,
                timestamp,
                children,
            }
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
        for (i, node) in tree.evidence_chain.iter().enumerate() {
            let src = node.source_session.as_deref().unwrap_or("-");
            out.push_str(&format!(
                "  {}. {} [session={}] — {}\n",
                i + 1,
                node.uri,
                src,
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

