//! LineageTracker + PhylogenyTree — 知识演化树追踪。

use agent_context_db_core::{ContextEntry, ContextUri, MvccVersion};
use chrono::{DateTime, Utc};

/// 血统追踪器。
pub struct LineageTracker;

/// 演化节点。
#[derive(Debug, Clone)]
pub struct PhylogenyNode {
    pub uri: ContextUri,
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub change_summary: String,
    pub children: Vec<PhylogenyNode>,
}

/// 系统发生树。
#[derive(Debug, Clone)]
pub struct PhylogenyTree {
    pub root: PhylogenyNode,
    pub total_nodes: usize,
    pub depth: usize,
}

impl LineageTracker {
    pub fn new() -> Self { Self }

    /// 记录一条血统变更。
    pub fn record_lineage(
        parent: &ContextEntry,
        child: &ContextEntry,
        change_summary: &str,
    ) -> PhylogenyNode {
        PhylogenyNode {
            uri: child.uri.clone(),
            version: child.mvcc_version.0,
            timestamp: child.created_at,
            change_summary: change_summary.to_string(),
            children: vec![],
        }
    }

    /// 从演化链构建系统发生树。
    pub fn build_tree(root: &ContextEntry, lineage: &[PhylogenyNode]) -> PhylogenyTree {
        let mut root_node = PhylogenyNode {
            uri: root.uri.clone(),
            version: 0,
            timestamp: Utc::now(),
            change_summary: "root".into(),
            children: lineage.to_vec(),
        };

        PhylogenyTree {
            total_nodes: 1 + lineage.len(),
            depth: 1 + lineage.len(),
            root: root_node.clone(),
        }
    }

    /// 格式化为可读的演化历史。
    pub fn format_tree(tree: &PhylogenyTree) -> String {
        let mut out = format!(
            "Phylogeny for {}\nTotal nodes: {}, depth: {}\n",
            tree.root.uri, tree.total_nodes, tree.depth
        );
        for node in &tree.root.children {
            out.push_str(&format!(
                "  v{} — {} ({})\n",
                node.version, node.change_summary, node.timestamp
            ));
        }
        out
    }
}
