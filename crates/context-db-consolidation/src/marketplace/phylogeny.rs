//! Cross-Agent Phylogeny — 追溯"这个知识最初是哪个 Agent 在哪个 session 学到的"。

use crate::marketplace::types::*;
use std::collections::{HashMap, HashSet};

/// 系统发生树节点。
#[derive(Debug, Clone)]
pub struct PhylogenyNode {
    pub entry_id: MarketId,
    pub publisher: AgentId,
    pub action: LineageAction,
    pub children: Vec<PhylogenyNode>,
    pub depth: usize,
}

/// 系统发生树 — 知识的完整演化史。
#[derive(Debug, Clone)]
pub struct PhylogeneticTree {
    pub root: PhylogenyNode,
    pub total_nodes: usize,
    pub max_depth: usize,
    pub agents_involved: Vec<AgentId>,
    /// 原始 Agent（最底层的 Origin）。
    pub origin_agent: AgentId,
    /// 创建时间。
    pub origin_time: chrono::DateTime<chrono::Utc>,
}

/// 跨 Agent 系统发生追踪器。
pub struct CrossAgentPhylogeny {
    /// 血统图缓存。
    lineages: parking_lot::RwLock<HashMap<MarketId, LineageNode>>,
}

impl CrossAgentPhylogeny {
    pub fn new() -> Self {
        Self {
            lineages: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// 注册一个血统节点。
    pub fn record(&self, node: LineageNode) {
        self.lineages.write().insert(node.market_id, node);
    }

    /// 构建系统发生树。
    /// 从 leaf 节点出发，递归追溯到所有 Origin 节点。
    pub fn phylogeny(&self, leaf_id: &MarketId) -> Option<PhylogeneticTree> {
        let lineages = self.lineages.read();
        let leaf = lineages.get(leaf_id)?;

        // 找根（Origin 节点）
        let root_node = self.find_origin(leaf, &lineages);
        let tree = self.build_tree(&root_node, &lineages, 0);

        // 收集所有涉及的 Agent
        let mut agents = Vec::new();
        self.collect_agents(&tree, &mut agents);

        let max_depth = self.max_depth_of(&tree);

        Some(PhylogeneticTree {
            origin_agent: root_node.publisher.clone(),
            origin_time: root_node.timestamp,
            root: tree,
            total_nodes: lineages.len(),
            max_depth,
            agents_involved: agents,
        })
    }

    fn find_origin(
        &self,
        node: &LineageNode,
        lineages: &HashMap<MarketId, LineageNode>,
    ) -> LineageNode {
        if node.parent_ids.is_empty() {
            return node.clone();
        }
        // 沿 parent_ids 追溯
        for parent_id in &node.parent_ids {
            if let Some(parent) = lineages.get(parent_id) {
                if matches!(parent.action, LineageAction::Origin) {
                    return parent.clone();
                }
                return self.find_origin(parent, lineages);
            }
        }
        node.clone()
    }

    fn build_tree(
        &self,
        node: &LineageNode,
        lineages: &HashMap<MarketId, LineageNode>,
        depth: usize,
    ) -> PhylogenyNode {
        let children: Vec<PhylogenyNode> = lineages
            .values()
            .filter(|n| n.parent_ids.contains(&node.market_id))
            .map(|n| self.build_tree(n, lineages, depth + 1))
            .collect();

        PhylogenyNode {
            entry_id: node.market_id,
            publisher: node.publisher.clone(),
            action: node.action.clone(),
            children,
            depth,
        }
    }

    fn collect_agents(&self, node: &PhylogenyNode, agents: &mut Vec<AgentId>) {
        if !agents.contains(&node.publisher) {
            agents.push(node.publisher.clone());
        }
        for child in &node.children {
            self.collect_agents(child, agents);
        }
    }

    fn max_depth_of(&self, node: &PhylogenyNode) -> usize {
        if node.children.is_empty() {
            node.depth
        } else {
            node.children
                .iter()
                .map(|c| self.max_depth_of(c))
                .max()
                .unwrap_or(node.depth)
        }
    }
}
