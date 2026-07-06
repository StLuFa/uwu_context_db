//! FederatedRegistry — Data Mesh 风格联邦注册表。
//!
//! 每个 Agent 维护自己的 registry shard，无中心目录。
//! 联邦查询通过 EventMesh broadcast + 各 shard 返回匹配结果。

use crate::marketplace::types::*;
use agent_context_db_core::{ContentType, ContextUri, EventMesh, Result, Topic, VectorIndex};
use std::collections::HashMap;
use std::sync::Arc;

/// 同伴信息。
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub agent_id: AgentId,
    pub domains: Vec<String>,
    pub bond_level: BondLevel,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// 联邦注册表 — 三注册合一，每个 Agent 独立维护。
pub struct FederatedRegistry {
    pub my_agent: AgentId,
    /// 我发布的条目。
    publications: parking_lot::RwLock<HashMap<MarketId, MarketEntry>>,
    /// 我采纳的外部条目（缓冲）。
    adoptions: parking_lot::RwLock<HashMap<MarketId, MarketEntry>>,
    /// 我的订阅（领域 → 发布者）。
    subscriptions: parking_lot::RwLock<HashMap<String, Vec<AgentId>>>,
    /// 已知同伴。
    peers: parking_lot::RwLock<HashMap<AgentId, PeerInfo>>,
    /// 本地向量索引。
    local_index: Arc<dyn VectorIndex>,
    /// 事件 mesh（用于联邦发现）。
    event_mesh: Option<EventMesh>,
    /// 血统图。
    lineage: parking_lot::RwLock<HashMap<MarketId, LineageNode>>,
}

impl FederatedRegistry {
    pub fn new(my_agent: AgentId, index: Arc<dyn VectorIndex>) -> Self {
        Self {
            my_agent,
            publications: parking_lot::RwLock::new(HashMap::new()),
            adoptions: parking_lot::RwLock::new(HashMap::new()),
            subscriptions: parking_lot::RwLock::new(HashMap::new()),
            peers: parking_lot::RwLock::new(HashMap::new()),
            local_index: index,
            event_mesh: None,
            lineage: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn with_event_mesh(mut self, mesh: EventMesh) -> Self {
        self.event_mesh = Some(mesh);
        self
    }

    // ── 发布 ──────────────────────────────

    pub fn publish(&self, entry: MarketEntry) {
        // 注册到向量索引
        let uri = agent_context_db_core::ContextUri::parse(
            format!("uwu://market/{}/{}", entry.publisher, entry.id.0)
        ).expect("marketplace URI is well-formed");
        let _ = self.local_index.upsert("market", agent_context_db_core::IndexPoint {
            uri: uri.clone(),
            vector: vec![0.0; 128], // actual embedding from product
            payload: serde_json::json!({
                "market_id": entry.id.0.to_string(),
                "domain": entry.domain,
                "quality": entry.quality_score,
            }),
        });
        self.publications.write().insert(entry.id, entry.clone());

        // 广播到 EventMesh
        if let Some(mesh) = self.event_mesh.clone() {
            let topic_str = format!("market.publish.{}", entry.domain);
            if let Ok(topic) = Topic::new(topic_str) {
                if let Ok(json) = serde_json::to_value(&entry) {
                    tokio::spawn(async move { let _ = mesh.emit(&topic, json).await; });
                }
            }
        }
    }

    /// 获取我的发布。
    pub fn my_publications(&self) -> Vec<MarketEntry> {
        self.publications.read().values().cloned().collect()
    }

    // ── 采纳 ──────────────────────────────

    pub fn adopt(&self, entry: MarketEntry) {
        self.adoptions.write().insert(entry.id, entry);
    }

    pub fn my_adoptions(&self) -> Vec<MarketEntry> {
        self.adoptions.read().values().cloned().collect()
    }

    // ── 同伴发现 ──────────────────────────────

    /// 注册一个同伴（通过 EventMesh 发现或手动添加）。
    pub fn register_peer(&self, peer: PeerInfo) {
        self.peers.write().insert(peer.agent_id.clone(), peer);
    }

    /// 获取某领域已知的同伴。
    pub fn peers_in_domain(&self, domain: &str) -> Vec<PeerInfo> {
        self.peers.read().values()
            .filter(|p| p.domains.iter().any(|d| d == domain))
            .cloned()
            .collect()
    }

    /// 所有已注册同伴。
    pub fn all_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().values().cloned().collect()
    }

    // ── 订阅 ──────────────────────────────

    pub fn subscribe(&self, domain: &str, publisher: AgentId) {
        self.subscriptions.write()
            .entry(domain.to_string())
            .or_default()
            .push(publisher);
    }

    pub fn subscriptions_for(&self, domain: &str) -> Vec<AgentId> {
        self.subscriptions.read().get(domain).cloned().unwrap_or_default()
    }

    // ── 血统 ──────────────────────────────

    pub fn record_lineage(&self, node: LineageNode) {
        self.lineage.write().insert(node.market_id, node);
    }

    pub fn get_lineage(&self, id: &MarketId) -> Option<LineageNode> {
        self.lineage.read().get(id).cloned()
    }

    /// 追溯完整血统链（递归）。
    pub fn trace_lineage(&self, id: &MarketId) -> Vec<LineageNode> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![*id];
        while let Some(current) = stack.pop() {
            if !visited.insert(current) { continue; }
            if let Some(node) = self.get_lineage(&current) {
                stack.extend(&node.parent_ids);
                result.push(node);
            }
        }
        result
    }

    // ── 联邦查询 ──────────────────────────────

    /// 本地搜索（向量索引）。
    pub async fn search_local(&self, embedding: &[f32], limit: usize) -> Vec<MarketEntry> {
        let hits = self.local_index.search("market", embedding.to_vec(), limit, None).await.unwrap_or_default();
        let pubs = self.publications.read();
        hits.iter()
            .filter_map(|h| {
                let id_str = h.payload.get("market_id")?.as_str()?;
                let uid = uuid::Uuid::parse_str(id_str).ok()?;
                pubs.get(&MarketId(uid)).cloned()
            })
            .collect()
    }

    /// 统计。
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            publications: self.publications.read().len(),
            adoptions: self.adoptions.read().len(),
            peers: self.peers.read().len(),
            subscriptions: self.subscriptions.read().len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub publications: usize,
    pub adoptions: usize,
    pub peers: usize,
    pub subscriptions: usize,
}
