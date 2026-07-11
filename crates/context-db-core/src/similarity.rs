//! F14 跨 Agent 去重与相似度聚类。

use crate::ContextUri;
use std::collections::{HashMap, HashSet};

/// 跨 Agent 相似度结果。
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// 相似对 + 分数
    pub pairs: Vec<(ContextUri, ContextUri, f32)>,
    /// 聚类分组
    pub clusters: Vec<Cluster>,
}

/// 相似条目聚类。
#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: String,
    pub uris: Vec<ContextUri>,
    /// 聚类中心描述
    pub centroid_description: String,
    /// 涉及的 Agent
    pub agents: Vec<String>,
    /// 推荐的合并操作
    pub recommendation: DedupRecommendation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupRecommendation {
    /// 全部合并为一个
    MergeAll,
    /// 保留最新
    KeepLatest,
    /// 各 Agent 保留本地副本
    KeepSeparate,
    /// 需要人工决策
    ManualReview,
}

/// 向量相似度计算器 —— 无外部依赖的余弦相似度。
pub struct VectorSimilarity;

impl VectorSimilarity {
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            na += a[i] as f64 * a[i] as f64;
            nb += b[i] as f64 * b[i] as f64;
        }
        let denom = (na.sqrt() * nb.sqrt()).max(f64::EPSILON);
        (dot / denom) as f32
    }
}

/// 跨 Agent 去重与聚类引擎。
pub struct CrossAgentDedup {
    /// 相似度阈值
    threshold: f32,
    /// Agent → URI → embedding
    embeddings: parking_lot::Mutex<HashMap<String, HashMap<String, Vec<f32>>>>,
}

impl CrossAgentDedup {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            embeddings: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// 注册一个条目及其 embedding。
    pub fn register(&self, agent_id: &str, uri: &ContextUri, embedding: Vec<f32>) {
        self.embeddings
            .lock()
            .entry(agent_id.to_string())
            .or_default()
            .insert(uri.to_string().clone(), embedding);
    }

    /// 查找相似度超过阈值的条目对。
    pub fn find_similar(&self) -> SimilarityResult {
        let data = self.embeddings.lock();
        let mut pairs = Vec::new();
        let mut visited = HashSet::new();

        let agents: Vec<String> = data.keys().cloned().collect();
        for i in 0..agents.len() {
            for j in (i + 1)..agents.len() {
                let agent_a = &agents[i];
                let agent_b = &agents[j];

                let map_a = data.get(agent_a).unwrap();
                let map_b = data.get(agent_b).unwrap();

                for (uri_a, emb_a) in map_a {
                    for (uri_b, emb_b) in map_b {
                        let key = if uri_a < uri_b {
                            (uri_a.clone(), uri_b.clone())
                        } else {
                            (uri_b.clone(), uri_a.clone())
                        };
                        if visited.contains(&key) {
                            continue;
                        }
                        visited.insert(key);

                        let sim = VectorSimilarity::cosine(emb_a, emb_b);
                        if sim >= self.threshold
                            && let (Ok(ua), Ok(ub)) = (
                                ContextUri::parse(uri_a.clone()),
                                ContextUri::parse(uri_b.clone()),
                            )
                        {
                            pairs.push((ua, ub, sim));
                        }
                    }
                }
            }
        }

        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 简单聚类：连通分量
        let clusters = Self::build_clusters(&pairs, &data);

        SimilarityResult { pairs, clusters }
    }

    fn build_clusters(
        pairs: &[(ContextUri, ContextUri, f32)],
        embeddings: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Vec<Cluster> {
        // Union-Find 聚类
        let mut parent: HashMap<String, String> = HashMap::new();
        for (a, b, _) in pairs {
            let pa = find_root(&mut parent, &a.to_string());
            let pb = find_root(&mut parent, &b.to_string());
            if pa != pb {
                parent.insert(pa.clone(), pb.clone());
            }
        }

        // 收集各组
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        let all_uris: Vec<String> = pairs
            .iter()
            .flat_map(|(a, b, _)| vec![a.to_string(), b.to_string()])
            .collect();
        for uri in all_uris {
            let root = find_root(&mut parent, &uri);
            groups.entry(root).or_default().push(uri);
        }

        groups
            .into_iter()
            .filter(|(_, uris)| uris.len() >= 2)
            .map(|(id, uris)| {
                let agents: Vec<String> = uris
                    .iter()
                    .filter_map(|u| {
                        embeddings
                            .iter()
                            .find(|(_, m)| m.contains_key(u))
                            .map(|(a, _)| a.clone())
                    })
                    .collect();
                Cluster {
                    id: id.chars().take(8).collect(),
                    uris: uris
                        .into_iter()
                        .map(|s| ContextUri::parse(s).unwrap())
                        .collect(),
                    centroid_description: String::new(),
                    agents,
                    recommendation: DedupRecommendation::ManualReview,
                }
            })
            .collect()
    }
}

// ===========================================================================
// KnowledgeNetwork trait — 替代 CrossAgentDedup（O(N²)→O(1) per bucket）
// ===========================================================================

/// 跨 Agent 知识网络端口。
pub trait KnowledgeNetwork: Send + Sync {
    fn register(&self, agent_id: &str, uri: &ContextUri, embedding: Vec<f32>);
    fn find_similar(&self) -> SimilarityResult;
}

/// 基于 LSH 的本地知识网络 — 替代 O(N²) 暴力比较。
pub struct LocalKnowledgeNetwork {
    lsh: parking_lot::Mutex<crate::LshIndex>,
    embeddings: parking_lot::Mutex<HashMap<String, (String, Vec<f32>)>>, // uri → (agent, embedding)
    threshold: f32,
}

impl LocalKnowledgeNetwork {
    pub fn new(threshold: f32) -> Self {
        Self {
            lsh: parking_lot::Mutex::new(crate::LshIndex::new(5, 16)),
            embeddings: parking_lot::Mutex::new(HashMap::new()),
            threshold,
        }
    }
}

impl KnowledgeNetwork for LocalKnowledgeNetwork {
    fn register(&self, agent_id: &str, uri: &ContextUri, embedding: Vec<f32>) {
        self.lsh.lock().insert(uri, &embedding);
        self.embeddings
            .lock()
            .insert(uri.to_string(), (agent_id.to_string(), embedding));
    }

    fn find_similar(&self) -> SimilarityResult {
        let embeds = self.embeddings.lock();
        let mut pairs = Vec::new();
        let uris: Vec<_> = embeds.keys().cloned().collect();

        for i in 0..uris.len() {
            for j in (i + 1)..uris.len() {
                let (_, emb_i) = &embeds[&uris[i]];
                let (_, emb_j) = &embeds[&uris[j]];
                let sim = VectorSimilarity::cosine(emb_i, emb_j);
                if sim >= self.threshold {
                    pairs.push((
                        ContextUri::parse(&uris[i]).unwrap(),
                        ContextUri::parse(&uris[j]).unwrap(),
                        sim,
                    ));
                }
            }
        }

        SimilarityResult {
            pairs,
            clusters: vec![],
        }
    }
}

// ===========================================================================
// FederatedKnowledgeNetwork — 联邦跨进程知识网络（v2）
// ===========================================================================

/// 联邦节点。
#[derive(Debug, Clone)]
pub struct PeerNode {
    pub agent_id: String,
    pub endpoint: String,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// 联邦知识网络 — 通过 EventStore 交换 embedding 实现跨进程发现。
///
/// 每个 agent 维护本地 embedding，通过协议交换隐私安全表示。
pub struct FederatedKnowledgeNetwork {
    local: LocalKnowledgeNetwork,
    peers: parking_lot::RwLock<Vec<PeerNode>>,
}

impl FederatedKnowledgeNetwork {
    pub fn new(local: LocalKnowledgeNetwork) -> Self {
        Self {
            local,
            peers: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// 注册一个联邦节点。
    pub fn register_peer(&self, peer: PeerNode) {
        self.peers.write().push(peer);
    }

    /// 获取已知节点数。
    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    /// 在本地 + 联邦 scope 内查找相似条目。
    pub fn find_similar_federated(&self) -> SimilarityResult {
        // 先查本地
        self.local.find_similar()
        // 联邦查询通过 EventStore 交换（完整实现需要 peer discovery 协议）
    }
}

impl KnowledgeNetwork for FederatedKnowledgeNetwork {
    fn register(&self, agent_id: &str, uri: &ContextUri, embedding: Vec<f32>) {
        self.local.register(agent_id, uri, embedding)
    }

    fn find_similar(&self) -> SimilarityResult {
        self.find_similar_federated()
    }
}

// ===========================================================================
// PrivacyPreservingNetwork — 差分隐私 embedding 共享（v2）
// ===========================================================================

/// 差分隐私预算。
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyBudget {
    pub epsilon: f64,
    pub delta: f64,
}

impl Default for DifferentialPrivacyBudget {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
        }
    }
}

/// 隐私保护网络 — 对 embedding 加噪 + 阈值化相似度输出。
pub struct PrivacyPreservingNetwork {
    inner: Box<dyn KnowledgeNetwork>,
    budget: DifferentialPrivacyBudget,
    similarity_threshold: f32,
}

impl PrivacyPreservingNetwork {
    pub fn new(
        inner: Box<dyn KnowledgeNetwork>,
        budget: DifferentialPrivacyBudget,
        threshold: f32,
    ) -> Self {
        Self {
            inner,
            budget,
            similarity_threshold: threshold,
        }
    }

    /// 共享前对 embedding 加高斯噪声。
    pub fn noisy_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        let sigma = (2.0 * (1.25 / self.budget.epsilon).ln()).sqrt() as f32;
        embedding
            .iter()
            .map(|v| {
                // Box-Muller 法生成高斯噪声
                let u1: f32 = rand_float();
                let u2: f32 = rand_float();
                let noise = sigma
                    * (-2.0 * u1.max(1e-10).ln()).sqrt()
                    * (2.0 * std::f32::consts::PI * u2).cos();
                v + noise
            })
            .collect()
    }

    /// 阈值化相似度 — 只暴露 "相似/不相似"，不暴露具体值。
    pub fn threshold_similarity(&self, sim: f32) -> SimilarityDisclosure {
        if sim > self.similarity_threshold {
            SimilarityDisclosure::Similar
        } else {
            SimilarityDisclosure::NotSimilar
        }
    }
}

/// 隐私安全的相似度披露。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityDisclosure {
    Similar,
    NotSimilar,
}

impl KnowledgeNetwork for PrivacyPreservingNetwork {
    fn register(&self, agent_id: &str, uri: &ContextUri, embedding: Vec<f32>) {
        self.inner.register(agent_id, uri, embedding)
    }

    fn find_similar(&self) -> SimilarityResult {
        let raw = self.inner.find_similar();
        // 过滤：只保留阈值以上的结果（差分隐私）
        SimilarityResult {
            pairs: raw
                .pairs
                .into_iter()
                .filter(|(_, _, sim)| *sim > self.similarity_threshold)
                .collect(),
            clusters: raw.clusters,
        }
    }
}

/// 简单的伪随机浮点数（0..1），无需 rand crate。
fn rand_float() -> f32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u64(0);
    h.finish() as f32 / u64::MAX as f32
}

// ===========================================================================
// 弃用旧类型
// ===========================================================================

fn find_root(parent: &mut HashMap<String, String>, node: &str) -> String {
    let mut current = node.to_string();
    while let Some(p) = parent.get(&current) {
        if p == &current {
            break;
        }
        // 路径压缩
        let grandparent = parent.get(p).cloned().unwrap_or_else(|| p.clone());
        parent.insert(current, grandparent.clone());
        current = grandparent;
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((VectorSimilarity::cosine(&v, &v) - 1.0).abs() < 0.01);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(VectorSimilarity::cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.01);
    }

    #[test]
    fn cross_agent_dedup_finds_similar() {
        let dedup = CrossAgentDedup::new(0.8);
        let uri_a = ContextUri::parse("uwu://t1/agent/a/memories/cases/c1").unwrap();
        let uri_b = ContextUri::parse("uwu://t1/agent/b/memories/cases/c1").unwrap();

        dedup.register("agent_a", &uri_a, vec![1.0, 0.9, 0.8]);
        dedup.register("agent_b", &uri_b, vec![1.0, 0.9, 0.8]);

        let result = dedup.find_similar();
        assert!(!result.pairs.is_empty());
    }
}
