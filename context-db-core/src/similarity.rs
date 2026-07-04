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
            .insert(uri.0.clone(), embedding);
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
                        let key = if uri_a < uri_b { (uri_a, uri_b) } else { (uri_b, uri_a) };
                        if visited.contains(&key) { continue; }
                        visited.insert(key);

                        let sim = VectorSimilarity::cosine(emb_a, emb_b);
                        if sim >= self.threshold {
                            pairs.push((
                                ContextUri(uri_a.clone()),
                                ContextUri(uri_b.clone()),
                                sim,
                            ));
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
            let pa = find_root(&mut parent, &a.0);
            let pb = find_root(&mut parent, &b.0);
            if pa != pb {
                parent.insert(pa.clone(), pb.clone());
            }
        }

        // 收集各组
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        let all_uris: Vec<&String> = pairs.iter()
            .flat_map(|(a, b, _)| vec![&a.0, &b.0])
            .collect();
        for uri in all_uris {
            let root = find_root(&mut parent, uri);
            groups.entry(root).or_default().push(uri.clone());
        }

        groups
            .into_iter()
            .filter(|(_, uris)| uris.len() >= 2)
            .map(|(id, uris)| {
                let agents: Vec<String> = uris.iter()
                    .filter_map(|u| embeddings.iter().find(|(_, m)| m.contains_key(u)).map(|(a, _)| a.clone()))
                    .collect();
                Cluster {
                    id: id.chars().take(8).collect(),
                    uris: uris.into_iter().map(ContextUri).collect(),
                    centroid_description: String::new(),
                    agents,
                    recommendation: DedupRecommendation::ManualReview,
                }
            })
            .collect()
    }
}

fn find_root(parent: &mut HashMap<String, String>, node: &str) -> String {
    let mut current = node.to_string();
    while let Some(p) = parent.get(&current) {
        if p == &current { break; }
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
