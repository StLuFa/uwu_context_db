//! SemanticAxis — 轴2 虚拟目录向量聚类。
//!
//! 语义路径不由手写，而是向量聚类自动生成。
//! Sleeptime 阶段对同类型晶体做增量聚类，类簇生成路径段。

use agent_context_db_core::{ContentType, ContextUri, VectorIndex};
use std::collections::HashMap;
use std::sync::Arc;

/// 类簇信息。
#[derive(Debug, Clone)]
struct ClusterInfo {
    centroid: Vec<f32>,
    members: Vec<String>, // URI strings
    label: String,
}

/// 虚拟语义目录 — 向量聚类自动生成的路径段。
pub struct SemanticAxis {
    index: Arc<dyn VectorIndex>,
    cluster_paths: parking_lot::RwLock<HashMap<String, Vec<String>>>,
    /// 存储的类簇（type → clusters）。
    clusters: parking_lot::RwLock<HashMap<ContentType, Vec<ClusterInfo>>>,
    /// 聚类阈值（余弦相似度低于此值 → 归入不同的类簇）。
    cluster_threshold: f32,
}

impl SemanticAxis {
    pub fn new(index: Arc<dyn VectorIndex>) -> Self {
        Self {
            index,
            cluster_paths: parking_lot::RwLock::new(HashMap::new()),
            clusters: parking_lot::RwLock::new(HashMap::new()),
            cluster_threshold: 0.7,
        }
    }

    /// 为新条目分配语义路径（基于向量最近类簇）。
    pub fn assign_path(&self, uri: &ContextUri, embedding: &[f32]) -> String {
        let segs = uri.segments();
        let content_type =
            ContentType::from_path_segment(segs.get(2).map(|s| s.as_str()).unwrap_or("fact"));

        if let Some(ct) = content_type {
            let clusters = self.clusters.read();
            if let Some(type_clusters) = clusters.get(&ct) {
                // 找最近的类簇
                let mut best_dist = f32::MAX;
                let mut best_idx = 0usize;
                for (i, c) in type_clusters.iter().enumerate() {
                    let dist = cosine_distance(embedding, &c.centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                if best_dist < self.cluster_threshold && !type_clusters.is_empty() {
                    return type_clusters[best_idx].label.clone();
                }
            }
        }

        // Fallback：基于 URI 路径段
        if segs.len() >= 4 {
            segs[1..segs.len().min(5)].join("/")
        } else {
            "unsorted".to_string()
        }
    }

    /// Sleeptime 阶段：使用向量索引近邻刷新类簇成员并重建语义路径。
    ///
    /// 每个已有类簇以当前质心查询同类型 top-k，按命中分数稳定排序、去重并更新成员。
    /// 由于 `VectorIndex` 只返回命中 URI/score，不暴露原始向量，质心只做质量归一化；
    /// 新类簇仍由上游 `add_cluster` 在有新标签或新质心时显式创建。
    pub async fn recluster(&self, content_type: ContentType) -> usize {
        // 第一阶段只复制查询所需快照，不让同步锁跨越 await。
        let cluster_snapshot = self
            .clusters
            .read()
            .get(&content_type)
            .cloned()
            .unwrap_or_default();
        let existing_count = cluster_snapshot.len();
        let mut updates = Vec::with_capacity(existing_count);

        for (index, cluster) in cluster_snapshot.into_iter().enumerate() {
            let hits = self
                .index
                .search(
                    "context",
                    cluster.centroid,
                    64,
                    Some(serde_json::json!({
                        "content_type": content_type.as_path_segment()
                    })),
                )
                .await
                .unwrap_or_default();
            updates.push((index, hits));
        }

        // 第二阶段重新加锁并应用结果；并发新增类簇会被保留。
        let mut clusters = self.clusters.write();
        let type_clusters = clusters.entry(content_type).or_default();
        for (index, hits) in updates {
            let Some(cluster) = type_clusters.get_mut(index) else {
                continue;
            };
            if !hits.is_empty() {
                let mut ranked = hits
                    .into_iter()
                    .filter(|hit| hit.score >= self.cluster_threshold)
                    .map(|hit| (hit.uri.to_string(), hit.score.clamp(0.0, 1.0)))
                    .collect::<Vec<_>>();
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                ranked.dedup_by(|a, b| a.0 == b.0);
                if !ranked.is_empty() {
                    let score_mass = ranked.iter().map(|(_, score)| *score).sum::<f32>();
                    cluster.members = ranked.into_iter().map(|(uri, _)| uri).collect();
                    normalize_centroid(
                        &mut cluster.centroid,
                        score_mass / cluster.members.len() as f32,
                    );
                }
            } else if cluster.members.len() > 1 {
                normalize_centroid(&mut cluster.centroid, 0.5);
            }
        }

        let rebuilt_paths = type_clusters
            .iter()
            .filter(|cluster| !cluster.members.is_empty())
            .map(|cluster| (cluster.label.clone(), cluster.members.clone()))
            .collect();
        drop(clusters);
        *self.cluster_paths.write() = rebuilt_paths;

        existing_count
    }

    /// 手动添加类簇（用于 LLM 生成的语义标签）。
    pub fn add_cluster(
        &self,
        content_type: ContentType,
        label: String,
        centroid: Vec<f32>,
        members: Vec<String>,
    ) {
        let mut clusters = self.clusters.write();
        clusters.entry(content_type).or_default().push(ClusterInfo {
            centroid,
            members: members.clone(),
            label: label.clone(),
        });
        self.cluster_paths.write().insert(label, members);
    }

    /// 当前类簇数量。
    pub fn cluster_count(&self, content_type: ContentType) -> usize {
        self.clusters
            .read()
            .get(&content_type)
            .map(|c| c.len())
            .unwrap_or(0)
    }
}

fn normalize_centroid(centroid: &mut [f32], confidence: f32) {
    let norm = centroid
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return;
    }
    let scale = confidence.clamp(0.25, 1.0) as f64 / norm;
    for value in centroid {
        *value = (*value as f64 * scale) as f32;
    }
}

/// 余弦距离 = 1 - 余弦相似度。
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let len = a.len().min(b.len());
    if len == 0 {
        return 1.0;
    }
    for i in 0..len {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f64::EPSILON);
    1.0 - (dot / denom) as f32
}
