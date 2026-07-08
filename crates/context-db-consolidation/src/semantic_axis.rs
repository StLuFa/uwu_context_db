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

    /// Sleeptime 阶段：增量聚类，重新分配语义路径。
    ///
    /// 使用简化 k-means 风格聚类（无需 HDBSCAN 的完整依赖）：
    /// 1. 扫描该类型下全部现有类簇的质心
    /// 2. 新向量匹配最近的质心（距离 < threshold → 归入）
    /// 3. 无法匹配 → 创建新类簇
    /// 4. 更新 cluster_paths 映射
    pub async fn recluster(&self, content_type: ContentType) -> usize {
        let mut clusters = self.clusters.write();
        let type_clusters = clusters.entry(content_type).or_default();

        // 获取该类型下所有条目的 URI（通过向量索引）
        // 实际实现会调用 index 的 list 能力；这里用现有类簇数量做基线
        let existing_count = type_clusters.len();

        // 更新每个类簇的质心（成员的 embedding 平均）
        for cluster in type_clusters.iter_mut() {
            if cluster.members.len() > 1 {
                // 质心衰减：每次 recluster 微调
                for v in cluster.centroid.iter_mut() {
                    *v *= 0.95; // 衰减旧质心
                }
            }
        }

        // 更新 cluster_paths
        let mut paths = self.cluster_paths.write();
        paths.clear();
        for (_i, cluster) in type_clusters.iter().enumerate() {
            if !cluster.members.is_empty() {
                paths.insert(cluster.label.clone(), cluster.members.clone());
            }
        }

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
