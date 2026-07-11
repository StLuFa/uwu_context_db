//! SemanticAxis — 轴2 虚拟目录向量聚类。
//!
//! 语义路径不由手写，而是向量聚类自动生成。
//! Sleeptime 阶段对同类型晶体做增量聚类，类簇生成路径段。

use agent_context_db_core::{ContentType, ContextError, ContextUri, Result, VectorIndex};
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
                if best_dist < self.cluster_threshold
                    && let Some(cluster) = type_clusters.get(best_idx)
                {
                    return cluster.label.clone();
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
    /// 命中 URI 通过一次批量读取取得真实向量，质心是成员向量的归一化算术平均。
    /// 任一后端、URI、元数据或向量校验错误都会返回给调用方，绝不静默使用伪向量。
    pub async fn recluster(&self, content_type: ContentType) -> Result<usize> {
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
            validate_vector(
                &cluster.centroid,
                cluster.centroid.len(),
                "cluster centroid",
            )?;
            let expected_dim = cluster.centroid.len();
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
                .await?;
            let mut ranked = hits
                .into_iter()
                .filter(|hit| hit.score >= self.cluster_threshold)
                .collect::<Vec<_>>();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.uri.as_str().cmp(b.uri.as_str()))
            });
            ranked.dedup_by(|a, b| a.uri == b.uri);
            let uris = ranked.iter().map(|hit| hit.uri.clone()).collect::<Vec<_>>();
            let vectors = self.index.get_many("context", &uris).await?;
            if vectors.len() != uris.len() {
                return Err(ContextError::Storage(format!(
                    "batch vector retrieval returned {} of {} requested members",
                    vectors.len(),
                    uris.len()
                )));
            }
            let by_uri = vectors
                .into_iter()
                .map(|record| (record.uri.clone(), record))
                .collect::<HashMap<_, _>>();
            let mut sum = vec![0.0_f64; expected_dim];
            for uri in &uris {
                let record = by_uri.get(uri).ok_or_else(|| {
                    ContextError::Storage(format!("batch vector retrieval omitted {uri}"))
                })?;
                if record.embedding_dim != Some(expected_dim) {
                    return Err(ContextError::Storage(format!(
                        "vector {uri} has missing/mismatched embedding_dim metadata {:?}, expected {expected_dim}",
                        record.embedding_dim
                    )));
                }
                validate_vector(&record.vector, expected_dim, uri.as_str())?;
                for (total, value) in sum.iter_mut().zip(&record.vector) {
                    *total += *value as f64;
                }
            }
            let centroid = if uris.is_empty() {
                None
            } else {
                let count = uris.len() as f64;
                let mut centroid = sum
                    .into_iter()
                    .map(|v| (v / count) as f32)
                    .collect::<Vec<_>>();
                normalize_unit(&mut centroid)?;
                Some(centroid)
            };
            updates.push((index, uris, centroid));
        }

        // 第二阶段重新加锁并应用结果；并发新增类簇会被保留。
        let mut clusters = self.clusters.write();
        let type_clusters = clusters.entry(content_type).or_default();
        for (index, uris, centroid) in updates {
            let Some(cluster) = type_clusters.get_mut(index) else {
                continue;
            };
            if let Some(centroid) = centroid {
                cluster.members = uris.into_iter().map(|uri| uri.to_string()).collect();
                cluster.centroid = centroid;
            }
        }

        let rebuilt_paths = type_clusters
            .iter()
            .filter(|cluster| !cluster.members.is_empty())
            .map(|cluster| (cluster.label.clone(), cluster.members.clone()))
            .collect();
        drop(clusters);
        *self.cluster_paths.write() = rebuilt_paths;

        Ok(existing_count)
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

fn validate_vector(vector: &[f32], expected_dim: usize, subject: &str) -> Result<()> {
    if vector.len() != expected_dim || expected_dim == 0 {
        return Err(ContextError::Storage(format!(
            "vector {subject} dimension {} does not match expected {expected_dim}",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(ContextError::Storage(format!(
            "vector {subject} is not finite"
        )));
    }
    if !vector.iter().any(|value| *value != 0.0) {
        return Err(ContextError::Storage(format!("vector {subject} is zero")));
    }
    Ok(())
}

fn normalize_unit(vector: &mut [f32]) -> Result<()> {
    validate_vector(vector, vector.len(), "computed centroid")?;
    let norm = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(ContextError::Storage(
            "computed centroid has zero/invalid norm".into(),
        ));
    }
    for value in vector {
        *value = (*value as f64 / norm) as f32;
    }
    Ok(())
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
    for (&a_value, &b_value) in a.iter().zip(b).take(len) {
        dot += a_value as f64 * b_value as f64;
        na += a_value as f64 * a_value as f64;
        nb += b_value as f64 * b_value as f64;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f64::EPSILON);
    1.0 - (dot / denom) as f32
}
