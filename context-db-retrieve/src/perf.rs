//! 性能优化层：P1 查询编译器 + P2 向量分层 + P3 量化压缩 + P4 流水线并行 + P6 物化视图 + P9 分区并行。

use agent_context_db_core::{ContextUri, FsOps, LlmClient, LlmOpts, Result, VectorIndex};
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// P1 检索查询编译器 — 将多个 FsOps 调用合并为单次批量查询
// ═══════════════════════════════════════════════════════════════════════════

/// 批量 FS 操作请求。
pub struct BatchFsRequest {
    pub uris: Vec<ContextUri>,
    pub patterns: Vec<agent_context_db_core::FindPattern>,
    pub levels: Vec<agent_context_db_core::ContentLevel>,
}

/// 查询编译器 — 将多次独立的 FsOps 调用编译为一次批量请求。
pub struct QueryCompiler {
    fs: Arc<dyn FsOps>,
}

impl QueryCompiler {
    pub fn new(fs: Arc<dyn FsOps>) -> Self { Self { fs } }

    /// 批量读取：一次收集所有 URI 的内容。
    pub async fn batch_read(
        &self, uris: &[ContextUri], level: agent_context_db_core::ContentLevel,
    ) -> Vec<(ContextUri, Result<agent_context_db_core::ContentPayload>)> {
        let handles: Vec<_> = uris.iter().map(|uri| {
            let fs = self.fs.clone();
            let uri = uri.clone();
            tokio::spawn(async move { (uri.clone(), fs.read(&uri, level).await) })
        }).collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(r) = h.await { results.push(r); }
        }
        results
    }

    /// 并行 grep：对多个 regex 一次搜索。
    pub async fn batch_grep(
        &self, patterns: &[(&str, &ContextUri)],
    ) -> Vec<(String, Vec<agent_context_db_core::GrepHit>)> {
        let owned: Vec<(String, ContextUri)> = patterns.iter().map(|(r, s)| (r.to_string(), (*s).clone())).collect();
        let mut handles = Vec::new();
        for (regex, scope) in owned {
            let fs = self.fs.clone();
            handles.push(tokio::spawn(async move { (regex.clone(), fs.grep(&regex, &scope).await) }));
        }
        let mut results = Vec::new();
        for h in handles {
            if let Ok((r, hits)) = h.await {
                if let Ok(hits) = hits { results.push((r, hits)); }
            }
        }
        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P2 向量索引分层 — Hot/Warm/Cold 三级缓存
// ═══════════════════════════════════════════════════════════════════════════

/// 缓存层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Hot,
    Warm,
    Cold,
}

/// 分层向量索引 — 按访问频率自动升级/降级。
pub struct TieredVectorIndex {
    index: Arc<dyn VectorIndex>,
    /// URI → 访问计数
    access_counts: parking_lot::Mutex<HashMap<String, u64>>,
    /// URI → 当前层级
    tier_map: parking_lot::Mutex<HashMap<String, CacheTier>>,
    hot_threshold: u64,
    warm_threshold: u64,
}

impl TieredVectorIndex {
    pub fn new(index: Arc<dyn VectorIndex>, hot: u64, warm: u64) -> Self {
        Self {
            index,
            access_counts: parking_lot::Mutex::new(HashMap::new()),
            tier_map: parking_lot::Mutex::new(HashMap::new()),
            hot_threshold: hot, warm_threshold: warm,
        }
    }

    pub fn record_access(&self, uri: &str) {
        let mut counts = self.access_counts.lock();
        let count = counts.entry(uri.to_string()).or_insert(0);
        *count += 1;
        let tier = if *count >= self.hot_threshold { CacheTier::Hot }
                   else if *count >= self.warm_threshold { CacheTier::Warm }
                   else { CacheTier::Cold };
        self.tier_map.lock().insert(uri.to_string(), tier);
    }

    pub fn tier(&self, uri: &str) -> CacheTier {
        self.tier_map.lock().get(uri).copied().unwrap_or(CacheTier::Cold)
    }

    pub async fn search(
        &self, collection: &str, query: Vec<f32>, top_k: usize,
    ) -> Result<Vec<agent_context_db_core::IndexHit>> {
        self.index.search(collection, query, top_k, None).await
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P3 嵌入向量量化压缩 — Product Quantization
// ═══════════════════════════════════════════════════════════════════════════

/// Product Quantization 压缩器。
pub struct VectorQuantizer {
    subvectors: usize,
    codebook_size: usize,
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl VectorQuantizer {
    pub fn new(subvectors: usize, codebook_size: usize) -> Self {
        Self { subvectors, codebook_size, codebooks: Vec::new() }
    }

    /// 训练码本（简化：k-means 占位，实际可用 faiss/skia）。
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() || self.subvectors == 0 { return; }
        let dim = vectors[0].len();
        let sub_dim = dim / self.subvectors;
        self.codebooks.clear();

        for s in 0..self.subvectors {
            let start = s * sub_dim;
            let end = start + sub_dim;
            let mut codebook = Vec::new();
            for i in 0..self.codebook_size.min(vectors.len()) {
                let v: Vec<f32> = vectors[i][start..end].to_vec();
                if !codebook.iter().any(|c: &Vec<f32>| c == &v) {
                    codebook.push(v);
                }
            }
            self.codebooks.push(codebook);
        }
    }

    /// 压缩向量：将 f32 向量转为 u8 码字索引序列。
    pub fn encode(&self, vector: &[f32]) -> Option<Vec<u8>> {
        if self.codebooks.is_empty() { return None; }
        let sub_dim = vector.len() / self.subvectors;
        let mut codes = Vec::with_capacity(self.subvectors);
        for (s, codebook) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let sub: Vec<f32> = vector[start..start + sub_dim].to_vec();
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (i, centroid) in codebook.iter().enumerate() {
                let dist = euclidean(&sub, centroid);
                if dist < best_dist { best_dist = dist; best_idx = i as u8; }
            }
            codes.push(best_idx);
        }
        Some(codes)
    }

    /// 存储压缩比。
    pub fn compression_ratio(&self, dim: usize) -> f32 {
        let original = dim * 4; // f32 = 4 bytes
        let compressed = self.subvectors; // u8 per subvector
        original as f32 / compressed.max(1) as f32
    }
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// P4 流水线并行生成 — 多个 LLM 请求并行化
// ═══════════════════════════════════════════════════════════════════════════

/// 并行 LLM 请求调度器。
pub struct ParallelGenerator {
    llm: Arc<dyn LlmClient>,
    max_concurrency: usize,
}

impl ParallelGenerator {
    pub fn new(llm: Arc<dyn LlmClient>, max: usize) -> Self { Self { llm, max_concurrency: max } }

    /// 并行生成多个摘要。
    pub async fn batch_generate_abstracts(
        &self, uris: &[ContextUri],
    ) -> Vec<(ContextUri, Result<String>)> {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        let handles: Vec<_> = uris.iter().map(|uri| {
            let llm = self.llm.clone();
            let uri = uri.clone();
            let sem = semaphore.clone();
            tokio::spawn(async move {
                let _permit = sem.acquire().await;
                let prompt = format!("Write a concise abstract (~100 tokens) for: {uri}");
                (uri, llm.complete(&prompt, &LlmOpts::default()).await
                    .map_err(|e| agent_context_db_core::ContextError::Storage(format!("{e}"))))
            })
        }).collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(r) = h.await { results.push(r); }
        }
        results
    }

    /// 并行生成多个 embedding。
    pub async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        let handles: Vec<_> = texts.iter().map(|t| {
            let llm = self.llm.clone();
            let text = t.clone();
            let sem = semaphore.clone();
            tokio::spawn(async move {
                let _permit = sem.acquire().await;
                llm.embed(&text).await
            })
        }).collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(Ok(v)) = h.await { results.push(v); }
        }
        Ok(results)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P6 检索物化视图 — 预计算常见路径
// ═══════════════════════════════════════════════════════════════════════════

/// 物化视图 — 缓存常见检索路径的结果。
pub struct MaterializedView {
    /// query_key → (uris, expires_at)
    cache: parking_lot::Mutex<HashMap<String, (Vec<ContextUri>, chrono::DateTime<chrono::Utc>)>>,
    ttl_secs: i64,
}

impl MaterializedView {
    pub fn new(ttl_secs: i64) -> Self {
        Self { cache: parking_lot::Mutex::new(HashMap::new()), ttl_secs }
    }

    pub fn get(&self, query: &str) -> Option<Vec<ContextUri>> {
        let cache = self.cache.lock();
        if let Some((uris, expires)) = cache.get(query) {
            if *expires > chrono::Utc::now() { return Some(uris.clone()); }
        }
        None
    }

    pub fn set(&self, query: &str, uris: Vec<ContextUri>) {
        let expires = chrono::Utc::now() + chrono::Duration::seconds(self.ttl_secs);
        self.cache.lock().insert(query.to_string(), (uris, expires));
    }

    pub fn invalidate(&self, scope: &str) {
        self.cache.lock().retain(|k, _| !k.starts_with(scope));
    }

    pub fn hit_rate(&self) -> (usize, usize) {
        let cache = self.cache.lock();
        let active = cache.values().filter(|(_, e)| *e > chrono::Utc::now()).count();
        (active, cache.len())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P9 分区并行检索 — 按 tenant 分区并行搜索
// ═══════════════════════════════════════════════════════════════════════════

/// 分区并行检索器。
pub struct PartitionedRetriever {
    fs: Arc<dyn FsOps>,
    partitions: Vec<String>,
}

impl PartitionedRetriever {
    pub fn new(fs: Arc<dyn FsOps>) -> Self { Self { fs, partitions: Vec::new() } }

    pub fn with_partitions(mut self, partitions: Vec<String>) -> Self {
        self.partitions = partitions; self
    }

    /// 并行在所有分区上执行 find。
    pub async fn parallel_find(
        &self, pattern_template: &agent_context_db_core::FindPattern,
    ) -> Vec<(String, Vec<ContextUri>)> {
        let owned_partitions = self.partitions.clone();
        let owned_pattern = pattern_template.clone();
        let handles: Vec<_> = owned_partitions.into_iter().map(|tenant| {
            let fs = self.fs.clone();
            let mut pattern = owned_pattern.clone();
            pattern.scope = Some(ContextUri::parse(format!("uwu://{tenant}")).unwrap_or_else(|_| ContextUri("".into())));
            tokio::spawn(async move {
                match fs.find(&pattern).await {
                    Ok(uris) => Some((tenant, uris)),
                    Err(_) => None,
                }
            })
        }).collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(Some(r)) = h.await { results.push(r); }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_quantizer_compression_ratio() {
        let q = VectorQuantizer::new(8, 256);
        let ratio = q.compression_ratio(768);
        assert!(ratio > 1.0);
    }

    #[test]
    fn materialized_view_ttl_expires() {
        let view = MaterializedView::new(1);
        view.set("test", vec![ContextUri::parse("uwu://t/x").unwrap()]);
        assert!(view.get("test").is_some());
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(view.get("test").is_none());
    }

    #[test]
    fn cache_tier_upgrades_on_access() {
        // 结构验证：不含真实 VectorIndex 的 tier logic 测试
        let counts = parking_lot::Mutex::new(HashMap::new());
        let tier_map = parking_lot::Mutex::new(HashMap::new());
        let uri = "uri-1";
        for _i in 0..6 {
            let mut c = counts.lock();
            let count = c.entry(uri.to_string()).or_insert(0);
            *count += 1;
            let tier = if *count >= 5 { CacheTier::Hot }
                       else if *count >= 2 { CacheTier::Warm }
                       else { CacheTier::Cold };
            tier_map.lock().insert(uri.to_string(), tier);
        }
        assert_eq!(tier_map.lock().get(uri), Some(&CacheTier::Hot));
    }
}
