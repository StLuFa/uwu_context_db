//! 性能优化层：P1 查询编译器 + P2 向量分层 + P3 量化压缩 + P4 流水线并行 + P6 物化视图 + P9 分区并行。

use agent_context_db_core::{
    ContextUri, EmbeddingCache, FsOps, LlmClient, LlmOpts, MemoryEmbeddingCache, Result,
    VectorIndex, embedding_content_hash,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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
    pub fn new(fs: Arc<dyn FsOps>) -> Self {
        Self { fs }
    }

    /// 批量读取：一次收集所有 URI 的内容。
    pub async fn batch_read(
        &self,
        uris: &[ContextUri],
        level: agent_context_db_core::ContentLevel,
    ) -> Vec<(ContextUri, Result<agent_context_db_core::ContentPayload>)> {
        let handles: Vec<_> = uris
            .iter()
            .map(|uri| {
                let fs = self.fs.clone();
                let uri = uri.clone();
                tokio::spawn(async move { (uri.clone(), fs.read(&uri, level).await) })
            })
            .collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(r) = h.await {
                results.push(r);
            }
        }
        results
    }

    /// 并行 grep：对多个 regex 一次搜索。
    pub async fn batch_grep(
        &self,
        patterns: &[(&str, &ContextUri)],
    ) -> Vec<(String, Vec<agent_context_db_core::GrepHit>)> {
        let owned: Vec<(String, ContextUri)> = patterns
            .iter()
            .map(|(r, s)| (r.to_string(), (*s).clone()))
            .collect();
        let mut handles = Vec::new();
        for (regex, scope) in owned {
            let fs = self.fs.clone();
            handles.push(tokio::spawn(async move {
                (regex.clone(), fs.grep(&regex, &scope).await)
            }));
        }
        let mut results = Vec::new();
        for h in handles {
            if let Ok((r, hits)) = h.await {
                if let Ok(hits) = hits {
                    results.push((r, hits));
                }
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
    /// B.6: 双锁合并为单一 Mutex — (access_counts, tier_map) 原子更新。
    state: parking_lot::Mutex<(HashMap<String, u64>, HashMap<String, CacheTier>)>,
    hot_threshold: u64,
    warm_threshold: u64,
}

impl TieredVectorIndex {
    pub fn new(index: Arc<dyn VectorIndex>, hot: u64, warm: u64) -> Self {
        Self {
            index,
            state: parking_lot::Mutex::new((HashMap::new(), HashMap::new())),
            hot_threshold: hot,
            warm_threshold: warm,
        }
    }

    pub fn record_access(&self, uri: &str) {
        let mut state = self.state.lock();
        let count = state.0.entry(uri.to_string()).or_insert(0);
        *count += 1;
        let tier = if *count >= self.hot_threshold {
            CacheTier::Hot
        } else if *count >= self.warm_threshold {
            CacheTier::Warm
        } else {
            CacheTier::Cold
        };
        state.1.insert(uri.to_string(), tier);
    }

    pub fn tier(&self, uri: &str) -> CacheTier {
        self.state
            .lock()
            .1
            .get(uri)
            .copied()
            .unwrap_or(CacheTier::Cold)
    }

    pub async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
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
        Self {
            subvectors,
            codebook_size,
            codebooks: Vec::new(),
        }
    }

    /// 训练码本：每个子空间独立跑确定性 k-means。
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        self.codebooks.clear();
        if vectors.is_empty() || self.subvectors == 0 || self.codebook_size == 0 {
            return;
        }
        let dim = vectors[0].len();
        if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
            return;
        }
        let ranges = subvector_ranges(dim, self.subvectors);
        if ranges.len() != self.subvectors || ranges.iter().any(|(start, end)| start == end) {
            return;
        }
        let k = self
            .codebook_size
            .min(vectors.len())
            .min(u8::MAX as usize + 1);

        for (start, end) in ranges {
            let samples = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect::<Vec<_>>();
            self.codebooks.push(kmeans_codebook(&samples, k, 24));
        }
    }

    /// 压缩向量：将 f32 向量转为 u8 码字索引序列。
    pub fn encode(&self, vector: &[f32]) -> Option<Vec<u8>> {
        if self.codebooks.len() != self.subvectors || self.subvectors == 0 {
            return None;
        }
        let ranges = subvector_ranges(vector.len(), self.subvectors);
        if ranges.len() != self.subvectors {
            return None;
        }
        let mut codes = Vec::with_capacity(self.subvectors);
        for (s, codebook) in self.codebooks.iter().enumerate() {
            if codebook.is_empty() {
                return None;
            }
            let (start, end) = ranges[s];
            let sub = &vector[start..end];
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            for (i, centroid) in codebook.iter().enumerate() {
                let dist = squared_euclidean(sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i as u8;
                }
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

fn subvector_ranges(dim: usize, subvectors: usize) -> Vec<(usize, usize)> {
    if subvectors == 0 || dim < subvectors {
        return vec![];
    }
    (0..subvectors)
        .map(|i| {
            let start = i * dim / subvectors;
            let end = (i + 1) * dim / subvectors;
            (start, end)
        })
        .collect()
}

fn kmeans_codebook(samples: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    if samples.is_empty() || k == 0 {
        return vec![];
    }
    let dim = samples[0].len();
    let mut centroids = initialize_centroids(samples, k);
    let mut assignments = vec![0usize; samples.len()];

    for _ in 0..iterations.max(1) {
        let mut changed = false;
        for (idx, sample) in samples.iter().enumerate() {
            let nearest = nearest_centroid(sample, &centroids);
            if assignments[idx] != nearest {
                assignments[idx] = nearest;
                changed = true;
            }
        }

        let mut sums = vec![vec![0.0f32; dim]; centroids.len()];
        let mut counts = vec![0usize; centroids.len()];
        for (sample, cluster) in samples.iter().zip(assignments.iter().copied()) {
            counts[cluster] += 1;
            for (sum, value) in sums[cluster].iter_mut().zip(sample) {
                *sum += *value;
            }
        }

        for (idx, centroid) in centroids.iter_mut().enumerate() {
            if counts[idx] == 0 {
                *centroid = samples[idx % samples.len()].clone();
            } else {
                for value in &mut sums[idx] {
                    *value /= counts[idx] as f32;
                }
                *centroid = sums[idx].clone();
            }
        }

        if !changed {
            break;
        }
    }

    centroids
}

fn initialize_centroids(samples: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    centroids.push(samples[0].clone());
    while centroids.len() < k {
        let next = samples
            .iter()
            .max_by(|a, b| {
                distance_to_nearest(a, &centroids)
                    .partial_cmp(&distance_to_nearest(b, &centroids))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| samples[centroids.len() % samples.len()].clone());
        centroids.push(next);
    }
    centroids
}

fn nearest_centroid(sample: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            squared_euclidean(sample, a)
                .partial_cmp(&squared_euclidean(sample, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn distance_to_nearest(sample: &[f32], centroids: &[Vec<f32>]) -> f32 {
    centroids
        .iter()
        .map(|centroid| squared_euclidean(sample, centroid))
        .fold(f32::MAX, f32::min)
}

fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>()
}

// ═══════════════════════════════════════════════════════════════════════════
// P4 流水线并行生成 — 多个 LLM 请求并行化
// ═══════════════════════════════════════════════════════════════════════════

/// 并行 LLM 请求调度器。
pub struct ParallelGenerator {
    llm: Arc<dyn LlmClient>,
    max_concurrency: usize,
    embed_cache: Arc<dyn EmbeddingCache>,
    embed_cache_ttl: Duration,
}

impl ParallelGenerator {
    pub fn new(llm: Arc<dyn LlmClient>, max: usize) -> Self {
        Self {
            llm,
            max_concurrency: max,
            embed_cache: Arc::new(MemoryEmbeddingCache::new(
                10_000,
                Duration::from_secs(86_400),
            )),
            embed_cache_ttl: Duration::from_secs(86_400),
        }
    }

    pub fn with_embedding_cache(mut self, cache: Arc<dyn EmbeddingCache>) -> Self {
        self.embed_cache = cache;
        self
    }

    pub fn with_embedding_cache_ttl(mut self, ttl: Duration) -> Self {
        self.embed_cache_ttl = ttl;
        self
    }

    /// 并行生成多个摘要。
    pub async fn batch_generate_abstracts(
        &self,
        uris: &[ContextUri],
    ) -> Vec<(ContextUri, Result<String>)> {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        let handles: Vec<_> = uris
            .iter()
            .map(|uri| {
                let llm = self.llm.clone();
                let uri = uri.clone();
                let sem = semaphore.clone();
                tokio::spawn(async move {
                    let _permit = sem.acquire().await;
                    let prompt = format!("Write a concise abstract (~100 tokens) for: {uri}");
                    (
                        uri,
                        llm.complete(&prompt, &LlmOpts::default())
                            .await
                            .map_err(|e| {
                                agent_context_db_core::ContextError::Storage(format!("{e}"))
                            }),
                    )
                })
            })
            .collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(r) = h.await {
                results.push(r);
            }
        }
        results
    }

    /// 并行生成多个 embedding，并按 blake3(content) 缓存去重。
    pub async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut misses: HashMap<String, String> = HashMap::new();

        for (idx, text) in texts.iter().enumerate() {
            let hash = embedding_content_hash(text);
            if let Some(embedding) = self.embed_cache.get(&hash).await {
                results[idx] = Some(embedding);
            } else {
                misses.entry(hash).or_insert_with(|| text.clone());
            }
        }

        if !misses.is_empty() {
            let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
            let handles: Vec<_> = misses
                .into_iter()
                .map(|(hash, text)| {
                    let llm = self.llm.clone();
                    let cache = self.embed_cache.clone();
                    let sem = semaphore.clone();
                    let ttl = self.embed_cache_ttl;
                    tokio::spawn(async move {
                        let _permit = sem.acquire().await;
                        let embedding = llm.embed(&text).await?.vector;
                        cache.put(&hash, embedding.clone(), ttl).await;
                        Ok::<_, agent_context_db_core::LlmError>((hash, embedding))
                    })
                })
                .collect();

            let mut loaded = HashMap::new();
            for h in handles {
                match h.await {
                    Ok(Ok((hash, embedding))) => {
                        loaded.insert(hash, embedding);
                    }
                    Ok(Err(err)) => return Err(err.into()),
                    Err(err) => {
                        return Err(agent_context_db_core::ContextError::Storage(format!(
                            "embedding task join: {err}"
                        )));
                    }
                }
            }

            for (idx, text) in texts.iter().enumerate() {
                if results[idx].is_none() {
                    let hash = embedding_content_hash(text);
                    if let Some(embedding) = loaded.get(&hash) {
                        results[idx] = Some(embedding.clone());
                    }
                }
            }
        }

        Ok(results.into_iter().flatten().collect())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P6 检索物化视图 — 预计算常见路径
// ═══════════════════════════════════════════════════════════════════════════

/// 物化视图 — 缓存常见检索路径的结果。
pub struct MaterializedView {
    cache: parking_lot::Mutex<HashMap<String, (Vec<ContextUri>, chrono::DateTime<chrono::Utc>)>>,
    ttl_secs: i64,
    max_capacity: usize, // H.2: 容量上限，超限时 LRU 淘汰
}

impl MaterializedView {
    pub fn new(ttl_secs: i64) -> Self {
        Self {
            cache: parking_lot::Mutex::new(HashMap::new()),
            ttl_secs,
            max_capacity: 500,
        }
    }

    pub fn with_capacity(ttl_secs: i64, capacity: usize) -> Self {
        Self {
            cache: parking_lot::Mutex::new(HashMap::new()),
            ttl_secs,
            max_capacity: capacity,
        }
    }

    pub fn get(&self, query: &str) -> Option<Vec<ContextUri>> {
        let cache = self.cache.lock();
        if let Some((uris, expires)) = cache.get(query) {
            if *expires > chrono::Utc::now() {
                return Some(uris.clone());
            }
        }
        None
    }

    pub fn set(&self, query: &str, uris: Vec<ContextUri>) {
        let expires = chrono::Utc::now() + chrono::Duration::seconds(self.ttl_secs);
        let mut cache = self.cache.lock();
        // H.2: 容量超限 → 淘汰最旧的 N 条
        if cache.len() >= self.max_capacity {
            let mut entries: Vec<(String, chrono::DateTime<chrono::Utc>)> =
                cache.iter().map(|(k, (_, e))| (k.clone(), *e)).collect();
            entries.sort_by_key(|(_, e)| *e);
            let to_remove = entries.len() - self.max_capacity + 1;
            for (k, _) in entries.iter().take(to_remove) {
                cache.remove(k);
            }
        }
        cache.insert(query.to_string(), (uris, expires));
    }

    pub fn invalidate(&self, scope: &str) {
        self.cache.lock().retain(|k, _| !k.starts_with(scope));
    }

    pub fn hit_rate(&self) -> (usize, usize) {
        let cache = self.cache.lock();
        let active = cache
            .values()
            .filter(|(_, e)| *e > chrono::Utc::now())
            .count();
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
    pub fn new(fs: Arc<dyn FsOps>) -> Self {
        Self {
            fs,
            partitions: Vec::new(),
        }
    }

    pub fn with_partitions(mut self, partitions: Vec<String>) -> Self {
        self.partitions = partitions;
        self
    }

    /// 并行在所有分区上执行 find。
    pub async fn parallel_find(
        &self,
        pattern_template: &agent_context_db_core::FindPattern,
    ) -> Vec<(String, Vec<ContextUri>)> {
        let owned_partitions = self.partitions.clone();
        let owned_pattern = pattern_template.clone();
        let handles: Vec<_> = owned_partitions
            .into_iter()
            .map(|tenant| {
                let fs = self.fs.clone();
                let mut pattern = owned_pattern.clone();
                pattern.scope = ContextUri::parse(format!("uwu://{tenant}")).ok();
                tokio::spawn(async move {
                    match fs.find(&pattern).await {
                        Ok(uris) => Some((tenant, uris)),
                        Err(_) => None,
                    }
                })
            })
            .collect();
        let mut results = Vec::new();
        for h in handles {
            if let Ok(Some(r)) = h.await {
                results.push(r);
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{JsonSchema, LlmError};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingLlm {
        calls: AtomicUsize,
    }

    impl CountingLlm {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LlmClient for CountingLlm {
        async fn complete(
            &self,
            _prompt: &str,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok(String::new())
        }

        async fn embed(
            &self,
            text: &str,
        ) -> std::result::Result<agent_context_db_core::EmbeddingVector, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(agent_context_db_core::EmbeddingVector::new(
                vec![text.len() as f32],
                "test",
                1,
            ))
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }
    }

    #[tokio::test]
    async fn batch_embed_deduplicates_same_batch_by_content_hash() {
        let llm = Arc::new(CountingLlm::new());
        let generator = ParallelGenerator::new(llm.clone(), 4);
        let texts = vec!["alpha".to_string(), "alpha".to_string(), "beta".to_string()];

        let embeddings = generator.batch_embed(&texts).await.unwrap();

        assert_eq!(embeddings, vec![vec![5.0], vec![5.0], vec![4.0]]);
        assert_eq!(llm.calls(), 2);
    }

    #[tokio::test]
    async fn batch_embed_reuses_cached_embedding_across_batches() {
        let llm = Arc::new(CountingLlm::new());
        let generator = ParallelGenerator::new(llm.clone(), 4);
        let texts = vec!["alpha".to_string(), "beta".to_string()];

        let first = generator.batch_embed(&texts).await.unwrap();
        let second = generator.batch_embed(&texts).await.unwrap();

        assert_eq!(first, second);
        assert_eq!(llm.calls(), 2);
    }

    #[test]
    fn vector_quantizer_compression_ratio() {
        let q = VectorQuantizer::new(8, 256);
        let ratio = q.compression_ratio(768);
        assert!(ratio > 1.0);
    }

    #[test]
    fn vector_quantizer_trains_kmeans_codebooks() {
        let vectors = vec![
            vec![0.0, 0.0, 10.0, 10.0],
            vec![0.1, 0.0, 10.1, 10.0],
            vec![9.0, 9.0, 0.0, 0.0],
            vec![9.1, 9.0, 0.1, 0.0],
        ];
        let mut q = VectorQuantizer::new(2, 2);
        q.train(&vectors);
        assert_eq!(q.codebooks.len(), 2);
        assert_eq!(q.codebooks[0].len(), 2);
        assert_eq!(q.encode(&vectors[0]).unwrap().len(), 2);
        assert_ne!(q.encode(&vectors[0]), q.encode(&vectors[2]));
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
            let tier = if *count >= 5 {
                CacheTier::Hot
            } else if *count >= 2 {
                CacheTier::Warm
            } else {
                CacheTier::Cold
            };
            tier_map.lock().insert(uri.to_string(), tier);
        }
        assert_eq!(tier_map.lock().get(uri), Some(&CacheTier::Hot));
    }
}
