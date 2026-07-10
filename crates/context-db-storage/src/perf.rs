//! 存储层性能优化：P5 WAL+批量提交 + P10 内容压缩去重。

use agent_context_db_core::{ContentRepo, ContextEntry, ContextError, MvccVersion, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// P5 Write-Ahead Log + 批量提交
// ═══════════════════════════════════════════════════════════════════════════

/// 批量写缓冲区。
pub struct BatchWriteBuffer {
    inner: Arc<dyn ContentRepo>,
    /// 待提交的写操作
    pending: Mutex<
        Vec<(
            ContextEntry,
            tokio::sync::oneshot::Sender<Result<MvccVersion>>,
        )>,
    >,
    /// 批量大小阈值
    batch_size: usize,
    /// 刷新间隔
    flush_interval: std::time::Duration,
    /// B.4: 后台刷新任务句柄，Drop 时 abort。
    _flush_handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Drop for BatchWriteBuffer {
    fn drop(&mut self) {
        if let Some(handle) = self._flush_handle.lock().take() {
            handle.abort();
        }
    }
}

impl BatchWriteBuffer {
    pub fn new(inner: Arc<dyn ContentRepo>, batch_size: usize) -> Arc<Self> {
        let this = Arc::new(Self {
            inner,
            pending: Mutex::new(Vec::new()),
            batch_size,
            flush_interval: std::time::Duration::from_millis(100),
            _flush_handle: Mutex::new(None),
        });
        // 启动后台刷新任务，保存 JoinHandle
        let this_clone = Arc::downgrade(&this);
        let handle = tokio::spawn(async move {
            loop {
                let Some(strong) = this_clone.upgrade() else {
                    break; // Arc 已 drop，退出循环
                };
                tokio::time::sleep(strong.flush_interval).await;
                strong.flush().await;
            }
        });
        *this._flush_handle.lock() = Some(handle);
        this
    }

    /// 提交单个写入（异步批量合并）。
    pub async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let should_flush = {
            let mut guard = self.pending.lock(); // E.3: 一次 lock
            guard.push((entry, tx));
            guard.len() >= self.batch_size
        };
        if should_flush {
            self.flush().await;
        }
        rx.await
            .map_err(|_| ContextError::Storage("batch write cancelled".into()))?
    }

    async fn flush(&self) {
        let batch: Vec<_> = {
            let mut guard = self.pending.lock();
            if guard.is_empty() {
                return;
            }
            std::mem::take(&mut *guard)
        };
        // 批量写入
        for (entry, tx) in batch {
            let result = self.inner.write(entry).await;
            let _ = tx.send(result);
        }
    }
}

/// Write-Ahead Log 条目。
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub sequence: u64,
    pub uri: String,
    pub entry_json: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// WAL + 批量提交组合。
pub struct WriteAheadLogger {
    entries: Mutex<Vec<WalEntry>>,
    seq: std::sync::atomic::AtomicU64,
}

impl WriteAheadLogger {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            seq: std::sync::atomic::AtomicU64::new(1),
        }
    }

    pub fn log(&self, uri: &str, entry: &ContextEntry) -> u64 {
        let seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Ok(json) = serde_json::to_string(entry) {
            self.entries.lock().push(WalEntry {
                sequence: seq,
                uri: uri.to_string(),
                entry_json: json,
                timestamp: chrono::Utc::now(),
            });
        }
        seq
    }

    pub fn drain(&self) -> Vec<WalEntry> {
        std::mem::take(&mut *self.entries.lock())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// P10 内容压缩去重 — blake3 内容寻址 + zstd 压缩
// ═══════════════════════════════════════════════════════════════════════════

/// 内容哈希（blake3）。
pub fn content_hash(data: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

/// zstd 压缩包装。
pub fn compress(data: &[u8], level: i32) -> Vec<u8> {
    zstd::encode_all(data, level).unwrap_or_else(|_| data.to_vec())
}

/// zstd 解压。
pub fn decompress(compressed: &[u8]) -> Option<Vec<u8>> {
    zstd::decode_all(compressed).ok()
}

/// 内容去重存储 — 相同内容只存一份（B.5: 三锁合并为 Mutex<Inner>）。
///
/// 支持可选持久化后端（H.5）：注入 `uwu_database::Cache` 后：
/// - `store`：内存写入 + 后端 write-through（best-effort）
/// - `load`：内存 miss 时回落到后端加载并回填内存
///
/// 用途：跨进程共享去重块（例如 Redis 共享 blob 池），进程重启后不丢历史 dedup 命中。
pub struct DedupStore {
    inner: Mutex<DedupInner>,
    /// 可选持久化后端（如 Redis / disk KV）。key 布局：`dedup:blob:{hash}`、`dedup:idx:{uri}`。
    persistence: Option<std::sync::Arc<dyn uwu_database::Cache>>,
}

struct DedupInner {
    blobs: HashMap<String, Vec<u8>>,
    index: HashMap<String, String>,
    stats: DedupStats,
}

#[derive(Debug, Clone, Default)]
pub struct DedupStats {
    pub total_writes: u64,
    pub dedup_hits: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
}

impl DedupStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(DedupInner {
                blobs: HashMap::new(),
                index: HashMap::new(),
                stats: DedupStats::default(),
            }),
            persistence: None,
        }
    }

    /// 挂载可选持久化后端（H.5）。传入 `uwu_database::Cache` 即可获得跨进程/重启后的
    /// dedup 命中保留能力。
    pub fn with_persistence(mut self, cache: std::sync::Arc<dyn uwu_database::Cache>) -> Self {
        self.persistence = Some(cache);
        self
    }

    fn blob_key(hash: &str) -> String {
        format!("dedup:blob:{hash}")
    }
    fn idx_key(uri: &str) -> String {
        format!("dedup:idx:{uri}")
    }

    /// 存储内容（自动去重+压缩）。若挂载了持久化后端，则 write-through 到后端（best-effort）。
    pub fn store(&self, uri: &str, data: &[u8]) -> String {
        let hash = content_hash(data);
        let compressed = {
            let mut inner = self.inner.lock();
            inner.stats.total_writes += 1;

            if inner.blobs.contains_key(&hash) {
                inner.stats.dedup_hits += 1;
                None
            } else {
                let compressed = compress(data, 3);
                inner.stats.raw_bytes += data.len() as u64;
                inner.stats.compressed_bytes += compressed.len() as u64;
                inner.blobs.insert(hash.clone(), compressed.clone());
                Some(compressed)
            }
        };
        {
            let mut inner = self.inner.lock();
            inner.index.insert(uri.to_string(), hash.clone());
        }

        // 持久化 write-through（锁外，避免阻塞其他写入）。
        if let Some(cache) = self.persistence.as_ref() {
            let cache = cache.clone();
            let hash_c = hash.clone();
            let uri_c = uri.to_string();
            // 在 tokio 运行时下 fire-and-forget；同步上下文则跳过（不阻塞调用方）。
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move {
                    if let Some(bytes) = compressed {
                        let _ = cache.set(&Self::blob_key(&hash_c), &bytes, None).await;
                    }
                    let _ = cache
                        .set(&Self::idx_key(&uri_c), hash_c.as_bytes(), None)
                        .await;
                });
            }
        }
        hash
    }

    /// 读取内容（自动解压）；内存 miss 时回落到持久化后端。
    pub fn load(&self, uri: &str) -> Option<Vec<u8>> {
        // 快路径：内存
        {
            let inner = self.inner.lock();
            if let Some(hash) = inner.index.get(uri) {
                if let Some(compressed) = inner.blobs.get(hash) {
                    return decompress(compressed);
                }
            }
        }
        // 慢路径：持久化后端
        let cache = self.persistence.as_ref()?.clone();
        let uri_owned = uri.to_string();
        // 若无 tokio 运行时则放弃（保持同步 API）
        let rt = tokio::runtime::Handle::try_current().ok()?;
        let (hash, compressed) = rt.block_on(async move {
            let hash_bytes = cache.get(&Self::idx_key(&uri_owned)).await.ok()??;
            let hash = String::from_utf8(hash_bytes).ok()?;
            let compressed = cache.get(&Self::blob_key(&hash)).await.ok()??;
            Some((hash, compressed))
        })?;
        let data = decompress(&compressed)?;
        // 回填内存
        let mut inner = self.inner.lock();
        inner.blobs.insert(hash.clone(), compressed);
        inner.index.insert(uri.to_string(), hash);
        Some(data)
    }

    pub fn stats(&self) -> DedupStats {
        self.inner.lock().stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_store_same_content_hashes_once() {
        let store = DedupStore::new();
        store.store("uri-1", b"hello world");
        store.store("uri-2", b"hello world");
        let stats = store.stats();
        assert_eq!(stats.total_writes, 2);
        assert_eq!(stats.dedup_hits, 1);
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let data = b"repeated content ".repeat(100);
        let c = compress(&data, 3);
        assert!(c.len() < data.len());
        let d = decompress(&c).unwrap();
        assert_eq!(d, data);
    }

    #[test]
    fn content_hash_is_deterministic() {
        assert_eq!(content_hash(b"hello"), content_hash(b"hello"));
        assert_ne!(content_hash(b"hello"), content_hash(b"world"));
    }

    #[test]
    fn wal_logs_and_drains() {
        let wal = WriteAheadLogger::new();
        let uri = agent_context_db_core::ContextUri::parse("uwu://t/x").unwrap();
        let entry = agent_context_db_core::ContextEntry::new_text(
            uri,
            agent_context_db_core::TenantId(uuid::Uuid::nil()),
            "test",
        );
        wal.log("uwu://t/x", &entry);
        assert_eq!(wal.drain().len(), 1);
    }
}
