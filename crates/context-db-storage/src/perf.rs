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
    pending: Mutex<Vec<(ContextEntry, tokio::sync::oneshot::Sender<Result<MvccVersion>>)>>,
    /// 批量大小阈值
    batch_size: usize,
    /// 刷新间隔
    flush_interval: std::time::Duration,
}

impl BatchWriteBuffer {
    pub fn new(inner: Arc<dyn ContentRepo>, batch_size: usize) -> Arc<Self> {
        let this = Arc::new(Self {
            inner,
            pending: Mutex::new(Vec::new()),
            batch_size,
            flush_interval: std::time::Duration::from_millis(100),
        });
        // 启动后台刷新任务
        let this_clone = this.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(this_clone.flush_interval).await;
                this_clone.flush().await;
            }
        });
        this
    }

    /// 提交单个写入（异步批量合并）。
    pub async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.lock().push((entry, tx));
        if self.pending.lock().len() >= self.batch_size {
            self.flush().await;
        }
        rx.await.map_err(|_| ContextError::Storage("batch write cancelled".into()))?
    }

    async fn flush(&self) {
        let batch: Vec<_> = {
            let mut guard = self.pending.lock();
            if guard.is_empty() { return; }
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
        Self { entries: Mutex::new(Vec::new()), seq: std::sync::atomic::AtomicU64::new(1) }
    }

    pub fn log(&self, uri: &str, entry: &ContextEntry) -> u64 {
        let seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Ok(json) = serde_json::to_string(entry) {
            self.entries.lock().push(WalEntry {
                sequence: seq, uri: uri.to_string(), entry_json: json,
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

/// 内容去重存储 —— 相同内容只存一份。
pub struct DedupStore {
    /// hash → compressed bytes
    blobs: Mutex<HashMap<String, Vec<u8>>>,
    /// uri → hash
    index: Mutex<HashMap<String, String>>,
    stats: Mutex<DedupStats>,
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
            blobs: Mutex::new(HashMap::new()),
            index: Mutex::new(HashMap::new()),
            stats: Mutex::new(DedupStats::default()),
        }
    }

    /// 存储内容（自动去重+压缩）。
    pub fn store(&self, uri: &str, data: &[u8]) -> String {
        let hash = content_hash(data);
        let mut stats = self.stats.lock();
        stats.total_writes += 1;

        let mut blobs = self.blobs.lock();
        if blobs.contains_key(&hash) {
            stats.dedup_hits += 1;
        } else {
            let compressed = compress(data, 3);
            stats.raw_bytes += data.len() as u64;
            stats.compressed_bytes += compressed.len() as u64;
            blobs.insert(hash.clone(), compressed);
        }

        self.index.lock().insert(uri.to_string(), hash.clone());
        hash
    }

    /// 读取内容（自动解压）。
    pub fn load(&self, uri: &str) -> Option<Vec<u8>> {
        let hash = self.index.lock().get(uri)?.clone();
        let compressed = self.blobs.lock().get(&hash)?.clone();
        decompress(&compressed)
    }

    pub fn stats(&self) -> DedupStats { self.stats.lock().clone() }
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
            uri, agent_context_db_core::TenantId(uuid::Uuid::nil()), "test",
        );
        wal.log("uwu://t/x", &entry);
        assert_eq!(wal.drain().len(), 1);
    }
}
