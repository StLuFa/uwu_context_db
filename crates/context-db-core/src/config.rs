//! K.1: UwuConfig — 集中管理配置常量。
//! K.2: ArcSwap 热更新 —— 无锁读，写时替换整份 Arc。

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

/// 可热更新的配置句柄。
///
/// - `load()` 无锁读，返回 `Guard<Arc<UwuConfig>>`
/// - `store(new)` 原子替换整份配置；旧配置在最后一个 Guard 释放后回收
/// - 读取远远多于写入的场景，性能优于 `RwLock`
pub type ConfigHandle = Arc<ArcSwap<UwuConfig>>;

/// 构造热更新句柄。
pub fn config_handle(config: UwuConfig) -> ConfigHandle {
    Arc::new(ArcSwap::from_pointee(config))
}

/// 便捷更新：读旧配置 → 应用变换 → 原子写入新配置。
pub fn update_config(handle: &ConfigHandle, mutate: impl FnOnce(&mut UwuConfig)) {
    let current = handle.load();
    let mut next = (**current).clone();
    mutate(&mut next);
    handle.store(Arc::new(next));
}

/// 上下文数据库统一配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UwuConfig {
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub cache: CacheConfig,
    pub lifecycle: LifecycleConfig,
    pub retrieval: RetrievalConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SqlStorageBackend {
    Sqlite,
    Postgres,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorStorageBackend {
    QdrantEdge,
    Memory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// SQL 内容后端。默认使用嵌入式 SQLite。
    pub backend: SqlStorageBackend,
    /// SQL 连接 URL。SQLite 默认持久化到当前目录下的 `uwu_context.db`。
    pub database_url: String,
    pub max_connections: usize,
    pub batch_size: usize,
    /// 向量后端。默认使用进程内 Qdrant Edge。
    pub vector_backend: VectorStorageBackend,
    /// Qdrant Edge 数据目录。
    pub vector_url: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: SqlStorageBackend::Sqlite,
            database_url: "sqlite://uwu_context.db?mode=rwc".into(),
            max_connections: 4,
            batch_size: 100,
            vector_backend: VectorStorageBackend::QdrantEdge,
            vector_url: Some("./uwu_context_vectors".into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String, // "openai" | "anthropic" | "generic_http"
    pub model: String,
    pub embedding_model: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub rpm_limit: u32, // requests per minute
    pub base_url: Option<String>,
    pub embedding_base_url: Option<String>,
    pub completion_path: Option<String>,
    pub embedding_path: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub timeout_secs: Option<u64>,
    pub headers: BTreeMap<String, String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "openai".into(),
            model: "gpt-4o-mini".into(),
            embedding_model: Some("text-embedding-3-small".into()),
            max_tokens: 4096,
            temperature: 0.0,
            rpm_limit: 60,
            base_url: None,
            embedding_base_url: None,
            completion_path: None,
            embedding_path: None,
            api_key: None,
            api_key_env: Some("OPENAI_API_KEY".into()),
            timeout_secs: Some(60),
            headers: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub backend: String, // "memory" | "redis"
    pub memory_capacity: usize,
    pub default_ttl_secs: u64,
    pub redis_url: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            backend: "memory".into(),
            memory_capacity: 1000,
            default_ttl_secs: 300,
            redis_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    pub default_stability: f64,
    pub archive_threshold: f32,
    pub delete_threshold: f32,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            default_stability: 7.0,
            archive_threshold: 0.15,
            delete_threshold: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub default_budget: usize,
    pub max_hops: usize,
    pub decay_factor: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_budget: 8000,
            max_hops: 2,
            decay_factor: 0.5,
        }
    }
}
