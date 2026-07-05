//! K.1: UwuConfig — 集中管理配置常量。

use serde::{Deserialize, Serialize};

// K.2: ArcSwap 热更新包装器。
use std::sync::Arc;

/// 可热更新的配置包装器 — clone 零成本（Arc）。
pub type ConfigHandle = Arc<parking_lot::RwLock<UwuConfig>>;

pub fn config_handle(config: UwuConfig) -> ConfigHandle {
    Arc::new(parking_lot::RwLock::new(config))
}

/// 上下文数据库统一配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UwuConfig {
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub cache: CacheConfig,
    pub lifecycle: LifecycleConfig,
    pub retrieval: RetrievalConfig,
}

impl Default for UwuConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            llm: LlmConfig::default(),
            cache: CacheConfig::default(),
            lifecycle: LifecycleConfig::default(),
            retrieval: RetrievalConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub backend: String,          // "postgres" | "sled" | "memory"
    pub max_connections: usize,
    pub batch_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self { backend: "memory".into(), max_connections: 10, batch_size: 100 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub rpm_limit: u32,           // requests per minute
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self { model: "gpt-4o-mini".into(), max_tokens: 4096, temperature: 0.0, rpm_limit: 60 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub backend: String,         // "memory" | "redis"
    pub memory_capacity: usize,
    pub default_ttl_secs: u64,
    pub redis_url: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { backend: "memory".into(), memory_capacity: 1000, default_ttl_secs: 300, redis_url: None }
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
        Self { default_stability: 7.0, archive_threshold: 0.15, delete_threshold: 0.05 }
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
        Self { default_budget: 8000, max_hops: 2, decay_factor: 0.5 }
    }
}
