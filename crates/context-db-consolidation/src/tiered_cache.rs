//! TieredCache — 三层冷热缓存分离（内存 hot → warm → PG 兜底）。

use agent_context_db_core::{ContentPayload, ContextUri};
use agent_context_db_retrieve::PrefetchPrediction;
use moka::policy::Expiry;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 缓存层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Hot,
    Warm,
    Cold,
}

/// 缓存条目。
#[derive(Clone)]
struct CacheEntry {
    payload: ContentPayload,
    inserted: Instant,
    ttl: Duration,
    access_count: u64,
    predicted_probability: f32,
}

#[derive(Clone)]
struct CacheEntryExpiry;

impl Expiry<String, CacheEntry> for CacheEntryExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &CacheEntry,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &CacheEntry,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PredictivePromotionConfig {
    pub hot_threshold: f32,
    pub warm_threshold: f32,
    pub hot_ttl_multiplier: f32,
    pub warm_ttl_multiplier: f32,
}

impl PredictivePromotionConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        crate::validate_unit_f32("hot_threshold", self.hot_threshold)?;
        crate::validate_unit_f32("warm_threshold", self.warm_threshold)?;
        if self.warm_threshold > self.hot_threshold {
            return Err(crate::ConfigError(
                "warm_threshold must not exceed hot_threshold".into(),
            ));
        }
        for (name, value) in [
            ("hot_ttl_multiplier", self.hot_ttl_multiplier),
            ("warm_ttl_multiplier", self.warm_ttl_multiplier),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(crate::ConfigError(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        Ok(())
    }
}

impl Default for PredictivePromotionConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 0.70,
            warm_threshold: 0.25,
            hot_ttl_multiplier: 1.5,
            warm_ttl_multiplier: 0.75,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CachePromotionReport {
    pub hot_promoted: usize,
    pub warm_promoted: usize,
    pub skipped: usize,
    pub warm_evicted: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TieredCacheStats {
    pub hot_entries: u64,
    pub warm_entries: usize,
    pub warm_accesses: u64,
}

/// 三层冷热缓存。
pub struct TieredCache {
    hot: moka::future::Cache<String, CacheEntry>,
    warm: parking_lot::Mutex<HashMap<String, CacheEntry>>,
    warm_capacity: usize,
    hot_ttl: Duration,
    warm_ttl: Duration,
    predictive_config: PredictivePromotionConfig,
}

impl TieredCache {
    pub fn new(
        hot_cap: usize,
        warm_cap: usize,
        predictive_config: PredictivePromotionConfig,
    ) -> Result<Self, crate::ConfigError> {
        if hot_cap == 0 || warm_cap == 0 {
            return Err(crate::ConfigError(
                "cache capacities must be nonzero".into(),
            ));
        }
        predictive_config.validate()?;
        Ok(Self {
            hot: moka::future::Cache::builder()
                .max_capacity(hot_cap.max(1) as u64)
                .expire_after(CacheEntryExpiry)
                .build(),
            warm: parking_lot::Mutex::new(HashMap::new()),
            warm_capacity: warm_cap.max(1),
            hot_ttl: Duration::from_secs(300),
            warm_ttl: Duration::from_secs(3600),
            predictive_config,
        })
    }

    /// 读取 — 先查 hot，再查 warm；warm 命中会提升到 hot，未命中由调用方回源。
    pub async fn get(&self, uri: &ContextUri) -> Option<ContentPayload> {
        let key = uri.to_string();
        if let Some(mut entry) = self.hot.get(&key).await {
            entry.access_count += 1;
            let payload = entry.payload.clone();
            self.hot.insert(key, entry).await;
            return Some(payload);
        }

        let warm_entry = {
            let mut warm = self.warm.lock();
            if let Some(entry) = warm.get_mut(&key) {
                if entry.inserted.elapsed() < entry.ttl {
                    entry.access_count += 1;
                    Some(entry.clone())
                } else {
                    warm.remove(&key);
                    None
                }
            } else {
                None
            }
        };
        if let Some(entry) = warm_entry {
            let payload = entry.payload.clone();
            self.promote_to_hot(
                &key,
                entry.payload,
                self.hot_ttl,
                entry.predicted_probability,
            )
            .await;
            return Some(payload);
        }
        None
    }

    /// 写入 — 显式访问直接进 hot。
    pub async fn put(&self, uri: &ContextUri, payload: ContentPayload) {
        self.promote_to_hot(&uri.to_string(), payload, self.hot_ttl, 1.0)
            .await;
    }

    /// 根据预测结果主动预热缓存。高概率进 hot，低概率进 warm，低于阈值跳过。
    pub async fn promote_predictions(
        &self,
        predictions: &[PrefetchPrediction],
        loaded: &[(ContextUri, ContentPayload)],
    ) -> CachePromotionReport {
        let mut by_uri: HashMap<String, (&PrefetchPrediction, ContentPayload)> = HashMap::new();
        for (uri, payload) in loaded {
            if let Some(prediction) = predictions.iter().find(|p| p.uri == *uri) {
                by_uri.insert(uri.to_string(), (prediction, payload.clone()));
            }
        }

        let mut report = CachePromotionReport::default();
        for (key, (prediction, payload)) in by_uri {
            let probability = prediction.probability.clamp(0.0, 1.0);
            if probability >= self.predictive_config.hot_threshold {
                let ttl = scaled_ttl(self.hot_ttl, self.predictive_config.hot_ttl_multiplier);
                self.promote_to_hot(&key, payload, ttl, probability).await;
                report.hot_promoted += 1;
            } else if probability >= self.predictive_config.warm_threshold {
                let ttl = scaled_ttl(self.warm_ttl, self.predictive_config.warm_ttl_multiplier);
                self.promote_to_warm(&key, payload, ttl, probability);
                report.warm_promoted += 1;
            } else {
                report.skipped += 1;
            }
        }
        report.warm_evicted = self.enforce_warm_capacity();
        report
    }

    async fn promote_to_hot(
        &self,
        key: &str,
        payload: ContentPayload,
        ttl: Duration,
        predicted_probability: f32,
    ) {
        self.hot
            .insert(
                key.to_string(),
                CacheEntry {
                    payload,
                    inserted: Instant::now(),
                    ttl,
                    access_count: 1,
                    predicted_probability,
                },
            )
            .await;
    }

    fn promote_to_warm(
        &self,
        key: &str,
        payload: ContentPayload,
        ttl: Duration,
        predicted_probability: f32,
    ) {
        self.warm.lock().insert(
            key.to_string(),
            CacheEntry {
                payload,
                inserted: Instant::now(),
                ttl,
                access_count: 0,
                predicted_probability,
            },
        );
    }

    /// Sleeptime 再平衡 — 清理过期项并按预测概率/访问次数淘汰 warm。
    pub async fn rebalance(&self) {
        self.hot.run_pending_tasks().await;
        let now = Instant::now();
        self.warm.lock().retain(|_, e| now - e.inserted < e.ttl);
        self.enforce_warm_capacity();
    }

    pub fn stats(&self) -> TieredCacheStats {
        let warm = self.warm.lock();
        TieredCacheStats {
            hot_entries: self.hot.entry_count(),
            warm_entries: warm.len(),
            warm_accesses: warm.values().map(|entry| entry.access_count).sum(),
        }
    }

    fn enforce_warm_capacity(&self) -> usize {
        let mut warm = self.warm.lock();
        if warm.len() <= self.warm_capacity {
            return 0;
        }
        let mut ranked: Vec<_> = warm
            .iter()
            .map(|(key, entry)| {
                let score =
                    entry.predicted_probability + (entry.access_count as f32).ln_1p() * 0.05;
                (key.clone(), score, entry.inserted)
            })
            .collect();
        ranked.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.2.elapsed().cmp(&b.2.elapsed()))
        });
        let evict_count = warm.len() - self.warm_capacity;
        for (key, _, _) in ranked.into_iter().take(evict_count) {
            warm.remove(&key);
        }
        evict_count
    }
}

fn scaled_ttl(base: Duration, multiplier: f32) -> Duration {
    let secs = (base.as_secs_f32() * multiplier.max(0.1)).round().max(1.0);
    Duration::from_secs(secs as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::ContentLevel;

    fn payload(text: &str) -> ContentPayload {
        ContentPayload::Text {
            sparse: text.into(),
            dense: text.into(),
            full: text.into(),
        }
    }

    fn prediction(uri: &str, probability: f32) -> PrefetchPrediction {
        PrefetchPrediction {
            uri: ContextUri::parse(uri).unwrap(),
            probability,
            pattern: agent_context_db_retrieve::AccessPattern::Sequential,
            prefetch_level: ContentLevel::L0,
        }
    }

    #[tokio::test]
    async fn predictive_promotion_splits_hot_warm_and_skip() {
        let cache = TieredCache::new(8, 8, PredictivePromotionConfig::default()).unwrap();
        let predictions = vec![
            prediction("uwu://t/agent/a/memory/fact/hot", 0.9),
            prediction("uwu://t/agent/a/memory/fact/warm", 0.4),
            prediction("uwu://t/agent/a/memory/fact/cold", 0.1),
        ];
        let loaded = vec![
            (predictions[0].uri.clone(), payload("hot")),
            (predictions[1].uri.clone(), payload("warm")),
            (predictions[2].uri.clone(), payload("cold")),
        ];

        let report = cache.promote_predictions(&predictions, &loaded).await;
        assert_eq!(report.hot_promoted, 1);
        assert_eq!(report.warm_promoted, 1);
        assert_eq!(report.skipped, 1);
        assert_eq!(cache.stats().warm_entries, 1);
        assert!(cache.get(&predictions[0].uri).await.is_some());
        assert!(cache.get(&predictions[1].uri).await.is_some());
        assert_eq!(cache.stats().warm_accesses, 1);
    }

    #[tokio::test]
    async fn warm_capacity_evicts_lowest_prediction() {
        let cache = TieredCache::new(8, 1, PredictivePromotionConfig::default()).unwrap();
        let predictions = vec![
            prediction("uwu://t/agent/a/memory/fact/a", 0.3),
            prediction("uwu://t/agent/a/memory/fact/b", 0.6),
        ];
        let loaded = vec![
            (predictions[0].uri.clone(), payload("a")),
            (predictions[1].uri.clone(), payload("b")),
        ];

        let report = cache.promote_predictions(&predictions, &loaded).await;
        assert_eq!(report.warm_evicted, 1);
        assert_eq!(cache.stats().warm_entries, 1);
        assert!(cache.get(&predictions[1].uri).await.is_some());
    }
}
