//! 可观测性模块（F13 质量评分 + F15 血缘图 + Metrics）。
//!
//! F9 ContextPubSub 已删除，使用 `crate::event_store`（基于 `uwu_event_mesh`）替代。

use crate::{ContentPayload, ContextEntry, ContextUri};
use chrono::{DateTime, Duration, Utc};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Data classes accepted by the observability policy. `Secret` can never be raw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitiveClass {
    Public,
    Operational,
    Identifier,
    Content,
    Endpoint,
    Secret,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "representation", rename_all = "snake_case")]
pub enum ObservedField {
    Raw {
        class: SensitiveClass,
        value: String,
    },
    Fingerprint {
        class: SensitiveClass,
        value: String,
        key_id: String,
        key_version: u32,
    },
    Omitted {
        class: SensitiveClass,
    },
}

#[derive(Clone)]
pub struct FingerprintKey {
    key: [u8; 32],
    pub id: String,
    pub version: u32,
}

impl FingerprintKey {
    pub fn new(key: [u8; 32], id: impl Into<String>, version: u32) -> Self {
        Self {
            key,
            id: id.into(),
            version,
        }
    }

    /// Keyed, domain-separated BLAKE3 fingerprint with 128 bits of output.
    pub fn observe(&self, class: SensitiveClass, domain: &str, value: &str) -> ObservedField {
        let mut hasher = blake3::Hasher::new_keyed(&self.key);
        hasher.update(b"uwu-context-db/observability/v1\0");
        hasher.update(domain.as_bytes());
        hasher.update(&[0]);
        hasher.update(format!("{class:?}").as_bytes());
        hasher.update(&[0]);
        hasher.update(value.as_bytes());
        let digest = hasher.finalize();
        ObservedField::Fingerprint {
            class,
            value: digest.to_hex()[..32].to_owned(),
            key_id: self.id.clone(),
            key_version: self.version,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticGrant {
    pub expires_at: DateTime<Utc>,
    pub operation: String,
    pub classes: Vec<SensitiveClass>,
    pub max_events: u64,
}

impl DiagnosticGrant {
    pub fn permits(
        &self,
        now: DateTime<Utc>,
        operation: &str,
        class: SensitiveClass,
        event: u64,
    ) -> bool {
        class != SensitiveClass::Secret
            && now < self.expires_at
            && self.operation == operation
            && self.classes.contains(&class)
            && event < self.max_events
    }

    pub fn with_ttl(
        operation: impl Into<String>,
        classes: Vec<SensitiveClass>,
        max_events: u64,
        ttl: Duration,
    ) -> Self {
        Self {
            expires_at: Utc::now() + ttl,
            operation: operation.into(),
            classes,
            max_events,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Io,
    Serialization,
    Timeout,
    Authentication,
    Authorization,
    RateLimited,
    Unavailable,
    InvalidInput,
    Conflict,
    Downstream,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorReport {
    pub kind: ErrorKind,
    pub retryable: bool,
    pub status: Option<u16>,
}

impl ErrorReport {
    pub const fn new(kind: ErrorKind, retryable: bool, status: Option<u16>) -> Self {
        Self {
            kind,
            retryable,
            status,
        }
    }

    pub fn downstream() -> Self {
        Self::new(ErrorKind::Downstream, false, None)
    }

    pub fn from_error(error: &(dyn std::error::Error + 'static)) -> Self {
        let kind = if let Some(io) = error.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::TimedOut {
                ErrorKind::Timeout
            } else {
                ErrorKind::Io
            }
        } else if error.is::<serde_json::Error>() {
            ErrorKind::Serialization
        } else {
            ErrorKind::Downstream
        };
        let retryable = matches!(
            kind,
            ErrorKind::Io | ErrorKind::Timeout | ErrorKind::RateLimited | ErrorKind::Unavailable
        );
        Self {
            kind,
            retryable,
            status: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub numerator: u64,
    pub denominator: u64,
    pub error_burst: u64,
    pub error_window: Duration,
}

pub struct DeterministicSampler {
    config: SamplingConfig,
    key: FingerprintKey,
    errors: Mutex<HashMap<ErrorKind, (DateTime<Utc>, u64)>>,
}

impl DeterministicSampler {
    pub fn new(config: SamplingConfig, key: FingerprintKey) -> Self {
        Self {
            config,
            key,
            errors: Mutex::new(HashMap::new()),
        }
    }
    pub fn sampled(&self, domain: &str, identity: &str) -> bool {
        if self.config.denominator == 0 {
            return false;
        }
        let ObservedField::Fingerprint { value, .. } =
            self.key
                .observe(SensitiveClass::Identifier, domain, identity)
        else {
            return false;
        };
        u64::from_str_radix(&value[..16], 16).unwrap_or(u64::MAX) % self.config.denominator
            < self.config.numerator.min(self.config.denominator)
    }
    pub fn allow_error(&self, kind: ErrorKind, now: DateTime<Utc>) -> bool {
        let mut errors = self.errors.lock().unwrap_or_else(|e| e.into_inner());
        let state = errors.entry(kind).or_insert((now, 0));
        if now - state.0 >= self.config.error_window {
            *state = (now, 0);
        }
        if state.1 >= self.config.error_burst {
            false
        } else {
            state.1 += 1;
            true
        }
    }
}

/// Runtime-injected tracing privacy policy. Missing key/grant always omits data.
pub struct ObservabilityPolicy {
    key: Option<FingerprintKey>,
    sampler: Option<Arc<DeterministicSampler>>,
    grant: Option<DiagnosticGrant>,
    events: AtomicU64,
}

impl ObservabilityPolicy {
    pub fn omit_all() -> Self {
        Self {
            key: None,
            sampler: None,
            grant: None,
            events: AtomicU64::new(0),
        }
    }

    pub fn new(key: FingerprintKey, sampler: Arc<DeterministicSampler>) -> Self {
        Self {
            key: Some(key),
            sampler: Some(sampler),
            grant: None,
            events: AtomicU64::new(0),
        }
    }

    pub fn with_grant(mut self, grant: DiagnosticGrant) -> Self {
        self.grant = Some(grant);
        self
    }

    pub fn observe(
        &self,
        operation: &str,
        class: SensitiveClass,
        domain: &str,
        value: &str,
    ) -> ObservedField {
        if class == SensitiveClass::Secret || class == SensitiveClass::Content {
            return ObservedField::Omitted { class };
        }
        let event = self.events.fetch_add(1, Ordering::Relaxed);
        if self
            .grant
            .as_ref()
            .is_some_and(|grant| grant.permits(Utc::now(), operation, class, event))
        {
            return ObservedField::Raw {
                class,
                value: value.to_owned(),
            };
        }
        match (&self.key, &self.sampler) {
            (Some(key), Some(sampler)) if sampler.sampled(domain, value) => {
                key.observe(class, domain, value)
            }
            _ => ObservedField::Omitted { class },
        }
    }

    pub fn report_error(&self, error: &(dyn std::error::Error + 'static)) -> Option<ErrorReport> {
        let report = ErrorReport::from_error(error);
        self.sampler
            .as_ref()
            .filter(|sampler| sampler.allow_error(report.kind, Utc::now()))
            .map(|_| report)
    }
}

impl Default for ObservabilityPolicy {
    fn default() -> Self {
        Self::omit_all()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F13 质量评分
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDimension {
    Completeness,
    Freshness,
    Consistency,
    Utility,
}

#[derive(Debug, Clone)]
pub struct QualityScore {
    pub dimensions: HashMap<QualityDimension, f32>,
    pub overall: f32,
    pub scored_at: DateTime<Utc>,
}

pub struct QualityScorer;

impl QualityScorer {
    pub fn score(entry: &ContextEntry, access_count: u64, now: DateTime<Utc>) -> QualityScore {
        let mut dims = HashMap::new();

        let completeness = if matches!(&entry.payload, ContentPayload::Text { dense, .. } if !dense.is_empty())
        {
            0.9
        } else {
            0.4
        };
        dims.insert(QualityDimension::Completeness, completeness);

        let age_days = (now - entry.updated_at).num_hours() as f32 / 24.0;
        let freshness = (-age_days / 30.0).exp().clamp(0.05, 1.0);
        dims.insert(QualityDimension::Freshness, freshness);

        let l0_text = entry.payload.sparse_text();
        let l0_len = l0_text.len() as f32;
        let consistency = if l0_len > 20.0 && l0_len < 2000.0 {
            0.85
        } else {
            0.5
        };
        dims.insert(QualityDimension::Consistency, consistency);

        let utility = ((access_count as f32 + 1.0).ln() / 5.0).clamp(0.1, 1.0);
        dims.insert(QualityDimension::Utility, utility);

        let overall = (completeness * 0.25 + freshness * 0.25 + consistency * 0.2 + utility * 0.3)
            .clamp(0.0, 1.0);

        QualityScore {
            dimensions: dims,
            overall,
            scored_at: now,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F15 血缘图（Provenance Graph）
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ProvenanceNode {
    pub uri: ContextUri,
    pub derived_from: Vec<ProvenanceEdge>,
    pub derived_to: Vec<ProvenanceEdge>,
}

#[derive(Debug, Clone)]
pub struct ProvenanceEdge {
    pub source: ContextUri,
    pub target: ContextUri,
    pub relation: ProvenanceRelationType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceRelationType {
    ExtractedFrom,
    MergedFrom,
    GeneratedBy,
    TriggeredBy,
    DerivedFrom,
}

pub struct ProvenanceGraph {
    nodes: parking_lot::RwLock<HashMap<String, ProvenanceNode>>, // B.1: 读多写少
}

impl ProvenanceGraph {
    pub fn new() -> Self {
        Self {
            nodes: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn add_derivation(
        &self,
        source: &ContextUri,
        target: &ContextUri,
        relation: ProvenanceRelationType,
    ) {
        let mut nodes = self.nodes.write();
        let edge = ProvenanceEdge {
            source: source.clone(),
            target: target.clone(),
            relation,
            timestamp: Utc::now(),
        };
        nodes
            .entry(source.to_string())
            .or_insert_with(|| ProvenanceNode {
                uri: source.clone(),
                derived_from: vec![],
                derived_to: vec![],
            })
            .derived_to
            .push(edge.clone());
        nodes
            .entry(target.to_string())
            .or_insert_with(|| ProvenanceNode {
                uri: target.clone(),
                derived_from: vec![],
                derived_to: vec![],
            })
            .derived_from
            .push(edge);
    }

    pub fn downstream(&self, root: &ContextUri, k: usize) -> Vec<ContextUri> {
        let nodes = self.nodes.read(); // B.1: 读锁
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![(root.clone(), 0)];
        while let Some((uri, depth)) = stack.pop() {
            if depth > k || !visited.insert(uri.to_string()) {
                continue;
            }
            if depth > 0 {
                result.push(uri.clone());
            }
            if let Some(node) = nodes.get(&uri.to_string()) {
                for edge in &node.derived_to {
                    stack.push((edge.target.clone(), depth + 1));
                }
            }
        }
        result
    }
}

impl Default for ProvenanceGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// I.2: 可观测性 — metrics crate 真实集成 + Prometheus exporter
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MetricsExporterConfig {
    pub endpoint: SocketAddr,
}

impl Default for MetricsExporterConfig {
    fn default() -> Self {
        Self {
            endpoint: SocketAddr::from(([0, 0, 0, 0], 9898)),
        }
    }
}

#[derive(Clone)]
pub struct MetricsExporter {
    handle: PrometheusHandle,
}

impl MetricsExporter {
    pub fn render(&self) -> String {
        self.handle.render()
    }
}

/// Install a Prometheus recorder and HTTP listener for `/metrics` scraping.
pub fn install_metrics_exporter(
    config: MetricsExporterConfig,
) -> Result<MetricsExporter, metrics_exporter_prometheus::BuildError> {
    let handle = PrometheusBuilder::new()
        .with_http_listener(config.endpoint)
        .install_recorder()?;
    Ok(MetricsExporter { handle })
}

/// Install an in-process Prometheus recorder. Call `render()` from any HTTP stack.
pub fn install_metrics_recorder() -> Result<MetricsExporter, metrics_exporter_prometheus::BuildError>
{
    let handle = PrometheusBuilder::new().install_recorder()?;
    Ok(MetricsExporter { handle })
}

/// 记录一次检索操作。
pub fn record_retrieval(hits: usize, duration_ms: u64, tokens: usize, cache_hit: bool) {
    metrics::counter!("uwu.retrieval.requests").increment(1);
    metrics::counter!("uwu.retrieval.hits").increment(hits as u64);
    metrics::histogram!("uwu.retrieval.duration_ms").record(duration_ms as f64);
    metrics::counter!("uwu.retrieval.tokens").increment(tokens as u64);
    if cache_hit {
        metrics::counter!("uwu.cache.hit").increment(1);
    } else {
        metrics::counter!("uwu.cache.miss").increment(1);
    }
}

/// 记录一次写入操作。
pub fn record_write(_uri: &ContextUri, success: bool) {
    metrics::counter!("uwu.write.requests").increment(1);
    if success {
        metrics::counter!("uwu.write.success").increment(1);
    } else {
        metrics::counter!("uwu.write.failure").increment(1);
    }
}

/// 记录一次巩固操作。
pub fn record_consolidation(entries: usize, products: usize, duration_ms: u64) {
    metrics::counter!("uwu.consolidation.requests").increment(1);
    metrics::counter!("uwu.consolidation.entries").increment(entries as u64);
    metrics::counter!("uwu.consolidation.products").increment(products as u64);
    metrics::histogram!("uwu.consolidation.duration_ms").record(duration_ms as f64);
}

/// 记录一次 LLM 调用。
pub fn record_llm_call(provider: &str, tokens: usize, duration_ms: u64, success: bool) {
    metrics::counter!("uwu.llm.calls", "provider" => provider.to_string()).increment(1);
    metrics::counter!("uwu.llm.tokens", "provider" => provider.to_string())
        .increment(tokens as u64);
    metrics::histogram!("uwu.llm.duration_ms", "provider" => provider.to_string())
        .record(duration_ms as f64);
    if !success {
        metrics::counter!("uwu.llm.errors", "provider" => provider.to_string()).increment(1);
    }
}

/// 记录缓存命中/未命中。
pub fn record_cache(hit: bool) {
    if hit {
        metrics::counter!("uwu.cache.hit").increment(1);
    } else {
        metrics::counter!("uwu.cache.miss").increment(1);
    }
}

#[cfg(test)]
mod privacy_tests {
    use super::*;

    fn key(byte: u8) -> FingerprintKey {
        FingerprintKey::new([byte; 32], "test", 7)
    }

    #[test]
    fn fingerprints_are_keyed_domain_separated_and_128_bit() {
        let a = key(1).observe(SensitiveClass::Identifier, "uri", "same");
        let b = key(2).observe(SensitiveClass::Identifier, "uri", "same");
        let c = key(1).observe(SensitiveClass::Identifier, "query", "same");
        assert_ne!(a, b);
        assert_ne!(a, c);
        let ObservedField::Fingerprint {
            value,
            key_id,
            key_version,
            ..
        } = a
        else {
            panic!()
        };
        assert_eq!(value.len(), 32);
        assert_eq!((key_id.as_str(), key_version), ("test", 7));
    }

    #[test]
    fn grants_expire_limit_events_and_never_reveal_secrets() {
        let now = Utc::now();
        let grant = DiagnosticGrant {
            expires_at: now + Duration::seconds(1),
            operation: "read".into(),
            classes: vec![SensitiveClass::Content, SensitiveClass::Secret],
            max_events: 1,
        };
        assert!(grant.permits(now, "read", SensitiveClass::Content, 0));
        assert!(!grant.permits(now, "read", SensitiveClass::Secret, 0));
        assert!(!grant.permits(now, "read", SensitiveClass::Content, 1));
        assert!(!grant.permits(
            now + Duration::seconds(2),
            "read",
            SensitiveClass::Content,
            0
        ));
    }

    #[test]
    fn sampling_and_error_limits_are_deterministic() {
        let sampler = DeterministicSampler::new(
            SamplingConfig {
                numerator: 1,
                denominator: 2,
                error_burst: 1,
                error_window: Duration::seconds(10),
            },
            key(3),
        );
        assert_eq!(
            sampler.sampled("request", "abc"),
            sampler.sampled("request", "abc")
        );
        let now = Utc::now();
        assert!(sampler.allow_error(ErrorKind::Timeout, now));
        assert!(!sampler.allow_error(ErrorKind::Timeout, now));
        assert!(sampler.allow_error(ErrorKind::Timeout, now + Duration::seconds(11)));
    }

    #[test]
    fn errors_are_structured_without_messages() {
        let timeout = std::io::Error::new(std::io::ErrorKind::TimedOut, "secret body");
        assert_eq!(
            ErrorReport::from_error(&timeout),
            ErrorReport {
                kind: ErrorKind::Timeout,
                retryable: true,
                status: None
            }
        );
        let json = serde_json::to_string(&ErrorReport::from_error(&timeout)).unwrap();
        assert!(!json.contains("secret body"));
    }
}
