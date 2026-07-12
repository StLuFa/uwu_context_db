//! 经过消息级认证的 EventMesh ↔ NATS 运行时桥。
//! Broker ACL 只负责传输层隔离；所有入站 envelope 在进入 mesh 前还必须通过
//! signer、签名、source/type allow-list、时效和 replay-id 验证。

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use agent_context_db_core::{
    Bridge, Envelope, EventMesh, EventTypeId, FlowChannel, SerializedEnvelope,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use tokio::sync::watch;
use uwu_event_mesh::core::error::{EventMeshError, Result as MeshResult};
use uwu_nats_bridge::{NatsConfig, NatsPublisher, NatsSubjects, NatsSubscriber};

const SIGNER_HEADER: &str = "context-db-signer";
const SIGNATURE_HEADER: &str = "context-db-signature-ed25519";
const SIGNATURE_DOMAIN: &str = "context-db-nats-envelope-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NatsChannelRoute {
    pub type_prefix: String,
    pub channel: FlowChannel,
}

/// 入站信任策略，同时持有本节点出站签名身份。
#[derive(Clone)]
pub struct NatsSecurityConfig {
    signer: String,
    signing_key: Arc<SigningKey>,
    trusted_signers: HashMap<String, VerifyingKey>,
    allowed_sources: HashSet<String>,
    allowed_type_prefixes: Vec<String>,
    max_age: Duration,
    max_clock_skew: Duration,
    replay_capacity: usize,
}

impl std::fmt::Debug for NatsSecurityConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NatsSecurityConfig")
            .field("signer", &self.signer)
            .field("trusted_signers", &self.trusted_signers.keys())
            .field("allowed_sources", &self.allowed_sources)
            .field("allowed_type_prefixes", &self.allowed_type_prefixes)
            .field("max_age", &self.max_age)
            .field("max_clock_skew", &self.max_clock_skew)
            .field("replay_capacity", &self.replay_capacity)
            .finish_non_exhaustive()
    }
}

impl NatsSecurityConfig {
    pub fn new(signer: impl Into<String>, signing_key: SigningKey) -> Self {
        let signer = signer.into();
        let mut trusted_signers = HashMap::new();
        trusted_signers.insert(signer.clone(), signing_key.verifying_key());
        Self {
            signer,
            signing_key: Arc::new(signing_key),
            trusted_signers,
            allowed_sources: HashSet::new(),
            allowed_type_prefixes: Vec::new(),
            max_age: Duration::from_secs(300),
            max_clock_skew: Duration::from_secs(30),
            replay_capacity: 100_000,
        }
    }

    pub fn trust_signer(mut self, signer: impl Into<String>, key: VerifyingKey) -> Self {
        self.trusted_signers.insert(signer.into(), key);
        self
    }

    pub fn allow_source(mut self, source: impl Into<String>) -> Self {
        self.allowed_sources.insert(source.into());
        self
    }

    pub fn allow_type_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.allowed_type_prefixes.push(prefix.into());
        self
    }

    pub fn with_freshness(mut self, max_age: Duration, max_clock_skew: Duration) -> Self {
        self.max_age = max_age;
        self.max_clock_skew = max_clock_skew;
        self
    }

    pub fn with_replay_capacity(mut self, capacity: usize) -> Self {
        self.replay_capacity = capacity.max(1);
        self
    }
}

#[derive(Debug, Clone)]
pub struct NatsBridgeConfig {
    pub url: String,
    pub correlation_id: String,
    pub connection_name: String,
    pub default_channel: FlowChannel,
    pub channel_routes: Vec<NatsChannelRoute>,
    pub security: NatsSecurityConfig,
    pub reconnect_initial: Duration,
    pub reconnect_max: Duration,
}

impl NatsBridgeConfig {
    /// 没有不安全默认值：调用方必须显式提供身份和信任策略。
    pub fn secure(security: NatsSecurityConfig) -> Self {
        Self {
            url: "nats://localhost:4222".into(),
            correlation_id: "default".into(),
            connection_name: "context-db".into(),
            default_channel: FlowChannel::Consolidation,
            channel_routes: default_channel_routes(),
            security,
            reconnect_initial: Duration::from_millis(250),
            reconnect_max: Duration::from_secs(30),
        }
    }

    pub fn route(mut self, type_prefix: impl Into<String>, channel: FlowChannel) -> Self {
        self.channel_routes.push(NatsChannelRoute {
            type_prefix: type_prefix.into(),
            channel,
        });
        self
    }

    fn to_nats(&self) -> NatsConfig {
        NatsConfig {
            url: self.url.clone(),
            connection_name: self.connection_name.clone(),
            max_reconnects: None,
            reconnect_delay: self.reconnect_initial,
            ..Default::default()
        }
    }
}

fn default_channel_routes() -> Vec<NatsChannelRoute> {
    vec![
        NatsChannelRoute {
            type_prefix: "context.entry_".into(),
            channel: FlowChannel::Main,
        },
        NatsChannelRoute {
            type_prefix: "context.consolidation_".into(),
            channel: FlowChannel::Consolidation,
        },
        NatsChannelRoute {
            type_prefix: "context.marketplace_".into(),
            channel: FlowChannel::Consolidation,
        },
        NatsChannelRoute {
            type_prefix: "intent.".into(),
            channel: FlowChannel::Monitoring,
        },
        NatsChannelRoute {
            type_prefix: "knowledge_network.".into(),
            channel: FlowChannel::Monitoring,
        },
    ]
}

pub struct NatsBridge {
    publisher: NatsPublisher,
    default_channel: FlowChannel,
    channel_routes: Vec<NatsChannelRoute>,
    security: NatsSecurityConfig,
}

impl NatsBridge {
    async fn connect(cfg: &NatsBridgeConfig) -> Result<Self, NatsBridgeError> {
        let publisher =
            NatsPublisher::connect(cfg.to_nats(), NatsSubjects::new(cfg.correlation_id.clone()))
                .await
                .map_err(|e| NatsBridgeError::Publisher(e.to_string()))?;
        Ok(Self {
            publisher,
            default_channel: cfg.default_channel,
            channel_routes: normalize_routes(cfg.channel_routes.clone()),
            security: cfg.security.clone(),
        })
    }

    fn route_channel(&self, env: &Envelope) -> FlowChannel {
        route_channel(
            env.type_id.as_ref(),
            &self.channel_routes,
            self.default_channel,
        )
    }
}

#[async_trait]
impl Bridge for NatsBridge {
    async fn publish_remote(&self, env: Arc<Envelope>) -> MeshResult<()> {
        let mut serialized = SerializedEnvelope::from_envelope(&env)?;
        sign_envelope(&mut serialized, &self.security).map_err(mesh_serialization_error)?;
        self.publisher
            .publish_envelope(self.route_channel(&env), serialized)
            .await
            .map_err(|e| mesh_serialization_error(NatsBridgeError::Publisher(e.to_string())))
    }
}

fn mesh_serialization_error(error: NatsBridgeError) -> EventMeshError {
    EventMeshError::Serialize(serde_json::Error::io(std::io::Error::other(
        error.to_string(),
    )))
}

fn normalize_routes(mut routes: Vec<NatsChannelRoute>) -> Vec<NatsChannelRoute> {
    routes.sort_by_key(|route| std::cmp::Reverse(route.type_prefix.len()));
    routes
}

fn route_channel(
    type_id: Option<&EventTypeId>,
    routes: &[NatsChannelRoute],
    fallback: FlowChannel,
) -> FlowChannel {
    let Some(type_id) = type_id else {
        return fallback;
    };
    let name = type_id.to_string();
    routes
        .iter()
        .find(|route| name.starts_with(&route.type_prefix))
        .map(|route| route.channel)
        .unwrap_or(fallback)
}

fn signing_bytes(env: &SerializedEnvelope) -> Result<Vec<u8>, NatsBridgeError> {
    let mut unsigned = env.clone();
    unsigned.headers.remove(SIGNATURE_HEADER);
    serde_json::to_vec(&(SIGNATURE_DOMAIN, unsigned)).map_err(NatsBridgeError::Serialize)
}

fn sign_envelope(
    env: &mut SerializedEnvelope,
    security: &NatsSecurityConfig,
) -> Result<(), NatsBridgeError> {
    env.headers
        .insert(SIGNER_HEADER.into(), security.signer.clone());
    env.headers.remove(SIGNATURE_HEADER);
    let signature = security.signing_key.sign(&signing_bytes(env)?);
    env.headers
        .insert(SIGNATURE_HEADER.into(), hex::encode(signature.to_bytes()));
    Ok(())
}

#[derive(Debug)]
struct ReplayCache {
    seen: Mutex<HashMap<String, DateTime<Utc>>>,
    capacity: usize,
    retention: chrono::Duration,
}

impl ReplayCache {
    fn new(policy: &NatsSecurityConfig) -> Result<Self, NatsBridgeError> {
        let retention = policy
            .max_age
            .checked_add(policy.max_clock_skew)
            .and_then(|duration| chrono::Duration::from_std(duration).ok())
            .ok_or_else(|| {
                NatsBridgeError::InvalidPolicy("freshness duration exceeds chrono range".into())
            })?;
        Ok(Self {
            seen: Mutex::new(HashMap::new()),
            capacity: policy.replay_capacity,
            retention,
        })
    }

    fn insert(&self, id: String, now: DateTime<Utc>) -> Result<(), NatsBridgeError> {
        let mut seen = self
            .seen
            .lock()
            .map_err(|_| NatsBridgeError::ReplayCachePoisoned)?;
        seen.retain(|_, at| now.signed_duration_since(*at) <= self.retention);
        if seen.contains_key(&id) {
            return Err(NatsBridgeError::Replay(id));
        }
        if seen.len() >= self.capacity {
            return Err(NatsBridgeError::ReplayCapacityExhausted);
        }
        seen.insert(id, now);
        Ok(())
    }
}

fn verify_envelope(
    env: &SerializedEnvelope,
    policy: &NatsSecurityConfig,
    replay: &ReplayCache,
    now: DateTime<Utc>,
) -> Result<(), NatsBridgeError> {
    let source = env
        .source
        .as_deref()
        .ok_or(NatsBridgeError::MissingSource)?;
    if !policy.allowed_sources.contains(source) {
        return Err(NatsBridgeError::SourceDenied(source.into()));
    }
    let type_name = env.type_id.to_string();
    if !policy
        .allowed_type_prefixes
        .iter()
        .any(|prefix| type_name.starts_with(prefix))
    {
        return Err(NatsBridgeError::TypeDenied(type_name));
    }
    let age = now.signed_duration_since(env.timestamp);
    let max_age = chrono::Duration::from_std(policy.max_age).map_err(|_| {
        NatsBridgeError::InvalidPolicy("max_age exceeds chrono duration range".into())
    })?;
    let skew = chrono::Duration::from_std(policy.max_clock_skew).map_err(|_| {
        NatsBridgeError::InvalidPolicy("max_clock_skew exceeds chrono duration range".into())
    })?;
    if age > max_age || age < -skew || env.is_expired() {
        return Err(NatsBridgeError::Expired);
    }
    let signer = env
        .headers
        .get(SIGNER_HEADER)
        .ok_or(NatsBridgeError::MissingSigner)?;
    let key = policy
        .trusted_signers
        .get(signer)
        .ok_or_else(|| NatsBridgeError::UnknownSigner(signer.clone()))?;
    let encoded = env
        .headers
        .get(SIGNATURE_HEADER)
        .ok_or(NatsBridgeError::MissingSignature)?;
    let bytes = hex::decode(encoded).map_err(|_| NatsBridgeError::InvalidSignature)?;
    let signature = Signature::from_slice(&bytes).map_err(|_| NatsBridgeError::InvalidSignature)?;
    key.verify(&signing_bytes(env)?, &signature)
        .map_err(|_| NatsBridgeError::InvalidSignature)?;
    // 只在所有认证检查通过后占用 replay id，伪造消息不能污染缓存。
    replay.insert(format!("{signer}\u{1f}{source}\u{1f}{}", env.id), now)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NatsHealthState {
    Connecting { attempt: u32 },
    Healthy,
    Reconnected { generation: u64 },
    Degraded { error: String, next_retry: Duration },
    RestartDetected { generation: u64 },
    Stopped,
}

struct HealthReporter {
    state: RwLock<NatsHealthState>,
    tx: watch::Sender<NatsHealthState>,
}

impl HealthReporter {
    fn set(&self, state: NatsHealthState) {
        match self.state.write() {
            Ok(mut current) => *current = state.clone(),
            Err(poisoned) => *poisoned.into_inner() = state.clone(),
        }
        self.tx.send_replace(state);
    }
}

/// 唯一公开的运行时入口。它强制安装安全策略、出站签名和可恢复入站订阅。
pub struct EventSystem {
    pub mesh: EventMesh,
    task: Option<tokio::task::JoinHandle<()>>,
    health: Arc<HealthReporter>,
}

impl EventSystem {
    pub async fn with_nats(cfg: NatsBridgeConfig) -> Result<Self, NatsBridgeError> {
        Self::attach(EventMesh::new(), cfg).await
    }

    pub async fn attach(mesh: EventMesh, cfg: NatsBridgeConfig) -> Result<Self, NatsBridgeError> {
        validate_policy(&cfg.security)?;
        let bridge = NatsBridge::connect(&cfg).await?;
        mesh.attach_bridge(Arc::new(bridge));
        let (tx, _) = watch::channel(NatsHealthState::Connecting { attempt: 0 });
        let health = Arc::new(HealthReporter {
            state: RwLock::new(NatsHealthState::Connecting { attempt: 0 }),
            tx,
        });
        let task = tokio::spawn(run_ingestor(cfg, mesh.clone(), health.clone()));
        Ok(Self {
            mesh,
            task: Some(task),
            health,
        })
    }

    pub fn health(&self) -> NatsHealthState {
        match self.health.state.read() {
            Ok(state) => state.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    pub fn subscribe_health(&self) -> watch::Receiver<NatsHealthState> {
        self.health.tx.subscribe()
    }

    pub fn shutdown(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        self.health.set(NatsHealthState::Stopped);
    }
}

impl Drop for EventSystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn validate_policy(policy: &NatsSecurityConfig) -> Result<(), NatsBridgeError> {
    if policy.signer.is_empty()
        || policy.allowed_sources.is_empty()
        || policy.allowed_type_prefixes.is_empty()
    {
        return Err(NatsBridgeError::InvalidPolicy(
            "signer and source/type allow-lists must be non-empty".into(),
        ));
    }
    Ok(())
}

async fn run_ingestor(cfg: NatsBridgeConfig, mesh: EventMesh, health: Arc<HealthReporter>) {
    let replay = match ReplayCache::new(&cfg.security) {
        Ok(replay) => replay,
        Err(error) => {
            health.set(NatsHealthState::Degraded {
                error: error.to_string(),
                next_retry: Duration::ZERO,
            });
            return;
        }
    };
    let mut attempt = 0u32;
    let mut generation = 0u64;
    let mut was_healthy = false;
    loop {
        health.set(NatsHealthState::Connecting { attempt });
        match NatsSubscriber::connect(cfg.to_nats(), cfg.correlation_id.clone()).await {
            Ok(mut subscriber) => {
                attempt = 0;
                generation = generation.saturating_add(1);
                if was_healthy {
                    health.set(NatsHealthState::RestartDetected { generation });
                    health.set(NatsHealthState::Reconnected { generation });
                } else {
                    health.set(NatsHealthState::Healthy);
                }
                was_healthy = true;
                while let Some((_channel, serialized)) = subscriber.recv_any().await {
                    match verify_envelope(&serialized, &cfg.security, &replay, Utc::now()).and_then(
                        |verified| {
                            serialized
                                .into_envelope()
                                .map(|env| (verified, env))
                                .map_err(NatsBridgeError::Mesh)
                        },
                    ) {
                        Ok(((), env)) => {
                            if let Err(error) = mesh.ingest_remote(Arc::new(env)).await {
                                tracing::error!(error = ?agent_context_db_core::ErrorReport::from_error(&error), "verified NATS envelope failed EventMesh ingestion");
                            }
                        }
                        Err(error) => {
                            tracing::warn!(error = ?agent_context_db_core::ErrorReport::from_error(&error), "rejected inbound NATS envelope")
                        }
                    }
                }
            }
            Err(error) => {
                tracing::warn!(error = ?agent_context_db_core::ErrorReport::from_error(&error), attempt, "NATS subscription connection failed")
            }
        }
        attempt = attempt.saturating_add(1);
        let delay = reconnect_delay(cfg.reconnect_initial, cfg.reconnect_max, attempt);
        health.set(NatsHealthState::Degraded {
            error: "subscriber disconnected".into(),
            next_retry: delay,
        });
        tokio::time::sleep(delay).await;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvidenceSource {
    /// In-process/synthetic contract evidence. It makes no claim about an external service.
    Contract,
    /// Evidence minted only after an environment-backed runner validates and probes its endpoint.
    VerifiedExternal(VerifiedExternalEvidence),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VerifiedExternalEvidence {
    endpoint_fingerprint: String,
    runner: ExternalProbeKind,
}

impl VerifiedExternalEvidence {
    pub fn endpoint_fingerprint(&self) -> &str {
        &self.endpoint_fingerprint
    }

    pub fn runner(&self) -> ExternalProbeKind {
        self.runner
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalProbeKind {
    PostgreSql,
    Nats,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvidenceRecord {
    pub run_id: String,
    pub sequence: u64,
    pub service: String,
    pub service_generation: u64,
    pub operation: String,
    pub outcome: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub elapsed_ms: u64,
    pub detail: Option<String>,
    pub source: EvidenceSource,
}

/// Environment-backed PostgreSQL probe. Connection metadata is validated before any I/O;
/// credentials and the raw endpoint are never copied into evidence records.
pub async fn run_postgres_env_probe(
    run_id: &str,
    service_generation: u64,
) -> Result<EvidenceRecord, NatsBridgeError> {
    let raw = std::env::var("DATABASE_URL")
        .map_err(|_| NatsBridgeError::ProbeConfiguration("DATABASE_URL is not set".into()))?;
    let (fingerprint, _) = validated_endpoint(&raw, &["postgres", "postgresql"])?;
    let started_at = Utc::now();
    let timer = std::time::Instant::now();
    let pool = sqlx::PgPool::connect(&raw).await.map_err(|error| {
        NatsBridgeError::Probe(format!("PostgreSQL connection failed: {error}"))
    })?;
    sqlx::query("SELECT 1")
        .execute(&pool)
        .await
        .map_err(|error| NatsBridgeError::Probe(format!("PostgreSQL probe failed: {error}")))?;
    pool.close().await;
    Ok(verified_probe_record(
        run_id,
        "postgresql",
        service_generation,
        started_at,
        timer,
        fingerprint,
        ExternalProbeKind::PostgreSql,
    ))
}

/// Environment-backed NATS probe using `NATS_URL` and the bridge's concrete publisher.
pub async fn run_nats_env_probe(
    run_id: &str,
    service_generation: u64,
) -> Result<EvidenceRecord, NatsBridgeError> {
    let raw = std::env::var("NATS_URL")
        .map_err(|_| NatsBridgeError::ProbeConfiguration("NATS_URL is not set".into()))?;
    let (fingerprint, _) = validated_endpoint(&raw, &["nats", "tls"])?;
    let started_at = Utc::now();
    let timer = std::time::Instant::now();
    let config = NatsConfig {
        url: raw,
        connection_name: "context-db-evidence-probe".into(),
        ..Default::default()
    };
    let _publisher =
        NatsPublisher::connect(config, NatsSubjects::new(format!("evidence-{run_id}")))
            .await
            .map_err(|error| NatsBridgeError::Probe(format!("NATS connection failed: {error}")))?;
    Ok(verified_probe_record(
        run_id,
        "nats",
        service_generation,
        started_at,
        timer,
        fingerprint,
        ExternalProbeKind::Nats,
    ))
}

fn validated_endpoint(raw: &str, schemes: &[&str]) -> Result<(String, String), NatsBridgeError> {
    let parsed = url::Url::parse(raw)
        .map_err(|_| NatsBridgeError::ProbeConfiguration("endpoint is not a valid URL".into()))?;
    if !schemes.contains(&parsed.scheme()) {
        return Err(NatsBridgeError::ProbeConfiguration(
            "endpoint scheme is invalid".into(),
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| NatsBridgeError::ProbeConfiguration("endpoint host is missing".into()))?;
    let canonical = format!(
        "{}://{}:{}",
        parsed.scheme(),
        host,
        parsed.port_or_known_default().unwrap_or(0)
    );
    let fingerprint = blake3::hash(canonical.as_bytes()).to_hex().to_string();
    Ok((fingerprint, canonical))
}

fn verified_probe_record(
    run_id: &str,
    service: &str,
    service_generation: u64,
    started_at: DateTime<Utc>,
    timer: std::time::Instant,
    endpoint_fingerprint: String,
    runner: ExternalProbeKind,
) -> EvidenceRecord {
    EvidenceRecord {
        run_id: run_id.into(),
        sequence: 0,
        service: service.into(),
        service_generation,
        operation: "connectivity-probe".into(),
        outcome: "success".into(),
        started_at,
        finished_at: Utc::now(),
        elapsed_ms: timer.elapsed().as_millis() as u64,
        detail: None,
        source: EvidenceSource::VerifiedExternal(VerifiedExternalEvidence {
            endpoint_fingerprint,
            runner,
        }),
    }
}

#[derive(Debug, Clone)]
pub struct BoundedLoadConfig {
    pub operations: usize,
    pub max_concurrency: usize,
    pub operation_timeout: Duration,
}

impl BoundedLoadConfig {
    pub fn validate(&self) -> Result<(), NatsBridgeError> {
        if self.operations == 0 || self.max_concurrency == 0 || self.operation_timeout.is_zero() {
            return Err(NatsBridgeError::InvalidPolicy(
                "load operations, concurrency, and timeout must be positive".into(),
            ));
        }
        if self.operations > 100_000 || self.max_concurrency > 1_024 {
            return Err(NatsBridgeError::InvalidPolicy(
                "load harness exceeds hard safety bounds".into(),
            ));
        }
        Ok(())
    }
}

pub async fn run_bounded_load<F, Fut>(
    run_id: &str,
    service: &str,
    config: BoundedLoadConfig,
    operation: F,
) -> Result<Vec<EvidenceRecord>, NatsBridgeError>
where
    F: Fn(usize) -> Fut + Sync,
    Fut: std::future::Future<Output = Result<(), String>>,
{
    use futures::{StreamExt, stream};
    config.validate()?;
    let timeout = config.operation_timeout;
    let records = stream::iter(0..config.operations)
        .map(|sequence| {
            let operation = &operation;
            async move {
                let started_at = Utc::now();
                let started = std::time::Instant::now();
                let result = tokio::time::timeout(timeout, operation(sequence)).await;
                let (outcome, detail) = match result {
                    Ok(Ok(())) => ("success", None),
                    Ok(Err(error)) => ("failure", Some(error)),
                    Err(_) => ("timeout", None),
                };
                EvidenceRecord {
                    run_id: run_id.into(),
                    sequence: sequence as u64,
                    service: service.into(),
                    service_generation: 0,
                    operation: "bounded-load".into(),
                    outcome: outcome.into(),
                    started_at,
                    finished_at: Utc::now(),
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    detail,
                    source: EvidenceSource::Contract,
                }
            }
        })
        .buffer_unordered(config.max_concurrency)
        .collect()
        .await;
    Ok(records)
}

fn reconnect_delay(initial: Duration, max: Duration, attempt: u32) -> Duration {
    initial
        .checked_mul(2u32.saturating_pow(attempt.saturating_sub(1).min(30)))
        .unwrap_or(max)
        .min(max)
}

#[derive(Debug, thiserror::Error)]
pub enum NatsBridgeError {
    #[error("nats publisher: {0}")]
    Publisher(String),
    #[error("event mesh: {0}")]
    Mesh(#[from] EventMeshError),
    #[error("serialize signed envelope: {0}")]
    Serialize(serde_json::Error),
    #[error("invalid NATS security policy: {0}")]
    InvalidPolicy(String),
    #[error("missing envelope source")]
    MissingSource,
    #[error("source denied: {0}")]
    SourceDenied(String),
    #[error("event type denied: {0}")]
    TypeDenied(String),
    #[error("envelope expired or timestamp is outside allowed skew")]
    Expired,
    #[error("missing signer")]
    MissingSigner,
    #[error("unknown signer: {0}")]
    UnknownSigner(String),
    #[error("missing signature")]
    MissingSignature,
    #[error("invalid signature")]
    InvalidSignature,
    #[error("replayed envelope: {0}")]
    Replay(String),
    #[error("replay cache capacity exhausted while all entries remain valid")]
    ReplayCapacityExhausted,
    #[error("replay cache lock poisoned")]
    ReplayCachePoisoned,
    #[error("invalid external probe configuration: {0}")]
    ProbeConfiguration(String),
    #[error("external probe failed: {0}")]
    Probe(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn security() -> NatsSecurityConfig {
        NatsSecurityConfig::new("node-a", SigningKey::from_bytes(&[7; 32]))
            .allow_source("worker-a")
            .allow_type_prefix("context.")
            .with_freshness(Duration::from_secs(60), Duration::from_secs(2))
    }

    fn envelope(policy: &NatsSecurityConfig) -> SerializedEnvelope {
        let mut env = SerializedEnvelope::new(
            EventTypeId::new("context", "entry_created"),
            "context.entry",
            "flow",
            "worker-a",
            &json!({"x": 1}),
        )
        .unwrap()
        .with_source("worker-a");
        sign_envelope(&mut env, policy).unwrap();
        env
    }

    #[test]
    fn rejects_forged_signature() -> Result<(), NatsBridgeError> {
        let policy = security();
        let mut env = envelope(&policy);
        env.payload_bytes = serde_json::to_vec(&json!({"x": 2})).unwrap();
        assert!(matches!(
            verify_envelope(&env, &policy, &ReplayCache::new(&policy)?, Utc::now()),
            Err(NatsBridgeError::InvalidSignature)
        ));
        Ok(())
    }

    #[test]
    fn rejects_replay() -> Result<(), NatsBridgeError> {
        let policy = security();
        let env = envelope(&policy);
        let cache = ReplayCache::new(&policy)?;
        verify_envelope(&env, &policy, &cache, Utc::now()).unwrap();
        assert!(matches!(
            verify_envelope(&env, &policy, &cache, Utc::now()),
            Err(NatsBridgeError::Replay(_))
        ));
        Ok(())
    }

    #[test]
    fn rejects_expired_envelope() -> Result<(), NatsBridgeError> {
        let policy = security();
        let mut env = envelope(&policy);
        env.timestamp = Utc::now() - chrono::Duration::minutes(5);
        sign_envelope(&mut env, &policy).unwrap();
        assert!(matches!(
            verify_envelope(&env, &policy, &ReplayCache::new(&policy)?, Utc::now()),
            Err(NatsBridgeError::Expired)
        ));
        Ok(())
    }

    #[test]
    fn rejects_unregistered_signer_and_disallowed_boundaries() -> Result<(), NatsBridgeError> {
        let trusted = security();
        let attacker = NatsSecurityConfig::new("attacker", SigningKey::from_bytes(&[9; 32]))
            .allow_source("worker-a")
            .allow_type_prefix("context.");
        let env = envelope(&attacker);
        assert!(matches!(
            verify_envelope(&env, &trusted, &ReplayCache::new(&trusted)?, Utc::now()),
            Err(NatsBridgeError::UnknownSigner(_))
        ));
        let mut env = envelope(&trusted);
        env.source = Some("other".into());
        sign_envelope(&mut env, &trusted).unwrap();
        assert!(matches!(
            verify_envelope(&env, &trusted, &ReplayCache::new(&trusted)?, Utc::now()),
            Err(NatsBridgeError::SourceDenied(_))
        ));
        Ok(())
    }

    #[test]
    fn reconnect_uses_capped_exponential_backoff() {
        let initial = Duration::from_millis(100);
        let max = Duration::from_secs(1);
        assert_eq!(reconnect_delay(initial, max, 1), Duration::from_millis(100));
        assert_eq!(reconnect_delay(initial, max, 4), Duration::from_millis(800));
        assert_eq!(reconnect_delay(initial, max, 8), max);
    }

    #[tokio::test]
    async fn bounded_load_records_contract_outcomes_without_external_evidence() {
        let records = run_bounded_load(
            "contract-run",
            "fake-nats",
            BoundedLoadConfig {
                operations: 3,
                max_concurrency: 2,
                operation_timeout: Duration::from_millis(5),
            },
            |index| async move {
                match index {
                    0 => Ok(()),
                    1 => Err("injected failure".into()),
                    _ => {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        Ok(())
                    }
                }
            },
        )
        .await
        .expect("bounded contract load");
        assert_eq!(records.len(), 3);
        assert!(
            records
                .iter()
                .all(|record| record.source == EvidenceSource::Contract)
        );
        for outcome in ["success", "failure", "timeout"] {
            assert!(records.iter().any(|record| record.outcome == outcome));
        }
    }

    #[test]
    fn endpoint_fingerprint_is_deterministic_and_excludes_credentials() {
        let (first, canonical) = validated_endpoint(
            "postgres://alice:secret@db.example:5432/context",
            &["postgres", "postgresql"],
        )
        .unwrap();
        let (second, _) = validated_endpoint(
            "postgres://bob:other@db.example:5432/another",
            &["postgres", "postgresql"],
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(canonical, "postgres://db.example:5432");
        assert!(!first.contains("secret"));
    }

    #[test]
    fn endpoint_validation_rejects_wrong_service_and_missing_host() {
        assert!(validated_endpoint("nats://broker:4222", &["postgres"]).is_err());
        assert!(validated_endpoint("postgres:///context", &["postgres"]).is_err());
    }

    #[test]
    fn longest_route_prefix_wins() {
        let routes = normalize_routes(vec![
            NatsChannelRoute {
                type_prefix: "context.".into(),
                channel: FlowChannel::Main,
            },
            NatsChannelRoute {
                type_prefix: "context.consolidation_".into(),
                channel: FlowChannel::Consolidation,
            },
        ]);
        assert_eq!(
            route_channel(
                Some(&EventTypeId::new("context", "consolidation_quality")),
                &routes,
                FlowChannel::Monitoring
            ),
            FlowChannel::Consolidation
        );
    }
}
