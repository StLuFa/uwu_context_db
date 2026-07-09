//! # agent-context-db-nats
//!
//! `NatsBridge` — 将本地 `EventMesh` 事件转发到 NATS/JetStream；
//! `NatsIngestor` — 从 NATS 订阅并回灌本地 mesh，实现跨进程/跨节点事件同步。
//!
//! ## 用法
//!
//! ```ignore
//! use agent_context_db_core::EventMesh;
//! use agent_context_db_nats::{NatsBridge, NatsIngestor, NatsBridgeConfig};
//! use std::sync::Arc;
//!
//! let mesh = EventMesh::new();
//!
//! // 1) 出站：本地 → NATS
//! let bridge = NatsBridge::connect(NatsBridgeConfig::default()).await?;
//! mesh.attach_bridge(Arc::new(bridge)).await?;
//!
//! // 2) 入站：NATS → 本地
//! let ingestor = NatsIngestor::connect(NatsBridgeConfig::default(), mesh.clone()).await?;
//! ingestor.spawn(); // 后台 task
//! ```

use std::sync::Arc;

use agent_context_db_core::{
    Bridge, Envelope, EventMesh, EventTypeId, FlowChannel, SerializedEnvelope,
};
use async_trait::async_trait;
use uwu_event_mesh::core::error::{EventMeshError, Result as MeshResult};
use uwu_nats_bridge::{NatsConfig, NatsPublisher, NatsSubjects, NatsSubscriber};

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

/// NatsBridge 配置。
#[derive(Debug, Clone)]
pub struct NatsBridgeConfig {
    /// NATS 服务器 URL。
    pub url: String,
    /// 关联 ID（session_id / agent_id），映射到 subject `agent.{cid}.*`。
    pub correlation_id: String,
    /// 连接名（观测用）。
    pub connection_name: String,
    /// 未匹配路由时使用的 fallback channel。
    pub default_channel: FlowChannel,
    /// 按事件类型前缀选择 channel，例如 `context.consolidation.` → Consolidation。
    pub channel_routes: Vec<NatsChannelRoute>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NatsChannelRoute {
    pub type_prefix: String,
    pub channel: FlowChannel,
}

impl Default for NatsBridgeConfig {
    fn default() -> Self {
        Self {
            url: "nats://localhost:4222".into(),
            correlation_id: "default".into(),
            connection_name: "context-db".into(),
            default_channel: FlowChannel::Consolidation,
            channel_routes: default_channel_routes(),
        }
    }
}

impl NatsBridgeConfig {
    pub fn route(mut self, type_prefix: impl Into<String>, channel: FlowChannel) -> Self {
        self.channel_routes.push(NatsChannelRoute {
            type_prefix: type_prefix.into(),
            channel,
        });
        self
    }

    fn to_nats(&self) -> NatsConfig {
        let mut c = NatsConfig::default();
        c.url = self.url.clone();
        c.connection_name = self.connection_name.clone();
        c
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

// ---------------------------------------------------------------------------
// 出站桥：本地 EventMesh → NATS
// ---------------------------------------------------------------------------

/// 出站桥：`impl Bridge for NatsBridge`，把本地 mesh publish 出去的信封转发到 NATS。
pub struct NatsBridge {
    publisher: NatsPublisher,
    default_channel: FlowChannel,
    channel_routes: Vec<NatsChannelRoute>,
}

impl NatsBridge {
    pub async fn connect(cfg: NatsBridgeConfig) -> Result<Self, NatsBridgeError> {
        let subjects = NatsSubjects::new(cfg.correlation_id.clone());
        let publisher = NatsPublisher::connect(cfg.to_nats(), subjects)
            .await
            .map_err(|e| NatsBridgeError::Publisher(e.to_string()))?;
        Ok(Self {
            publisher,
            default_channel: cfg.default_channel,
            channel_routes: normalize_routes(cfg.channel_routes),
        })
    }

    /// 允许用户根据 Envelope 内容动态选择 channel（例如 monitoring 事件走 Monitoring）。
    pub fn with_channel(mut self, ch: FlowChannel) -> Self {
        self.default_channel = ch;
        self
    }

    pub fn route_channel(&self, env: &Envelope) -> FlowChannel {
        route_channel(
            env.type_id.as_ref(),
            &self.channel_routes,
            self.default_channel,
        )
    }

    /// 底层 publisher（进阶用法）。
    pub fn publisher(&self) -> &NatsPublisher {
        &self.publisher
    }
}

#[async_trait]
impl Bridge for NatsBridge {
    async fn publish_remote(&self, env: Arc<Envelope>) -> MeshResult<()> {
        let serialized = SerializedEnvelope::from_envelope(&env)?;
        self.publisher
            .publish_envelope(self.route_channel(&env), serialized)
            .await
            .map_err(|e| {
                EventMeshError::Serialize(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("nats publish: {e}"),
                )))
            })?;
        Ok(())
    }
}

fn normalize_routes(mut routes: Vec<NatsChannelRoute>) -> Vec<NatsChannelRoute> {
    routes.sort_by(|a, b| b.type_prefix.len().cmp(&a.type_prefix.len()));
    routes
}

fn route_channel(
    type_id: Option<&EventTypeId>,
    routes: &[NatsChannelRoute],
    default_channel: FlowChannel,
) -> FlowChannel {
    let Some(type_id) = type_id else {
        return default_channel;
    };
    let type_name = type_id.to_string();
    routes
        .iter()
        .find(|route| type_name.starts_with(&route.type_prefix))
        .map(|route| route.channel)
        .unwrap_or(default_channel)
}

// ---------------------------------------------------------------------------
// 入站：NATS → 本地 EventMesh
// ---------------------------------------------------------------------------

/// 入站 ingestor：把 NATS 订阅到的事件回灌到本地 mesh。
pub struct NatsIngestor {
    subscriber: NatsSubscriber,
    mesh: EventMesh,
}

impl NatsIngestor {
    pub async fn connect(cfg: NatsBridgeConfig, mesh: EventMesh) -> Result<Self, NatsBridgeError> {
        let subscriber = NatsSubscriber::connect(cfg.to_nats(), cfg.correlation_id)
            .await
            .map_err(|e| NatsBridgeError::Subscriber(e.to_string()))?;
        Ok(Self { subscriber, mesh })
    }

    /// 后台运行 —— 一直接收直到 mesh 关闭或 subscriber 断开。
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        let mut subscriber = self.subscriber;
        let mesh = self.mesh;
        tokio::spawn(async move {
            while let Some((_channel, serialized)) = subscriber.recv_any().await {
                match serialized.into_envelope() {
                    Ok(env) => {
                        if let Err(e) = mesh.ingest_remote(Arc::new(env)).await {
                            tracing::warn!("ingest_remote failed: {e}");
                        }
                    }
                    Err(e) => tracing::warn!("deserialize envelope: {e}"),
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// 一体化装配：EventSystem
// ---------------------------------------------------------------------------

/// 已接线的事件系统：本地 `EventMesh` + 出站 `NatsBridge` + 入站 `NatsIngestor` 后台 task。
///
/// 上层应用只需持有此结构即可获得跨进程事件能力。Drop 时不会自动关闭 ingestor；
/// 若需受控停机，调用 [`EventSystem::shutdown`]。
pub struct EventSystem {
    /// 已 attach 出站桥的本地 mesh，可传给 marketplace/consolidation 等模块。
    pub mesh: EventMesh,
    ingestor: Option<tokio::task::JoinHandle<()>>,
}

impl EventSystem {
    /// 新建 mesh 并接入 NATS。
    pub async fn with_nats(cfg: NatsBridgeConfig) -> Result<Self, NatsBridgeError> {
        Self::attach(EventMesh::new(), cfg).await
    }

    /// 在既有 mesh 上接入 NATS（用于调用方已通过 builder 定制 store/dedup 的场景）。
    pub async fn attach(mesh: EventMesh, cfg: NatsBridgeConfig) -> Result<Self, NatsBridgeError> {
        let bridge = NatsBridge::connect(cfg.clone()).await?;
        mesh.attach_bridge(Arc::new(bridge));
        let ingestor = NatsIngestor::connect(cfg, mesh.clone()).await?;
        let handle = ingestor.spawn();
        Ok(Self {
            mesh,
            ingestor: Some(handle),
        })
    }

    /// 停止入站 task（出站桥随 mesh Drop 释放）。
    pub fn shutdown(&mut self) {
        if let Some(h) = self.ingestor.take() {
            h.abort();
        }
    }
}

impl Drop for EventSystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::EventTypeId;

    #[test]
    fn routes_context_events_by_type_prefix() {
        let routes = default_channel_routes();
        assert_eq!(
            route_channel(
                Some(&EventTypeId::new("context", "entry_created")),
                &routes,
                FlowChannel::Consolidation
            ),
            FlowChannel::Main
        );
        assert_eq!(
            route_channel(
                Some(&EventTypeId::new("context", "consolidation_consolidated")),
                &routes,
                FlowChannel::Main
            ),
            FlowChannel::Consolidation
        );
        assert_eq!(
            route_channel(
                Some(&EventTypeId::new("intent", "decision")),
                &routes,
                FlowChannel::Consolidation
            ),
            FlowChannel::Monitoring
        );
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
                Some(&EventTypeId::new(
                    "context",
                    "consolidation_quality_assessed"
                )),
                &routes,
                FlowChannel::Monitoring,
            ),
            FlowChannel::Consolidation
        );
    }
}

// ---------------------------------------------------------------------------
// 错误
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum NatsBridgeError {
    #[error("nats publisher: {0}")]
    Publisher(String),

    #[error("nats subscriber: {0}")]
    Subscriber(String),

    #[error("mesh error: {0}")]
    Mesh(#[from] uwu_event_mesh::core::error::EventMeshError),
}
