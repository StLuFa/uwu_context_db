//! 统一事件存储 — EventStore + 事件溯源 + 因果 DAG + 快照优化。
//!
//! 合并旧 `ContextPubSub`（observability.rs）和 `EventEmitter`（event.rs），
//! 事件持久化到 append-only log，支持从任意 offset 回放。

use crate::{ContentPayload, ContextEntry, ContextUri, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use uuid::Uuid;

// ===========================================================================
// DomainEvent
// ===========================================================================

/// 事件 ID。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub Uuid);

impl EventId {
    pub fn new() -> Self { Self(Uuid::new_v4()) }
}

/// 统一领域事件 — 合并 ChangeEvent + ChangeEventStream。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainEvent {
    pub id: EventId,
    pub timestamp: DateTime<Utc>,
    pub kind: EventKind,
    pub uri: ContextUri,
    pub payload: EventPayload,
    pub causality: CausalLink,
    pub metadata: EventMetadata,
}

/// 事件类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Created,
    Updated,
    Deleted,
    Archived,
    Merged,
    Branched,
    Tagged,
    QualityAssessed,
    Consolidated,
}

/// 事件负载。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPayload {
    pub before: Option<serde_json::Value>,
    pub after: Option<serde_json::Value>,
    pub diff_summary: Option<String>,
}

/// 事件元数据。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    pub source: Option<String>,
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub tags: Vec<String>,
}

// ===========================================================================
// 因果 DAG
// ===========================================================================

/// 因果链 — 支持多因一果（替代旧的单链表 CausalLink）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    pub event_id: EventId,
    /// 父事件（多因一果）。
    pub parents: Vec<EventId>,
    pub causality_type: CausalityType,
}

impl CausalLink {
    pub fn new(event_id: EventId) -> Self {
        Self {
            event_id,
            parents: vec![],
            causality_type: CausalityType::Direct,
        }
    }

    pub fn with_parent(mut self, parent: EventId, ct: CausalityType) -> Self {
        self.parents.push(parent);
        self.causality_type = ct;
        self
    }
}

/// 因果类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalityType {
    /// 直接因果。
    Direct,
    /// 贡献性因果（多个事件共同导致）。
    Contributory,
    /// 必要条件。
    Necessary,
    /// 充分条件。
    Sufficient,
}

/// 因果 DAG — 用于因果追溯查询。
#[derive(Debug, Clone, Default)]
pub struct CausalDag {
    pub nodes: HashMap<EventId, DomainEvent>,
    pub edges: HashMap<EventId, Vec<EventId>>,
}

impl CausalDag {
    /// 哪些事件的组合导致了 target。
    pub fn root_causes(&self, event_id: &EventId) -> Vec<EventId> {
        let mut roots = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![*event_id];
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(event) = self.nodes.get(&id) {
                if event.causality.parents.is_empty() {
                    roots.push(id);
                } else {
                    stack.extend(&event.causality.parents);
                }
            }
        }
        roots
    }

    /// target 导致了哪些后续事件。
    pub fn effects(&self, event_id: &EventId) -> Vec<EventId> {
        self.edges.get(event_id).cloned().unwrap_or_default()
    }
}

// ===========================================================================
// EventStore trait
// ===========================================================================

/// 统一事件存储端口 — 合并 ContextPubSub + EventEmitter。
#[async_trait]
pub trait EventStore: Send + Sync {
    /// 追加事件到 append-only log。
    async fn append(&self, event: DomainEvent) -> Result<EventId>;

    /// 从指定 offset 读取事件。
    async fn read_from(&self, offset: u64, limit: usize) -> Result<Vec<DomainEvent>>;

    /// 订阅事件流（实时推送）。
    async fn subscribe(
        &self,
        filter: EventFilter,
    ) -> Result<Box<dyn EventStream>>;

    /// 因果追溯 — 沿因果 DAG 遍历。
    async fn trace_causality(&self, event_id: &EventId) -> Result<CausalDag>;

    /// 事件折叠 — 从事件流重建当前状态。
    async fn fold(&self, uri: &ContextUri, from: Option<EventId>) -> Result<ContextEntry>;
}

// ===========================================================================
// EventFilter + EventStream
// ===========================================================================

/// 事件订阅过滤条件。
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    pub uri_prefix: Option<String>,
    pub kinds: Vec<EventKind>,
    pub agent_id: Option<String>,
}

/// 事件流 — 异步迭代器。
#[async_trait]
pub trait EventStream: Send + Sync {
    async fn next(&mut self) -> Option<DomainEvent>;
}

// ===========================================================================
// 快照存储
// ===========================================================================

/// 快照存储 — 定期物化当前状态，避免每次从头回放。
pub struct SnapshotStore {
    /// uri → (last_applied_event_id, snapshot)
    snapshots: RwLock<HashMap<String, (EventId, ContextEntry)>>,
}

impl SnapshotStore {
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(HashMap::new()),
        }
    }

    /// 读取快照（如果存在）。
    pub fn get(&self, uri: &ContextUri) -> Option<(EventId, ContextEntry)> {
        self.snapshots.read().get(&uri.to_string()).cloned()
    }

    /// 更新快照。
    pub fn set(&self, uri: &ContextUri, event_id: EventId, entry: ContextEntry) {
        self.snapshots
            .write()
            .insert(uri.to_string(), (event_id, entry));
    }

    /// 删除快照。
    pub fn remove(&self, uri: &ContextUri) {
        self.snapshots.write().remove(&uri.to_string());
    }
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// 内存实现（测试用）
// ===========================================================================

/// 内存事件存储。
pub struct MemoryEventStore {
    events: RwLock<Vec<DomainEvent>>,
    snapshots: SnapshotStore,
}

impl MemoryEventStore {
    pub fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
            snapshots: SnapshotStore::new(),
        }
    }
}

#[async_trait]
impl EventStore for MemoryEventStore {
    async fn append(&self, event: DomainEvent) -> Result<EventId> {
        let id = event.id;
        self.events.write().push(event);
        Ok(id)
    }

    async fn read_from(&self, offset: u64, limit: usize) -> Result<Vec<DomainEvent>> {
        let events = self.events.read();
        let start = offset as usize;
        Ok(events
            .iter()
            .skip(start)
            .take(limit)
            .cloned()
            .collect())
    }

    async fn subscribe(
        &self,
        _filter: EventFilter,
    ) -> Result<Box<dyn EventStream>> {
        Ok(Box::new(MemoryEventStream {
            events: self.events.read().clone(),
            offset: 0,
        }))
    }

    async fn trace_causality(&self, event_id: &EventId) -> Result<CausalDag> {
        let events = self.events.read();
        let mut dag = CausalDag::default();
        // 构建因果 DAG
        for event in events.iter().filter(|e| e.causality.event_id == *event_id
            || e.causality.parents.contains(event_id))
        {
            dag.nodes.insert(event.id, event.clone());
            dag.edges
                .entry(event.causality.event_id)
                .or_default()
                .extend(event.causality.parents.iter().copied());
        }
        Ok(dag)
    }

    async fn fold(&self, uri: &ContextUri, from: Option<EventId>) -> Result<ContextEntry> {
        let uri_str = uri.to_string();
        let events = self.events.read();
        let relevant: Vec<&DomainEvent> = events
            .iter()
            .filter(|e| e.uri.to_string() == uri_str)
            .collect();

        if relevant.is_empty() {
            return Err(crate::ContextError::NotFound(uri_str));
        }

        // 从最后一个事件重建状态（简化版）
        let last = relevant.last().unwrap();
        if let Some(after) = &last.payload.after {
            serde_json::from_value(after.clone())
                .map_err(|e| crate::ContextError::Serialization(e.to_string()))
        } else {
            Err(crate::ContextError::NotFound(uri_str))
        }
    }
}

/// 内存事件流。
struct MemoryEventStream {
    events: Vec<DomainEvent>,
    offset: usize,
}

#[async_trait]
impl EventStream for MemoryEventStream {
    async fn next(&mut self) -> Option<DomainEvent> {
        if self.offset < self.events.len() {
            let event = self.events[self.offset].clone();
            self.offset += 1;
            Some(event)
        } else {
            None
        }
    }
}
