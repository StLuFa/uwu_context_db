//! CDC / watch API for content writes.
use crate::{Page, PageRequest};

use crate::{
    BrowsingOps, ContentPayload, ContentRepo, ContentStore, ContextEntry, ContextError, ContextUri,
    DirEntry, FindPattern, FsOps, GrepHit, MvccVersion, Result, TreeNode,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WatchCheckpoint(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeKind {
    Write,
    Delete,
    Rename,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeEvent {
    pub checkpoint: WatchCheckpoint,
    pub kind: ChangeKind,
    pub uri: ContextUri,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_uri: Option<ContextUri>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<MvccVersion>,
    pub tenant: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    pub path: String,
    pub occurred_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WatchOptions {
    pub prefix: Option<String>,
    pub from_checkpoint: Option<WatchCheckpoint>,
    pub include_current: bool,
}

pub struct WatchStream {
    rx: broadcast::Receiver<ChangeEvent>,
    backlog: std::collections::VecDeque<ChangeEvent>,
    prefix: Option<String>,
}

impl WatchStream {
    pub async fn recv(&mut self) -> Result<ChangeEvent> {
        loop {
            if let Some(event) = self.backlog.pop_front()
                && self.matches(&event)
            {
                return Ok(event);
            }

            match self.rx.recv().await {
                Ok(event) if self.matches(&event) => return Ok(event),
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    return Err(ContextError::Storage(format!(
                        "watch stream lagged by {skipped} events"
                    )));
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(ContextError::Storage("watch stream closed".into()));
                }
            }
        }
    }

    fn matches(&self, event: &ChangeEvent) -> bool {
        self.prefix
            .as_ref()
            .map(|prefix| {
                event.uri.to_string().starts_with(prefix)
                    || event
                        .to_uri
                        .as_ref()
                        .map(|uri| uri.to_string().starts_with(prefix))
                        .unwrap_or(false)
            })
            .unwrap_or(true)
    }
}

pub trait WatchSource: Send + Sync {
    fn watch(&self, options: WatchOptions) -> WatchStream;
    fn current_checkpoint(&self) -> WatchCheckpoint;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchDelivery {
    pub event: ChangeEvent,
    pub live_receivers: usize,
}

impl WatchDelivery {
    pub fn delivered_live(&self) -> bool {
        self.live_receivers > 0
    }
}

pub struct WatchHub {
    next: AtomicU64,
    tx: broadcast::Sender<ChangeEvent>,
    history: parking_lot::Mutex<Vec<ChangeEvent>>,
    history_capacity: usize,
}

impl WatchHub {
    pub fn new(buffer: usize) -> Self {
        let capacity = buffer.max(1);
        let (tx, _) = broadcast::channel(capacity);
        Self {
            next: AtomicU64::new(1),
            tx,
            history: parking_lot::Mutex::new(Vec::with_capacity(capacity)),
            history_capacity: capacity,
        }
    }

    pub fn emit(
        &self,
        kind: ChangeKind,
        uri: ContextUri,
        to_uri: Option<ContextUri>,
        version: Option<MvccVersion>,
    ) -> WatchDelivery {
        let checkpoint = WatchCheckpoint(self.next.fetch_add(1, Ordering::SeqCst));
        let event = ChangeEvent::new(checkpoint, kind, uri, to_uri, version);
        {
            let mut history = self.history.lock();
            history.push(event.clone());
            if history.len() > self.history_capacity {
                let overflow = history.len() - self.history_capacity;
                history.drain(0..overflow);
            }
        }
        let live_receivers = self.tx.send(event.clone()).unwrap_or(0);
        WatchDelivery {
            event,
            live_receivers,
        }
    }
}

impl WatchSource for WatchHub {
    fn watch(&self, options: WatchOptions) -> WatchStream {
        let from = options.from_checkpoint.unwrap_or(WatchCheckpoint(0));
        let backlog = if options.include_current || options.from_checkpoint.is_some() {
            self.history
                .lock()
                .iter()
                .filter(|event| event.checkpoint > from)
                .cloned()
                .collect()
        } else {
            std::collections::VecDeque::new()
        };
        WatchStream {
            rx: self.tx.subscribe(),
            backlog,
            prefix: options.prefix,
        }
    }

    fn current_checkpoint(&self) -> WatchCheckpoint {
        WatchCheckpoint(self.next.load(Ordering::SeqCst).saturating_sub(1))
    }
}

impl Default for WatchHub {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[derive(Clone)]
pub struct WatchableStore<R> {
    inner: R,
    hub: Arc<WatchHub>,
}

impl<R> WatchableStore<R> {
    pub fn new(inner: R, hub: Arc<WatchHub>) -> Self {
        Self { inner, hub }
    }

    pub fn hub(&self) -> Arc<WatchHub> {
        self.hub.clone()
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R> WatchSource for WatchableStore<R>
where
    R: Send + Sync,
{
    fn watch(&self, options: WatchOptions) -> WatchStream {
        self.hub.watch(options)
    }

    fn current_checkpoint(&self) -> WatchCheckpoint {
        self.hub.current_checkpoint()
    }
}

#[async_trait]
impl<R> FsOps for WatchableStore<R>
where
    R: FsOps + Send + Sync,
{
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        self.inner.ls(dir, page).await
    }

    async fn find(&self, pattern: &FindPattern, page: PageRequest) -> Result<Page<ContextUri>> {
        self.inner.find(pattern, page).await
    }

    async fn grep(&self, regex: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        self.inner.grep(regex, scope).await
    }

    async fn tree(
        &self,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        self.inner.tree(root, depth, page).await
    }

    async fn read(&self, uri: &ContextUri, level: crate::ContentLevel) -> Result<ContentPayload> {
        self.inner.read(uri, level).await
    }
}

#[async_trait]
impl<R> BrowsingOps for WatchableStore<R>
where
    R: BrowsingOps + Send + Sync,
{
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        self.inner.ls(dir, page).await
    }

    async fn tree(
        &self,
        dir: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        self.inner.tree(dir, depth, page).await
    }

    async fn find(
        &self,
        scope: &ContextUri,
        pattern: &str,
        page: PageRequest,
    ) -> Result<Page<ContextUri>> {
        self.inner.find(scope, pattern, page).await
    }

    async fn grep(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<GrepHit>> {
        self.inner.grep(scope, pattern).await
    }
}

#[async_trait]
impl<R> ContentRepo for WatchableStore<R>
where
    R: ContentRepo + Send + Sync,
{
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let uri = entry.uri.clone();
        let version = self.inner.write(entry).await?;
        self.hub.emit(ChangeKind::Write, uri, None, Some(version));
        Ok(version)
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.inner.delete(uri).await?;
        self.hub.emit(ChangeKind::Delete, uri.clone(), None, None);
        Ok(())
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        self.inner.rename(from, to).await?;
        self.hub
            .emit(ChangeKind::Rename, from.clone(), Some(to.clone()), None);
        Ok(())
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        let versions = self.inner.batch_write(entries).await?;
        for (entry, version) in entries.iter().zip(versions.iter().copied()) {
            self.hub
                .emit(ChangeKind::Write, entry.uri.clone(), None, Some(version));
        }
        Ok(versions)
    }
}

#[async_trait]
impl<R> ContentStore for WatchableStore<R>
where
    R: ContentStore + Send + Sync,
{
    async fn read(&self, uri: &ContextUri, level: crate::ContentLevel) -> Result<ContentPayload> {
        self.inner.read(uri, level).await
    }

    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let uri = entry.uri.clone();
        let version = self.inner.write(entry).await?;
        self.hub.emit(ChangeKind::Write, uri, None, Some(version));
        Ok(version)
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.inner.delete(uri).await?;
        self.hub.emit(ChangeKind::Delete, uri.clone(), None, None);
        Ok(())
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        self.inner.rename(from, to).await?;
        self.hub
            .emit(ChangeKind::Rename, from.clone(), Some(to.clone()), None);
        Ok(())
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        let versions = self.inner.batch_write(entries).await?;
        for (entry, version) in entries.iter().zip(versions.iter().copied()) {
            self.hub
                .emit(ChangeKind::Write, entry.uri.clone(), None, Some(version));
        }
        Ok(versions)
    }

    async fn scan_by_prefix(&self, prefix: &str, page: PageRequest) -> Result<Page<ContextEntry>> {
        self.inner.scan_by_prefix(prefix, page).await
    }

    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: crate::ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        self.inner.scan_by_type(prefix, content_type, page).await
    }
}

impl ChangeEvent {
    fn new(
        checkpoint: WatchCheckpoint,
        kind: ChangeKind,
        uri: ContextUri,
        to_uri: Option<ContextUri>,
        version: Option<MvccVersion>,
    ) -> Self {
        let tenant = uri.tenant().to_string();
        let agent = agent_segment(&uri).map(str::to_string);
        let path = uri.to_string();
        Self {
            checkpoint,
            kind,
            uri,
            to_uri,
            version,
            tenant,
            agent,
            path,
            occurred_at: chrono::Utc::now(),
        }
    }
}

fn agent_segment(uri: &ContextUri) -> Option<&str> {
    uri.segments()
        .windows(2)
        .find_map(|pair| (pair[0] == "agent").then_some(pair[1].as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TenantId;
    use std::collections::HashMap;
    use tokio::time::{Duration, timeout};
    use uuid::Uuid;

    #[derive(Default)]
    struct MemoryRepo {
        entries: parking_lot::Mutex<HashMap<String, ContextEntry>>,
        next: AtomicU64,
    }

    #[async_trait]
    impl ContentRepo for MemoryRepo {
        async fn write(&self, mut entry: ContextEntry) -> Result<MvccVersion> {
            let version = MvccVersion(self.next.fetch_add(1, Ordering::SeqCst) + 1);
            entry.mvcc_version = version;
            self.entries.lock().insert(entry.uri.to_string(), entry);
            Ok(version)
        }

        async fn delete(&self, uri: &ContextUri) -> Result<()> {
            self.entries.lock().remove(&uri.to_string());
            Ok(())
        }

        async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
            let mut entries = self.entries.lock();
            if let Some(mut entry) = entries.remove(&from.to_string()) {
                entry.uri = to.clone();
                entries.insert(to.to_string(), entry);
            }
            Ok(())
        }
    }

    fn entry(uri: &str, text: &str) -> ContextEntry {
        ContextEntry::new_text(ContextUri::parse(uri).unwrap(), TenantId(Uuid::nil()), text)
    }

    #[tokio::test]
    async fn watchable_store_emits_write_delete_rename_and_replay() {
        let hub = Arc::new(WatchHub::new(16));
        let store = WatchableStore::new(MemoryRepo::default(), hub.clone());
        let uri = ContextUri::parse("uwu://t/agent/a/memories/fact/one").unwrap();
        let renamed = ContextUri::parse("uwu://t/agent/a/memories/fact/two").unwrap();
        let mut stream = store.watch(WatchOptions {
            prefix: Some("uwu://t/agent/a".into()),
            ..Default::default()
        });
        let mut renamed_prefix_stream = store.watch(WatchOptions {
            prefix: Some("uwu://t/agent/a/memories/fact/two".into()),
            ..Default::default()
        });

        store.write(entry(&uri.to_string(), "one")).await.unwrap();
        store.rename(&uri, &renamed).await.unwrap();
        store.delete(&renamed).await.unwrap();

        let write = recv_test_event(&mut stream).await;
        assert_eq!(write.kind, ChangeKind::Write);
        assert_eq!(write.uri, uri);
        assert_eq!(write.agent.as_deref(), Some("a"));
        assert!(write.version.is_some());

        let rename = recv_test_event(&mut stream).await;
        assert_eq!(rename.kind, ChangeKind::Rename);
        assert_eq!(rename.to_uri, Some(renamed.clone()));

        let renamed_prefix_rename = recv_test_event(&mut renamed_prefix_stream).await;
        assert_eq!(renamed_prefix_rename.kind, ChangeKind::Rename);
        assert_eq!(renamed_prefix_rename.to_uri, Some(renamed.clone()));

        let delete = recv_test_event(&mut stream).await;
        assert_eq!(delete.kind, ChangeKind::Delete);

        let mut replay = hub.watch(WatchOptions {
            from_checkpoint: Some(WatchCheckpoint(1)),
            include_current: true,
            ..Default::default()
        });
        assert_eq!(recv_test_event(&mut replay).await.kind, ChangeKind::Rename);
        assert_eq!(recv_test_event(&mut replay).await.kind, ChangeKind::Delete);
    }

    async fn recv_test_event(stream: &mut WatchStream) -> ChangeEvent {
        timeout(Duration::from_secs(1), stream.recv())
            .await
            .expect("watch stream timed out")
            .expect("watch stream returned error")
    }
}
