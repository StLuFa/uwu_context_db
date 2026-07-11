//! # agent-context-db-testkit
//!
//! 存储端口内存实现，仅用于测试/开发：
//! - [`MemoryContextStore`]：`FsOps` + `ContentRepo` + `VersionOps` + `TenantOps` 四端口
//! - [`MemoryVersionStore`]：`VersionStore`（Commit/Branch/Tag DAG）
//!
//! 生产环境由 `agent-context-db-storage` 注入 PG + Qdrant 后端。
use agent_context_db_core::{Page, PageRequest};

pub mod version;

pub use version::MemoryVersionStore;

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentRepo, ContentStore, ContentType, ContextDiff,
    ContextEntry, ContextError, ContextUri, DirEntry, FindPattern, FsOps, GraphRelation,
    GraphStore, GrepHit, MvccVersion, Result, TenantId, TenantOps, TreeNode, VersionEntry,
    VersionOps, sanitize_entry_for_write,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;

/// 内存版存储 —— 同时实现四个窄端口，故自动满足 `ContextStore`。
#[derive(Default)]
pub struct MemoryContextStore {
    // uri -> 版本序列（末尾为最新）
    entries: Mutex<HashMap<String, Vec<ContextEntry>>>,
    // uri -> L2 原始字节（模拟 AGFS blob 存储）
    l2_blobs: Mutex<HashMap<String, Vec<u8>>>,
    // from uri -> typed outgoing edges
    graph_edges: Mutex<HashMap<String, Vec<(String, GraphRelation)>>>,
}

impl MemoryContextStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn latest(&self, uri: &str) -> Option<ContextEntry> {
        self.entries.lock().get(uri).and_then(|v| v.last().cloned())
    }

    /// 存入 L2 原始字节（模拟 AGFS blob 写）。
    pub fn put_l2_blob(&self, uri: &str, bytes: Vec<u8>) {
        self.l2_blobs.lock().insert(uri.to_string(), bytes);
    }
}

/// 对已经按稳定键排序的结果执行 keyset 分页。
///
/// 多取一条仅用于判断后续页是否存在；游标始终是本页最后一条的键。
fn keyset_page<T>(items: Vec<T>, page: &PageRequest, key: impl Fn(&T) -> String) -> Page<T> {
    let limit = page.effective_limit();
    let mut selected: Vec<T> = items
        .into_iter()
        .filter(|item| page.after.as_ref().is_none_or(|after| key(item) > *after))
        .take(limit + 1)
        .collect();
    let has_more = selected.len() > limit;
    if has_more {
        selected.pop();
    }
    let next_cursor = if has_more {
        selected.last().map(&key)
    } else {
        None
    };
    Page::new(selected, next_cursor)
}

#[async_trait]
impl GraphStore for MemoryContextStore {
    async fn add_edge(
        &self,
        from: &ContextUri,
        to: &ContextUri,
        kind: GraphRelation,
    ) -> Result<()> {
        self.graph_edges
            .lock()
            .entry(from.to_string())
            .or_default()
            .push((to.to_string(), kind));
        Ok(())
    }

    async fn remove_edge(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        if let Some(edges) = self.graph_edges.lock().get_mut(&from.to_string()) {
            edges.retain(|(target, _)| target != &to.to_string());
        }
        Ok(())
    }

    async fn outgoing_neighbors(
        &self,
        uri: &ContextUri,
        kind: Option<GraphRelation>,
    ) -> Result<Vec<ContextUri>> {
        let edges = self.graph_edges.lock();
        Ok(edges
            .get(&uri.to_string())
            .into_iter()
            .flat_map(|out| out.iter())
            .filter(|(_, relation)| kind.is_none_or(|k| k == *relation))
            .filter_map(|(target, _)| ContextUri::parse(target.clone()).ok())
            .collect())
    }

    async fn incoming_neighbors(
        &self,
        uri: &ContextUri,
        kind: Option<GraphRelation>,
    ) -> Result<Vec<ContextUri>> {
        let target = uri.to_string();
        let edges = self.graph_edges.lock();
        Ok(edges
            .iter()
            .flat_map(|(source, out)| {
                out.iter().filter_map(|(candidate, relation)| {
                    (candidate == &target && kind.is_none_or(|k| k == *relation))
                        .then(|| ContextUri::parse(source.clone()).ok())
                        .flatten()
                })
            })
            .collect())
    }

    async fn batch_traverse(
        &self,
        seeds: &[ContextUri],
        kinds: &[GraphRelation],
        max_hops: usize,
    ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
        let edges = self.graph_edges.lock().clone();
        let allowed: std::collections::HashSet<_> = kinds.iter().copied().collect();
        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        for seed in seeds {
            visited.insert(seed.to_string());
            queue.push_back((seed.clone(), 0_usize));
        }
        while let Some((from, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }
            if let Some(next) = edges.get(&from.to_string()) {
                for (target, relation) in next {
                    if !allowed.is_empty() && !allowed.contains(relation) {
                        continue;
                    }
                    let Ok(to) = ContextUri::parse(target.clone()) else {
                        continue;
                    };
                    out.push((from.clone(), to.clone(), *relation));
                    if visited.insert(target.clone()) {
                        queue.push_back((to, depth + 1));
                    }
                }
            }
        }
        Ok(out)
    }

    async fn centrality(&self, uri: &ContextUri) -> Result<f32> {
        let edges = self.graph_edges.lock();
        let out_degree = edges.get(&uri.to_string()).map(|v| v.len()).unwrap_or(0);
        let in_degree = edges
            .values()
            .flatten()
            .filter(|(target, _)| target == &uri.to_string())
            .count();
        Ok(((out_degree + in_degree) as f32 / 16.0).min(1.0))
    }
}

#[async_trait]
impl ContentRepo for MemoryContextStore {
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        let mut entry = sanitize_entry_for_write(&entry)?;
        let mut map = self.entries.lock();
        let list = map.entry(entry.uri.to_string().clone()).or_default();
        let next = MvccVersion(list.len() as u64 + 1);
        entry.mvcc_version = next;
        entry.updated_at = chrono::Utc::now();
        list.push(entry);
        Ok(next)
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.entries.lock().remove(&uri.to_string());
        Ok(())
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        let mut map = self.entries.lock();
        let val = map
            .remove(&from.to_string())
            .ok_or_else(|| ContextError::NotFound(from.to_string()))?;
        map.insert(to.to_string(), val);
        Ok(())
    }
}

#[async_trait]
impl FsOps for MemoryContextStore {
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        let prefix = format!("{}/", dir.to_string().trim_end_matches('/'));
        let map = self.entries.lock();
        let mut out = Vec::new();
        for (uri, versions) in map.iter() {
            if let Some(rest) = uri.strip_prefix(&prefix) {
                let is_dir = rest.contains('/');
                let Some(latest) = versions.last() else {
                    continue;
                };
                let Ok(u) = ContextUri::parse(uri.clone()) else {
                    continue;
                };
                out.push(DirEntry {
                    uri: u,
                    is_dir,
                    abstract_: latest.l0_text().to_string(),
                    content_type: latest.metadata.content_type,
                });
            }
        }
        out.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(keyset_page(out, &page, |entry| entry.uri.to_string()))
    }

    async fn find(&self, pattern: &FindPattern, page: PageRequest) -> Result<Page<ContextUri>> {
        let map = self.entries.lock();
        let scope = pattern
            .scope
            .as_ref()
            .map(|u| u.to_string())
            .unwrap_or_default();
        let mut uris: Vec<_> = map
            .iter()
            .filter(|(uri, _)| uri.starts_with(&scope))
            .filter(|(_, versions)| match pattern.content_type {
                Some(ct) => versions
                    .last()
                    .and_then(|e| e.metadata.content_type)
                    .map(|c| c == ct)
                    .unwrap_or(false),
                None => true,
            })
            .filter_map(|(uri, _)| ContextUri::parse(uri.clone()).ok())
            .collect();
        uris.sort();
        Ok(keyset_page(uris, &page, ToString::to_string))
    }

    async fn grep(&self, regex: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        let needle = regex.to_lowercase();
        let map = self.entries.lock();
        let mut hits = Vec::new();
        let scope_str = scope.to_string();
        for (uri, versions) in map.iter() {
            if !uri.starts_with(&scope_str) {
                continue;
            }
            if let Some(e) = versions.last() {
                let l0 = e.l0_text();
                if l0.to_lowercase().contains(&needle) {
                    let Ok(u) = ContextUri::parse(uri.clone()) else {
                        continue;
                    };
                    hits.push(GrepHit {
                        uri: u,
                        line: l0.to_string(),
                        level: ContentLevel::L0,
                    });
                }
            }
        }
        Ok(hits)
    }

    async fn tree(
        &self,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        let prefix = format!("{}/", root.to_string().trim_end_matches('/'));
        let map = self.entries.lock();

        // 收集所有 root 下的 URI
        let uris: Vec<String> = map
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();

        let children = build_memory_tree(&prefix, &uris, 0, depth);
        Ok(keyset_page(
            vec![TreeNode {
                uri: root.clone(),
                is_dir: true,
                children,
            }],
            &page,
            |node| node.uri.to_string(),
        ))
    }

    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        let e = self
            .latest(&uri.to_string())
            .ok_or_else(|| ContextError::NotFound(uri.to_string().clone()))?;
        let l0 = e.l0_text().to_string();
        let l1 = match &e.payload {
            ContentPayload::Text { dense, .. } => dense.clone(),
            _ => String::new(),
        };
        Ok(match level {
            ContentLevel::L0 => ContentPayload::Text {
                sparse: l0.clone(),
                dense: l1.clone(),
                full: l0,
            },
            ContentLevel::L1 => ContentPayload::Text {
                sparse: l0.clone(),
                dense: l1.clone(),
                full: l0,
            },
            ContentLevel::L2 => {
                if let Some(bytes) = self.l2_blobs.lock().get(&uri.to_string())
                    && !bytes.is_empty()
                {
                    return Ok(ContentPayload::Text {
                        sparse: l0,
                        dense: l1,
                        full: String::from_utf8(bytes.clone()).unwrap_or_default(),
                    });
                }
                match &e.payload {
                    ContentPayload::Text { full, .. } => ContentPayload::Text {
                        sparse: l0,
                        dense: l1,
                        full: full.clone(),
                    },
                    other => other.clone(),
                }
            }
        })
    }
}

#[async_trait]
impl ContentStore for MemoryContextStore {
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        <Self as FsOps>::read(self, uri, level).await
    }

    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        <Self as ContentRepo>::write(self, entry).await
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        <Self as ContentRepo>::delete(self, uri).await
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        <Self as ContentRepo>::rename(self, from, to).await
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        <Self as ContentRepo>::batch_write(self, entries).await
    }

    async fn scan_by_prefix(&self, prefix: &str, page: PageRequest) -> Result<Page<ContextEntry>> {
        let map = self.entries.lock();
        let mut entries: Vec<_> = map
            .iter()
            .filter(|(uri, _)| uri.starts_with(prefix))
            .filter_map(|(_, versions)| versions.last().cloned())
            .collect();
        entries.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(keyset_page(entries, &page, |entry| entry.uri.to_string()))
    }

    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        let map = self.entries.lock();
        let mut entries: Vec<_> = map
            .iter()
            .filter(|(uri, _)| uri.starts_with(prefix))
            .filter_map(|(_, versions)| versions.last().cloned())
            .filter(|entry| entry.metadata.content_type == Some(content_type))
            .collect();
        entries.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(keyset_page(entries, &page, |entry| entry.uri.to_string()))
    }
}

#[async_trait]
impl VersionOps for MemoryContextStore {
    async fn version_history(
        &self,
        uri: &ContextUri,
        page: PageRequest,
    ) -> Result<Page<VersionEntry>> {
        let versions = self
            .entries
            .lock()
            .get(&uri.to_string())
            .map(|list| {
                list.iter()
                    .map(|e| VersionEntry {
                        version: e.mvcc_version,
                        message: e.l0_text().to_string(),
                        ts: e.updated_at,
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(keyset_page(versions, &page, |entry| {
            format!("{:020}", entry.version.0)
        }))
    }

    async fn rollback(&self, uri: &ContextUri, to: MvccVersion) -> Result<()> {
        let mut map = self.entries.lock();
        let list = map
            .get_mut(&uri.to_string())
            .ok_or_else(|| ContextError::NotFound(uri.to_string().clone()))?;
        let target = list
            .iter()
            .find(|e| e.mvcc_version == to)
            .cloned()
            .ok_or_else(|| ContextError::VersionConflict(format!("no version {:?}", to)))?;
        // rollback = 以旧版内容追加新版
        let mut restored = target;
        restored.mvcc_version = MvccVersion(list.len() as u64 + 1);
        list.push(restored);
        Ok(())
    }

    async fn diff(&self, uri: &ContextUri, a: MvccVersion, b: MvccVersion) -> Result<ContextDiff> {
        Ok(ContextDiff {
            summary: format!("{}: v{:?} → v{:?}", uri, a, b),
        })
    }
}

#[async_trait]
impl TenantOps for MemoryContextStore {
    async fn list_tenants(&self, page: PageRequest) -> Result<Page<TenantId>> {
        let map = self.entries.lock();
        let mut set: Vec<TenantId> = map
            .values()
            .filter_map(|v| v.last().map(|e| e.tenant))
            .collect();
        set.sort_by_key(|t| t.0);
        set.dedup_by_key(|t| t.0);
        Ok(keyset_page(set, &page, |tenant| tenant.0.to_string()))
    }
}

/// 递归构建内存树节点。
fn build_memory_tree(
    prefix: &str,
    all_uris: &[String],
    current_depth: usize,
    max_depth: usize,
) -> Vec<TreeNode> {
    if current_depth >= max_depth {
        return vec![];
    }

    // 提取当前层级的直接子项名称
    let mut seen: std::collections::BTreeMap<String, bool> = std::collections::BTreeMap::new();
    // name -> is_dir

    for uri_str in all_uris {
        let rest = match uri_str.strip_prefix(prefix) {
            Some(r) => r,
            None => continue,
        };
        if rest.is_empty() {
            continue;
        }
        let slash_pos = rest.find('/');
        if let Some(pos) = slash_pos {
            let dir_name = &rest[..pos];
            seen.entry(dir_name.to_string()).or_insert(true);
        } else {
            seen.entry(rest.to_string()).or_insert(false);
        }
    }

    let mut children = Vec::new();
    for (name, is_dir) in seen {
        let Ok(child_uri) = ContextUri::parse(format!("{}{}", prefix, name)) else {
            continue;
        };
        if is_dir {
            let child_prefix = format!("{}{}/", prefix, name);
            let sub_children =
                build_memory_tree(&child_prefix, all_uris, current_depth + 1, max_depth);
            children.push(TreeNode {
                uri: child_uri,
                is_dir: true,
                children: sub_children,
            });
        } else {
            children.push(TreeNode {
                uri: child_uri,
                is_dir: false,
                children: vec![],
            });
        }
    }
    children
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::ContextStore;
    use uuid::Uuid;

    fn entry(uri: &str, text: &str) -> ContextEntry {
        ContextEntry::new_text(ContextUri::parse(uri).unwrap(), TenantId(Uuid::nil()), text)
    }

    #[tokio::test]
    async fn write_read_ls_roundtrip() {
        let store = MemoryContextStore::new();
        ContentRepo::write(
            &store,
            entry("uwu://t/agent/a/memories/cases/c1", "solved bug X"),
        )
        .await
        .unwrap();

        // read L0
        let p = FsOps::read(
            &store,
            &ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap(),
            ContentLevel::L0,
        )
        .await
        .unwrap();
        assert!(matches!(p, ContentPayload::Text { sparse, .. } if sparse == "solved bug X"));

        // ls parent dir
        let dir = ContextUri::parse("uwu://t/agent/a/memories/cases").unwrap();
        assert_eq!(
            store.ls(&dir, PageRequest::default()).await.unwrap().len(),
            1
        );

        // grep
        let hits = store
            .grep("bug", &ContextUri::parse("uwu://t").unwrap())
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn write_sanitizes_sensitive_payload_before_storage() {
        let store = MemoryContextStore::new();
        let uri = ContextUri::parse("uwu://t/agent/a/memories/evidence/pii").unwrap();
        ContentRepo::write(
            &store,
            entry(
                &uri.to_string(),
                "contact user@example.com with sk-secret12345678901234567890",
            ),
        )
        .await
        .unwrap();

        let payload = FsOps::read(&store, &uri, ContentLevel::L2).await.unwrap();
        let ContentPayload::Text { sparse, full, .. } = payload else {
            panic!("expected text payload");
        };
        assert!(!sparse.contains("user@example.com"));
        assert!(!full.contains("sk-secret12345678901234567890"));
        assert!(sparse.contains("[REDACTED_EMAIL]"));
        assert!(full.contains("[REDACTED_SECRET]"));
        assert!(
            store
                .latest(&uri.to_string())
                .unwrap()
                .metadata
                .custom
                .get("security_redactions")
                .is_some()
        );
    }

    #[tokio::test]
    async fn version_history_and_rollback() {
        let store = MemoryContextStore::new();
        let uri = ContextUri::parse("uwu://t/agent/a/state/mid/s1").unwrap();
        let v1 = ContentRepo::write(&store, entry(&uri.to_string(), "v1"))
            .await
            .unwrap();
        ContentRepo::write(&store, entry(&uri.to_string(), "v2"))
            .await
            .unwrap();
        assert_eq!(
            store
                .version_history(&uri, PageRequest::default())
                .await
                .unwrap()
                .len(),
            2
        );

        store.rollback(&uri, v1).await.unwrap();
        assert_eq!(
            store
                .version_history(&uri, PageRequest::default())
                .await
                .unwrap()
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn all_page_requests_use_stable_keyset_pagination() {
        let store = MemoryContextStore::new();
        let tenant_a = TenantId(Uuid::from_u128(1));
        let tenant_b = TenantId(Uuid::from_u128(2));
        for (index, tenant) in [tenant_a, tenant_a, tenant_b, tenant_b, tenant_b]
            .into_iter()
            .enumerate()
        {
            let mut value = entry(
                &format!("uwu://t/agent/a/memories/cases/c{index}"),
                &format!("v{index}"),
            );
            value.tenant = tenant;
            value.metadata.content_type = Some(if index % 2 == 0 {
                ContentType::Fact
            } else {
                ContentType::Belief
            });
            ContentRepo::write(&store, value).await.unwrap();
        }

        async fn collect_uris<F, Fut>(mut load: F) -> Vec<String>
        where
            F: FnMut(PageRequest) -> Fut,
            Fut: std::future::Future<Output = Result<Page<ContextUri>>>,
        {
            let mut cursor = None;
            let mut all = Vec::new();
            loop {
                let request = cursor.as_ref().map_or_else(
                    || PageRequest::new(2),
                    |value| PageRequest::new(2).after(value),
                );
                let page = load(request).await.unwrap();
                all.extend(page.items.into_iter().map(|uri| uri.to_string()));
                match page.next_cursor {
                    Some(next) => cursor = Some(next),
                    None => break,
                }
            }
            all
        }

        let scope = ContextUri::parse("uwu://t/agent/a/memories/cases").unwrap();
        let pattern = FindPattern {
            scope: Some(scope.clone()),
            content_type: None,
            name_glob: None,
            max_depth: None,
        };
        let found = collect_uris(|page| store.find(&pattern, page)).await;
        assert_eq!(found.len(), 5);
        assert!(found.windows(2).all(|pair| pair[0] < pair[1]));

        let first = store.ls(&scope, PageRequest::new(2)).await.unwrap();
        assert_eq!(first.len(), 2);
        let second = store
            .ls(
                &scope,
                PageRequest::new(2).after(first.next_cursor.unwrap()),
            )
            .await
            .unwrap();
        let third = store
            .ls(
                &scope,
                PageRequest::new(2).after(second.next_cursor.clone().unwrap()),
            )
            .await
            .unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(third.len(), 1);
        assert!(third.next_cursor.is_none());

        let prefix = scope.to_string();
        let scan = store
            .scan_by_prefix(&prefix, PageRequest::new(2))
            .await
            .unwrap();
        assert_eq!(scan.len(), 2);
        assert!(scan.next_cursor.is_some());
        let typed = store
            .scan_by_type(&prefix, ContentType::Fact, PageRequest::new(2))
            .await
            .unwrap();
        assert_eq!(typed.len(), 2);
        let typed_end = store
            .scan_by_type(
                &prefix,
                ContentType::Fact,
                PageRequest::new(2).after(typed.next_cursor.unwrap()),
            )
            .await
            .unwrap();
        assert_eq!(typed_end.len(), 1);
        assert!(typed_end.next_cursor.is_none());

        let tenants = store.list_tenants(PageRequest::new(1)).await.unwrap();
        assert_eq!(tenants.len(), 1);
        let tenant_end = store
            .list_tenants(PageRequest::new(1).after(tenants.next_cursor.unwrap()))
            .await
            .unwrap();
        assert_eq!(tenant_end.len(), 1);
        assert!(tenant_end.next_cursor.is_none());

        let version_uri = ContextUri::parse(&found[0]).unwrap();
        for text in ["v2", "v3"] {
            ContentRepo::write(&store, entry(version_uri.as_str(), text))
                .await
                .unwrap();
        }
        let versions = store
            .version_history(&version_uri, PageRequest::new(2))
            .await
            .unwrap();
        assert_eq!(versions.len(), 2);
        let version_end = store
            .version_history(
                &version_uri,
                PageRequest::new(2).after(versions.next_cursor.unwrap()),
            )
            .await
            .unwrap();
        assert_eq!(version_end.len(), 1);
        assert!(version_end.next_cursor.is_none());

        let tree = store
            .tree(&scope, 2, PageRequest::new(usize::MAX))
            .await
            .unwrap();
        assert_eq!(tree.len(), 1);
        assert!(tree.next_cursor.is_none());
        let tree_end = store
            .tree(&scope, 2, PageRequest::new(1).after(scope.to_string()))
            .await
            .unwrap();
        assert!(tree_end.is_empty());
        assert!(tree_end.next_cursor.is_none());

        let hard_limited = keyset_page(
            (0..1_001).collect::<Vec<_>>(),
            &PageRequest {
                after: None,
                limit: usize::MAX,
            },
            |value| format!("{value:04}"),
        );
        assert_eq!(hard_limited.len(), agent_context_db_core::MAX_PAGE_SIZE);
        assert!(hard_limited.next_cursor.is_some());
    }

    #[tokio::test]
    async fn context_store_supertrait_is_satisfied() {
        // 编译期验证：MemoryContextStore 自动实现聚合 ContextStore。
        fn assert_store<T: ContextStore>() {}
        assert_store::<MemoryContextStore>();
    }
}
