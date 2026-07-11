//! Write-path semantic deduplication for content stores.
//!
//! The decorator is intentionally store-agnostic: it wraps any `ContentStore`,
//! scans the target URI scope, and suppresses near-duplicate writes before they
//! can trigger downstream embedding/consolidation work.
use crate::{Page, PageRequest};

use async_trait::async_trait;

use crate::{
    BrowsingOps, ContentLevel, ContentPayload, ContentRepo, ContentStore, ContentType, ContextDiff,
    ContextEntry, ContextError, ContextUri, DirEntry, FindPattern, FsOps, GrepHit, MvccVersion,
    Result, TenantId, TenantOps, TreeNode, VersionEntry, VersionOps,
};

#[derive(Debug, Clone)]
pub struct SemanticWriteDedupConfig {
    pub scan_limit: usize,
    pub jaccard_threshold: f32,
    pub min_tokens: usize,
}

impl Default for SemanticWriteDedupConfig {
    fn default() -> Self {
        Self {
            scan_limit: 256,
            jaccard_threshold: 0.86,
            min_tokens: 5,
        }
    }
}

impl SemanticWriteDedupConfig {
    pub fn validate(&self) -> Result<()> {
        if self.scan_limit == 0 {
            return Err(ContextError::Unsupported(
                "semantic dedup scan_limit must be greater than zero".into(),
            ));
        }
        if !self.jaccard_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.jaccard_threshold)
            || self.jaccard_threshold == 0.0
        {
            return Err(ContextError::Unsupported(
                "semantic dedup jaccard_threshold must be finite and in (0, 1]".into(),
            ));
        }
        if self.min_tokens == 0 {
            return Err(ContextError::Unsupported(
                "semantic dedup min_tokens must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WriteDedupDecision {
    pub duplicate_of: ContextUri,
    pub similarity: f32,
    pub existing_version: MvccVersion,
}

/// Decorates a `ContentStore` with near-duplicate suppression on write.
pub struct SemanticWriteDedupStore<R> {
    inner: R,
    config: SemanticWriteDedupConfig,
}

impl<R> SemanticWriteDedupStore<R> {
    pub fn new(inner: R) -> Result<Self> {
        Self::with_config(inner, SemanticWriteDedupConfig::default())
    }

    pub fn with_config(inner: R, config: SemanticWriteDedupConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { inner, config })
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

#[async_trait]
impl<R> FsOps for SemanticWriteDedupStore<R>
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

    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        self.inner.read(uri, level).await
    }
}

#[async_trait]
impl<R> BrowsingOps for SemanticWriteDedupStore<R>
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
impl<R> VersionOps for SemanticWriteDedupStore<R>
where
    R: VersionOps + Send + Sync,
{
    async fn version_history(
        &self,
        uri: &ContextUri,
        page: PageRequest,
    ) -> Result<Page<VersionEntry>> {
        self.inner.version_history(uri, page).await
    }

    async fn rollback(&self, uri: &ContextUri, to: MvccVersion) -> Result<()> {
        self.inner.rollback(uri, to).await
    }

    async fn diff(&self, uri: &ContextUri, a: MvccVersion, b: MvccVersion) -> Result<ContextDiff> {
        self.inner.diff(uri, a, b).await
    }
}

#[async_trait]
impl<R> TenantOps for SemanticWriteDedupStore<R>
where
    R: TenantOps + Send + Sync,
{
    async fn list_tenants(&self, page: PageRequest) -> Result<Page<TenantId>> {
        self.inner.list_tenants(page).await
    }
}

#[async_trait]
impl<R> ContentStore for SemanticWriteDedupStore<R>
where
    R: ContentStore + Send + Sync,
{
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        self.inner.read(uri, level).await
    }

    async fn write(&self, mut entry: ContextEntry) -> Result<MvccVersion> {
        if let Some(decision) = find_duplicate(&self.inner, &entry, &self.config).await? {
            set_dedup_metadata(&mut entry, &decision)?;
            return Ok(decision.existing_version);
        }
        self.inner.write(entry).await
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.inner.delete(uri).await
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        self.inner.rename(from, to).await
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        let mut versions = Vec::with_capacity(entries.len());
        for entry in entries {
            versions.push(self.write(entry.clone()).await?);
        }
        Ok(versions)
    }

    async fn scan_by_prefix(&self, prefix: &str, page: PageRequest) -> Result<Page<ContextEntry>> {
        self.inner.scan_by_prefix(prefix, page).await
    }

    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        self.inner.scan_by_type(prefix, content_type, page).await
    }
}

#[async_trait]
impl<R> ContentRepo for SemanticWriteDedupStore<R>
where
    R: ContentRepo + ContentStore + Send + Sync,
{
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        ContentStore::write(self, entry).await
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        ContentStore::delete(self, uri).await
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        ContentStore::rename(self, from, to).await
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        ContentStore::batch_write(self, entries).await
    }
}

async fn find_duplicate(
    store: &dyn ContentStore,
    entry: &ContextEntry,
    config: &SemanticWriteDedupConfig,
) -> Result<Option<WriteDedupDecision>> {
    if config.scan_limit == 0 || config.jaccard_threshold <= 0.0 {
        return Ok(None);
    }

    let candidate_text = semantic_text(&entry.payload);
    let candidate_tokens = tokenize_for_dedup(&candidate_text);
    if candidate_tokens.len() < config.min_tokens {
        return Ok(None);
    }

    let prefix = dedup_scope_prefix(&entry.uri);
    let existing_entries = store
        .scan_by_prefix(&prefix, PageRequest::new(config.scan_limit))
        .await?;

    let mut best: Option<WriteDedupDecision> = None;
    for existing in existing_entries {
        if existing.uri == entry.uri
            || existing.metadata.content_type != entry.metadata.content_type
        {
            continue;
        }

        let existing_tokens = tokenize_for_dedup(&semantic_text(&existing.payload));
        if existing_tokens.len() < config.min_tokens {
            continue;
        }

        let similarity = jaccard_sorted(&candidate_tokens, &existing_tokens);
        if similarity >= config.jaccard_threshold
            && best
                .as_ref()
                .map(|current| similarity > current.similarity)
                .unwrap_or(true)
        {
            best = Some(WriteDedupDecision {
                duplicate_of: existing.uri,
                similarity,
                existing_version: existing.mvcc_version,
            });
        }
    }

    Ok(best)
}

fn set_dedup_metadata(entry: &mut ContextEntry, decision: &WriteDedupDecision) -> Result<()> {
    entry
        .metadata
        .set_custom_field("dedup_skipped", &true)
        .map_err(ContextError::Serialization)?;
    entry
        .metadata
        .set_custom_field("duplicate_of", &decision.duplicate_of.to_string())
        .map_err(ContextError::Serialization)?;
    entry
        .metadata
        .set_custom_field("dedup_similarity", &decision.similarity)
        .map_err(ContextError::Serialization)?;
    Ok(())
}

fn dedup_scope_prefix(uri: &ContextUri) -> String {
    uri.parent()
        .map(|parent| parent.to_string())
        .unwrap_or_else(|| uri.to_string())
}

fn semantic_text(payload: &ContentPayload) -> String {
    match payload {
        ContentPayload::Text {
            sparse,
            dense,
            full,
        } => {
            let strongest = if !full.trim().is_empty() {
                full
            } else if !dense.trim().is_empty() {
                dense
            } else {
                sparse
            };
            strongest.to_string()
        }
        ContentPayload::Image { .. } => "[image]".into(),
        ContentPayload::Audio { transcript, .. } => transcript.clone(),
        ContentPayload::Structured { summary, data, .. } => format!("{summary}\n{data}"),
        ContentPayload::Composite { summary, .. } => summary.clone(),
    }
}

fn tokenize_for_dedup(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn jaccard_sorted(left: &[String], right: &[String]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let mut i = 0;
    let mut j = 0;
    let mut intersection = 0usize;
    let mut union = 0usize;

    while i < left.len() && j < right.len() {
        union += 1;
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    union += left.len().saturating_sub(i) + right.len().saturating_sub(j);

    intersection as f32 / union.max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentType, ContextMeta, MediaType, TenantId};
    use std::sync::Mutex;
    use uuid::Uuid;

    #[derive(Default)]
    struct MemoryContentStore {
        entries: Mutex<Vec<ContextEntry>>,
    }

    #[async_trait]
    impl ContentStore for MemoryContentStore {
        async fn read(&self, uri: &ContextUri, _level: ContentLevel) -> Result<ContentPayload> {
            self.entries
                .lock()
                .unwrap()
                .iter()
                .find(|entry| &entry.uri == uri)
                .map(|entry| entry.payload.clone())
                .ok_or_else(|| ContextError::NotFound(uri.to_string()))
        }

        async fn write(&self, mut entry: ContextEntry) -> Result<MvccVersion> {
            let mut entries = self.entries.lock().unwrap();
            let version = MvccVersion(entries.len() as u64 + 1);
            entry.mvcc_version = version;
            entries.push(entry);
            Ok(version)
        }

        async fn delete(&self, _uri: &ContextUri) -> Result<()> {
            Ok(())
        }

        async fn rename(&self, _from: &ContextUri, _to: &ContextUri) -> Result<()> {
            Ok(())
        }

        async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
            let mut versions = Vec::with_capacity(entries.len());
            for entry in entries {
                versions.push(self.write(entry.clone()).await?);
            }
            Ok(versions)
        }

        async fn scan_by_prefix(
            &self,
            prefix: &str,
            page: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            let entries = self
                .entries
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.uri.to_string().starts_with(prefix))
                .take(page.effective_limit())
                .cloned()
                .collect();
            Ok(Page::new(entries, None))
        }

        async fn scan_by_type(
            &self,
            prefix: &str,
            content_type: ContentType,
            page: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            let entries = self
                .entries
                .lock()
                .unwrap()
                .iter()
                .filter(|entry| entry.uri.to_string().starts_with(prefix))
                .filter(|entry| entry.metadata.content_type == Some(content_type))
                .take(page.effective_limit())
                .cloned()
                .collect();
            Ok(Page::new(entries, None))
        }
    }

    fn entry(uri: &str, text: &str) -> ContextEntry {
        let metadata = ContextMeta {
            content_type: Some(ContentType::Fact),
            ..Default::default()
        };
        ContextEntry {
            uri: ContextUri::parse(uri).unwrap(),
            tenant: TenantId(Uuid::nil()),
            payload: ContentPayload::Text {
                sparse: text.into(),
                dense: text.into(),
                full: text.into(),
            },
            media_type: MediaType::Text,
            metadata,
            mvcc_version: MvccVersion(0),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            derivation: None,
        }
    }

    #[tokio::test]
    async fn suppresses_near_duplicate_write_in_same_scope() {
        let store = SemanticWriteDedupStore::with_config(
            MemoryContentStore::default(),
            SemanticWriteDedupConfig {
                jaccard_threshold: 0.8,
                min_tokens: 3,
                ..Default::default()
            },
        )
        .unwrap();

        let first = store
            .write(entry(
                "uwu://tenant/agent/a/fact/topic/one",
                "rust async traits use Send Sync futures for safe storage adapters",
            ))
            .await
            .unwrap();
        let second = store
            .write(entry(
                "uwu://tenant/agent/a/fact/topic/two",
                "rust async traits use Send Sync futures for safe storage adapters",
            ))
            .await
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(store.inner.entries.lock().unwrap().len(), 1);
    }

    #[test]
    fn jaccard_handles_sorted_unique_tokens() {
        let left = tokenize_for_dedup("alpha beta beta gamma");
        let right = tokenize_for_dedup("alpha beta delta");
        assert!((jaccard_sorted(&left, &right) - 0.5).abs() < f32::EPSILON);
    }
}
