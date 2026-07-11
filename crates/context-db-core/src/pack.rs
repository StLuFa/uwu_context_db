//! ContextPack 导出导入（F7）+ 路径级 ACL（F8）。
use crate::{Page, PageRequest};

use crate::{
    BrowsingOps, ContentLevel, ContentPayload, ContentRepo, ContentStore, ContentType,
    ContextEntry, ContextError, ContextUri, DirEntry, FindPattern, FsOps, GrepHit, MvccVersion,
    Result, TreeNode,
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// F7 ContextPack — 子树导出/导入/打包分享
// ═══════════════════════════════════════════════════════════════════════════

/// ContextPack 元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeta {
    pub name: String,
    pub description: Option<String>,
    pub exported_at: chrono::DateTime<chrono::Utc>,
    pub source_agent: Option<String>,
    pub entry_count: usize,
}

/// ContextPack — 可导出的上下文子树（K.6: entries 去冗余，用 Vec 替代 HashMap）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub meta: PackMeta,
    /// 根 scope
    pub scope: ContextUri,
    /// 条目列表（URI 在 entry 内部，不重复存储）
    pub entries: Vec<ContextEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<PackSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackSignature {
    pub algorithm: String,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackTrustPolicy {
    RequireValidSignature,
    RequireTrustedPublicKey(Vec<String>),
}

impl PackTrustPolicy {
    fn validate(&self, pack: &ContextPack) -> Result<()> {
        match self {
            Self::RequireValidSignature => {
                if pack.verify_signature() {
                    Ok(())
                } else {
                    Err(ContextError::TrustPolicy(
                        "context pack requires a valid ed25519 signature".into(),
                    ))
                }
            }
            Self::RequireTrustedPublicKey(keys) => {
                Self::RequireValidSignature.validate(pack)?;
                let Some(signature) = &pack.signature else {
                    return Err(ContextError::TrustPolicy(
                        "context pack signature is missing".into(),
                    ));
                };
                if keys.iter().any(|key| key == &signature.public_key) {
                    Ok(())
                } else {
                    Err(ContextError::TrustPolicy(
                        "context pack signer is not trusted".into(),
                    ))
                }
            }
        }
    }
}

impl ContextPack {
    pub fn new(scope: ContextUri, name: impl Into<String>) -> Self {
        Self {
            meta: PackMeta {
                name: name.into(),
                description: None,
                exported_at: chrono::Utc::now(),
                source_agent: None,
                entry_count: 0,
            },
            scope,
            entries: Vec::new(),
            signature: None,
        }
    }

    pub fn with_source(mut self, agent: impl Into<String>) -> Self {
        self.meta.source_agent = Some(agent.into());
        self
    }
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.meta.description = Some(desc.into());
        self
    }

    pub fn add_entry(&mut self, entry: ContextEntry) {
        self.entries.push(entry);
        self.meta.entry_count = self.entries.len();
    }

    pub fn to_signed_json(&self, signing_key: &SigningKey) -> Result<String> {
        let mut signed = self.clone();
        signed.sign(signing_key)?;
        Ok(serde_json::to_string_pretty(&signed)?)
    }

    pub fn from_trusted_json(json: &str) -> Result<Self> {
        Self::from_json_with_policy(json, PackTrustPolicy::RequireValidSignature)
    }

    pub fn from_json_with_policy(json: &str, policy: PackTrustPolicy) -> Result<Self> {
        let pack: Self = serde_json::from_str(json)?;
        policy.validate(&pack)?;
        Ok(pack)
    }

    pub fn signing_payload(&self) -> std::result::Result<Vec<u8>, serde_json::Error> {
        let mut unsigned = self.clone();
        unsigned.signature = None;
        serde_json::to_vec(&unsigned)
    }

    pub fn sign(&mut self, signing_key: &SigningKey) -> std::result::Result<(), serde_json::Error> {
        let payload = self.signing_payload()?;
        let signature = signing_key.sign(&payload);
        self.signature = Some(PackSignature {
            algorithm: "ed25519".into(),
            public_key: BASE64.encode(signing_key.verifying_key().to_bytes()),
            signature: BASE64.encode(signature.to_bytes()),
        });
        Ok(())
    }

    pub fn verify_signature(&self) -> bool {
        let Some(signature) = &self.signature else {
            return false;
        };
        if signature.algorithm != "ed25519" {
            return false;
        }
        let Ok(public_key_bytes) = BASE64.decode(&signature.public_key) else {
            return false;
        };
        let Ok(signature_bytes) = BASE64.decode(&signature.signature) else {
            return false;
        };
        let Ok(public_key_array) = <[u8; 32]>::try_from(public_key_bytes.as_slice()) else {
            return false;
        };
        let Ok(signature_array) = <[u8; 64]>::try_from(signature_bytes.as_slice()) else {
            return false;
        };
        let Ok(verifying_key) = VerifyingKey::from_bytes(&public_key_array) else {
            return false;
        };
        let sig = Signature::from_bytes(&signature_array);
        let Ok(payload) = self.signing_payload() else {
            return false;
        };
        verifying_key.verify(&payload, &sig).is_ok()
    }

    pub fn filter_by_scope(&self, prefix: &ContextUri) -> Vec<&ContextEntry> {
        self.entries
            .iter()
            .filter(|e| e.uri.to_string().starts_with(&prefix.to_string()))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F8 路径级 ACL
// ═══════════════════════════════════════════════════════════════════════════

/// 权限位。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    pub delete: bool,
    pub share: bool,
}

impl Permissions {
    pub const fn full() -> Self {
        Self {
            read: true,
            write: true,
            delete: true,
            share: true,
        }
    }
    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            delete: false,
            share: false,
        }
    }
    pub const fn none() -> Self {
        Self {
            read: false,
            write: false,
            delete: false,
            share: false,
        }
    }
}

/// 访问主体。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Principal {
    User(String),
    Agent(String),
    Role(String),
    Anonymous,
}

/// ACL 规则。
#[derive(Debug, Clone)]
pub struct AclRule {
    /// URI 模式（前缀匹配）
    pub path_pattern: String,
    /// 主体
    pub principal: Principal,
    /// 权限
    pub permissions: Permissions,
    /// 优先级（越大越优先）
    pub priority: u32,
}

/// 路径级 ACL 引擎。
pub struct PathAcl {
    rules: parking_lot::Mutex<Vec<AclRule>>,
}

impl PathAcl {
    pub fn new() -> Self {
        Self {
            rules: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// 添加规则。
    pub fn add_rule(&self, rule: AclRule) {
        let mut rules = self.rules.lock();
        rules.push(rule);
        rules.sort_by_key(|r| -(r.priority as i64));
    }

    /// 检查主体对 URI 是否有指定权限。
    pub fn check(&self, uri: &ContextUri, principal: &Principal, required: Permissions) -> bool {
        let rules = self.rules.lock();
        let uri_str = uri.to_string();

        for rule in rules.iter() {
            if &rule.principal != principal {
                continue;
            }
            if !uri_str.starts_with(&rule.path_pattern) {
                continue;
            }
            let p = rule.permissions;
            if (!required.read || p.read)
                && (!required.write || p.write)
                && (!required.delete || p.delete)
                && (!required.share || p.share)
            {
                return true;
            }
            return false;
        }
        false
    }
}

impl Default for PathAcl {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AclProtectedStore<R> {
    inner: R,
    acl: std::sync::Arc<PathAcl>,
    principal: Principal,
}

impl<R> AclProtectedStore<R> {
    pub fn new(inner: R, acl: std::sync::Arc<PathAcl>, principal: Principal) -> Self {
        Self {
            inner,
            acl,
            principal,
        }
    }

    fn require(&self, uri: &ContextUri, permissions: Permissions) -> Result<()> {
        if self.acl.check(uri, &self.principal, permissions) {
            Ok(())
        } else {
            Err(crate::ContextError::PermissionDenied(format!(
                "principal {:?} lacks permission for {uri}",
                self.principal
            )))
        }
    }

    fn can_read(&self, uri: &ContextUri) -> bool {
        self.acl
            .check(uri, &self.principal, Permissions::read_only())
    }

    fn require_read(&self, uri: &ContextUri) -> Result<()> {
        if self.can_read(uri) {
            Ok(())
        } else {
            Err(crate::ContextError::PermissionDenied(format!(
                "principal {:?} lacks read permission for {uri}",
                self.principal
            )))
        }
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

#[async_trait]
impl<R> ContentRepo for AclProtectedStore<R>
where
    R: ContentRepo + Send + Sync,
{
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        self.require(
            &entry.uri,
            Permissions {
                read: false,
                write: true,
                delete: false,
                share: false,
            },
        )?;
        self.inner.write(entry).await
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.require(
            uri,
            Permissions {
                read: false,
                write: false,
                delete: true,
                share: false,
            },
        )?;
        self.inner.delete(uri).await
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        self.require(
            from,
            Permissions {
                read: false,
                write: false,
                delete: true,
                share: false,
            },
        )?;
        self.require(
            to,
            Permissions {
                read: false,
                write: true,
                delete: false,
                share: false,
            },
        )?;
        self.inner.rename(from, to).await
    }
}

#[async_trait]
impl<R> FsOps for AclProtectedStore<R>
where
    R: FsOps + Send + Sync,
{
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        self.require_read(dir)?;
        let entries = self.inner.ls(dir, page).await?;
        Ok(entries
            .into_iter()
            .filter(|entry| self.can_read(&entry.uri))
            .collect())
    }

    async fn find(&self, pattern: &FindPattern, page: PageRequest) -> Result<Page<ContextUri>> {
        if let Some(scope) = &pattern.scope {
            self.require_read(scope)?;
        }
        let uris = self.inner.find(pattern, page).await?;
        Ok(uris.into_iter().filter(|uri| self.can_read(uri)).collect())
    }

    async fn grep(&self, regex: &str, scope: &ContextUri) -> Result<Vec<GrepHit>> {
        self.require_read(scope)?;
        let hits = self.inner.grep(regex, scope).await?;
        Ok(hits
            .into_iter()
            .filter(|hit| self.can_read(&hit.uri))
            .collect())
    }

    async fn tree(
        &self,
        root: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        self.require_read(root)?;
        let mut tree = self.inner.tree(root, depth, page).await?;
        for node in &mut tree.items {
            filter_tree(node, |uri| self.can_read(uri));
        }
        Ok(tree)
    }

    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        self.require_read(uri)?;
        self.inner.read(uri, level).await
    }
}

#[async_trait]
impl<R> ContentStore for AclProtectedStore<R>
where
    R: ContentStore + Send + Sync,
{
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
        self.require_read(uri)?;
        self.inner.read(uri, level).await
    }

    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion> {
        self.require(
            &entry.uri,
            Permissions {
                read: false,
                write: true,
                delete: false,
                share: false,
            },
        )?;
        self.inner.write(entry).await
    }

    async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.require(
            uri,
            Permissions {
                read: false,
                write: false,
                delete: true,
                share: false,
            },
        )?;
        self.inner.delete(uri).await
    }

    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
        self.require(
            from,
            Permissions {
                read: false,
                write: false,
                delete: true,
                share: false,
            },
        )?;
        self.require(
            to,
            Permissions {
                read: false,
                write: true,
                delete: false,
                share: false,
            },
        )?;
        self.inner.rename(from, to).await
    }

    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        for entry in entries {
            self.require(
                &entry.uri,
                Permissions {
                    read: false,
                    write: true,
                    delete: false,
                    share: false,
                },
            )?;
        }
        self.inner.batch_write(entries).await
    }

    async fn scan_by_prefix(&self, prefix: &str, page: PageRequest) -> Result<Page<ContextEntry>> {
        let scope = ContextUri::parse(prefix.trim_end_matches('/')).map_err(|err| {
            ContextError::InvalidUri(format!("ACL scan_by_prefix scope parse failed: {err}"))
        })?;
        self.require_read(&scope)?;
        Ok(self
            .inner
            .scan_by_prefix(prefix, page)
            .await?
            .into_iter()
            .filter(|entry| self.require_read(&entry.uri).is_ok())
            .collect())
    }

    async fn scan_by_type(
        &self,
        prefix: &str,
        content_type: ContentType,
        page: PageRequest,
    ) -> Result<Page<ContextEntry>> {
        let scope = ContextUri::parse(prefix.trim_end_matches('/')).map_err(|err| {
            ContextError::InvalidUri(format!("ACL scan_by_type scope parse failed: {err}"))
        })?;
        self.require_read(&scope)?;
        Ok(self
            .inner
            .scan_by_type(prefix, content_type, page)
            .await?
            .into_iter()
            .filter(|entry| self.require_read(&entry.uri).is_ok())
            .collect())
    }
}

#[async_trait]
impl<R> BrowsingOps for AclProtectedStore<R>
where
    R: BrowsingOps + Send + Sync,
{
    async fn ls(&self, dir: &ContextUri, page: PageRequest) -> Result<Page<DirEntry>> {
        self.require_read(dir)?;
        let entries = self.inner.ls(dir, page).await?;
        Ok(entries
            .into_iter()
            .filter(|entry| self.can_read(&entry.uri))
            .collect())
    }

    async fn tree(
        &self,
        dir: &ContextUri,
        depth: usize,
        page: PageRequest,
    ) -> Result<Page<TreeNode>> {
        self.require_read(dir)?;
        let mut tree = self.inner.tree(dir, depth, page).await?;
        for node in &mut tree.items {
            filter_tree(node, |uri| self.can_read(uri));
        }
        Ok(tree)
    }

    async fn find(
        &self,
        scope: &ContextUri,
        pattern: &str,
        page: PageRequest,
    ) -> Result<Page<ContextUri>> {
        self.require_read(scope)?;
        let uris = self.inner.find(scope, pattern, page).await?;
        Ok(uris.into_iter().filter(|uri| self.can_read(uri)).collect())
    }

    async fn grep(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<GrepHit>> {
        self.require_read(scope)?;
        let hits = self.inner.grep(scope, pattern).await?;
        Ok(hits
            .into_iter()
            .filter(|hit| self.can_read(&hit.uri))
            .collect())
    }
}

fn filter_tree<F>(node: &mut TreeNode, can_read: F)
where
    F: Fn(&ContextUri) -> bool + Copy,
{
    node.children.retain(|child| can_read(&child.uri));
    for child in &mut node.children {
        filter_tree(child, can_read);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_pack_roundtrip() {
        let mut pack =
            ContextPack::new(ContextUri::parse("uwu://t1/agent/a1").unwrap(), "test-pack");
        let entry = ContextEntry::new_text(
            ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap(),
            crate::TenantId(uuid::Uuid::nil()),
            "test case",
        );
        pack.add_entry(entry);

        let signing_key = SigningKey::from_bytes(&[8_u8; 32]);
        let json = pack.to_signed_json(&signing_key).unwrap();
        let restored = ContextPack::from_trusted_json(&json).unwrap();
        assert_eq!(restored.meta.entry_count, 1);
    }

    #[test]
    fn context_pack_import_requires_valid_signature_by_default() {
        let mut pack =
            ContextPack::new(ContextUri::parse("uwu://t1/agent/a1").unwrap(), "test-pack");
        pack.add_entry(ContextEntry::new_text(
            ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap(),
            crate::TenantId(uuid::Uuid::nil()),
            "test case",
        ));
        let unsigned = serde_json::to_string_pretty(&pack).unwrap();
        assert!(matches!(
            ContextPack::from_trusted_json(&unsigned).unwrap_err(),
            crate::ContextError::TrustPolicy(_)
        ));

        let signing_key = SigningKey::from_bytes(&[9_u8; 32]);
        let signed = pack.to_signed_json(&signing_key).unwrap();
        assert!(ContextPack::from_trusted_json(&signed).is_ok());
    }

    #[test]
    fn context_pack_signs_and_rejects_tampering() {
        let mut pack =
            ContextPack::new(ContextUri::parse("uwu://t1/agent/a1").unwrap(), "test-pack");
        pack.add_entry(ContextEntry::new_text(
            ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap(),
            crate::TenantId(uuid::Uuid::nil()),
            "test case",
        ));
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);

        pack.sign(&signing_key).unwrap();
        assert!(pack.verify_signature());

        pack.add_entry(ContextEntry::new_text(
            ContextUri::parse("uwu://t1/agent/a1/memories/cases/c2").unwrap(),
            crate::TenantId(uuid::Uuid::nil()),
            "tampered",
        ));
        assert!(!pack.verify_signature());
    }

    #[test]
    fn path_acl_enforces_permissions() {
        let acl = PathAcl::new();
        acl.add_rule(AclRule {
            path_pattern: "uwu://t1/agent/a1".into(),
            principal: Principal::User("u1".into()),
            permissions: Permissions::read_only(),
            priority: 10,
        });

        let uri = ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap();
        assert!(acl.check(
            &uri,
            &Principal::User("u1".into()),
            Permissions::read_only()
        ));
        assert!(!acl.check(&uri, &Principal::User("u1".into()), Permissions::full()));
        assert!(!acl.check(&uri, &Principal::Anonymous, Permissions::read_only()));
    }

    #[tokio::test]
    async fn acl_protected_store_blocks_content_store_reads_and_writes() {
        struct NullStore;

        #[async_trait]
        impl ContentRepo for NullStore {
            async fn write(&self, _entry: ContextEntry) -> Result<MvccVersion> {
                Ok(MvccVersion(1))
            }
            async fn delete(&self, _uri: &ContextUri) -> Result<()> {
                Ok(())
            }
            async fn rename(&self, _from: &ContextUri, _to: &ContextUri) -> Result<()> {
                Ok(())
            }
        }

        #[async_trait]
        impl ContentStore for NullStore {
            async fn read(
                &self,
                _uri: &ContextUri,
                _level: ContentLevel,
            ) -> Result<ContentPayload> {
                Ok(ContentPayload::Text {
                    sparse: "ok".into(),
                    dense: "ok".into(),
                    full: "ok".into(),
                })
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
                Ok(vec![MvccVersion(1); entries.len()])
            }
            async fn scan_by_prefix(
                &self,
                _prefix: &str,
                _page: PageRequest,
            ) -> Result<Page<ContextEntry>> {
                Ok(Page::new(Vec::new(), None))
            }
            async fn scan_by_type(
                &self,
                _prefix: &str,
                _content_type: ContentType,
                _page: PageRequest,
            ) -> Result<Page<ContextEntry>> {
                Ok(Page::new(Vec::new(), None))
            }
        }

        let acl = std::sync::Arc::new(PathAcl::new());
        acl.add_rule(AclRule {
            path_pattern: "uwu://t1/agent/a1".into(),
            principal: Principal::User("u1".into()),
            permissions: Permissions::read_only(),
            priority: 10,
        });
        let store = AclProtectedStore::new(NullStore, acl, Principal::User("u1".into()));
        let uri = ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap();
        assert!(
            ContentStore::read(&store, &uri, ContentLevel::L0)
                .await
                .is_ok()
        );

        let entry = ContextEntry::new_text(uri, crate::TenantId(uuid::Uuid::nil()), "test case");
        let err = ContentStore::write(&store, entry).await.unwrap_err();
        assert!(matches!(err, crate::ContextError::PermissionDenied(_)));
    }
}
