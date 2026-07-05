//! ContextPack 导出导入（F7）+ 路径级 ACL（F8）。

use crate::{ContextEntry, ContextUri};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        }
    }

    pub fn with_source(mut self, agent: impl Into<String>) -> Self { self.meta.source_agent = Some(agent.into()); self }
    pub fn with_description(mut self, desc: impl Into<String>) -> Self { self.meta.description = Some(desc.into()); self }

    pub fn add_entry(&mut self, entry: ContextEntry) {
        self.entries.push(entry);
        self.meta.entry_count = self.entries.len();
    }

    pub fn to_json(&self) -> String { serde_json::to_string_pretty(self).unwrap_or_default() }
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> { serde_json::from_str(json) }

    pub fn filter_by_scope(&self, prefix: &ContextUri) -> Vec<&ContextEntry> {
        self.entries.iter().filter(|e| e.uri.to_string().starts_with(&prefix.to_string())).collect()
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
        Self { read: true, write: true, delete: true, share: true }
    }
    pub const fn read_only() -> Self {
        Self { read: true, write: false, delete: false, share: false }
    }
    pub const fn none() -> Self {
        Self { read: false, write: false, delete: false, share: false }
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
        Self { rules: parking_lot::Mutex::new(Vec::new()) }
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
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_pack_roundtrip() {
        let mut pack = ContextPack::new(
            ContextUri::parse("uwu://t1/agent/a1").unwrap(),
            "test-pack",
        );
        let entry = ContextEntry::new_text(
            ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap(),
            crate::TenantId(uuid::Uuid::nil()),
            "test case",
        );
        pack.add_entry(entry);

        let json = pack.to_json();
        let restored = ContextPack::from_json(&json).unwrap();
        assert_eq!(restored.meta.entry_count, 1);
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
        assert!(acl.check(&uri, &Principal::User("u1".into()), Permissions::read_only()));
        assert!(!acl.check(&uri, &Principal::User("u1".into()), Permissions::full()));
        assert!(!acl.check(&uri, &Principal::Anonymous, Permissions::read_only()));
    }
}
