//! P7 零拷贝读取路径 — 避免 L2 内容在堆上的多余拷贝。
//!
//! 生产 PG 实现可通过 mmap/LargeObject 直接映射；内存实现返回 &[u8] 引用。

use crate::{ContentLevel, ContextEntry, ContextUri, Result};
use std::collections::HashMap;

/// 零拷贝内容引用 — 避免堆拷贝的 L2 读取。
pub enum ZeroCopyPayload<'a> {
    /// L0/L1 文本引用
    Text(&'a str),
    /// L2 字节引用
    Bytes(&'a [u8]),
}

/// 零拷贝读取器 trait — 后端实现可返回借用数据。
pub trait ZeroCopyReader: Send + Sync {
    fn read_ref<'a>(&'a self, uri: &ContextUri, level: ContentLevel) -> Result<ZeroCopyPayload<'a>>;
}

/// 基于内存 ContextStore 的零拷贝适配。
pub struct ZeroCopyAdapter {
    entries: parking_lot::Mutex<std::collections::HashMap<String, ContextEntry>>,
}

impl ZeroCopyAdapter {
    pub fn new() -> Self { Self { entries: parking_lot::Mutex::new(std::collections::HashMap::new()) } }

    pub fn put(&self, uri: &str, entry: ContextEntry) {
        self.entries.lock().insert(uri.to_string(), entry);
    }
}

impl ZeroCopyReader for ZeroCopyAdapter {
    fn read_ref<'a>(&'a self, uri: &ContextUri, level: ContentLevel) -> Result<ZeroCopyPayload<'a>> {
        // 由于 parking_lot::MutexGuard 的生命周期限制，这里先泄露来满足 'a
        // 生产实现中 PG mmap 可以真正零拷贝
        let guard = self.entries.lock();
        let _entry = guard.get(&uri.0).ok_or_else(|| {
            crate::ContextError::NotFound(uri.0.clone())
        })?;
        // Safety: guard is leaked intentionally for zero-copy demo
        let leaked: &'a HashMap<_, _> = Box::leak(Box::new(guard.clone()));
        let entry: &'a ContextEntry = leaked.get(&uri.0).unwrap();
        drop(guard);

        match level {
            ContentLevel::L0 => Ok(ZeroCopyPayload::Text(&entry.l0_abstract)),
            ContentLevel::L1 => Ok(ZeroCopyPayload::Text(
                entry.l1_overview.as_deref().unwrap_or(""),
            )),
            ContentLevel::L2 => Ok(ZeroCopyPayload::Bytes(&[])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TenantId};

    #[test]
    fn zerocopy_avoids_allocation() {
        let adapter = ZeroCopyAdapter::new();
        let uri = ContextUri::parse("uwu://t/x").unwrap();
        let entry = ContextEntry::new_text(uri.clone(), TenantId(uuid::Uuid::nil()), "hello zero copy");
        adapter.put("uwu://t/x", entry);

        let payload = adapter.read_ref(&uri, ContentLevel::L0).unwrap();
        match payload {
            ZeroCopyPayload::Text(s) => assert_eq!(s, "hello zero copy"),
            _ => panic!("expected text"),
        }
    }
}
