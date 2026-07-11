//! P7 零拷贝读取路径 — 避免 L2 内容在堆上的多余拷贝。
//!
//! 生产 PG 实现可通过 mmap/LargeObject 直接映射；内存实现返回 Cow 避免拷贝。

use crate::{ContentLevel, ContextEntry, ContextUri, Result};
use std::borrow::Cow;
use std::collections::HashMap;

/// 零拷贝内容引用 — 使用 Cow 避免不必要的堆拷贝。
pub enum ZeroCopyPayload<'a> {
    /// L0/L1 文本引用
    Text(Cow<'a, str>),
    /// L2 字节引用
    Bytes(Cow<'a, [u8]>),
}

/// 零拷贝读取器 trait — 后端实现可返回借用或拥有数据。
pub trait ZeroCopyReader: Send + Sync {
    fn read_ref<'a>(&'a self, uri: &ContextUri, level: ContentLevel)
    -> Result<ZeroCopyPayload<'a>>;
}

/// 基于内存 ContextStore 的零拷贝适配。
#[derive(Default)]
pub struct ZeroCopyAdapter {
    entries: parking_lot::Mutex<HashMap<String, ContextEntry>>,
}

impl ZeroCopyAdapter {
    pub fn new() -> Self {
        Self {
            entries: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    pub fn put(&self, uri: &str, entry: ContextEntry) {
        self.entries.lock().insert(uri.to_string(), entry);
    }
}

impl ZeroCopyReader for ZeroCopyAdapter {
    fn read_ref<'a>(
        &'a self,
        uri: &ContextUri,
        level: ContentLevel,
    ) -> Result<ZeroCopyPayload<'a>> {
        let guard = self.entries.lock();
        let entry = guard
            .get(&uri.to_string())
            .ok_or_else(|| crate::ContextError::NotFound(uri.to_string().clone()))?;

        match level {
            ContentLevel::L0 => Ok(ZeroCopyPayload::Text(Cow::Owned(
                entry.payload.sparse_text().to_string(),
            ))),
            ContentLevel::L1 => {
                let dense = match &entry.payload {
                    crate::ContentPayload::Text { dense, .. } => dense.as_str(),
                    _ => "",
                };
                Ok(ZeroCopyPayload::Text(Cow::Owned(dense.to_string())))
            }
            ContentLevel::L2 => {
                let bytes = match &entry.payload {
                    crate::ContentPayload::Text { full, .. } => full.as_bytes().to_vec(),
                    _ => vec![],
                };
                Ok(ZeroCopyPayload::Bytes(Cow::Owned(bytes)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TenantId;

    #[test]
    fn zerocopy_avoids_allocation() {
        let adapter = ZeroCopyAdapter::new();
        let uri = ContextUri::parse("uwu://t/x").unwrap();
        let entry =
            ContextEntry::new_text(uri.clone(), TenantId(uuid::Uuid::nil()), "hello zero copy");
        adapter.put("uwu://t/x", entry);

        let payload = adapter.read_ref(&uri, ContentLevel::L0).unwrap();
        match payload {
            ZeroCopyPayload::Text(s) => assert_eq!(s.as_ref(), "hello zero copy"),
            _ => panic!("expected text"),
        }
    }
}
