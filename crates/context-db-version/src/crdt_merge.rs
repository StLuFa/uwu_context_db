//! CRDT 加速合并（U4 集成）—— 多 Agent 并发写入的无冲突自动合并。
//!
//! 使用 `uwu-crdt` 的 LwwMap / ORSet 等原语，为 `mergeable()` 的 MemoryClass
//! 提供零冲突合并策略，替代 ThreeWay 合并中的人工仲裁。

use agent_context_db_core::{ContextEntry, ContextUri, MemoryClass};
use std::collections::HashMap;
use uwu_crdt::LwwMap;

/// CRDT 合并结果。
#[derive(Debug, Clone)]
pub struct CrdtMergeResult {
    /// 合并后的条目
    pub merged: ContextEntry,
    /// 是否有冲突被自动解决
    pub conflicts_resolved: usize,
    /// 合并方式
    pub strategy: CrdtStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrdtStrategy {
    /// LWW-Map：键值覆盖（用于 Profile/Preferences）
    LwwKeyValue,
    /// 集合合并：去重并集（用于 Skills/Tools/Patterns）
    SetUnion,
}

/// CRDT 合并器 —— 利用 uwu-crdt 为 context-db 条目做无冲突合并。
pub struct CrdtMerger {
    node_id: String,
}

impl CrdtMerger {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
        }
    }

    /// 判断两个条目是否可以 CRDT 合并（仅 mergeable 类型）。
    pub fn can_merge(class: MemoryClass) -> bool {
        class.mergeable()
    }

    /// 对两个条目做 CRDT 合并。
    pub fn merge(
        &self,
        uri: &ContextUri,
        class: MemoryClass,
        entry_a: &ContextEntry,
        entry_b: &ContextEntry,
    ) -> CrdtMergeResult {
        let clock_a = entry_a.mvcc_version.0;
        let clock_b = entry_b.mvcc_version.0;

        let (merged, strategy) = match class {
            MemoryClass::Profile | MemoryClass::Preferences => {
                self.merge_lww(uri, entry_a, clock_a, entry_b, clock_b)
            }
            _ => self.merge_set_union(uri, entry_a, entry_b),
        };

        CrdtMergeResult {
            merged,
            conflicts_resolved: 1,
            strategy,
        }
    }

    /// LWW 合并：解析条目内容为键值对，逐 key 取 winner。
    fn merge_lww(
        &self,
        uri: &ContextUri,
        a: &ContextEntry,
        clock_a: u64,
        b: &ContextEntry,
        clock_b: u64,
    ) -> (ContextEntry, CrdtStrategy) {
        let l0_text_a = a.l0_text();
        let l0_text_b = b.l0_text();
        let parsed_a = parse_kv_pairs(l0_text_a);
        let parsed_b = parse_kv_pairs(l0_text_b);

        let mut map: LwwMap<String, String> = LwwMap::new();
        for (k, v) in &parsed_a {
            map.set(k.clone(), v.clone(), clock_a, &self.node_id);
        }
        for (k, v) in &parsed_b {
            map.set(k.clone(), v.clone(), clock_b, &self.node_id);
        }

        let merged_text: String = map
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join("; ");

        let mut merged = a.clone();
        merged.uri = uri.clone();
        merged.mvcc_version = agent_context_db_core::MvccVersion(clock_a.max(clock_b) + 1);
        // 将合并文本写入 payload
        merged.payload = agent_context_db_core::ContentPayload::Text {
            sparse: merged_text.clone(),
            dense: merged_text.clone(),
            full: merged_text,
        };

        (merged, CrdtStrategy::LwwKeyValue)
    }

    /// Set union 合并：将文本按分隔符拆为 items，去重合并。
    fn merge_set_union(
        &self,
        uri: &ContextUri,
        a: &ContextEntry,
        b: &ContextEntry,
    ) -> (ContextEntry, CrdtStrategy) {
        let l0_a = a.l0_text();
        let l0_b = b.l0_text();
        let items_a: Vec<&str> = l0_a
            .split(&[',', ';', '\n'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        let items_b: Vec<&str> = l0_b
            .split(&[',', ';', '\n'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut all = items_a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for item in items_b {
            if !all.iter().any(|x| x == item) {
                all.push(item.to_string());
            }
        }

        let merged_text = all.join("; ");
        let mut merged = a.clone();
        merged.uri = uri.clone();
        merged.mvcc_version =
            agent_context_db_core::MvccVersion(a.mvcc_version.0.max(b.mvcc_version.0) + 1);
        merged.payload = agent_context_db_core::ContentPayload::Text {
            sparse: merged_text.clone(),
            dense: merged_text.clone(),
            full: merged_text,
        };

        (merged, CrdtStrategy::SetUnion)
    }
}

/// 极简 key:value 解析（`key: value` 或 `key = value` 格式）。
fn parse_kv_pairs(text: &str) -> HashMap<String, String> {
    text.split(&[',', ';', '\n'][..])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| {
            if let Some(pos) = s.find(':').or_else(|| s.find('=')) {
                let key = s[..pos].trim().to_string();
                let val = s[pos + 1..].trim().to_string();
                Some((key, val))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::TenantId;

    fn entry(uri_str: &str, text: &str) -> ContextEntry {
        ContextEntry::new_text(
            ContextUri::parse(uri_str).unwrap(),
            TenantId(uuid::Uuid::nil()),
            text,
        )
    }

    #[test]
    fn lww_merge_higher_clock_wins() {
        let merger = CrdtMerger::new("agent-a");
        let uri = ContextUri::parse("uwu://t1/agent/a/memories/preferences/p1").unwrap();

        let mut a = entry(
            "uwu://t1/agent/a/memories/preferences/p1",
            "theme: dark; lang: en",
        );
        let mut b = entry(
            "uwu://t1/agent/a/memories/preferences/p1",
            "theme: light; lang: zh",
        );
        a.mvcc_version = agent_context_db_core::MvccVersion(1);
        b.mvcc_version = agent_context_db_core::MvccVersion(5); // b wins

        let result = merger.merge(&uri, MemoryClass::Preferences, &a, &b);
        assert!(matches!(
            &result.merged.payload,
            agent_context_db_core::ContentPayload::Text { sparse, .. } if sparse.contains("theme: light")
        ));
        assert_eq!(result.strategy, CrdtStrategy::LwwKeyValue);
    }

    #[test]
    fn set_union_dedup_merges_skills() {
        let merger = CrdtMerger::new("agent-a");
        let uri = ContextUri::parse("uwu://t1/agent/a/memories/skills/s1").unwrap();

        let a = entry("uwu://t1/agent/a/memories/skills/s1", "docker; git");
        let b = entry("uwu://t1/agent/a/memories/skills/s1", "git; kubernetes");

        let result = merger.merge(&uri, MemoryClass::Skills, &a, &b);
        // docker; git; kubernetes — git 去重
        assert!(matches!(
            &result.merged.payload,
            agent_context_db_core::ContentPayload::Text { sparse, .. } if sparse.contains("docker")
        ));
        assert!(matches!(
            &result.merged.payload,
            agent_context_db_core::ContentPayload::Text { sparse, .. } if sparse.contains("kubernetes")
        ));
        assert_eq!(result.strategy, CrdtStrategy::SetUnion);
    }

    #[test]
    fn non_mergeable_class_cannot_merge() {
        assert!(CrdtMerger::can_merge(MemoryClass::Preferences));
        assert!(!CrdtMerger::can_merge(MemoryClass::Events));
    }
}
