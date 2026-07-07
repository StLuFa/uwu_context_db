//! ProgressiveLoader — 四层渐进式加载（索引→摘要→完整→证据）。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextEntry, ContextUri};

/// 加载层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoadLevel {
    Index = 0,    // name + 类型 (~20 tokens)
    Abstract = 1, // L0 摘要 (~100 tokens)
    Full = 2,     // L1 概览 (~2k tokens)
    Evidence = 3, // 完整证据树（按需）
}

/// 渐进式加载器。
pub struct ProgressiveLoader {
    budget: usize,
    used: usize,
}

impl ProgressiveLoader {
    pub fn new(budget: usize) -> Self {
        Self { budget, used: 0 }
    }

    /// 按预算逐层加载 — 预算紧张时降级。
    pub fn load_level(&self, entry: &ContextEntry) -> LoadLevel {
        let remaining = self.budget.saturating_sub(self.used);
        match remaining {
            r if r >= 2000 => LoadLevel::Full,
            r if r >= 100 => LoadLevel::Abstract,
            _ => LoadLevel::Index,
        }
    }

    /// 按层级返回对应内容。
    pub fn content_at(&self, entry: &ContextEntry, level: LoadLevel) -> String {
        match level {
            LoadLevel::Index => {
                format!(
                    "{} ({})",
                    entry.uri,
                    entry
                        .content_type()
                        .map(|c| c.as_path_segment())
                        .unwrap_or("?"),
                )
            }
            LoadLevel::Abstract => entry.l0_text().to_string(),
            LoadLevel::Full => match &entry.payload {
                ContentPayload::Text { dense, .. } => dense.clone(),
                _ => entry.l0_text().to_string(),
            },
            LoadLevel::Evidence => {
                // 完整证据树需查询 GraphStore
                format!("[evidence tree for {}]", entry.uri)
            }
        }
    }

    /// 消费 token 预算。
    pub fn consume(&mut self, tokens: usize) {
        self.used = self.used.saturating_add(tokens);
    }

    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.used)
    }
}
