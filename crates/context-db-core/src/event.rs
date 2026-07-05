//! 变更事件流 + 因果链（F5）+ 上下文继承（F11）+ 上下文模板（F12）。

use crate::{ContextEntry, ContextUri, MemoryClass};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// F5 变更事件流 + 因果链
// ═══════════════════════════════════════════════════════════════════════════

/// 变更来源。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeSource {
    SessionCommit { session_id: String, compression_index: u64 },
    AgentWrite { agent_id: String },
    ForkPromotion { fork_name: String },
    AutoConsolidation,
    Import { pack_name: String },
}

/// 因果链节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// 触发源 URI
    pub source_uri: ContextUri,
    /// 来源类型
    pub source: ChangeSource,
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 上游因果链
    pub parent: Option<Box<CausalLink>>,
}

/// 流式变更事件 + 完整因果链。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEventStream {
    /// 事件 ID
    pub event_id: String,
    /// 变更的条目
    pub entries: Vec<ContextEntry>,
    /// 因果链
    pub causal_chain: CausalLink,
    /// 发生时间
    pub occurred_at: DateTime<Utc>,
}

/// 事件流发射器。
pub struct EventEmitter {
    /// 事件历史
    history: parking_lot::Mutex<Vec<ChangeEventStream>>,
    /// 因果链索引：URI → 最近的因果链
    causal_index: parking_lot::Mutex<HashMap<String, CausalLink>>,
}

impl EventEmitter {
    pub fn new() -> Self {
        Self {
            history: parking_lot::Mutex::new(Vec::new()),
            causal_index: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// 发射一个变更事件。
    pub fn emit(
        &self,
        entries: Vec<ContextEntry>,
        source: ChangeSource,
        parent_uri: Option<ContextUri>,
    ) -> ChangeEventStream {
        let parent = parent_uri
            .and_then(|uri| self.causal_index.lock().get(&uri.to_string()).cloned());

        let causal_link = CausalLink {
            source_uri: entries.first().map(|e| e.uri.clone()).unwrap_or_else(|| ContextUri::parse("uwu://_/empty").unwrap()),
            source: source.clone(),
            timestamp: Utc::now(),
            parent: parent.map(Box::new),
        };

        let event = ChangeEventStream {
            event_id: uuid::Uuid::new_v4().to_string(),
            entries,
            causal_chain: causal_link.clone(),
            occurred_at: Utc::now(),
        };

        // 更新索引
        for entry in &event.entries {
            self.causal_index.lock().insert(entry.uri.to_string().clone(), causal_link.clone());
        }
        self.history.lock().push(event.clone());
        event
    }

    /// 追溯某个 URI 的完整因果链。
    pub fn trace_causality(&self, uri: &ContextUri) -> Vec<CausalLink> {
        let mut chain = Vec::new();
        let mut current = self.causal_index.lock().get(&uri.to_string()).cloned();
        while let Some(link) = current {
            current = link.parent.as_ref().map(|p| *p.clone());
            chain.push(link);
        }
        chain
    }

    /// 最近 N 个事件。
    pub fn recent(&self, n: usize) -> Vec<ChangeEventStream> {
        self.history.lock().iter().rev().take(n).cloned().collect()
    }
}

impl Default for EventEmitter {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════════════
// F11 上下文继承与覆盖
// ═══════════════════════════════════════════════════════════════════════════

/// 继承链节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritanceNode {
    /// 当前 Agent
    pub agent_id: String,
    /// 父 Agent（继承来源）
    pub parent: Option<String>,
    /// 继承的 scope
    pub scope: ContextUri,
    /// 覆盖规则
    pub overrides: HashMap<String, OverrideRule>,
}

/// 覆盖规则。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverrideRule {
    /// 覆盖的 URI 模式
    pub uri_pattern: String,
    /// 覆盖类型
    pub action: OverrideAction,
    /// 优先级
    pub priority: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverrideAction {
    /// 完全覆盖父级
    Replace,
    /// 合并（子优先）
    Merge,
    /// 仅追加
    Append,
    /// 隐藏父级条目
    Hide,
}

/// 继承链管理器。
///
/// 运行时维护 Agent 继承 DAG；支持序列化到 context-db 和从 context-db 恢复。
#[derive(Default)]
pub struct InheritanceChain {
    nodes: parking_lot::Mutex<Vec<InheritanceNode>>,
}

impl InheritanceChain {
    pub fn new() -> Self { Self::default() }

    /// 注册一个继承关系。
    pub fn register(&self, node: InheritanceNode) {
        self.nodes.lock().push(node);
    }

    /// 解析一个 URI 在继承链中的有效来源。
    /// 返回 (有效的条目 URI, 是否来自父级)。
    pub fn resolve(&self, uri: &ContextUri, agent_id: &str) -> Option<(ContextUri, bool)> {
        let nodes = self.nodes.lock();
        let node = nodes.iter().find(|n| n.agent_id == agent_id)?;

        // 先查自己的覆盖
        for (pattern, rule) in &node.overrides {
            if uri.to_string().starts_with(pattern) {
                return match rule.action {
                    OverrideAction::Replace => Some((uri.clone(), false)),
                    OverrideAction::Hide => None,
                    OverrideAction::Merge | OverrideAction::Append => Some((uri.clone(), false)),
                };
            }
        }

        // 查父级
        if let Some(ref parent_id) = node.parent {
            let parent_node = nodes.iter().find(|n| &n.agent_id == parent_id)?;
            // 将 URI 映射到父级 scope
            let mapped = uri.to_string().replace(&node.scope.0, &parent_node.scope.0);
            Some((ContextUri::parse(mapped).unwrap(), true))
        } else {
            Some((uri.clone(), false))
        }
    }

    /// 序列化为 JSON（用于持久化到 context-db）。
    pub fn serialize_to_json(&self) -> String {
        let nodes = self.nodes.lock();
        serde_json::to_string(&*nodes).unwrap_or_else(|_| "[]".to_string())
    }

    /// 从 JSON 反序列化（从 context-db 恢复）。
    pub fn deserialize_from_json(&self, json: &str) -> Result<(), String> {
        let parsed: Vec<InheritanceNode> =
            serde_json::from_str(json).map_err(|e| format!("deserialize: {e}"))?;
        let mut nodes = self.nodes.lock();
        nodes.clear();
        nodes.extend(parsed);
        Ok(())
    }

    /// 返回所有节点的快照（只读）。
    pub fn snapshot(&self) -> Vec<InheritanceNode> {
        self.nodes.lock().clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F12 上下文模板与实例化
// ═══════════════════════════════════════════════════════════════════════════

/// 上下文模板。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextTemplate {
    pub name: String,
    pub description: String,
    /// 模板条目（占位符用 {key} 标记）
    pub entries: Vec<TemplateEntry>,
    /// 默认变量值
    pub defaults: HashMap<String, String>,
}

/// 模板中的条目定义。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEntry {
    /// URI 模板（含占位符）
    pub uri_template: String,
    /// L0 模板
    pub abstract_template: String,
    /// 记忆类型
    pub memory_class: Option<MemoryClass>,
}

/// 模板引擎。
pub struct TemplateEngine;

impl TemplateEngine {
    /// 用变量填充模板，生成实例化条目。
    pub fn instantiate(
        template: &ContextTemplate,
        variables: &HashMap<String, String>,
        scope: &ContextUri,
    ) -> Vec<ContextEntry> {
        let mut entries = Vec::new();

        // 合并默认值 + 传入变量（传入优先）
        let mut vars = template.defaults.clone();
        for (k, v) in variables {
            vars.insert(k.clone(), v.clone());
        }

        for tpl in &template.entries {
            let uri_str = TemplateEngine::interpolate(&tpl.uri_template, &vars);
            let abstract_ = TemplateEngine::interpolate(&tpl.abstract_template, &vars);

            let full_uri = format!("{}/{}", scope, uri_str);
            let uri = ContextUri::parse(full_uri).unwrap_or_else(|_| scope.clone());

            let mut entry = ContextEntry::new_text(
                uri,
                crate::TenantId(uuid::Uuid::nil()),
                abstract_,
            );
            if let Some(mc) = tpl.memory_class {
                entry.metadata.memory_class = Some(mc);
            }
            entries.push(entry);
        }
        entries
    }

    fn interpolate(template: &str, vars: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in vars {
            result = result.replace(&format!("{{{key}}}"), value);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_emitter_traces_causality_chain() {
        let emitter = EventEmitter::new();
        let uri = ContextUri::parse("uwu://t1/agent/a1/memories/cases/c1").unwrap();
        let entry = ContextEntry::new_text(uri.clone(), crate::TenantId(uuid::Uuid::nil()), "test");

        let e1 = emitter.emit(vec![entry], ChangeSource::SessionCommit {
            session_id: "s1".into(), compression_index: 0,
        }, None);
        assert_eq!(e1.event_id.len(), 36);

        let chain = emitter.trace_causality(&uri);
        assert!(!chain.is_empty());
    }

    #[test]
    fn inheritance_resolves_to_parent() {
        let chain = InheritanceChain::new();
        chain.register(InheritanceNode {
            agent_id: "child".into(),
            parent: Some("parent".into()),
            scope: ContextUri::parse("uwu://t1/agent/child").unwrap(),
            overrides: HashMap::new(),
        });
        chain.register(InheritanceNode {
            agent_id: "parent".into(),
            parent: None,
            scope: ContextUri::parse("uwu://t1/agent/parent").unwrap(),
            overrides: HashMap::new(),
        });

        let result = chain.resolve(
            &ContextUri::parse("uwu://t1/agent/child/memories/preferences/p1").unwrap(),
            "child",
        );
        assert!(result.is_some());
    }

    #[test]
    fn template_instantiation_fills_variables() {
        let template = ContextTemplate {
            name: "new-agent-init".into(),
            description: "初始化包".into(),
            entries: vec![TemplateEntry {
                uri_template: "memories/profile/{name}".into(),
                abstract_template: "Agent {name} initialized with role {role}".into(),
                memory_class: Some(MemoryClass::Profile),
            }],
            defaults: HashMap::from([("role".into(), "assistant".into())]),
        };

        let vars = HashMap::from([("name".into(), "helper-bot".into())]);
        let scope = ContextUri::parse("uwu://t1/agent/helper-bot").unwrap();
        let entries = TemplateEngine::instantiate(&template, &vars, &scope);

        assert_eq!(entries.len(), 1);
        assert!(entries[0].l0_text().contains("helper-bot"));
        assert!(entries[0].l0_text().contains("assistant"));
    }
}
