//! F11 上下文继承 + F12 上下文模板。
//!
//! F5 变更事件流 / EventEmitter / CausalLink 已迁移到 `event_store.rs`
//! （基于 `uwu_event_mesh`）。

use crate::{ContextEntry, ContextUri, MemoryClass};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[derive(Default)]
pub struct InheritanceChain {
    nodes: parking_lot::Mutex<Vec<InheritanceNode>>,
}

impl InheritanceChain {
    pub fn new() -> Self {
        Self::default()
    }

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
            let mapped = uri
                .to_string()
                .replace(node.scope.as_str(), parent_node.scope.as_str());
            Some((ContextUri::parse(mapped).unwrap(), true))
        } else {
            Some((uri.clone(), false))
        }
    }

    pub fn serialize_to_json(&self) -> String {
        let nodes = self.nodes.lock();
        serde_json::to_string(&*nodes).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn deserialize_from_json(&self, json: &str) -> Result<(), String> {
        let parsed: Vec<InheritanceNode> =
            serde_json::from_str(json).map_err(|e| format!("deserialize: {e}"))?;
        let mut nodes = self.nodes.lock();
        nodes.clear();
        nodes.extend(parsed);
        Ok(())
    }

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
    pub entries: Vec<TemplateEntry>,
    pub defaults: HashMap<String, String>,
}

/// 模板中的条目定义。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEntry {
    pub uri_template: String,
    pub abstract_template: String,
    pub memory_class: Option<MemoryClass>,
}

/// 模板引擎。
pub struct TemplateEngine;

impl TemplateEngine {
    pub fn instantiate(
        template: &ContextTemplate,
        variables: &HashMap<String, String>,
        scope: &ContextUri,
    ) -> Vec<ContextEntry> {
        let mut entries = Vec::new();
        let mut vars = template.defaults.clone();
        for (k, v) in variables {
            vars.insert(k.clone(), v.clone());
        }

        for tpl in &template.entries {
            let uri_str = TemplateEngine::interpolate(&tpl.uri_template, &vars);
            let abstract_ = TemplateEngine::interpolate(&tpl.abstract_template, &vars);

            let full_uri = format!("{}/{}", scope, uri_str);
            let uri = ContextUri::parse(full_uri).unwrap_or_else(|_| scope.clone());

            let mut entry =
                ContextEntry::new_text(uri, crate::TenantId(uuid::Uuid::nil()), abstract_);
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
