//! 意图分析器：将自然语言查询拆为 0-N 个类型化查询。
//!
//! - [`RuleBasedIntentAnalyzer`]：关键词匹配版（先行跑通链路）
//! - [`LlmIntentAnalyzer`]：LLM 驱动版（结构化意图分类，生产级）

use agent_context_db_core::{
    ContentType, ContextUri, LlmClient, LlmOpts, Result,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

use crate::{IntentAnalyzer, QueryKind, RetrieveContext, TypedQuery};

// ═══════════════════════════════════════════════════════════════════════════
// LlmIntentAnalyzer
// ═══════════════════════════════════════════════════════════════════════════

/// 基于 LLM 的意图分析器。
///
/// 使用 `LlmClient` 将自然语言查询拆为结构化 `TypedQuery`，
/// 自动推断 `QueryKind` 和 `target_dirs`。
pub struct LlmIntentAnalyzer {
    llm: Arc<dyn LlmClient>,
    /// 默认用户 ID（当 ctx 未指定时使用）
    default_tenant: String,
    /// 默认 Agent ID
    default_agent: String,
}

impl LlmIntentAnalyzer {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        default_tenant: impl Into<String>,
        default_agent: impl Into<String>,
    ) -> Self {
        Self {
            llm,
            default_tenant: default_tenant.into(),
            default_agent: default_agent.into(),
        }
    }
}

#[async_trait]
impl IntentAnalyzer for LlmIntentAnalyzer {
    async fn analyze(&self, query: &str, ctx: &RetrieveContext) -> Result<Vec<TypedQuery>> {
        let tenant = ctx.user_id.as_deref().unwrap_or(&self.default_tenant);
        let agent = ctx.agent_id.as_deref().unwrap_or(&self.default_agent);

        let prompt = format!(
            r#"Analyze the following user query and break it into one or more typed sub-queries.

User query: "{query}"

Available query kinds (pick the most appropriate for each sub-query):
- "SemanticSearch": broad meaning search across all memories
- "EntityLookup": looking for a specific person, project, or entity
- "EventRecall": recalling past events or timeline queries
- "SkillReuse": asking how to do something, tutorials, methods
- "PatternMatch": looking for reusable patterns or templates
- "StateSnapshot": asking about current/recent state
- "PersonaRelation": asking about relationships or social context

Available memory classes (assign the best matching one, or null):
- "preferences", "profile", "entities", "events", "cases", "patterns", "tools", "skills"

Target directories follow this convention:
- uwu://<tenant>/agent/<agent>/memories/<class>
- uwu://<tenant>/agent/<agent>/state/short|mid|long
- uwu://<tenant>/agent/<agent>/persona/relations

Return a JSON array of objects with:
- "kind": one of the query kinds above
- "text": the rewritten search text
- "target_dirs": array of directory URIs
- "expected_class": memory class or null

Current context: tenant="{tenant}", agent="{agent}"
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(1024),
            temperature: Some(0.0),
            ..Default::default()
        };

        let response = self
            .llm
            .complete(&prompt, &opts)
            .await
            .map_err(|e| agent_context_db_core::ContextError::Storage(
                format!("llm intent: {e}")
            ))?;

        // 解析 LLM JSON 输出
        let raw: Vec<RawTypedQuery> = serde_json::from_str(&response).unwrap_or_default();

        if raw.is_empty() {
            // Fallback: 返回一个宽泛的 SemanticSearch
            return Ok(vec![TypedQuery {
                kind: QueryKind::SemanticSearch,
                text: query.to_string(),
                target_dirs: default_memory_dirs(tenant, agent),
                expected_type: None,
            }]);
        }

        Ok(raw
            .into_iter()
            .map(|r| TypedQuery {
                kind: parse_kind(&r.kind),
                text: r.text,
                target_dirs: r
                    .target_dirs
                    .into_iter()
                    .filter_map(|d| ContextUri::parse(d).ok())
                    .collect(),
                expected_type: r.expected_class.as_deref().and_then(parse_class),
            })
            .collect())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RuleBasedIntentAnalyzer
// ═══════════════════════════════════════════════════════════════════════════

pub struct RuleBasedIntentAnalyzer {
    default_user_id: String,
    default_agent_id: String,
}

impl RuleBasedIntentAnalyzer {
    pub fn new(default_user_id: impl Into<String>, default_agent_id: impl Into<String>) -> Self {
        Self {
            default_user_id: default_user_id.into(),
            default_agent_id: default_agent_id.into(),
        }
    }

    fn classify(&self, query: &str, ctx: &RetrieveContext) -> Vec<TypedQuery> {
        let lower = query.to_lowercase();
        let user_id = ctx.user_id.as_deref().unwrap_or(&self.default_user_id);
        let agent_id = ctx.agent_id.as_deref().unwrap_or(&self.default_agent_id);

        let mut results = Vec::new();

        if contains_any(&lower, &["when", "happened", "event", "那天", "之前", "上次"]) {
            results.push(tq(QueryKind::EventRecall, query, &mem_dirs(user_id, agent_id, &["events", "cases"]), Some(ContentType::Fact)));
        }
        if contains_any(&lower, &["who", "what is", "entity", "project", "是谁", "什么是", "哪个"]) {
            results.push(tq(QueryKind::EntityLookup, query, &mem_dirs(user_id, agent_id, &["entities", "profile"]), Some(ContentType::Fact)));
        }
        if contains_any(&lower, &["how to", "how do", "步骤", "方法", "怎么", "如何", "教程"]) {
            results.push(tq(QueryKind::SkillReuse, query, &[
                memories_dir(user_id, agent_id, "skills"),
                memories_dir(user_id, agent_id, "tools"),
                uri(format!("uwu://{}/agent/{}/experiences", user_id, agent_id)),
            ], Some(ContentType::Skill)));
        }
        if contains_any(&lower, &["pattern", "template", "模式", "模板", "惯例", "typically"]) {
            results.push(tq(QueryKind::PatternMatch, query, &mem_dirs(user_id, agent_id, &["patterns", "cases"]), Some(ContentType::Heuristic)));
        }
        if contains_any(&lower, &["state", "snapshot", "状态", "当前", "now", "recently", "最近"]) {
            results.push(tq(QueryKind::StateSnapshot, query, &[
                uri(format!("uwu://{}/agent/{}/state/short", user_id, agent_id)),
                uri(format!("uwu://{}/agent/{}/state/mid", user_id, agent_id)),
            ], None));
        }
        if contains_any(&lower, &["relation", "persona", "关系", "朋友", "信任", "trust"]) {
            results.push(tq(QueryKind::PersonaRelation, query, &[
                uri(format!("uwu://{}/agent/{}/persona/relations", user_id, agent_id)),
            ], None));
        }
        if results.is_empty() || contains_any(&lower, &["prefer", "like", "dislike", "喜欢", "偏好", "remember", "记得"]) {
            results.push(tq(QueryKind::SemanticSearch, query, &default_memory_dirs(user_id, agent_id), None));
        }

        results
    }
}

#[async_trait]
impl IntentAnalyzer for RuleBasedIntentAnalyzer {
    async fn analyze(&self, query: &str, ctx: &RetrieveContext) -> Result<Vec<TypedQuery>> {
        Ok(self.classify(query, ctx))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 辅助
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct RawTypedQuery {
    kind: String,
    text: String,
    #[serde(default)]
    target_dirs: Vec<String>,
    expected_class: Option<String>,
}

fn parse_kind(s: &str) -> QueryKind {
    match s {
        "EntityLookup" => QueryKind::EntityLookup,
        "EventRecall" => QueryKind::EventRecall,
        "SkillReuse" => QueryKind::SkillReuse,
        "PatternMatch" => QueryKind::PatternMatch,
        "StateSnapshot" => QueryKind::StateSnapshot,
        "PersonaRelation" => QueryKind::PersonaRelation,
        _ => QueryKind::SemanticSearch,
    }
}

fn parse_class(s: &str) -> Option<ContentType> {
    match s {
        "profile" => Some(ContentType::Profile),
        "preferences" => Some(ContentType::Preference),
        "entities" => Some(ContentType::Fact),
        "events" => Some(ContentType::Fact),
        "cases" => Some(ContentType::Error),
        "patterns" => Some(ContentType::Heuristic),
        "tools" => Some(ContentType::Skill),
        "skills" => Some(ContentType::Skill),
        _ => ContentType::from_path_segment(s),
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn uri(s: impl Into<String>) -> ContextUri {
    ContextUri::parse(s.into()).expect("invalid URI in intent.rs helper")
}

fn memories_dir(tenant: &str, agent_id: &str, sub: &str) -> ContextUri {
    uri(format!("uwu://{}/agent/{}/memories/{}", tenant, agent_id, sub))
}

fn mem_dirs(tenant: &str, agent: &str, subs: &[&str]) -> Vec<ContextUri> {
    subs.iter().map(|s| memories_dir(tenant, agent, s)).collect()
}

fn default_memory_dirs(tenant: &str, agent: &str) -> Vec<ContextUri> {
    mem_dirs(tenant, agent, &["preferences", "profile", "cases", "events", "skills", "tools", "patterns", "entities"])
}

fn tq(kind: QueryKind, text: &str, dirs: &[ContextUri], class: Option<ContentType>) -> TypedQuery {
    TypedQuery {
        kind,
        text: text.to_string(),
        target_dirs: dirs.to_vec(),
        expected_type: class,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RetrieveContext {
        RetrieveContext {
            user_id: Some("u1".into()),
            agent_id: Some("a1".into()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn event_query_is_classified() {
        let ia = RuleBasedIntentAnalyzer::new("u1", "a1");
        let tqs = ia.analyze("what happened last week?", &ctx()).await.unwrap();
        assert_eq!(tqs[0].kind, QueryKind::EventRecall);
    }

    #[tokio::test]
    async fn howto_query_targets_skills() {
        let ia = RuleBasedIntentAnalyzer::new("u1", "a1");
        let tqs = ia.analyze("how to deploy the app?", &ctx()).await.unwrap();
        assert_eq!(tqs[0].kind, QueryKind::SkillReuse);
    }

    #[tokio::test]
    async fn ambiguous_query_gets_semantic_search() {
        let ia = RuleBasedIntentAnalyzer::new("u1", "a1");
        let tqs = ia.analyze("rust async patterns", &ctx()).await.unwrap();
        assert!(tqs.iter().any(|t| t.kind == QueryKind::PatternMatch));
    }

    #[tokio::test]
    async fn preference_query_falls_back_to_semantic() {
        let ia = RuleBasedIntentAnalyzer::new("u1", "a1");
        let tqs = ia.analyze("what does the user like?", &ctx()).await.unwrap();
        assert_eq!(tqs[0].kind, QueryKind::SemanticSearch);
    }

    #[tokio::test]
    async fn llm_intent_has_valid_target_dirs() {
        // Mock LLM returning structured intent
        struct MockIntentLlm;
        #[async_trait]
        impl LlmClient for MockIntentLlm {
            async fn complete(&self, _: &str, _: &LlmOpts) -> std::result::Result<String, agent_context_db_core::LlmError> {
                Ok(r#"[{"kind":"SemanticSearch","text":"user preference","target_dirs":["uwu://u1/agent/a1/memories/preferences"],"expected_class":"preferences"}]"#.into())
            }
            async fn embed(&self, _: &str) -> std::result::Result<Vec<f32>, agent_context_db_core::LlmError> {
                Ok(vec![])
            }
        }

        let ia = LlmIntentAnalyzer::new(Arc::new(MockIntentLlm), "u1", "a1");
        let tqs = ia.analyze("what does the user like?", &ctx()).await.unwrap();
        assert_eq!(tqs.len(), 1);
        assert_eq!(tqs[0].kind, QueryKind::SemanticSearch);
        assert!(!tqs[0].target_dirs.is_empty());
        assert_eq!(tqs[0].expected_type, Some(ContentType::Preference));
    }
}
