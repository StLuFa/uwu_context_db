//! `uwu://` URI — 上下文条目的强类型地址（M0 + Arc 零拷贝）。
//!
//! 形如 `uwu://<tenant>/<category>/<...segments>[?as_of=<time>&level=l0&limit=N]`，
//! clone 为零成本（Arc）。查询参数支持时态查询和层级选择。

use crate::error::{ContextError, Result};
use crate::model::ContentLevel;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const SCHEME: &str = "uwu://";

// ===========================================================================
// QueryParams + AsOfTime — 时态查询支持
// ===========================================================================

/// URI 查询参数 — 支持时态查询、层级选择、分支切换。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QueryParams {
    /// 时态查询：获取特定时间点的版本。
    pub as_of: Option<AsOfTime>,
    /// 内容层级过滤。
    pub level: Option<ContentLevel>,
    /// 版本分支。
    pub branch: Option<String>,
    /// 结果数量限制。
    pub limit: Option<usize>,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            as_of: None,
            level: None,
            branch: None,
            limit: None,
        }
    }
}

/// 时态定位点。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AsOfTime {
    /// 指定 commit hash。
    Commit(String),
    /// 指定时间戳。
    Timestamp(DateTime<Utc>),
    /// 最新版本（默认）。
    Latest,
}

// ===========================================================================
// ContextUri — Arc<UriInner> 零拷贝
// ===========================================================================

/// 结构化 URI 内部表示。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)] // F.6
struct UriInner {
    /// 租户 ID（第一段）。
    tenant: String,
    /// 路径段（不含 scheme，预解析）。
    path: Vec<String>,
    /// 查询参数（可选）。
    query: Option<QueryParams>,
    /// 规范化字符串（缓存）。
    canonical: String,
}

/// 上下文条目唯一标识（`uwu://` URI 的强类型封装，clone 零成本）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)] // F.6
pub struct ContextUri(#[serde(with = "uri_serde")] Arc<UriInner>);

/// Serde 适配：序列化为字符串，反序列化从字符串 parse。
mod uri_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        inner: &Arc<UriInner>,
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&inner.canonical)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> std::result::Result<Arc<UriInner>, D::Error> {
        let s = String::deserialize(d)?;
        let inner = UriInner::from_string(&s).map_err(serde::de::Error::custom)?;
        Ok(Arc::new(inner))
    }
}

impl UriInner {
    fn from_string(s: &str) -> Result<Self> {
        if !s.starts_with(SCHEME) {
            return Err(ContextError::InvalidUri(format!(
                "missing `{SCHEME}` scheme: {s}"
            )));
        }
        let rest = &s[SCHEME.len()..];
        if rest.is_empty() {
            return Err(ContextError::InvalidUri(format!("empty path: {s}")));
        }

        // Split path and query string
        let (path_part, query_str) = match rest.find('?') {
            Some(pos) => (&rest[..pos], Some(&rest[pos + 1..])),
            None => (rest, None),
        };

        let path: Vec<String> = path_part
            .split('/')
            .filter(|seg| !seg.is_empty())
            .map(String::from)
            .collect();

        if path.is_empty() {
            return Err(ContextError::InvalidUri(format!("empty path: {s}")));
        }

        let tenant = path[0].clone();

        // Parse query parameters
        let query = query_str.and_then(|qs| parse_query_string(qs));

        Ok(Self {
            tenant,
            path,
            query,
            canonical: s.to_string(),
        })
    }
}

/// 解析 URI 查询字符串 `as_of=...&level=l0&limit=50`。
fn parse_query_string(qs: &str) -> Option<QueryParams> {
    if qs.is_empty() {
        return None;
    }
    let mut params = QueryParams::default();
    for pair in qs.split('&') {
        let mut kv = pair.splitn(2, '=');
        let key = kv.next()?;
        let val = kv.next().unwrap_or("");
        match key {
            "as_of" => {
                params.as_of = if val.is_empty() || val == "latest" {
                    Some(AsOfTime::Latest)
                } else if let Ok(ts) = val.parse::<chrono::DateTime<Utc>>() {
                    Some(AsOfTime::Timestamp(ts))
                } else {
                    Some(AsOfTime::Commit(val.to_string()))
                };
            }
            "level" => {
                params.level = match val {
                    "l0" => Some(ContentLevel::L0),
                    "l1" => Some(ContentLevel::L1),
                    "l2" => Some(ContentLevel::L2),
                    _ => None,
                };
            }
            "branch" => {
                if !val.is_empty() {
                    params.branch = Some(val.to_string());
                }
            }
            "limit" => {
                if let Ok(n) = val.parse::<usize>() {
                    params.limit = Some(n);
                }
            }
            _ => {} // ignore unknown params
        }
    }
    Some(params)
}

impl ContextUri {
    /// 解析并校验一个 `uwu://` URI。
    pub fn parse(s: impl Into<String>) -> Result<Self> {
        let inner = UriInner::from_string(&s.into())?;
        Ok(Self(Arc::new(inner)))
    }

    /// 路径段（不含 scheme）。
    pub fn segments(&self) -> &[String] {
        &self.0.path
    }

    /// 租户段（第一段）。
    pub fn tenant_segment(&self) -> Option<&str> {
        self.0.path.first().map(|s| s.as_str())
    }

    /// 父目录 URI（去掉最后一段）。根级返回 None。
    pub fn parent(&self) -> Option<ContextUri> {
        let path = &self.0.path;
        if path.len() <= 1 {
            return None;
        }
        let parent_path = &path[..path.len() - 1];
        let canonical = format!("{SCHEME}{}", parent_path.join("/"));
        let inner = UriInner {
            tenant: path[0].clone(),
            path: parent_path.to_vec(),
            query: None,
            canonical,
        };
        Some(ContextUri(Arc::new(inner)))
    }

    /// 追加子段。
    pub fn join(&self, seg: &str) -> ContextUri {
        let canonical = format!("{}/{}", self.0.canonical.trim_end_matches('/'), seg);
        let mut path = self.0.path.clone();
        path.push(seg.to_string());
        let inner = UriInner {
            tenant: self.0.tenant.clone(),
            path,
            query: None,
            canonical,
        };
        ContextUri(Arc::new(inner))
    }

    /// 路径深度（段数）。
    pub fn depth(&self) -> usize {
        self.0.path.len()
    }

    /// 分类（解析第二段：跳过 tenant）。
    pub fn category(&self) -> UriCategory {
        let cat = self.0.path.get(1).map(|s| s.as_str()).unwrap_or("");
        match cat {
            "user" => UriCategory::User,
            "agent" => UriCategory::Agent,
            "resources" => UriCategory::Resources,
            "skills" => UriCategory::Skills,
            "wiki" => UriCategory::Wiki,
            "sessions" => UriCategory::Sessions,
            "state" => UriCategory::State,
            "persona" => UriCategory::Persona,
            "metacog" => UriCategory::Metacog,
            "character" => UriCategory::Character,
            _ => UriCategory::Unknown,
        }
    }

    /// 租户名。
    pub fn tenant(&self) -> &str {
        &self.0.tenant
    }

    /// 查询参数（如果有）。
    pub fn query(&self) -> Option<&QueryParams> {
        self.0.query.as_ref()
    }

    /// 时态查询定位点。
    pub fn as_of(&self) -> Option<&AsOfTime> {
        self.0.query.as_ref()?.as_of.as_ref()
    }

    /// 请求的内容层级。
    pub fn level(&self) -> Option<ContentLevel> {
        self.0.query.as_ref()?.level
    }

    /// 返回带查询参数的 URI（不可变更新）。
    pub fn with_query(&self, query: QueryParams) -> Self {
        let base = self
            .0
            .canonical
            .split('?')
            .next()
            .unwrap_or(&self.0.canonical);
        let canonical = format!("{}?{}", base, query_to_string(&query));
        let inner = UriInner {
            tenant: self.0.tenant.clone(),
            path: self.0.path.clone(),
            query: Some(query),
            canonical,
        };
        ContextUri(Arc::new(inner))
    }

    /// 去掉所有查询参数，返回纯路径 URI。
    pub fn without_query(&self) -> Self {
        if self.0.query.is_none() {
            return self.clone();
        }
        let base = self
            .0
            .canonical
            .split('?')
            .next()
            .unwrap_or(&self.0.canonical);
        ContextUri::parse(base).unwrap_or_else(|_| self.clone())
    }

    /// 规范化字符串。
    pub fn as_str(&self) -> &str {
        &self.0.canonical
    }
}

/// 将 QueryParams 序列化为查询字符串。
fn query_to_string(q: &QueryParams) -> String {
    let mut parts = Vec::new();
    if let Some(ref as_of) = q.as_of {
        parts.push(match as_of {
            AsOfTime::Latest => "as_of=latest".to_string(),
            AsOfTime::Commit(h) => format!("as_of={}", h),
            AsOfTime::Timestamp(ts) => format!("as_of={}", ts.to_rfc3339()),
        });
    }
    if let Some(level) = q.level {
        parts.push(format!("level={}", level.as_str()));
    }
    if let Some(ref branch) = q.branch {
        parts.push(format!("branch={}", branch));
    }
    if let Some(limit) = q.limit {
        parts.push(format!("limit={}", limit));
    }
    parts.join("&")
}

impl std::fmt::Display for ContextUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.canonical)
    }
}

// ===========================================================================
// UriCategory
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)] // F.6: Ord for BTreeMap
pub enum UriCategory {
    User,
    Agent,
    Resources,
    Skills,
    Wiki,
    Sessions,
    State,
    Persona,
    Metacog,
    Character,
    Unknown,
}

// ===========================================================================
// 旧 API 兼容：提供 `uri.0` → `uri.to_string()` 的桥接
// ===========================================================================

/// 临时兼容辅助：从 &str 构造（跳过校验，仅测试用）。
#[doc(hidden)]
pub fn uri_from_str(s: &str) -> ContextUri {
    ContextUri::parse(s).unwrap()
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_bad_scheme() {
        assert!(ContextUri::parse("http://x/y").is_err());
        assert!(ContextUri::parse("uwu://").is_err());
    }

    #[test]
    fn parent_and_depth() {
        let u = ContextUri::parse("uwu://t1/user/u1/memories/preferences/p1").unwrap();
        assert_eq!(u.depth(), 6);
        assert_eq!(u.tenant_segment(), Some("t1"));
        let p = u.parent().unwrap();
        assert_eq!(p.to_string(), "uwu://t1/user/u1/memories/preferences");
        assert_eq!(p.category(), UriCategory::User);
    }

    #[test]
    fn join_appends_segment() {
        let u = ContextUri::parse("uwu://t1/agent/a1").unwrap();
        assert_eq!(u.join("state").to_string(), "uwu://t1/agent/a1/state");
    }

    #[test]
    fn clone_is_cheap() {
        let u = ContextUri::parse("uwu://t1/a/b/c").unwrap();
        let cloned = u.clone();
        assert_eq!(u.to_string(), cloned.to_string());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn uri_roundtrip(
            tenant in "[a-z][a-z0-9]*",
            segs in proptest::collection::vec("[a-z][a-z0-9_]*", 1..6)
        ) {
            let uri_str = format!("uwu://{}/{}", tenant, segs.join("/"));
            let u = ContextUri::parse(&uri_str).unwrap();
            prop_assert_eq!(u.to_string(), uri_str);
            prop_assert_eq!(u.depth(), segs.len() + 1);
        }

        #[test]
        fn parse_then_clone_identical(
            s in "uwu://[a-z]+(/[a-z0-9_]+){0,4}"
        ) {
            let u = ContextUri::parse(&s).unwrap();
            prop_assert_eq!(u.to_string(), u.clone().to_string());
        }
    }
}
