//! `uwu://` URI —— 上下文条目的强类型地址（M0）。
//!
//! 形如 `uwu://<tenant>/<category>/<...segments>`，例：
//! `uwu://tenant1/user/u1/memories/preferences/p1`。

use crate::error::{ContextError, Result};
use serde::{Deserialize, Serialize};

pub const SCHEME: &str = "uwu://";

/// 上下文条目唯一标识（`uwu://` URI 的强类型封装）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextUri(pub String);

impl ContextUri {
    /// 解析并校验一个 `uwu://` URI。
    pub fn parse(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        if !s.starts_with(SCHEME) {
            return Err(ContextError::InvalidUri(format!(
                "missing `{}` scheme: {}",
                SCHEME, s
            )));
        }
        if s[SCHEME.len()..].is_empty() {
            return Err(ContextError::InvalidUri(format!("empty path: {}", s)));
        }
        Ok(Self(s))
    }

    /// 去掉 scheme 后的路径段（不含空段）。
    pub fn segments(&self) -> Vec<&str> {
        self.0
            .strip_prefix(SCHEME)
            .unwrap_or(&self.0)
            .split('/')
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// 租户段（第一段）。
    pub fn tenant_segment(&self) -> Option<&str> {
        self.segments().first().copied()
    }

    /// 父目录 URI（去掉最后一段）。根级返回 `None`。
    pub fn parent(&self) -> Option<ContextUri> {
        let segs = self.segments();
        if segs.len() <= 1 {
            return None;
        }
        let parent = &segs[..segs.len() - 1];
        Some(ContextUri(format!("{}{}", SCHEME, parent.join("/"))))
    }

    /// 追加子段。
    pub fn join(&self, seg: &str) -> ContextUri {
        ContextUri(format!("{}/{}", self.0.trim_end_matches('/'), seg))
    }

    /// 路径深度（段数）。
    pub fn depth(&self) -> usize {
        self.segments().len()
    }

    /// 分类（解析第二段：跳过 tenant）。
    pub fn category(&self) -> UriCategory {
        let segs = self.segments();
        let cat = segs.get(1).copied().unwrap_or("");
        // memories 下的具体分类由 metadata 表达，这里只区分顶层域。
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
}

impl std::fmt::Display for ContextUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        assert_eq!(p.0, "uwu://t1/user/u1/memories/preferences");
        assert_eq!(p.category(), UriCategory::User);
    }

    #[test]
    fn join_appends_segment() {
        let u = ContextUri::parse("uwu://t1/agent/a1").unwrap();
        assert_eq!(u.join("state").0, "uwu://t1/agent/a1/state");
    }
}
