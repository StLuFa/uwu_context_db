//! 核心数据模型（M0 → 认知数据库重构）：
//! - 分层编码 ContentPayload（替代 L0/L1/L2 三独立字段）
//! - 13 种 ContentType 内容分类
//! - 强类型 ContextMeta（epistemic / validity / consolidation）
//! - 结构化 ContextUri（Arc<UriInner>）

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ===========================================================================
// 基础标识类型
// ===========================================================================

/// 单调递增版本号（M0 用；M2 起版本管理迁移到 `agent-context-db-version` 的 CommitId）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MvccVersion(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub Uuid);

/// Blob 引用 — 指向 BlobStore 中的原始载荷。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub hash: ContentHash,
    pub size: usize,
    pub mime_type: String,
}

/// 内容哈希（blake3）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

/// Schema 引用 —— 指向 JSON Schema / Avro / Protobuf 定义。
///
/// `format` 声明 schema 语法（`"json-schema"` / `"avro"` / `"protobuf"` 等），
/// `blob` 指向实际定义文件。空 `blob` 表示纯内联式使用 `format` 作为轻标记。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaRef {
    pub format: String,
    pub blob: Option<BlobRef>,
}

impl SchemaRef {
    pub fn json_schema(blob: BlobRef) -> Self {
        Self {
            format: "json-schema".into(),
            blob: Some(blob),
        }
    }
}

// ===========================================================================
// 媒体格式（payload 的物理格式）
// ===========================================================================

/// 内容媒体格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaType {
    Text,
    Image,
    Audio,
    Video,
    Binary,
}

// ===========================================================================
// 三层内容级别
// ===========================================================================

/// 三层内容级别：L0 摘要 / L1 概览 / L2 原始详情。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum ContentLevel {
    #[default]
    L0,
    L1,
    L2,
}

impl ContentLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ContentLevel::L0 => "l0",
            ContentLevel::L1 => "l1",
            ContentLevel::L2 => "l2",
        }
    }
}

// ===========================================================================
// 分层编码 ContentPayload（替代旧的 Abstract/Overview/Detail）
// ===========================================================================

/// 分层编码内容 — 每个变体自带三级编码。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPayload {
    /// 文本内容：稀疏 → 稠密 → 完整
    Text {
        sparse: String, // L0 ~100 tokens，可由 LLM 从 dense 自动生成
        dense: String,  // L1 ~2k tokens，可由 LLM 从 full 自动生成
        full: String,   // L2 完整文本
    },
    /// 图像内容：缩略图 → 特征向量 → 原始像素
    Image {
        thumbnail: Vec<u8>,                            // L0 ~256x256 JPEG
        features: crate::multimodal::EncodedEmbedding, // L1 image embedding with explicit space identity
        raw: BlobRef,                                  // L2 原始像素（存 BlobStore）
    },
    /// 音频内容：转写 → 语音 embedding → 原始波形
    Audio {
        transcript: String,                             // L0 ASR 转写
        embedding: crate::multimodal::EncodedEmbedding, // L1 audio embedding with explicit space identity
        raw: BlobRef,                                   // L2 原始波形
    },
    /// 结构化内容：JSON 原生存储 + 可选 schema 引用
    Structured {
        summary: String, // L0 人类可读摘要
        /// L1 可选 schema 描述（BlobRef 指向 JSON Schema / Avro / Protobuf 定义）。
        /// 用于校验 `data` 结构、驱动 UI 渲染。`None` 视为 schemaless。
        schema: Option<SchemaRef>,
        data: serde_json::Value, // L2 完整 JSON
    },
    /// 多部分组合（如带图的文章）
    Composite {
        summary: String,
        parts: Vec<ContentPart>,
    },
}

/// 组合内容的一部分 — 可独立解码或引用其他条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text(Box<ContentPayload>),
    Image(Box<ContentPayload>),
    Audio(Box<ContentPayload>),
    Reference(crate::uri::ContextUri),
}

/// Searchable text projected from a payload's three storage levels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentIndexProjection {
    pub l0: String,
    pub l1: Option<String>,
    pub l2: String,
}

impl ContentPayload {
    /// Produces the canonical searchable L0/L1/L2 representation used by every backend.
    pub fn index_projection(&self) -> ContentIndexProjection {
        let mut projection = ContentIndexProjection {
            l0: String::new(),
            l1: None,
            l2: String::new(),
        };
        self.append_index_projection(&mut projection);
        projection
    }

    fn append_index_projection(&self, output: &mut ContentIndexProjection) {
        fn append(target: &mut String, value: &str) {
            if !value.is_empty() {
                if !target.is_empty() {
                    target.push('\n');
                }
                target.push_str(value);
            }
        }
        fn append_optional(target: &mut Option<String>, value: &str) {
            if value.is_empty() {
                return;
            }
            append(target.get_or_insert_with(String::new), value);
        }

        match self {
            Self::Text {
                sparse,
                dense,
                full,
            } => {
                append(&mut output.l0, sparse);
                append_optional(&mut output.l1, dense);
                append(&mut output.l2, full);
            }
            Self::Image {
                thumbnail,
                features,
                raw,
            } => {
                append(
                    &mut output.l0,
                    &format!("[image thumbnail_bytes={}]", thumbnail.len()),
                );
                append_optional(
                    &mut output.l1,
                    &format!("[image feature_dimensions={}]", features.values.len()),
                );
                append(
                    &mut output.l2,
                    &format!(
                        "[image blob={} size={} mime={}]",
                        raw.hash.0, raw.size, raw.mime_type
                    ),
                );
            }
            Self::Audio {
                transcript,
                embedding,
                raw,
            } => {
                append(&mut output.l0, transcript);
                append_optional(
                    &mut output.l1,
                    &format!("[audio embedding_dimensions={}]", embedding.values.len()),
                );
                append(
                    &mut output.l2,
                    &format!(
                        "{}\n[audio blob={} size={} mime={}]",
                        transcript, raw.hash.0, raw.size, raw.mime_type
                    ),
                );
            }
            Self::Structured {
                summary,
                schema,
                data,
            } => {
                append(&mut output.l0, summary);
                if let Some(schema) = schema {
                    append_optional(
                        &mut output.l1,
                        &format!("[schema format={}]", schema.format),
                    );
                }
                let json = serde_json::to_string(data).unwrap_or_else(|_| data.to_string());
                append_optional(&mut output.l1, &json);
                append(&mut output.l2, &json);
            }
            Self::Composite { summary, parts } => {
                append(&mut output.l0, summary);
                append(&mut output.l2, summary);
                for part in parts {
                    match part {
                        ContentPart::Text(payload)
                        | ContentPart::Image(payload)
                        | ContentPart::Audio(payload) => payload.append_index_projection(output),
                        ContentPart::Reference(uri) => {
                            let reference = format!("[reference {uri}]");
                            append_optional(&mut output.l1, &reference);
                            append(&mut output.l2, &reference);
                        }
                    }
                }
            }
        }
    }

    /// 获取 L0 级文本摘要（所有变体通用）。
    pub fn sparse_text(&self) -> &str {
        match self {
            ContentPayload::Text { sparse, .. } => sparse,
            ContentPayload::Image { .. } => "[image]",
            ContentPayload::Audio { transcript, .. } => transcript,
            ContentPayload::Structured { summary, .. } => summary,
            ContentPayload::Composite { summary, .. } => summary,
        }
    }

    /// 按 token 预算逐层解码。
    pub fn decode_within_budget(&self, budget: usize) -> DecodedContent {
        match self {
            ContentPayload::Text {
                sparse,
                dense,
                full,
            } => {
                let l0 = sparse.clone();
                let l1 = crate::tokenizer::count_tokens(dense)
                    .ok()
                    .filter(|tokens| budget >= *tokens)
                    .map(|_| dense.clone());
                let l2 = crate::tokenizer::count_tokens(full)
                    .ok()
                    .filter(|tokens| budget >= *tokens)
                    .map(|_| full.clone());
                DecodedContent::Text { l0, l1, l2 }
            }
            ContentPayload::Image { thumbnail, .. } => DecodedContent::Binary(thumbnail.clone()),
            ContentPayload::Audio { transcript, .. } => DecodedContent::Text {
                l0: transcript.clone(),
                l1: None,
                l2: None,
            },
            ContentPayload::Structured { summary, data, .. } => DecodedContent::Text {
                l0: summary.clone(),
                l1: Some(data.to_string()),
                l2: None,
            },
            ContentPayload::Composite { summary, .. } => DecodedContent::Text {
                l0: summary.clone(),
                l1: None,
                l2: None,
            },
        }
    }
}

/// 按预算解码后的内容。
#[derive(Debug, Clone)]
pub enum DecodedContent {
    Text {
        l0: String,
        l1: Option<String>,
        l2: Option<String>,
    },
    Binary(Vec<u8>),
}

// ===========================================================================
// 记忆分类（13 种内容类型）
// ===========================================================================

/// 13 种内容类型 — URI 路径段原生的记忆分类。
///
/// 类型进 URI：`uwu://t/{agent}/memory/{type}/{semantic_path}/{id}`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    // === 认识论类型 ===
    /// 可验证事实，需证据支撑
    Fact,
    /// 主观判断，带置信度
    Belief,
    /// 待验证假设
    Hypothesis,
    /// 经验法则
    Heuristic,
    /// 可执行步骤
    Procedure,

    // === 外部对齐类型 ===
    /// 用户偏好
    Preference,
    /// 用户画像
    Profile,
    /// 目标/意图
    Goal,

    // === 补充类型 ===
    /// 已验证的能力/工具用法
    Skill,
    /// 反思/元认知产物
    Reflection,
    /// 原始证据（不蒸馏）
    Evidence,
    /// 失败案例/踩坑
    Error,
    /// 系统元记忆
    Meta,
}

impl ContentType {
    /// URI 路径段名称。
    pub fn as_path_segment(&self) -> &'static str {
        match self {
            ContentType::Fact => "fact",
            ContentType::Belief => "belief",
            ContentType::Hypothesis => "hypothesis",
            ContentType::Heuristic => "heuristic",
            ContentType::Procedure => "procedure",
            ContentType::Preference => "preference",
            ContentType::Profile => "profile",
            ContentType::Goal => "goal",
            ContentType::Skill => "skill",
            ContentType::Reflection => "reflection",
            ContentType::Evidence => "evidence",
            ContentType::Error => "error",
            ContentType::Meta => "meta",
        }
    }

    /// 从 URI 路径段反解类型。
    pub fn from_path_segment(s: &str) -> Option<Self> {
        match s {
            "fact" => Some(Self::Fact),
            "belief" => Some(Self::Belief),
            "hypothesis" => Some(Self::Hypothesis),
            "heuristic" => Some(Self::Heuristic),
            "procedure" => Some(Self::Procedure),
            "preference" => Some(Self::Preference),
            "profile" => Some(Self::Profile),
            "goal" => Some(Self::Goal),
            "skill" => Some(Self::Skill),
            "reflection" => Some(Self::Reflection),
            "evidence" => Some(Self::Evidence),
            "error" => Some(Self::Error),
            "meta" => Some(Self::Meta),
            _ => None,
        }
    }

    /// 是否可合并（CRDT mergeable）。
    pub fn mergeable(&self) -> bool {
        matches!(
            self,
            Self::Fact
                | Self::Belief
                | Self::Preference
                | Self::Profile
                | Self::Skill
                | Self::Procedure
                | Self::Heuristic
        )
    }
}

// ===========================================================================
// 认识论类型（EpistemicType — 五类知识属性）
// ===========================================================================

/// 认识论类型 — 知识的可信度属性。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicType {
    /// 可验证事实，需要证据支撑，可被 invalidate
    Fact,
    /// 主观信念，带个人置信度，可被用户修正
    Belief,
    /// 待验证假设，初始为叠加态，经验证后退相干
    Hypothesis,
    /// 经验法则，容忍模糊，质量分驱动
    Heuristic,
    /// 程序步骤，需要步骤验证
    Procedure,
}

// ===========================================================================
// 元数据类型
// ===========================================================================

/// 双时序有效期记录（）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityRecord {
    pub valid_from: chrono::DateTime<chrono::Utc>,
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
    pub invalidated_by: Option<crate::uri::ContextUri>,
    pub invalidation_reason: Option<String>,
}

/// Explicit knowledge decay horizon. `Infinite` is never scheduled for age-based
/// review; finite values are validated before entering persistence or APIs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HalfLife {
    Infinite,
    Finite { days: f64 },
}

impl HalfLife {
    pub const MAX_FINITE_DAYS: f64 = 365_000.0;

    pub fn finite(days: f64) -> Option<Self> {
        if days.is_finite() && days > 0.0 {
            Some(Self::Finite {
                days: days.min(Self::MAX_FINITE_DAYS),
            })
        } else {
            None
        }
    }
}

/// 巩固元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationMeta {
    pub source: String,
    pub generation: usize,
    pub status: ConsolidationStatus,
    pub patch_count: usize,
    pub lineage: Vec<LineageEntry>,
    pub evidence_uris: Vec<crate::uri::ContextUri>,
    pub corroboration: usize,
    pub half_life: Option<HalfLife>,
    pub entangled_with: Vec<crate::uri::ContextUri>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationStatus {
    Pending,
    InProgress,
    Converged,
    Stale,
}

/// 血统条目 — 版本演化链的一环。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEntry {
    pub version: MvccVersion,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub change_summary: String,
}

// ===========================================================================
// 派生链 — 记录 L0/L1 如何从 L2 派生
// ===========================================================================

/// 派生链 — L2 变更时自动触发 L0/L1 重算。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivationChain {
    pub full_source: ContentHash,
    pub dense_rule: DerivationRule,
    pub sparse_rule: DerivationRule,
    pub last_recomputed: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DerivationRule {
    Llm {
        prompt_template: String,
        model: String,
    },
    Extractive {
        algorithm: String,
    },
    Manual,
}

// ===========================================================================
// F.4 MetaKind — 统一内容类型、状态作用域与系统元数据
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaKind {
    Memory(ContentType),
    State(StateScope),
    System,
}

// State 作用域
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateScope {
    Short,
    Mid,
    Long,
}

// ===========================================================================
// ContextMeta（强类型化）
// ===========================================================================

/// 条目元数据 — 强类型字段替代 custom JSON。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextMeta {
    /// 内容类型（URI 路径段原生分类）。
    pub content_type: Option<ContentType>,
    pub state_scope: Option<StateScope>,
    pub tags: Vec<String>,

    // === 新增强类型字段 ===
    /// 认识论类型。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epistemic_type: Option<EpistemicType>,
    /// 质量分缓存。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f32>,
    /// 双时序有效期。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validity: Option<ValidityRecord>,
    /// 巩固元数据。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consolidation: Option<ConsolidationMeta>,

    /// 扩展字段（保留向后兼容）。
    #[serde(default)]
    pub custom: serde_json::Value,
}

impl ContextMeta {
    /// 获取认识论类型（从 content_type 推断）。
    pub fn epistemic_type(&self) -> Option<EpistemicType> {
        self.epistemic_type.or_else(|| {
            self.content_type.and_then(|ct| match ct {
                ContentType::Fact => Some(EpistemicType::Fact),
                ContentType::Belief => Some(EpistemicType::Belief),
                ContentType::Hypothesis => Some(EpistemicType::Hypothesis),
                ContentType::Heuristic => Some(EpistemicType::Heuristic),
                ContentType::Procedure => Some(EpistemicType::Procedure),
                _ => None,
            })
        })
    }

    /// 把 `custom` 反序列化为具体类型 `T`。
    ///
    /// 存储层始终保留 `serde_json::Value`，此方法只在应用层提供强类型访问，
    /// 避免因引入泛型污染 `ContextMeta` / `ContextEntry` 和所有窄端口 trait。
    pub fn custom_as<T>(&self) -> serde_json::Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        serde_json::from_value(self.custom.clone())
    }

    /// 用具体类型 `T` 覆写 `custom`。
    ///
    /// 如果序列化失败会返回错误并保持 `custom` 不变。
    pub fn set_custom<T>(&mut self, value: &T) -> serde_json::Result<()>
    where
        T: serde::Serialize,
    {
        let v = serde_json::to_value(value)?;
        self.custom = v;
        Ok(())
    }

    /// 便捷方法：读取 `custom` 对象的某个字段并反序列化为 `T`。
    ///
    /// 当 `custom` 不是 JSON 对象或缺少该字段时返回 `None`。
    pub fn custom_field<T>(&self, key: &str) -> Option<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.custom
            .as_object()
            .and_then(|obj| obj.get(key))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// 便捷方法：在 `custom` 对象中写入/覆盖一个字段。
    ///
    /// 若 `custom` 目前不是 JSON 对象，会先将其重置为空对象再写入。
    pub fn set_custom_field<T>(&mut self, key: &str, value: &T) -> serde_json::Result<()>
    where
        T: serde::Serialize,
    {
        let v = serde_json::to_value(value)?;
        if !self.custom.is_object() {
            self.custom = serde_json::Value::Object(serde_json::Map::new());
        }
        if let Some(obj) = self.custom.as_object_mut() {
            obj.insert(key.to_string(), v);
        }
        Ok(())
    }
}

// ===========================================================================
// ContextEntry（重写）
// ===========================================================================

/// 认知数据库核心条目 — 使用分层编码 ContentPayload。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntry {
    pub uri: crate::uri::ContextUri,
    pub tenant: TenantId,
    /// 分层编码内容（替代 l0_abstract / l1_overview / l2_detail_uri）。
    pub payload: ContentPayload,
    pub media_type: MediaType,
    pub metadata: ContextMeta,
    pub mvcc_version: MvccVersion,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// 派生链 — 记录 L0/L1 如何从 L2 派生。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derivation: Option<DerivationChain>,
}

impl ContextEntry {
    /// 校验 URI 中明确使用 UUID 的租户段与结构化租户 ID 一致。
    ///
    /// 历史 URI 允许使用租户 slug；只有 tenant 段本身是 UUID 时才具有可直接比较的身份语义。
    pub fn validate_tenant_binding(&self) -> crate::Result<()> {
        if let Ok(uri_tenant) = uuid::Uuid::parse_str(self.uri.tenant())
            && uri_tenant != self.tenant.0
        {
            return Err(crate::ContextError::InvalidUri(format!(
                "URI tenant {} does not match entry tenant {}",
                self.uri.tenant(),
                self.tenant.0
            )));
        }
        Ok(())
    }

    /// 构造一个最小 L0 文本条目（便于测试 / 快速写入）。
    pub fn new_text(
        uri: crate::uri::ContextUri,
        tenant: TenantId,
        abstract_: impl Into<String>,
    ) -> Self {
        let text = abstract_.into();
        let now = chrono::Utc::now();
        Self {
            uri,
            tenant,
            payload: ContentPayload::Text {
                sparse: text.clone(),
                dense: text.clone(),
                full: text,
            },
            media_type: MediaType::Text,
            metadata: ContextMeta::default(),
            mvcc_version: MvccVersion(0),
            created_at: now,
            updated_at: now,
            derivation: None,
        }
    }

    /// 获取 L0 摘要文本。
    pub fn l0_text(&self) -> &str {
        self.payload.sparse_text()
    }

    /// 获取 ContentType（从 metadata 中）。
    pub fn content_type(&self) -> Option<ContentType> {
        self.metadata.content_type
    }
}

#[cfg(test)]
mod context_meta_custom_tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Priority {
        level: String,
        score: u32,
    }

    #[test]
    fn set_and_get_typed_custom() {
        let mut meta = ContextMeta::default();
        let p = Priority {
            level: "high".into(),
            score: 9,
        };
        meta.set_custom(&p).unwrap();
        let back: Priority = meta.custom_as().unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn custom_as_fails_on_shape_mismatch() {
        let meta = ContextMeta {
            custom: serde_json::json!("not an object"),
            ..Default::default()
        };
        let r: serde_json::Result<Priority> = meta.custom_as();
        assert!(r.is_err());
    }

    #[test]
    fn set_and_get_custom_field() {
        let mut meta = ContextMeta::default();
        meta.set_custom_field("priority", &"high".to_string())
            .unwrap();
        meta.set_custom_field("score", &9u32).unwrap();

        let level: Option<String> = meta.custom_field("priority");
        let score: Option<u32> = meta.custom_field("score");
        let missing: Option<String> = meta.custom_field("nope");

        assert_eq!(level.as_deref(), Some("high"));
        assert_eq!(score, Some(9));
        assert!(missing.is_none());
    }

    #[test]
    fn set_custom_field_replaces_non_object() {
        let mut meta = ContextMeta {
            custom: serde_json::json!("was string"),
            ..Default::default()
        };
        meta.set_custom_field("k", &1u32).unwrap();
        assert_eq!(meta.custom_field::<u32>("k"), Some(1));
    }
}

// ===========================================================================
// FS 寻址返回类型
// ===========================================================================

/// 目录项（`ls` 返回）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub uri: crate::uri::ContextUri,
    pub is_dir: bool,
    /// L0 摘要。
    pub abstract_: String,
    /// 内容类型（从条目元数据中获取）。
    pub content_type: Option<ContentType>,
}

/// `find` 匹配模式。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FindPattern {
    pub scope: Option<crate::uri::ContextUri>,
    pub name_glob: Option<String>,
    /// 按 ContentType 过滤。
    pub content_type: Option<ContentType>,
    pub max_depth: Option<usize>,
}

/// `grep` 命中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepHit {
    pub uri: crate::uri::ContextUri,
    pub line: String,
    pub level: ContentLevel,
}

/// `tree` 返回的节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub uri: crate::uri::ContextUri,
    pub is_dir: bool,
    pub children: Vec<TreeNode>,
}

/// 版本历史条目（M0 占位）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    pub version: MvccVersion,
    pub message: String,
    pub ts: chrono::DateTime<chrono::Utc>,
}

/// 两版本间差异（M0 占位）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextDiff {
    pub summary: String,
}
