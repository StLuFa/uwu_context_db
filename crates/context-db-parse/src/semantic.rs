//! `SemanticProcessorImpl`：L0/L1 生成 + 自底向上聚合。
//!
//! 使用 `LlmClient` 为条目生成摘要（L0 ~100 tokens）和概览（L1 ~2k tokens）。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContextUri, FsOps, JsonSchema, LlmClient, LlmOpts, Result,
};
use async_trait::async_trait;
use moka::future::Cache;
use std::sync::Arc;
use std::time::Duration;

use crate::SemanticProcessor;

#[derive(Debug, Clone)]
pub struct SemanticProcessorCacheConfig {
    pub capacity: u64,
    pub ttl: Duration,
}

impl Default for SemanticProcessorCacheConfig {
    fn default() -> Self {
        Self {
            capacity: 100_000,
            ttl: Duration::from_secs(60 * 60 * 24 * 7),
        }
    }
}

/// 基于 `LlmClient` 的语义处理器实现。
pub struct SemanticProcessorImpl {
    llm: Arc<dyn LlmClient>,
    fs: Arc<dyn FsOps>,
    summaries: Cache<String, String>,
}

impl SemanticProcessorImpl {
    pub fn new(llm: Arc<dyn LlmClient>, fs: Arc<dyn FsOps>) -> Self {
        Self::with_cache_config(llm, fs, SemanticProcessorCacheConfig::default())
    }

    pub fn with_cache_config(
        llm: Arc<dyn LlmClient>,
        fs: Arc<dyn FsOps>,
        config: SemanticProcessorCacheConfig,
    ) -> Self {
        Self {
            llm,
            fs,
            summaries: Cache::builder()
                .max_capacity(config.capacity.max(1))
                .time_to_live(config.ttl)
                .build(),
        }
    }
}

#[async_trait]
impl SemanticProcessor for SemanticProcessorImpl {
    async fn generate_abstract(&self, uri: &ContextUri) -> Result<String> {
        let payload = self.fs.read(uri, ContentLevel::L2).await?;
        let content = strongest_text(&payload);
        let key = summary_cache_key("abstract", uri, &content);
        if let Some(cached) = self.summaries.get(&key).await {
            return Ok(cached);
        }

        let prompt = format!(
            r#"You are a context summarizer. Write a concise L0 abstract (~100 tokens) for:
URI: {uri}

Content:
{content}

An abstract should capture: what this entry is about, its category, and key information.
Respond with ONLY the abstract text, no additional commentary.
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(150),
            temperature: Some(0.1),
            ..Default::default()
        };

        let summary = self
            .llm
            .complete(&prompt, &opts)
            .await
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                agent_context_db_core::ContextError::Storage(format!("llm generate_abstract: {e}"))
            })?;
        self.summaries.insert(key, summary.clone()).await;
        Ok(summary)
    }

    async fn generate_overview(&self, uri: &ContextUri) -> Result<String> {
        let payload = self.fs.read(uri, ContentLevel::L2).await?;
        let content = strongest_text(&payload);
        let key = summary_cache_key("overview", uri, &content);
        if let Some(cached) = self.summaries.get(&key).await {
            return Ok(cached);
        }

        let prompt = format!(
            r#"You are a context organizer. Write an L1 overview (~1000 tokens) for:
URI: {uri}

Content:
{content}

An overview should include:
1. A structured table of contents with sections
2. Key concepts and their relationships
3. Navigation hints for related entries

Format as Markdown with ## section headers.
Respond with ONLY the overview text.
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(1500),
            temperature: Some(0.2),
            ..Default::default()
        };

        let overview = self
            .llm
            .complete(&prompt, &opts)
            .await
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                agent_context_db_core::ContextError::Storage(format!("llm generate_overview: {e}"))
            })?;
        self.summaries.insert(key, overview.clone()).await;
        Ok(overview)
    }

    async fn aggregate_upward(&self, root: &ContextUri) -> Result<String> {
        // 自底向上聚合：
        // 1. 遍历 root 的直接子条目（文件），读取其 L0 摘要
        // 2. 遍历 root 的直接子目录，递归收集摘要
        // 3. 将所有子摘要合成父目录的 L1 概览
        // 4. 返回生成的 L1 概览，由调用方决定是否写入存储

        let entries = self
            .fs
            .ls(root, agent_context_db_core::PageRequest::default())
            .await?;

        let mut child_abstracts: Vec<(ContextUri, String)> = Vec::new();

        for entry in &entries {
            match self.fs.read(&entry.uri, ContentLevel::L0).await {
                Ok(payload) => {
                    let abs = payload.sparse_text().to_string();
                    if !abs.is_empty() {
                        child_abstracts.push((entry.uri.clone(), abs));
                    }
                }
                Err(_) => {
                    // 目录：递归聚合
                    if entry.is_dir {
                        // 递归聚合子目录
                        match Box::pin(self.aggregate_upward(&entry.uri)).await {
                            Ok(overview) => {
                                child_abstracts.push((entry.uri.clone(), overview));
                            }
                            Err(_) => {
                                // 子目录聚合失败，跳过
                            }
                        }
                    }
                }
            }
        }

        if child_abstracts.is_empty() {
            return Ok(format!("(empty directory: {root})"));
        }

        // 构建 LLM 合成提示
        let children_text: Vec<String> = child_abstracts
            .iter()
            .map(|(uri, abs)| format!("- {uri}: {abs}"))
            .collect();
        let joined = children_text.join("\n");
        let key = summary_cache_key("aggregate", root, &joined);
        if let Some(cached) = self.summaries.get(&key).await {
            return Ok(cached);
        }

        let prompt = format!(
            r#"You are a context aggregator. Synthesize an L1 overview for directory:
URI: {root}

Its child entries:
{joined}

Write a structured overview (~1000 tokens) that:
1. Groups related children under ## section headers
2. Highlights cross-references and relationships between entries
3. Provides a table-of-contents style navigation summary

Format as Markdown. Respond with ONLY the overview text.
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(1500),
            temperature: Some(0.2),
            ..Default::default()
        };

        let overview = self
            .llm
            .complete(&prompt, &opts)
            .await
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                agent_context_db_core::ContextError::Storage(format!("llm aggregate_upward: {e}"))
            })?;
        self.summaries.insert(key, overview.clone()).await;
        Ok(overview)
    }

    async fn multimodal_to_text(&self, uri: &ContextUri) -> Result<(String, String)> {
        // 多模态转文本：读取 L2 Detail 原始字节，调用 LLM 描述为文本
        match self.fs.read(uri, ContentLevel::L2).await {
            Ok(ContentPayload::Text { full, .. }) if !full.is_empty() => {
                let bytes = full.as_bytes().to_vec();
                // 尝试检测内容类型并转 base64 描述
                let content_hint = detect_content_type(&bytes);

                let prompt = format!(
                    r#"Describe this {content_hint} content located at URI: {uri}

The content is provided as raw bytes of length {len}.

Generate TWO outputs:
1. A concise L0 abstract (~100 tokens) describing what this content is
2. A detailed L1 overview (~1000 tokens) describing key elements, structure, and meaning

Return your response as a JSON object:
{{"abstract": "...", "overview": "..."}}
"#,
                    content_hint = content_hint,
                    len = bytes.len()
                );

                let opts = LlmOpts {
                    max_tokens: Some(1500),
                    temperature: Some(0.2),
                    ..Default::default()
                };

                let response = self
                    .llm
                    .complete_json(&prompt, &multimodal_result_schema(), &opts)
                    .await
                    .map_err(|e| {
                        agent_context_db_core::ContextError::Storage(format!(
                            "llm multimodal_to_text: {e}"
                        ))
                    })?;
                let result: MultimodalResult = serde_json::from_str(&response).map_err(|e| {
                    agent_context_db_core::ContextError::Storage(format!(
                        "llm multimodal_to_text returned invalid structured response: {e}"
                    ))
                })?;
                if result.abstract_text.trim().is_empty() || result.overview.trim().is_empty() {
                    return Err(agent_context_db_core::ContextError::Storage(
                        "llm multimodal_to_text returned empty abstract or overview".into(),
                    ));
                }
                Ok((result.abstract_text, result.overview))
            }
            Ok(ContentPayload::Text { sparse, dense, .. }) => {
                Ok((sparse.chars().take(200).collect(), dense))
            }
            Ok(_) => Err(agent_context_db_core::ContextError::Unsupported(format!(
                "multimodal_to_text requires L2 Detail content for {uri}"
            ))),
            Err(e) => Err(e),
        }
    }
}

// ===========================================================================
// 辅助函数
// ===========================================================================

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MultimodalResult {
    #[serde(rename = "abstract")]
    abstract_text: String,
    overview: String,
}

fn multimodal_result_schema() -> JsonSchema {
    JsonSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "abstract": { "type": "string", "minLength": 1 },
            "overview": { "type": "string", "minLength": 1 }
        },
        "required": ["abstract", "overview"],
        "additionalProperties": false
    }))
}

fn summary_cache_key(kind: &str, uri: &ContextUri, content: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(kind.as_bytes());
    hasher.update(b"\0");
    hasher.update(uri.to_string().as_bytes());
    hasher.update(b"\0");
    hasher.update(content.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn strongest_text(payload: &ContentPayload) -> String {
    match payload {
        ContentPayload::Text {
            sparse,
            dense,
            full,
        } => {
            if !full.trim().is_empty() {
                full.clone()
            } else if !dense.trim().is_empty() {
                dense.clone()
            } else {
                sparse.clone()
            }
        }
        ContentPayload::Image { .. } => "[image]".into(),
        ContentPayload::Audio { transcript, .. } => transcript.clone(),
        ContentPayload::Structured { summary, data, .. } => format!("{summary}\n{data}"),
        ContentPayload::Composite { summary, .. } => summary.clone(),
    }
}

/// 检测字节内容的基本类型提示。
fn detect_content_type(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 4 {
        // PNG magic
        if &bytes[..4] == b"\x89PNG" {
            return "PNG image";
        }
        // JPEG magic
        if &bytes[..2] == b"\xff\xd8" {
            return "JPEG image";
        }
        // GIF magic
        if &bytes[..3] == b"GIF" {
            return "GIF image";
        }
        // WebP
        if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            return "WebP image";
        }
        // PDF
        if &bytes[..4] == b"%PDF" {
            return "PDF document";
        }
        // WAV audio
        if &bytes[..4] == b"RIFF" && bytes.len() >= 12 && &bytes[8..12] == b"WAVE" {
            return "WAV audio";
        }
        // MP4 video
        if bytes.len() >= 12 {
            // ftyp box
            if &bytes[4..8] == b"ftyp" {
                return "MP4 video";
            }
        }
    }
    // 尝试 UTF-8 检测
    if std::str::from_utf8(bytes).is_ok() {
        return "text";
    }
    "binary"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_png_magic() {
        let png = b"\x89PNG\r\n\x1a\n";
        assert_eq!(detect_content_type(png), "PNG image");
    }

    #[test]
    fn detect_jpeg_magic() {
        let jpg = b"\xff\xd8\xff\xe0";
        assert_eq!(detect_content_type(jpg), "JPEG image");
    }

    #[test]
    fn detect_text_fallback() {
        assert_eq!(detect_content_type(b"hello world"), "text");
    }

    #[test]
    fn summary_cache_key_changes_when_content_changes() {
        let uri = ContextUri::parse("uwu://t/agent/a/fact/x").unwrap();
        assert_ne!(
            summary_cache_key("abstract", &uri, "old content"),
            summary_cache_key("abstract", &uri, "new content")
        );
    }

    #[test]
    fn strongest_text_prefers_full_payload() {
        let payload = ContentPayload::Text {
            sparse: "sparse".into(),
            dense: "dense".into(),
            full: "full".into(),
        };
        assert_eq!(strongest_text(&payload), "full");
    }

    #[test]
    fn multimodal_contract_uses_abstract_json_name() {
        let result: MultimodalResult =
            serde_json::from_str(r#"{"abstract":"concise","overview":"detailed"}"#).unwrap();
        assert_eq!(result.abstract_text, "concise");
        assert_eq!(result.overview, "detailed");

        let schema = multimodal_result_schema().schema;
        assert!(schema["properties"].get("abstract").is_some());
        assert!(schema["properties"].get("abstract_").is_none());
        assert_eq!(
            schema["required"],
            serde_json::json!(["abstract", "overview"])
        );
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn multimodal_contract_rejects_legacy_or_incomplete_payloads() {
        assert!(
            serde_json::from_str::<MultimodalResult>(
                r#"{"abstract_":"legacy","overview":"detailed"}"#
            )
            .is_err()
        );
        assert!(serde_json::from_str::<MultimodalResult>(r#"{"abstract":"concise"}"#).is_err());
    }
}
