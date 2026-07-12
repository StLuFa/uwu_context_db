//! Strongly typed multimodal embedding port and embedding-space identity.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::{BlobRef, ContextError, LlmClient, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingNormalization {
    None,
    L2,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingSpaceId {
    pub model: String,
    pub checkpoint: String,
    pub preprocess: String,
    pub dim: usize,
    pub normalization: EmbeddingNormalization,
}

impl EmbeddingSpaceId {
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty()
            || self.checkpoint.trim().is_empty()
            || self.preprocess.trim().is_empty()
            || self.dim == 0
        {
            return Err(ContextError::Unsupported(
                "invalid embedding space identity".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodedEmbedding {
    pub space: EmbeddingSpaceId,
    pub values: Vec<f32>,
}

impl EncodedEmbedding {
    pub fn new(space: EmbeddingSpaceId, values: Vec<f32>) -> Result<Self> {
        space.validate()?;
        if values.len() != space.dim || values.iter().any(|v| !v.is_finite()) {
            return Err(ContextError::Unsupported(format!(
                "invalid embedding: expected {} finite values, got {}",
                space.dim,
                values.len()
            )));
        }
        if space.normalization == EmbeddingNormalization::L2 {
            let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-3 {
                return Err(ContextError::Unsupported(format!(
                    "embedding is not L2 normalized (norm={norm})"
                )));
            }
        }
        Ok(Self { space, values })
    }

    pub fn ensure_space(&self, expected: &EmbeddingSpaceId) -> Result<()> {
        if &self.space != expected {
            return Err(ContextError::Unsupported("embedding space mismatch".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
pub enum EmbeddingInput {
    Text { text: String },
    Image { bytes: Vec<u8>, mime_type: String },
    Audio { bytes: Vec<u8>, mime_type: String },
}

impl EmbeddingInput {
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text { .. } => Modality::Text,
            Self::Image { .. } => Modality::Image,
            Self::Audio { .. } => Modality::Audio,
        }
    }
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Text { text } if text.is_empty() => Err(ContextError::Unsupported(
                "empty text embedding input".into(),
            )),
            Self::Image { bytes, mime_type }
                if bytes.is_empty() || !mime_type.starts_with("image/") =>
            {
                Err(ContextError::Unsupported(
                    "invalid image bytes or mime type".into(),
                ))
            }
            Self::Audio { bytes, mime_type }
                if bytes.is_empty() || !mime_type.starts_with("audio/") =>
            {
                Err(ContextError::Unsupported(
                    "invalid audio bytes or mime type".into(),
                ))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderCapabilities {
    pub modalities: BTreeSet<Modality>,
    pub batch: bool,
    pub max_batch_size: usize,
}

#[async_trait]
pub trait MultimodalEncoder: Send + Sync {
    fn space(&self) -> &EmbeddingSpaceId;
    fn capabilities(&self) -> EncoderCapabilities;
    async fn encode_batch(&self, inputs: &[EmbeddingInput]) -> Result<Vec<EncodedEmbedding>>;
    async fn encode(&self, input: EmbeddingInput) -> Result<EncodedEmbedding> {
        let mut values = self.encode_batch(&[input]).await?;
        if values.len() != 1 {
            return Err(ContextError::Unsupported(
                "encoder returned incorrect result count".into(),
            ));
        }
        Ok(values.remove(0))
    }
}

/// Adapts the existing LLM text embedding API without pretending it supports binary modalities.
pub struct LlmTextEncoder<L> {
    llm: L,
    space: EmbeddingSpaceId,
}
impl<L> LlmTextEncoder<L> {
    pub fn new(llm: L, space: EmbeddingSpaceId) -> Result<Self> {
        space.validate()?;
        Ok(Self { llm, space })
    }
}
#[async_trait]
impl<L: LlmClient> MultimodalEncoder for LlmTextEncoder<L> {
    fn space(&self) -> &EmbeddingSpaceId {
        &self.space
    }
    fn capabilities(&self) -> EncoderCapabilities {
        EncoderCapabilities {
            modalities: [Modality::Text].into_iter().collect(),
            batch: true,
            max_batch_size: usize::MAX,
        }
    }
    async fn encode_batch(&self, inputs: &[EmbeddingInput]) -> Result<Vec<EncodedEmbedding>> {
        let texts: Vec<String> = inputs
            .iter()
            .map(|input| match input {
                EmbeddingInput::Text { text } => Ok(text.clone()),
                _ => Err(ContextError::Unsupported(
                    "LLM text adapter only supports text".into(),
                )),
            })
            .collect::<Result<_>>()?;
        let vectors = self.llm.embed_batch(&texts).await?;
        if vectors.len() != inputs.len() {
            return Err(ContextError::Unsupported(
                "LLM returned incorrect embedding count".into(),
            ));
        }
        vectors
            .into_iter()
            .map(|v| EncodedEmbedding::new(self.space.clone(), v.vector))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedEmbedding {
    pub modality: Modality,
    pub embedding: EncodedEmbedding,
    pub source: BlobRef,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingVector, JsonSchema, LlmError, LlmOpts};

    struct MemoryLlm;
    #[async_trait]
    impl LlmClient for MemoryLlm {
        async fn complete(&self, _: &str, _: &LlmOpts) -> std::result::Result<String, LlmError> {
            Ok(String::new())
        }
        async fn complete_json(
            &self,
            _: &str,
            _: &JsonSchema,
            _: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }
        async fn embed(&self, text: &str) -> std::result::Result<EmbeddingVector, LlmError> {
            Ok(EmbeddingVector::new(
                vec![text.len() as f32, 1.0],
                "memory",
                1,
            ))
        }
        async fn embed_batch(
            &self,
            texts: &[String],
        ) -> std::result::Result<Vec<EmbeddingVector>, LlmError> {
            Ok(texts
                .iter()
                .map(|text| EmbeddingVector::new(vec![text.len() as f32, 1.0], "memory", 1))
                .collect())
        }
    }

    fn space() -> EmbeddingSpaceId {
        EmbeddingSpaceId {
            model: "memory".into(),
            checkpoint: "v1".into(),
            preprocess: "utf8-v1".into(),
            dim: 2,
            normalization: EmbeddingNormalization::None,
        }
    }

    #[tokio::test]
    async fn memory_llm_text_encoder_batches_and_rejects_binary_modalities() {
        let encoder = LlmTextEncoder::new(MemoryLlm, space()).unwrap();
        let values = encoder
            .encode_batch(&[
                EmbeddingInput::Text { text: "a".into() },
                EmbeddingInput::Text {
                    text: "abcd".into(),
                },
            ])
            .await
            .unwrap();
        assert_eq!(values[0].values, vec![1.0, 1.0]);
        assert_eq!(values[1].values, vec![4.0, 1.0]);
        assert!(
            encoder
                .encode(EmbeddingInput::Image {
                    bytes: vec![0x89, b'P', b'N', b'G'],
                    mime_type: "image/png".into(),
                })
                .await
                .is_err()
        );
    }
}
