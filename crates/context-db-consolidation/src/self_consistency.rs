//! Self-consistency voting for consolidation products.
//!
//! 对高价值或高不确定条目进行多次采样，把结论按语义近似聚类后采用多数派；
//! 票型分散会降低置信度，便于后续写回 ConfidenceCalibrator 或主动核验。

use std::collections::HashSet;
use std::sync::Arc;

use agent_context_db_core::{
    ConsolidationStatus, ContentType, ContextEntry, EpistemicType, LlmClient, LlmOpts, LlmTaskKind,
    PromptOptimization,
};
use serde::{Deserialize, Serialize};

use crate::{ConsolidationProduct, ConsolidationProductMeta};

#[derive(Debug, Clone)]
pub struct SelfConsistencyConfig {
    pub samples: usize,
    /// Minimum number of non-empty provider responses required for consensus.
    pub min_valid_samples: usize,
    pub min_majority_ratio: f32,
    pub similarity_threshold: f32,
    pub base_temperature: f32,
    pub max_tokens: u32,
}

impl SelfConsistencyConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.samples == 0 || self.min_valid_samples == 0 || self.max_tokens == 0 {
            return Err(crate::ConfigError(
                "samples, min_valid_samples, and max_tokens must be nonzero".into(),
            ));
        }
        if self.min_valid_samples > self.samples {
            return Err(crate::ConfigError(
                "min_valid_samples must not exceed samples".into(),
            ));
        }
        crate::validate_unit_f32("min_majority_ratio", self.min_majority_ratio)?;
        crate::validate_unit_f32("similarity_threshold", self.similarity_threshold)?;
        if !self.base_temperature.is_finite() || self.base_temperature < 0.0 {
            return Err(crate::ConfigError(
                "base_temperature must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}

impl Default for SelfConsistencyConfig {
    fn default() -> Self {
        Self {
            samples: 5,
            min_valid_samples: 3,
            min_majority_ratio: 0.55,
            similarity_threshold: 0.62,
            base_temperature: 0.75,
            max_tokens: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyVoteCluster {
    pub representative: String,
    pub votes: usize,
    pub members: Vec<String>,
    pub mean_similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfConsistencyReport {
    /// Number requested from the provider.
    pub requested_samples: usize,
    /// Number of non-empty responses actually returned.
    pub sample_count: usize,
    /// Fraction of requested samples that produced usable responses.
    pub completeness: f32,
    pub majority_votes: usize,
    pub majority_ratio: f32,
    pub vote_entropy: f32,
    pub confidence_variance: f32,
    pub accepted: bool,
    pub clusters: Vec<ConsistencyVoteCluster>,
}

pub struct SelfConsistencyConsolidator {
    llm: Arc<dyn LlmClient>,
    config: SelfConsistencyConfig,
}

impl SelfConsistencyConsolidator {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        config: SelfConsistencyConfig,
    ) -> Result<Self, crate::ConfigError> {
        config.validate()?;
        Ok(Self { llm, config })
    }

    pub async fn consolidate(
        &self,
        entry: &ContextEntry,
    ) -> (ConsolidationProduct, SelfConsistencyReport) {
        let content = entry.l0_text().to_string();
        let content_type = entry.content_type().unwrap_or(ContentType::Fact);
        let epistemic_type = entry
            .metadata
            .epistemic_type()
            .unwrap_or(EpistemicType::Fact);
        let prompts = (0..self.config.samples.max(1))
            .map(|sample| self.sample_prompt(&content, content_type, epistemic_type, sample))
            .collect::<Vec<_>>();
        let opts = LlmOpts {
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.base_temperature),
            task: LlmTaskKind::Synthesis,
            prompt: PromptOptimization::default().target_tokens(1_200),
            ..Default::default()
        };
        let requested_samples = prompts.len();
        let samples = match self.llm.batch_complete(&prompts, &opts).await {
            Ok(values) => values
                .into_iter()
                .take(requested_samples)
                .map(|value| normalize_answer(&value))
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>(),
            // A provider failure is zero evidence. Falling back to the input as
            // a vote would let one synthetic sample masquerade as consensus.
            Err(_) => Vec::new(),
        };
        let clusters = cluster_samples(&samples, self.config.similarity_threshold);
        let report = report_from_clusters(
            &clusters,
            requested_samples,
            samples.len(),
            self.config.min_valid_samples,
            self.config.min_majority_ratio,
        );
        let winner = clusters
            .first()
            .map(|cluster| cluster.representative.clone())
            .unwrap_or(content);
        let majority_ratio = report.majority_ratio;
        let completeness = report.completeness;
        let calibrated_confidence = if report.accepted {
            (entry.metadata.quality_score.unwrap_or(0.5) * 0.35
                + majority_ratio * 0.45
                + (1.0 - report.confidence_variance).clamp(0.0, 1.0) * 0.20)
                .clamp(0.0, 1.0)
                * completeness
        } else {
            (majority_ratio * completeness * 0.55).clamp(0.0, 0.55)
        };
        let quality_score = (entry.metadata.quality_score.unwrap_or(0.5) * 0.5
            + majority_ratio * 0.35
            + (1.0 - report.vote_entropy).clamp(0.0, 1.0) * 0.15)
            .clamp(0.0, 1.0);

        let product = ConsolidationProduct {
            uri: entry.uri.clone(),
            content_type,
            epistemic_type,
            content: winner,
            quality_score,
            confidence: calibrated_confidence,
            evidence_required: false,
            superseded_claim: None,
            evidence_uris: vec![],
            contradiction_uris: if report.accepted {
                vec![]
            } else {
                vec![entry.uri.clone()]
            },
            error_pattern: None,
            hypothesis_outcome: None,
            preconditions: None,
            expected_outcome: None,
            related_policy_uris: vec![],
            provenance: None,
            metadata: ConsolidationProductMeta {
                source_session: None,
                generation: report.sample_count,
                status: if report.accepted {
                    ConsolidationStatus::Converged
                } else {
                    ConsolidationStatus::Stale
                },
                patch_count: report.clusters.len(),
                lineage: vec![],
                validity: entry.metadata.validity.clone(),
                half_life: None,
            },
        };
        (product, report)
    }

    fn sample_prompt(
        &self,
        content: &str,
        content_type: ContentType,
        epistemic_type: EpistemicType,
        sample: usize,
    ) -> String {
        format!(
            r#"Consolidate this memory into one precise, self-contained principle.

Memory: "{content}"
Content type: {content_type:?}
Epistemic type: {epistemic_type:?}
Sample path: {sample}

Vary the reasoning path, but keep the output factual and evidence-grounded.
Return only the principle text, 1-3 sentences, no JSON and no markdown."#
        )
    }
}

fn report_from_clusters(
    clusters: &[ConsistencyVoteCluster],
    requested_samples: usize,
    sample_count: usize,
    min_valid_samples: usize,
    min_majority_ratio: f32,
) -> SelfConsistencyReport {
    let majority_votes = clusters.first().map(|c| c.votes).unwrap_or(0);
    // Missing responses count against consensus; provider completeness is part
    // of the evidence contract, rather than an implementation detail.
    let majority_ratio = majority_votes as f32 / requested_samples.max(1) as f32;
    let completeness = sample_count as f32 / requested_samples.max(1) as f32;
    let vote_entropy = entropy(clusters, sample_count);
    let confidence_variance = clusters
        .first()
        .map(|cluster| 1.0 - cluster.mean_similarity)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    SelfConsistencyReport {
        requested_samples,
        sample_count,
        completeness,
        majority_votes,
        majority_ratio,
        vote_entropy,
        confidence_variance,
        accepted: sample_count >= min_valid_samples.max(2) && majority_ratio >= min_majority_ratio,
        clusters: clusters.to_vec(),
    }
}

fn cluster_samples(samples: &[String], threshold: f32) -> Vec<ConsistencyVoteCluster> {
    let mut clusters: Vec<ConsistencyVoteCluster> = Vec::new();
    for sample in samples {
        if let Some(cluster) = clusters
            .iter_mut()
            .max_by(|a, b| {
                similarity(sample, &a.representative)
                    .total_cmp(&similarity(sample, &b.representative))
            })
            .filter(|cluster| similarity(sample, &cluster.representative) >= threshold)
        {
            cluster.members.push(sample.clone());
            cluster.votes += 1;
            cluster.mean_similarity = cluster
                .members
                .iter()
                .map(|member| similarity(member, &cluster.representative))
                .sum::<f32>()
                / cluster.members.len().max(1) as f32;
        } else {
            clusters.push(ConsistencyVoteCluster {
                representative: sample.clone(),
                votes: 1,
                members: vec![sample.clone()],
                mean_similarity: 1.0,
            });
        }
    }
    clusters.sort_by(|a, b| {
        b.votes
            .cmp(&a.votes)
            .then_with(|| b.mean_similarity.total_cmp(&a.mean_similarity))
    });
    clusters
}

fn entropy(clusters: &[ConsistencyVoteCluster], sample_count: usize) -> f32 {
    if sample_count <= 1 {
        return 0.0;
    }
    let total = sample_count as f32;
    let raw = clusters
        .iter()
        .map(|cluster| {
            let p = cluster.votes as f32 / total;
            if p <= 0.0 { 0.0 } else { -p * p.ln() }
        })
        .sum::<f32>();
    let max_entropy = (sample_count as f32).ln().max(1e-6);
    (raw / max_entropy).clamp(0.0, 1.0)
}

fn similarity(left: &str, right: &str) -> f32 {
    let left_tokens = tokens(left);
    let right_tokens = tokens(right);
    if left_tokens.is_empty() && right_tokens.is_empty() {
        return 1.0;
    }
    let intersection = left_tokens.intersection(&right_tokens).count() as f32;
    let union = left_tokens.union(&right_tokens).count() as f32;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn tokens(value: &str) -> HashSet<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| part.len() > 2)
        .map(|part| part.to_ascii_lowercase())
        .collect()
}

fn normalize_answer(value: &str) -> String {
    value
        .trim()
        .trim_matches('`')
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContextEntry, ContextUri, LlmError, TenantId};
    use async_trait::async_trait;
    use uuid::Uuid;

    struct VotingLlm;

    #[async_trait]
    impl LlmClient for VotingLlm {
        async fn complete(&self, _prompt: &str, _opts: &LlmOpts) -> Result<String, LlmError> {
            Ok("Use bounded graph traversal for relationship recall".into())
        }

        async fn batch_complete(
            &self,
            prompts: &[String],
            _opts: &LlmOpts,
        ) -> Result<Vec<String>, LlmError> {
            Ok(prompts
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    if idx < 4 {
                        "Use bounded graph traversal for relationship recall".to_string()
                    } else {
                        "Prefer unrelated cache eviction heuristics".to_string()
                    }
                })
                .collect())
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &agent_context_db_core::JsonSchema,
            _opts: &LlmOpts,
        ) -> Result<String, LlmError> {
            Ok("{}".into())
        }
    }

    fn entry() -> ContextEntry {
        let mut entry = ContextEntry::new_text(
            ContextUri::parse("uwu://tenant/agent/a/memory/fact/topic/entry").unwrap(),
            TenantId(Uuid::nil()),
            "relationship recall should use bounded graph traversal",
        );
        entry.metadata.content_type = Some(ContentType::Fact);
        entry.metadata.quality_score = Some(0.7);
        entry
    }

    struct PartialLlm;

    #[async_trait]
    impl LlmClient for PartialLlm {
        async fn complete(&self, _prompt: &str, _opts: &LlmOpts) -> Result<String, LlmError> {
            unreachable!()
        }

        async fn batch_complete(
            &self,
            _prompts: &[String],
            _opts: &LlmOpts,
        ) -> Result<Vec<String>, LlmError> {
            Ok(vec!["one apparently unanimous answer".into()])
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &agent_context_db_core::JsonSchema,
            _opts: &LlmOpts,
        ) -> Result<String, LlmError> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn partial_single_response_cannot_masquerade_as_consensus() {
        let consolidator = SelfConsistencyConsolidator::new(
            Arc::new(PartialLlm),
            SelfConsistencyConfig::default(),
        )
        .unwrap();
        let (product, report) = consolidator.consolidate(&entry()).await;

        assert_eq!(report.requested_samples, 5);
        assert_eq!(report.sample_count, 1);
        assert_eq!(report.completeness, 0.2);
        assert_eq!(report.majority_ratio, 0.2);
        assert!(!report.accepted);
        assert!(product.confidence < 0.1);
        assert_eq!(product.metadata.status, ConsolidationStatus::Stale);
    }

    #[tokio::test]
    async fn self_consistency_accepts_majority_cluster_and_calibrates_confidence() {
        let consolidator = SelfConsistencyConsolidator::new(
            Arc::new(VotingLlm),
            SelfConsistencyConfig {
                samples: 5,
                min_majority_ratio: 0.6,
                similarity_threshold: 0.5,
                ..Default::default()
            },
        )
        .unwrap();
        let (product, report) = consolidator.consolidate(&entry()).await;

        assert!(report.accepted);
        assert_eq!(report.majority_votes, 4);
        assert!(product.content.contains("bounded graph traversal"));
        assert!(product.confidence > 0.6);
        assert_eq!(product.metadata.status, ConsolidationStatus::Converged);
    }
}
