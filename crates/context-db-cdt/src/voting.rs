//! 投票演化 — Insight 持续投票，低分自动淘汰。

use crate::reflection::SemanticGradient;
use agent_context_db_core::{ContentType, ContextUri};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// 可演化 Insight。
#[derive(Debug, Clone)]
pub struct EvolvableInsight {
    pub uri: ContextUri,
    pub content: String,
    pub epistemic_type: ContentType,
    pub votes: VoteRecord,
    pub last_updated: DateTime<Utc>,
    pub evidence: Vec<String>,
    pub generation: usize,
}

#[derive(Debug, Clone, Default)]
pub struct VoteRecord {
    pub upvotes: Vec<Vote>,
    pub downvotes: Vec<Vote>,
    pub net_score: f32,
}

#[derive(Debug, Clone)]
pub struct Vote {
    pub voter_uri: ContextUri,
    pub weight: f32,
    pub timestamp: DateTime<Utc>,
    pub evidence: Option<String>,
}

/// 投票操作。
pub enum VoteOp {
    Add(EvolvableInsight),
    Upvote(ContextUri, Vote),
    Downvote(ContextUri, Vote),
    Edit(ContextUri, String),
}

/// 一轮 ExpeL 风格演化的结果。
#[derive(Debug, Clone, Default)]
pub struct EvolutionReport {
    pub added: usize,
    pub merged: usize,
    pub deprecated: usize,
    pub surviving: usize,
}

impl EvolvableInsight {
    pub fn new(uri: ContextUri, content: String, epistemic_type: ContentType) -> Self {
        Self {
            uri,
            content,
            epistemic_type,
            votes: VoteRecord::default(),
            last_updated: Utc::now(),
            evidence: vec![],
            generation: 0,
        }
    }

    /// 从 Reflexion 语义梯度生成可演化 insight。
    pub fn from_gradient(index: usize, gradient: &SemanticGradient) -> Self {
        let uri = gradient.source_uri.clone().unwrap_or_else(|| {
            ContextUri::parse(format!(
                "uwu://t/agent/a/memories/reflection/generated-{index}"
            ))
            .expect("generated reflection URI must parse")
        });
        let mut insight = Self::new(
            uri,
            format!(
                "{}\nACTION: {}",
                gradient.reflection_text, gradient.action_improvement
            ),
            gradient.error_type,
        );
        insight.evidence = gradient.epistemic_tags.clone();
        insight.votes.upvotes.push(Vote {
            voter_uri: ContextUri::parse("uwu://t/agent/cdt/reflexion").unwrap(),
            weight: gradient.priority.clamp(0.0, 1.0),
            timestamp: Utc::now(),
            evidence: Some("generated from semantic gradient".into()),
        });
        insight.recompute_score();
        insight
    }

    /// 加权净分 = Σ(upvote.weight) - Σ(downvote.weight)。
    pub fn recompute_score(&mut self) {
        let up: f32 = self.votes.upvotes.iter().map(|v| v.weight).sum();
        let down: f32 = self.votes.downvotes.iter().map(|v| v.weight).sum();
        self.votes.net_score = up - down;
    }

    /// 净分 ≤ 0 → 淘汰。
    pub fn should_deprecate(&self) -> bool {
        self.votes.net_score <= 0.0
    }

    pub fn similarity_key(&self) -> String {
        normalize_key(&self.content)
    }
}

/// 投票演化引擎。
pub struct InsightEvolutionEngine {
    deprecate_threshold: f32,
    merge_threshold: f32,
}

impl Default for InsightEvolutionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InsightEvolutionEngine {
    pub fn new() -> Self {
        Self {
            deprecate_threshold: 0.0,
            merge_threshold: 0.72,
        }
    }

    pub fn with_thresholds(mut self, deprecate_threshold: f32, merge_threshold: f32) -> Self {
        self.deprecate_threshold = deprecate_threshold;
        self.merge_threshold = merge_threshold.clamp(0.0, 1.0);
        self
    }

    pub fn vote(&self, insight: &mut EvolvableInsight, op: VoteOp) {
        match op {
            VoteOp::Add(i) => *insight = i,
            VoteOp::Upvote(target, v) if target == insight.uri => {
                insight.votes.upvotes.push(v);
                insight.recompute_score();
            }
            VoteOp::Downvote(target, v) if target == insight.uri => {
                insight.votes.downvotes.push(v);
                insight.recompute_score();
            }
            VoteOp::Edit(target, new) if target == insight.uri => {
                insight.content = new;
                insight.generation += 1;
                insight.last_updated = Utc::now();
            }
            _ => {}
        }
    }

    /// ExpeL 风格：从新反思生成候选、合并相似 insight、按投票淘汰低分候选。
    pub fn evolve_from_gradients(
        &self,
        insights: &mut Vec<EvolvableInsight>,
        gradients: &[SemanticGradient],
    ) -> EvolutionReport {
        let before = insights.len();
        let mut report = EvolutionReport::default();
        let mut index: HashMap<String, usize> = insights
            .iter()
            .enumerate()
            .map(|(idx, insight)| (insight.similarity_key(), idx))
            .collect();

        for (i, gradient) in gradients.iter().enumerate() {
            let candidate = EvolvableInsight::from_gradient(i, gradient);
            if let Some(existing_idx) = find_similar(insights, &candidate, self.merge_threshold)
                .or_else(|| index.get(&candidate.similarity_key()).copied())
            {
                let existing = &mut insights[existing_idx];
                existing.content = merge_content(&existing.content, &candidate.content);
                existing.evidence.extend(candidate.evidence.clone());
                existing
                    .votes
                    .upvotes
                    .extend(candidate.votes.upvotes.clone());
                existing.generation += 1;
                existing.last_updated = Utc::now();
                existing.recompute_score();
                report.merged += 1;
            } else {
                index.insert(candidate.similarity_key(), insights.len());
                insights.push(candidate);
                report.added += 1;
            }
        }

        let before_cleanup = insights.len();
        insights.retain(|i| i.votes.net_score > self.deprecate_threshold);
        report.deprecated = before_cleanup - insights.len();
        report.surviving = insights.len();
        if before == 0 && report.added == 0 {
            report.surviving = 0;
        }
        report
    }

    pub fn cleanup(insights: &mut Vec<EvolvableInsight>) -> usize {
        let before = insights.len();
        insights.retain(|i| !i.should_deprecate());
        before - insights.len()
    }
}

fn normalize_key(content: &str) -> String {
    content
        .split_whitespace()
        .take(8)
        .map(|s| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_set(content: &str) -> std::collections::HashSet<String> {
    content
        .split_whitespace()
        .map(|s| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn jaccard(a: &str, b: &str) -> f32 {
    let a = token_set(a);
    let b = token_set(b);
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(&b).count() as f32;
    let union = a.union(&b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn find_similar(
    insights: &[EvolvableInsight],
    candidate: &EvolvableInsight,
    threshold: f32,
) -> Option<usize> {
    insights
        .iter()
        .enumerate()
        .filter(|(_, i)| i.epistemic_type == candidate.epistemic_type)
        .map(|(idx, i)| (idx, jaccard(&i.content, &candidate.content)))
        .filter(|(_, score)| *score >= threshold)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
}

fn merge_content(existing: &str, candidate: &str) -> String {
    if existing.contains(candidate) {
        existing.to_string()
    } else if candidate.contains(existing) {
        candidate.to_string()
    } else {
        format!("{existing}\nEVOLVED: {candidate}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(text: &str, action: &str, priority: f32) -> SemanticGradient {
        SemanticGradient {
            error_type: ContentType::Error,
            reflection_text: text.into(),
            action_improvement: action.into(),
            epistemic_tags: vec!["procedure".into()],
            source_uri: None,
            priority,
        }
    }

    #[test]
    fn gradient_creates_voted_insight() {
        let insight = EvolvableInsight::from_gradient(0, &gradient("timeout", "retry", 0.8));
        assert!(insight.votes.net_score > 0.0);
        assert!(insight.content.contains("ACTION"));
    }

    #[test]
    fn evolution_merges_similar_gradients_and_keeps_positive_votes() {
        let engine = InsightEvolutionEngine::new().with_thresholds(0.0, 0.2);
        let mut insights = Vec::new();
        let report = engine.evolve_from_gradients(
            &mut insights,
            &[
                gradient("timeout during deploy", "add retry", 0.8),
                gradient("timeout during deploy", "add backoff", 0.7),
            ],
        );
        assert_eq!(report.added, 1);
        assert_eq!(report.merged, 1);
        assert_eq!(insights.len(), 1);
        assert!(insights[0].votes.net_score > 1.0);
    }
}
