//! Memetic evolution — 用质量、采纳和佐证信号驱动知识种群演化。
//!
//! 该模块把 phylogeny 从“只记录血统”升级为可执行的选择压力：高 fitness
//! 条目会产生变异或交叉后代，低 fitness 条目会被标记淘汰，并为每个后代写入谱系节点。

use std::sync::Arc;

use agent_context_db_core::{LlmClient, LlmOpts, LlmTaskKind, PromptOptimization};
use chrono::Utc;

use crate::crdt::{PatchSet, SemanticCrdtMerger};
use crate::types::*;

#[derive(Debug, Clone)]
pub struct MemeticEvolutionConfig {
    pub elite_fraction: f32,
    pub mutation_threshold: f32,
    pub crossover_threshold: f32,
    pub cull_threshold: f32,
    pub max_offspring: usize,
    pub quality_weight: f32,
    pub confidence_weight: f32,
    pub adoption_weight: f32,
    pub corroboration_weight: f32,
    pub hit_weight: f32,
    pub novelty_weight: f32,
    pub recency_weight: f32,
    pub hit_log_scale: f32,
    pub recency_half_life_days: f32,
    pub contradiction_penalty: f32,
    pub downvote_penalty: f32,
    pub max_penalty: f32,
}

impl Default for MemeticEvolutionConfig {
    fn default() -> Self {
        Self {
            elite_fraction: 0.35,
            mutation_threshold: 0.68,
            crossover_threshold: 0.76,
            cull_threshold: 0.22,
            max_offspring: 8,
            quality_weight: 0.30,
            confidence_weight: 0.16,
            adoption_weight: 0.20,
            corroboration_weight: 0.14,
            hit_weight: 0.08,
            novelty_weight: 0.07,
            recency_weight: 0.05,
            hit_log_scale: 8.0,
            recency_half_life_days: 90.0,
            contradiction_penalty: 0.12,
            downvote_penalty: 0.06,
            max_penalty: 0.55,
        }
    }
}

impl MemeticEvolutionConfig {
    pub fn validate(&self) -> Result<(), String> {
        for value in [
            self.elite_fraction,
            self.mutation_threshold,
            self.crossover_threshold,
            self.cull_threshold,
            self.quality_weight,
            self.confidence_weight,
            self.adoption_weight,
            self.corroboration_weight,
            self.hit_weight,
            self.novelty_weight,
            self.recency_weight,
            self.contradiction_penalty,
            self.downvote_penalty,
            self.max_penalty,
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err("memetic fractions, weights, and penalties must be in [0, 1]".into());
            }
        }
        let weight_sum = self.quality_weight
            + self.confidence_weight
            + self.adoption_weight
            + self.corroboration_weight
            + self.hit_weight
            + self.novelty_weight
            + self.recency_weight;
        if (weight_sum - 1.0).abs() > f32::EPSILON * 8.0 {
            return Err("memetic fitness weights must sum to 1".into());
        }
        if self.max_offspring == 0
            || self.hit_log_scale <= 0.0
            || self.recency_half_life_days <= 0.0
            || !self.hit_log_scale.is_finite()
            || !self.recency_half_life_days.is_finite()
        {
            return Err("memetic limits and scales must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionAction {
    Mutation,
    Crossover,
    Cull,
}

#[derive(Debug, Clone)]
pub struct FitnessSignals {
    pub adoption_rate: f32,
    pub hit_count: usize,
    pub contradiction_count: usize,
    pub downvote_count: usize,
    pub recency_days: f32,
}

impl Default for FitnessSignals {
    fn default() -> Self {
        Self {
            adoption_rate: 0.0,
            hit_count: 0,
            contradiction_count: 0,
            downvote_count: 0,
            recency_days: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvolutionCandidate {
    pub entry: MarketEntry,
    pub signals: FitnessSignals,
}

#[derive(Debug, Clone)]
pub struct FitnessScore {
    pub entry_id: MarketId,
    pub score: f32,
    pub novelty: f32,
    pub penalty: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffspringValidationStatus {
    Pending,
}

#[derive(Debug, Clone)]
pub struct EvolutionOffspring {
    pub action: EvolutionAction,
    pub entry: MarketEntry,
    pub lineage: LineageNode,
    pub parent_ids: Vec<MarketId>,
    pub fitness: f32,
    pub validation_status: OffspringValidationStatus,
    /// Publication must independently satisfy all checks after the text changed.
    pub required_validations: Vec<&'static str>,
}

#[derive(Debug, Clone, Default)]
pub struct EvolutionRunReport {
    pub fitness: Vec<FitnessScore>,
    pub offspring: Vec<EvolutionOffspring>,
    pub culled: Vec<LineageNode>,
    pub skipped_license: usize,
}

pub struct MemeticEvolutionEngine {
    config: MemeticEvolutionConfig,
    merger: SemanticCrdtMerger,
    llm: Option<Arc<dyn LlmClient>>,
}

impl MemeticEvolutionEngine {
    pub fn new(node_id: impl Into<String>, config: MemeticEvolutionConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            merger: SemanticCrdtMerger::new(node_id.into()),
            llm: None,
        })
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.merger = SemanticCrdtMerger::new("memetic-evolution").with_llm(llm.clone());
        self.llm = Some(llm);
        self
    }

    pub async fn evolve(&self, candidates: &[EvolutionCandidate]) -> EvolutionRunReport {
        let mut ranked = candidates
            .iter()
            .map(|candidate| (self.fitness(candidate, candidates), candidate))
            .collect::<Vec<_>>();
        ranked.sort_by(|(a, _), (b, _)| b.score.total_cmp(&a.score));

        let mut report = EvolutionRunReport {
            fitness: ranked.iter().map(|(score, _)| score.clone()).collect(),
            ..Default::default()
        };

        for (fitness, candidate) in ranked.iter().rev() {
            if fitness.score <= self.config.cull_threshold {
                report.culled.push(LineageNode {
                    market_id: candidate.entry.id,
                    publisher: candidate.entry.publisher.clone(),
                    action: LineageAction::Deprecated,
                    parent_ids: vec![candidate.entry.id],
                    timestamp: Utc::now(),
                });
            }
        }

        let elite_count = ((ranked.len() as f32 * self.config.elite_fraction).ceil() as usize)
            .clamp(1, ranked.len().max(1));
        let elites = ranked.iter().take(elite_count).collect::<Vec<_>>();

        for (fitness, candidate) in &elites {
            if report.offspring.len() >= self.config.max_offspring {
                break;
            }
            if fitness.score >= self.config.mutation_threshold {
                if !LicenseInfo::from(candidate.entry.license.clone()).derivative_allowed {
                    report.skipped_license += 1;
                    continue;
                }
                report
                    .offspring
                    .push(self.mutate(&candidate.entry, fitness.score).await);
            }
        }

        for pair in elites.windows(2) {
            if report.offspring.len() >= self.config.max_offspring {
                break;
            }
            let (left_score, left) = pair[0];
            let (right_score, right) = pair[1];
            let pair_score = (left_score.score + right_score.score) / 2.0;
            if pair_score < self.config.crossover_threshold {
                continue;
            }
            if !LicenseInfo::from(left.entry.license.clone()).derivative_allowed
                || !LicenseInfo::from(right.entry.license.clone()).derivative_allowed
            {
                report.skipped_license += 1;
                continue;
            }
            report
                .offspring
                .push(self.crossover(&left.entry, &right.entry, pair_score).await);
        }

        report
    }

    pub fn fitness(
        &self,
        candidate: &EvolutionCandidate,
        population: &[EvolutionCandidate],
    ) -> FitnessScore {
        let entry = &candidate.entry;
        let signals = &candidate.signals;
        let corroboration = match entry.corroboration.level {
            CorroborationLevel::Unverified => 0.0,
            CorroborationLevel::SingleSession => 0.25,
            CorroborationLevel::CrossSession => 0.55,
            CorroborationLevel::CrossAgent => 0.8,
            CorroborationLevel::Established => 1.0,
        };
        let novelty = domain_novelty(entry, population);
        let hit_signal =
            ((signals.hit_count as f32 + 1.0).ln() / self.config.hit_log_scale).clamp(0.0, 1.0);
        let recency = (1.0
            / (1.0 + signals.recency_days.max(0.0) / self.config.recency_half_life_days))
            .clamp(0.0, 1.0);
        let penalty = (signals.contradiction_count as f32 * self.config.contradiction_penalty
            + signals.downvote_count as f32 * self.config.downvote_penalty)
            .clamp(0.0, self.config.max_penalty);
        let score = (entry.quality_score * self.config.quality_weight
            + entry.confidence * self.config.confidence_weight
            + signals.adoption_rate.clamp(0.0, 1.0) * self.config.adoption_weight
            + corroboration * self.config.corroboration_weight
            + hit_signal * self.config.hit_weight
            + novelty * self.config.novelty_weight
            + recency * self.config.recency_weight
            - penalty)
            .clamp(0.0, 1.0);

        FitnessScore {
            entry_id: entry.id,
            score,
            novelty,
            penalty,
        }
    }

    async fn mutate(&self, parent: &MarketEntry, fitness: f32) -> EvolutionOffspring {
        let mut child = parent.clone();
        child.id = MarketId::new();
        child.principle = self.mutated_principle(parent).await;
        // Changed content cannot inherit evidence, maturity, quality, or confidence.
        child.evidence_uris.clear();
        child.corroboration = CorroborationProof::new();
        child.quality_score = 0.0;
        child.confidence = 0.0;
        child.domain = format!("{}/variant", parent.domain);
        child.created_at = Utc::now();
        child.provenance = None;

        let lineage = LineageNode {
            market_id: child.id,
            publisher: child.publisher.clone(),
            action: LineageAction::Derived,
            parent_ids: vec![parent.id],
            timestamp: child.created_at,
        };
        EvolutionOffspring {
            action: EvolutionAction::Mutation,
            entry: child,
            lineage,
            parent_ids: vec![parent.id],
            fitness,
            validation_status: OffspringValidationStatus::Pending,
            required_validations: vec!["consistency", "provenance", "adoption"],
        }
    }

    async fn crossover(
        &self,
        left: &MarketEntry,
        right: &MarketEntry,
        fitness: f32,
    ) -> EvolutionOffspring {
        let left_patch = patch_from_entry(left, 1);
        let right_patch = patch_from_entry(right, 2);
        let merged = self.merger.merge(&left_patch, &right_patch).await;
        let mut child = left.clone();
        child.id = MarketId::new();
        child.publisher = left.publisher.clone();
        child.domain = common_domain(&left.domain, &right.domain);
        child.principle = merged.principle;
        // Parent citations may guide generation but do not validate the merged claim.
        child.evidence_uris.clear();
        child.quality_score = 0.0;
        child.confidence = 0.0;
        child.created_at = Utc::now();
        child.provenance = None;
        child.corroboration = CorroborationProof::new();

        let parent_ids = vec![left.id, right.id];
        let lineage = LineageNode {
            market_id: child.id,
            publisher: child.publisher.clone(),
            action: LineageAction::Merged,
            parent_ids: parent_ids.clone(),
            timestamp: child.created_at,
        };
        EvolutionOffspring {
            action: EvolutionAction::Crossover,
            entry: child,
            lineage,
            parent_ids,
            fitness,
            validation_status: OffspringValidationStatus::Pending,
            required_validations: vec!["consistency", "provenance", "adoption"],
        }
    }

    async fn mutated_principle(&self, parent: &MarketEntry) -> String {
        if let Some(llm) = &self.llm {
            let prompt = format!(
                r#"Improve this marketplace knowledge principle by making it more precise and transferable.

Domain: {}
Principle: {}

Return only the improved principle text. Do not add markdown."#,
                parent.domain, parent.principle
            );
            if let Ok(text) = llm
                .complete(
                    &prompt,
                    &LlmOpts {
                        max_tokens: Some(256),
                        temperature: Some(0.7),
                        task: LlmTaskKind::Synthesis,
                        prompt: PromptOptimization::default().target_tokens(900),
                        ..Default::default()
                    },
                )
                .await
            {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
        structured_mutation(parent)
    }
}

fn patch_from_entry(entry: &MarketEntry, clock: u64) -> PatchSet {
    PatchSet {
        entry_id: entry.id,
        patcher: entry.publisher.clone(),
        clock,
        principle: Some(entry.principle.clone()),
        preconditions: vec![entry.domain.clone()],
        evidence_uris: entry.evidence_uris.clone(),
        confidence: entry.confidence,
        quality_score: entry.quality_score,
    }
}

fn structured_mutation(parent: &MarketEntry) -> String {
    let principle = parent.principle.trim();
    let lower = principle.to_ascii_lowercase();
    let evidence_clause = if parent.evidence_uris.is_empty() {
        "after recording at least one supporting evidence URI"
    } else {
        "when the cited evidence still matches the current context"
    };
    let corroboration_clause = match parent.corroboration.level {
        CorroborationLevel::Unverified | CorroborationLevel::SingleSession => {
            "treat it as provisional and request independent corroboration"
        }
        CorroborationLevel::CrossSession | CorroborationLevel::CrossAgent => {
            "prefer it when a new case has comparable cross-context signals"
        }
        CorroborationLevel::Established => "apply it by default unless a contradiction is attached",
    };

    if lower.contains("always") || lower.contains("never") {
        return format!(
            "Within domain '{}', replace absolute wording in '{}' with bounded preconditions; {}, and {}.",
            parent.domain, principle, evidence_clause, corroboration_clause
        );
    }
    if lower.contains("when ")
        || lower.contains("if ")
        || principle.contains('当')
        || principle.contains("如果")
    {
        return format!(
            "Within domain '{}', apply '{}' only after checking preconditions, evidence freshness, and contradiction links; {}.",
            parent.domain, principle, corroboration_clause
        );
    }
    if parent.confidence < 0.55 || parent.quality_score < 0.55 {
        return format!(
            "Within domain '{}', keep '{}' as a candidate rule; {} before promotion, and {}.",
            parent.domain, principle, evidence_clause, corroboration_clause
        );
    }
    format!(
        "Within domain '{}', use '{}' as the default rule for matching cases; {}, and record exceptions as contradiction evidence.",
        parent.domain, principle, evidence_clause
    )
}

fn domain_novelty(entry: &MarketEntry, population: &[EvolutionCandidate]) -> f32 {
    let same_domain = population
        .iter()
        .filter(|candidate| candidate.entry.domain == entry.domain)
        .count();
    (1.0 / same_domain.max(1) as f32).sqrt().clamp(0.0, 1.0)
}

fn common_domain(left: &str, right: &str) -> String {
    if left == right {
        return format!("{left}/hybrid");
    }
    let prefix = left
        .split('/')
        .zip(right.split('/'))
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a)
        .collect::<Vec<_>>()
        .join("/");
    if prefix.is_empty() {
        "hybrid".to_string()
    } else {
        format!("{prefix}/hybrid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, EpistemicType};

    fn entry(domain: &str, quality: f32, confidence: f32, publisher: &str) -> MarketEntry {
        let mut corroboration = CorroborationProof::new();
        corroboration.add_corroboration(AgentId::new("peer-a"), 2);
        MarketEntry {
            id: MarketId::new(),
            publisher: AgentId::new(publisher),
            domain: domain.into(),
            entry_type: MarketEntryType::Skill,
            principle: format!("Use bounded traversal for {domain}"),
            evidence_uris: vec![],
            quality_score: quality,
            confidence,
            corroboration,
            provenance: None,
            license: KnowledgeLicense::Attribution,
            epistemic_type: EpistemicType::Heuristic,
            content_type: ContentType::Skill,
            half_life: Some(agent_context_db_core::HalfLife::Finite { days: 120.0 }),
            created_at: Utc::now(),
            expires_at: None,
        }
    }

    #[tokio::test]
    async fn memetic_engine_mutates_crosses_and_culls_by_fitness() {
        let a = entry("rust/retrieval", 0.92, 0.88, "agent-a");
        let b = entry("rust/retrieval", 0.86, 0.82, "agent-b");
        let weak = entry("rust/obsolete", 0.12, 0.2, "agent-c");
        let candidates = vec![
            EvolutionCandidate {
                entry: a.clone(),
                signals: FitnessSignals {
                    adoption_rate: 0.95,
                    hit_count: 80,
                    ..Default::default()
                },
            },
            EvolutionCandidate {
                entry: b.clone(),
                signals: FitnessSignals {
                    adoption_rate: 0.88,
                    hit_count: 60,
                    ..Default::default()
                },
            },
            EvolutionCandidate {
                entry: weak.clone(),
                signals: FitnessSignals {
                    contradiction_count: 3,
                    downvote_count: 4,
                    ..Default::default()
                },
            },
        ];

        let report = MemeticEvolutionEngine::new(
            "test-node",
            MemeticEvolutionConfig {
                elite_fraction: 0.67,
                max_offspring: 4,
                ..Default::default()
            },
        )
        .unwrap()
        .evolve(&candidates)
        .await;

        assert!(report.fitness[0].score >= report.fitness[1].score);
        assert!(
            report
                .offspring
                .iter()
                .any(|o| o.action == EvolutionAction::Mutation && o.parent_ids == vec![a.id])
        );
        assert!(report
            .offspring
            .iter()
            .any(|o| o.action == EvolutionAction::Crossover && o.parent_ids == vec![a.id, b.id]));
        assert_eq!(report.culled.len(), 1);
        assert_eq!(report.culled[0].action, LineageAction::Deprecated);
        for offspring in &report.offspring {
            assert_eq!(
                offspring.validation_status,
                OffspringValidationStatus::Pending
            );
            assert_eq!(
                offspring.required_validations,
                vec!["consistency", "provenance", "adoption"]
            );
            assert!(offspring.entry.evidence_uris.is_empty());
            assert_eq!(
                offspring.entry.corroboration.level,
                CorroborationLevel::Unverified
            );
            assert_eq!(offspring.entry.confidence, 0.0);
            assert_eq!(offspring.entry.quality_score, 0.0);
            assert!(offspring.entry.provenance.is_none());
        }
    }
}
