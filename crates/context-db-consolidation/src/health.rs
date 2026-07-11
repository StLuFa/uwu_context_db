//! Knowledge health, consistency guarding, and active learning loops.
//!
//! These components turn quality signals into concrete maintenance actions during
//! Sleeptime: fragile bridge memories are revalidated first, contradictions are
//! quarantined or invalidated according to policy, and high-uncertainty memories
//! emit learning probes whose outcomes can be written back into the belief state.

use crate::quality::{
    ActiveLearningAction, ActiveLearningPlanner, ActiveLearningTask, DimensionBelief,
    HorizonAwareQualityScorer, HorizonQualitySignals, QualityBeliefDimension, QualityBeliefState,
};
use agent_context_db_core::{
    ContentType, ContextEntry, ContextError, ContextUri, GraphStore, ValidityRecord,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const HEALTH_REPORT_KEY: &str = "knowledge_health";
const ACTIVE_LEARNING_KEY: &str = "active_learning";
const QUALITY_BELIEF_KEY: &str = "quality_belief_state";
const CONSISTENCY_GUARD_KEY: &str = "consistency_guard";
const EMBEDDING_SNAPSHOT_KEY: &str = "embedding_snapshot";
const EMBEDDING_VECTOR_KEY: &str = "embedding_vector";
const CURIOSITY_TASKS_KEY: &str = "curiosity_tasks";

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KnowledgeHealthConfig {
    pub max_nodes: usize,
    pub max_issues: usize,
    pub risky_quality_threshold: f32,
    pub bridge_score_threshold: f32,
    pub min_dependency_count: usize,
    pub graph_centrality_weight: f32,
}

impl KnowledgeHealthConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.max_nodes == 0 || self.max_issues == 0 || self.min_dependency_count == 0 {
            return Err(crate::ConfigError(
                "knowledge health limits must be nonzero".into(),
            ));
        }
        crate::validate_unit_f32("risky_quality_threshold", self.risky_quality_threshold)?;
        crate::validate_unit_f32("bridge_score_threshold", self.bridge_score_threshold)?;
        crate::validate_unit_f32("graph_centrality_weight", self.graph_centrality_weight)
    }
}

impl Default for KnowledgeHealthConfig {
    fn default() -> Self {
        Self {
            max_nodes: 512,
            max_issues: 64,
            risky_quality_threshold: 0.55,
            bridge_score_threshold: 0.32,
            min_dependency_count: 2,
            graph_centrality_weight: 0.45,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthIssueKind {
    DangerousBridge,
    MissingEvidence,
    InvalidatedDependency,
    StaleLowConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthRepairAction {
    Revalidate,
    StrengthenEvidence,
    CascadeRepair,
    DemoteRetrieval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    pub uri: ContextUri,
    pub issue: HealthIssueKind,
    pub action: HealthRepairAction,
    pub priority: f32,
    pub quality: f32,
    pub dependency_count: usize,
    pub centrality: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeHealthReport {
    pub generated_at: DateTime<Utc>,
    pub issues: Vec<NodeHealth>,
}

pub struct KnowledgeHealthDiagnostician {
    config: KnowledgeHealthConfig,
    graph: Option<Arc<dyn GraphStore>>,
}

impl KnowledgeHealthDiagnostician {
    pub fn new(config: KnowledgeHealthConfig) -> Result<Self, crate::ConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            graph: None,
        })
    }

    pub fn with_graph(mut self, graph: Arc<dyn GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    pub async fn diagnose(
        &self,
        entries: &[ContextEntry],
        now: DateTime<Utc>,
    ) -> Result<KnowledgeHealthReport, ContextError> {
        let limited = entries
            .iter()
            .take(self.config.max_nodes)
            .collect::<Vec<_>>();
        let dependencies = dependency_counts(&limited);
        let max_degree = dependencies.values().copied().max().unwrap_or(1).max(1) as f32;
        let mut issues = Vec::new();

        for entry in limited {
            let quality = entry.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
            let dependency_count = *dependencies.get(&entry.uri).unwrap_or(&0);
            let graph_centrality = self.graph_centrality(&entry.uri).await?;
            let local_centrality = dependency_count as f32 / max_degree;
            let centrality = (local_centrality * (1.0 - self.config.graph_centrality_weight)
                + graph_centrality * self.config.graph_centrality_weight)
                .clamp(0.0, 1.0);
            let bridge_score = centrality * (1.0 - quality);

            if dependency_count >= self.config.min_dependency_count
                && quality <= self.config.risky_quality_threshold
                && bridge_score >= self.config.bridge_score_threshold
            {
                issues.push(NodeHealth {
                    uri: entry.uri.clone(),
                    issue: HealthIssueKind::DangerousBridge,
                    action: HealthRepairAction::Revalidate,
                    priority: bridge_score,
                    quality,
                    dependency_count,
                    centrality,
                    reason: format!(
                        "central bridge with {dependency_count} dependents, quality {quality:.2}, centrality {centrality:.2}"
                    ),
                });
            }

            if entry.content_type() == Some(ContentType::Fact)
                && evidence_count(entry) == 0
                && quality < 0.72
            {
                issues.push(NodeHealth {
                    uri: entry.uri.clone(),
                    issue: HealthIssueKind::MissingEvidence,
                    action: HealthRepairAction::StrengthenEvidence,
                    priority: (0.72 - quality + centrality * 0.25).clamp(0.0, 1.0),
                    quality,
                    dependency_count,
                    centrality,
                    reason: "fact memory lacks evidence and should be corroborated".into(),
                });
            }

            if entry
                .metadata
                .validity
                .as_ref()
                .and_then(|v| v.valid_until)
                .is_some()
                && dependency_count > 0
            {
                issues.push(NodeHealth {
                    uri: entry.uri.clone(),
                    issue: HealthIssueKind::InvalidatedDependency,
                    action: HealthRepairAction::CascadeRepair,
                    priority: (0.55 + centrality * 0.45).clamp(0.0, 1.0),
                    quality,
                    dependency_count,
                    centrality,
                    reason: "invalidated memory is still referenced by dependent knowledge".into(),
                });
            }

            let age_days = (now - entry.updated_at).num_hours().max(0) as f32 / 24.0;
            if age_days > 30.0 && quality < 0.45 && dependency_count == 0 {
                issues.push(NodeHealth {
                    uri: entry.uri.clone(),
                    issue: HealthIssueKind::StaleLowConfidence,
                    action: HealthRepairAction::DemoteRetrieval,
                    priority: ((age_days / 180.0).min(0.5) + (0.45 - quality)).clamp(0.0, 1.0),
                    quality,
                    dependency_count,
                    centrality,
                    reason: format!("stale low-confidence leaf memory, age {age_days:.1} days"),
                });
            }
        }

        issues.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(Ordering::Equal)
        });
        issues.truncate(self.config.max_issues);
        Ok(KnowledgeHealthReport {
            generated_at: now,
            issues,
        })
    }

    pub fn apply_repairs(
        &self,
        entries: &mut [ContextEntry],
        report: &KnowledgeHealthReport,
    ) -> Result<usize, ContextError> {
        let by_uri = report
            .issues
            .iter()
            .map(|issue| (issue.uri.clone(), issue))
            .collect::<HashMap<_, _>>();
        let mut updated = 0usize;

        for entry in entries {
            let Some(issue) = by_uri.get(&entry.uri) else {
                continue;
            };
            push_tag(&mut entry.metadata.tags, "health:diagnosed");
            push_tag(
                &mut entry.metadata.tags,
                match issue.action {
                    HealthRepairAction::Revalidate => "health:revalidate",
                    HealthRepairAction::StrengthenEvidence => "health:strengthen-evidence",
                    HealthRepairAction::CascadeRepair => "health:cascade-repair",
                    HealthRepairAction::DemoteRetrieval => "health:demote-retrieval",
                },
            );
            entry
                .metadata
                .set_custom_field(HEALTH_REPORT_KEY, issue)
                .map_err(ContextError::Serialization)?;
            updated += 1;
        }
        Ok(updated)
    }

    async fn graph_centrality(&self, uri: &ContextUri) -> Result<f32, ContextError> {
        let Some(graph) = &self.graph else {
            return Ok(0.0);
        };
        Ok(graph.centrality(uri).await?.clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConsistencyGuardianConfig {
    pub max_pairs: usize,
    pub overlap_threshold: f32,
    pub strong_quality_delta: f32,
    pub allow_auto_invalidate: bool,
}

impl ConsistencyGuardianConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.max_pairs == 0 {
            return Err(crate::ConfigError("max_pairs must be nonzero".into()));
        }
        crate::validate_unit_f32("overlap_threshold", self.overlap_threshold)?;
        crate::validate_unit_f32("strong_quality_delta", self.strong_quality_delta)
    }
}

impl Default for ConsistencyGuardianConfig {
    fn default() -> Self {
        Self {
            max_pairs: 4096,
            overlap_threshold: 0.36,
            strong_quality_delta: 0.18,
            allow_auto_invalidate: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyRepairAction {
    RevalidateBoth,
    InvalidateWeaker,
    KeepBothContextual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyGuardTask {
    pub uri_a: ContextUri,
    pub uri_b: ContextUri,
    pub action: ConsistencyRepairAction,
    pub priority: f32,
    pub overlap: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsistencyGuardPlan {
    pub generated_at: DateTime<Utc>,
    pub tasks: Vec<ConsistencyGuardTask>,
}

pub struct ActiveConsistencyGuardian {
    config: ConsistencyGuardianConfig,
}

impl ActiveConsistencyGuardian {
    pub fn new(config: ConsistencyGuardianConfig) -> Result<Self, crate::ConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn plan(&self, entries: &[ContextEntry], now: DateTime<Utc>) -> ConsistencyGuardPlan {
        let candidates = entries
            .iter()
            .filter(|entry| is_epistemic_claim(entry) && is_currently_valid(entry, now))
            .collect::<Vec<_>>();
        let mut tasks = Vec::new();
        let mut checked = 0usize;

        'outer: for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                if checked >= self.config.max_pairs {
                    break 'outer;
                }
                checked += 1;
                let a = candidates[i];
                let b = candidates[j];
                if a.content_type() != b.content_type() {
                    continue;
                }
                let overlap = token_overlap(a.l0_text(), b.l0_text());
                if overlap < self.config.overlap_threshold
                    || !has_contradiction_marker(a.l0_text(), b.l0_text())
                {
                    continue;
                }
                let qa = a.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
                let qb = b.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
                let delta = (qa - qb).abs();
                let action = if self.config.allow_auto_invalidate
                    && delta >= self.config.strong_quality_delta
                {
                    ConsistencyRepairAction::InvalidateWeaker
                } else if same_evidence(a, b) {
                    ConsistencyRepairAction::RevalidateBoth
                } else {
                    ConsistencyRepairAction::KeepBothContextual
                };
                tasks.push(ConsistencyGuardTask {
                    uri_a: a.uri.clone(),
                    uri_b: b.uri.clone(),
                    action,
                    priority: (overlap * 0.6 + delta * 0.4).clamp(0.0, 1.0),
                    overlap,
                    reason: format!("contradiction markers with token overlap {overlap:.2} and quality delta {delta:.2}"),
                });
            }
        }

        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(Ordering::Equal)
        });
        ConsistencyGuardPlan {
            generated_at: now,
            tasks,
        }
    }

    pub fn apply(
        &self,
        entries: &mut [ContextEntry],
        plan: &ConsistencyGuardPlan,
        now: DateTime<Utc>,
    ) -> Result<usize, ContextError> {
        let by_uri = entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.uri.clone(), idx))
            .collect::<HashMap<_, _>>();
        let mut touched = HashSet::new();

        for task in &plan.tasks {
            let Some(&idx_a) = by_uri.get(&task.uri_a) else {
                continue;
            };
            let Some(&idx_b) = by_uri.get(&task.uri_b) else {
                continue;
            };
            let qa = entries[idx_a].metadata.quality_score.unwrap_or(0.5);
            let qb = entries[idx_b].metadata.quality_score.unwrap_or(0.5);
            match task.action {
                ConsistencyRepairAction::InvalidateWeaker => {
                    let weaker = if qa <= qb { idx_a } else { idx_b };
                    let stronger_uri = if weaker == idx_a {
                        &task.uri_b
                    } else {
                        &task.uri_a
                    };
                    entries[weaker].metadata.validity = Some(ValidityRecord {
                        valid_from: entries[weaker].created_at,
                        valid_until: Some(now),
                        invalidated_by: Some(stronger_uri.clone()),
                        invalidation_reason: Some(task.reason.clone()),
                    });
                    push_tag(
                        &mut entries[weaker].metadata.tags,
                        "consistency:auto-invalidated",
                    );
                    entries[weaker]
                        .metadata
                        .set_custom_field(CONSISTENCY_GUARD_KEY, task)
                        .map_err(ContextError::Serialization)?;
                    touched.insert(weaker);
                }
                ConsistencyRepairAction::RevalidateBoth => {
                    for idx in [idx_a, idx_b] {
                        push_tag(&mut entries[idx].metadata.tags, "consistency:revalidate");
                        entries[idx]
                            .metadata
                            .set_custom_field(CONSISTENCY_GUARD_KEY, task)
                            .map_err(ContextError::Serialization)?;
                        touched.insert(idx);
                    }
                }
                ConsistencyRepairAction::KeepBothContextual => {
                    for idx in [idx_a, idx_b] {
                        push_tag(
                            &mut entries[idx].metadata.tags,
                            "consistency:contextual-conflict",
                        );
                        entries[idx]
                            .metadata
                            .set_custom_field(CONSISTENCY_GUARD_KEY, task)
                            .map_err(ContextError::Serialization)?;
                        touched.insert(idx);
                    }
                }
            }
        }
        Ok(touched.len())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ActiveLearningExecutionConfig {
    pub max_entries: usize,
    pub max_tasks_total: usize,
    pub min_quality_for_learning: f32,
}

impl ActiveLearningExecutionConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.max_entries == 0 || self.max_tasks_total == 0 {
            return Err(crate::ConfigError(
                "active learning limits must be nonzero".into(),
            ));
        }
        crate::validate_unit_f32("min_quality_for_learning", self.min_quality_for_learning)
    }
}

impl Default for ActiveLearningExecutionConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_tasks_total: 128,
            min_quality_for_learning: 0.20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningObservation {
    pub target_uri: ContextUri,
    pub dimension: QualityBeliefDimension,
    pub observation: f32,
    pub reliability: f32,
    pub note: String,
}

pub struct ActiveLearningLoop {
    planner: ActiveLearningPlanner,
    scorer: HorizonAwareQualityScorer,
    config: ActiveLearningExecutionConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EmbeddingDriftConfig {
    pub max_entries: usize,
    pub min_sample_size: usize,
    pub psi_threshold: f32,
    pub centroid_shift_threshold: f32,
    pub norm_kl_threshold: f32,
    pub bins: usize,
}

impl EmbeddingDriftConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.max_entries == 0 || self.min_sample_size == 0 || self.bins < 2 {
            return Err(crate::ConfigError(
                "embedding limits must be nonzero and bins must be at least 2".into(),
            ));
        }
        if self.min_sample_size > self.max_entries {
            return Err(crate::ConfigError(
                "min_sample_size must not exceed max_entries".into(),
            ));
        }
        for (name, value) in [
            ("psi_threshold", self.psi_threshold),
            ("centroid_shift_threshold", self.centroid_shift_threshold),
            ("norm_kl_threshold", self.norm_kl_threshold),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(crate::ConfigError(format!(
                    "{name} must be finite and nonnegative"
                )));
            }
        }
        Ok(())
    }
}

impl Default for EmbeddingDriftConfig {
    fn default() -> Self {
        Self {
            max_entries: 512,
            min_sample_size: 8,
            psi_threshold: 0.20,
            centroid_shift_threshold: 0.18,
            norm_kl_threshold: 0.15,
            bins: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingDriftAction {
    Observe,
    RebuildIndex,
    RetrainReranker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDistributionSnapshot {
    pub dim: usize,
    pub count: usize,
    pub centroid: Vec<f32>,
    pub dimension_histograms: Vec<Vec<f32>>,
    pub norm_histogram: Vec<f32>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDriftReport {
    pub generated_at: DateTime<Utc>,
    pub sample_size: usize,
    pub centroid_shift: f32,
    pub psi: f32,
    pub norm_kl: f32,
    pub action: EmbeddingDriftAction,
    pub reason: String,
    pub snapshot: Option<EmbeddingDistributionSnapshot>,
}

pub struct EmbeddingDriftMonitor {
    config: EmbeddingDriftConfig,
}

impl EmbeddingDriftMonitor {
    pub fn new(config: EmbeddingDriftConfig) -> Result<Self, crate::ConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn analyze(&self, entries: &[ContextEntry], now: DateTime<Utc>) -> EmbeddingDriftReport {
        let vectors = collect_embedding_vectors(entries, self.config.max_entries);
        if vectors.len() < self.config.min_sample_size {
            return EmbeddingDriftReport {
                generated_at: now,
                sample_size: vectors.len(),
                centroid_shift: 0.0,
                psi: 0.0,
                norm_kl: 0.0,
                action: EmbeddingDriftAction::Observe,
                reason: "not enough embedding samples to establish drift".into(),
                snapshot: None,
            };
        }

        let current = snapshot_from_vectors(&vectors, self.config.bins, now);
        let baseline = entries.iter().find_map(|entry| {
            entry
                .metadata
                .custom_field::<EmbeddingDistributionSnapshot>(EMBEDDING_SNAPSHOT_KEY)
        });
        let Some(baseline) = baseline else {
            return EmbeddingDriftReport {
                generated_at: now,
                sample_size: current.count,
                centroid_shift: 0.0,
                psi: 0.0,
                norm_kl: 0.0,
                action: EmbeddingDriftAction::Observe,
                reason: "recording initial embedding distribution baseline".into(),
                snapshot: Some(current),
            };
        };

        let centroid_shift = cosine_distance(&baseline.centroid, &current.centroid);
        let psi = average_psi(
            &baseline.dimension_histograms,
            &current.dimension_histograms,
        );
        let norm_kl = symmetric_kl(&baseline.norm_histogram, &current.norm_histogram);
        let action = if psi >= self.config.psi_threshold
            || centroid_shift >= self.config.centroid_shift_threshold
        {
            EmbeddingDriftAction::RebuildIndex
        } else if norm_kl >= self.config.norm_kl_threshold {
            EmbeddingDriftAction::RetrainReranker
        } else {
            EmbeddingDriftAction::Observe
        };
        EmbeddingDriftReport {
            generated_at: now,
            sample_size: current.count,
            centroid_shift,
            psi,
            norm_kl,
            action,
            reason: format!(
                "centroid_shift={centroid_shift:.3}, psi={psi:.3}, norm_kl={norm_kl:.3}"
            ),
            snapshot: Some(current),
        }
    }

    pub fn apply(
        &self,
        entries: &mut [ContextEntry],
        report: &EmbeddingDriftReport,
    ) -> Result<usize, ContextError> {
        let Some(snapshot) = &report.snapshot else {
            return Ok(0);
        };
        let mut updated = 0usize;
        for entry in entries.iter_mut().take(1) {
            entry
                .metadata
                .set_custom_field(EMBEDDING_SNAPSHOT_KEY, snapshot)
                .map_err(ContextError::Serialization)?;
            push_tag(&mut entry.metadata.tags, "embedding-drift:baseline");
            match report.action {
                EmbeddingDriftAction::Observe => {
                    push_tag(&mut entry.metadata.tags, "embedding-drift:observe")
                }
                EmbeddingDriftAction::RebuildIndex => {
                    push_tag(&mut entry.metadata.tags, "embedding-drift:rebuild-index")
                }
                EmbeddingDriftAction::RetrainReranker => {
                    push_tag(&mut entry.metadata.tags, "embedding-drift:retrain-reranker")
                }
            }
            updated += 1;
        }
        Ok(updated)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CuriosityExplorerConfig {
    pub max_tasks: usize,
    pub min_gap_severity: f32,
    pub min_uncertainty: f32,
}

impl CuriosityExplorerConfig {
    pub fn validate(&self) -> Result<(), crate::ConfigError> {
        if self.max_tasks == 0 {
            return Err(crate::ConfigError("max_tasks must be nonzero".into()));
        }
        crate::validate_unit_f32("min_gap_severity", self.min_gap_severity)?;
        crate::validate_unit_f32("min_uncertainty", self.min_uncertainty)
    }
}

impl Default for CuriosityExplorerConfig {
    fn default() -> Self {
        Self {
            max_tasks: 64,
            min_gap_severity: 0.35,
            min_uncertainty: 0.45,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CuriosityAction {
    LocalSearch,
    FederatedDiscovery,
    StormSynthesis,
    UserVerification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuriosityTask {
    pub target_uri: ContextUri,
    pub query: String,
    pub action: CuriosityAction,
    pub priority: f32,
    pub source: String,
    pub reason: String,
}

pub struct CuriosityExplorer {
    config: CuriosityExplorerConfig,
}

impl CuriosityExplorer {
    pub fn new(config: CuriosityExplorerConfig) -> Result<Self, crate::ConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn plan(&self, entries: &[ContextEntry], now: DateTime<Utc>) -> Vec<CuriosityTask> {
        let mut tasks = Vec::new();
        for entry in entries {
            if !is_currently_valid(entry, now) {
                continue;
            }
            let uncertainty = entry
                .metadata
                .custom_field::<QualityBeliefState>(QUALITY_BELIEF_KEY)
                .map(|state| state.uncertainty)
                .unwrap_or_else(|| 1.0 - entry.metadata.quality_score.unwrap_or(0.5));
            if uncertainty >= self.config.min_uncertainty {
                tasks.push(CuriosityTask {
                    target_uri: entry.uri.clone(),
                    query: curiosity_query(entry),
                    action: if uncertainty > 0.68 {
                        CuriosityAction::FederatedDiscovery
                    } else {
                        CuriosityAction::LocalSearch
                    },
                    priority: uncertainty.clamp(0.0, 1.0),
                    source: "quality_uncertainty".into(),
                    reason: format!("quality belief uncertainty {uncertainty:.2}"),
                });
            }
            if let Some(gaps) = entry
                .metadata
                .custom_field::<Vec<CuriosityGap>>("discovered_gaps")
            {
                for gap in gaps {
                    if gap.severity < self.config.min_gap_severity {
                        continue;
                    }
                    tasks.push(CuriosityTask {
                        target_uri: entry.uri.clone(),
                        query: gap.description.clone(),
                        action: CuriosityAction::StormSynthesis,
                        priority: gap.severity.clamp(0.0, 1.0),
                        source: "storm_gap".into(),
                        reason: gap.suggested_exploration,
                    });
                }
            }
            if entry
                .metadata
                .tags
                .iter()
                .any(|tag| tag == "active-learning:federated-discovery")
            {
                tasks.push(CuriosityTask {
                    target_uri: entry.uri.clone(),
                    query: curiosity_query(entry),
                    action: CuriosityAction::FederatedDiscovery,
                    priority: 0.72,
                    source: "active_learning".into(),
                    reason: "active learning requested federated corroboration".into(),
                });
            }
        }
        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(Ordering::Equal)
        });
        tasks.dedup_by(|a, b| {
            a.target_uri == b.target_uri && a.query == b.query && a.action == b.action
        });
        tasks.truncate(self.config.max_tasks);
        tasks
    }

    pub fn apply_tasks(
        &self,
        entries: &mut [ContextEntry],
        tasks: &[CuriosityTask],
    ) -> Result<usize, ContextError> {
        let grouped = tasks.iter().fold(
            HashMap::<ContextUri, Vec<CuriosityTask>>::new(),
            |mut acc, task| {
                acc.entry(task.target_uri.clone())
                    .or_default()
                    .push(task.clone());
                acc
            },
        );
        let mut updated = 0usize;
        for entry in entries {
            let Some(tasks) = grouped.get(&entry.uri) else {
                continue;
            };
            push_tag(&mut entry.metadata.tags, "curiosity:planned");
            for task in tasks {
                push_tag(&mut entry.metadata.tags, curiosity_tag(task.action));
            }
            entry
                .metadata
                .set_custom_field(CURIOSITY_TASKS_KEY, tasks)
                .map_err(ContextError::Serialization)?;
            updated += 1;
        }
        Ok(updated)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CuriosityGap {
    description: String,
    severity: f32,
    suggested_exploration: String,
}

impl ActiveLearningLoop {
    pub fn new(
        planner: ActiveLearningPlanner,
        scorer: HorizonAwareQualityScorer,
        config: ActiveLearningExecutionConfig,
    ) -> Result<Self, crate::ConfigError> {
        config.validate()?;
        Ok(Self {
            planner,
            scorer,
            config,
        })
    }

    pub fn plan(&self, entries: &[ContextEntry], now: DateTime<Utc>) -> Vec<ActiveLearningTask> {
        let mut tasks = Vec::new();
        for entry in entries.iter().take(self.config.max_entries) {
            let quality = entry.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
            if quality < self.config.min_quality_for_learning || !is_currently_valid(entry, now) {
                continue;
            }
            let state = entry
                .metadata
                .custom_field::<QualityBeliefState>(QUALITY_BELIEF_KEY)
                .unwrap_or_else(|| {
                    self.scorer
                        .reassess(entry, signals_from_entry(entry, now))
                        .posterior
                });
            tasks.extend(self.planner.plan(entry, &state));
            if tasks.len() >= self.config.max_tasks_total {
                break;
            }
        }
        tasks.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(Ordering::Equal)
        });
        tasks.truncate(self.config.max_tasks_total);
        tasks
    }

    pub fn apply_tasks(
        &self,
        entries: &mut [ContextEntry],
        tasks: &[ActiveLearningTask],
    ) -> Result<usize, ContextError> {
        let mut grouped: HashMap<ContextUri, Vec<ActiveLearningTask>> = HashMap::new();
        for task in tasks {
            grouped
                .entry(task.target_uri.clone())
                .or_default()
                .push(task.clone());
        }
        let mut updated = 0usize;
        for entry in entries {
            let Some(tasks) = grouped.get(&entry.uri) else {
                continue;
            };
            push_tag(&mut entry.metadata.tags, "active-learning:planned");
            for task in tasks {
                push_tag(&mut entry.metadata.tags, action_tag(task.action));
            }
            entry
                .metadata
                .set_custom_field(ACTIVE_LEARNING_KEY, tasks)
                .map_err(ContextError::Serialization)?;
            updated += 1;
        }
        Ok(updated)
    }

    pub fn apply_observations(
        &self,
        entries: &mut [ContextEntry],
        observations: &[ActiveLearningObservation],
        now: DateTime<Utc>,
    ) -> Result<usize, ContextError> {
        let mut by_uri: HashMap<ContextUri, Vec<&ActiveLearningObservation>> = HashMap::new();
        for observation in observations {
            by_uri
                .entry(observation.target_uri.clone())
                .or_default()
                .push(observation);
        }
        let mut updated = 0usize;
        for entry in entries {
            let Some(observations) = by_uri.get(&entry.uri) else {
                continue;
            };
            let mut state = entry
                .metadata
                .custom_field::<QualityBeliefState>(QUALITY_BELIEF_KEY)
                .unwrap_or_else(|| {
                    self.scorer
                        .reassess(entry, signals_from_entry(entry, now))
                        .posterior
                });
            for observation in observations {
                update_dimension(
                    &mut state,
                    observation.dimension,
                    observation.observation,
                    observation.reliability,
                );
            }
            state.last_scored_at = now;
            state.uncertainty = average_uncertainty(&state);
            state.confidence = (1.0 - state.uncertainty).clamp(0.0, 1.0);
            state.overall = average_mean(&state);
            entry.metadata.quality_score = Some(state.overall);
            entry
                .metadata
                .set_custom_field(QUALITY_BELIEF_KEY, &state)
                .map_err(ContextError::Serialization)?;
            push_tag(&mut entry.metadata.tags, "active-learning:observed");
            updated += 1;
        }
        Ok(updated)
    }
}

fn collect_embedding_vectors(entries: &[ContextEntry], max_entries: usize) -> Vec<Vec<f32>> {
    entries
        .iter()
        .take(max_entries)
        .filter_map(|entry| {
            entry
                .metadata
                .custom_field::<Vec<f32>>(EMBEDDING_VECTOR_KEY)
        })
        .filter(|vector| !vector.is_empty() && vector.iter().all(|v| v.is_finite()))
        .collect()
}

fn snapshot_from_vectors(
    vectors: &[Vec<f32>],
    bins: usize,
    now: DateTime<Utc>,
) -> EmbeddingDistributionSnapshot {
    let dim = vectors.iter().map(Vec::len).min().unwrap_or(0);
    let count = vectors.len();
    let mut centroid = vec![0.0; dim];
    for vector in vectors {
        for (idx, value) in vector.iter().take(dim).enumerate() {
            centroid[idx] += *value;
        }
    }
    for value in &mut centroid {
        *value /= count.max(1) as f32;
    }
    let inspected_dims = dim.min(16);
    let dimension_histograms = (0..inspected_dims)
        .map(|idx| histogram(vectors.iter().map(|v| v[idx]), bins, -1.0, 1.0))
        .collect::<Vec<_>>();
    let norm_histogram = histogram(
        vectors.iter().map(|v| l2_norm(&v[..dim])),
        bins,
        0.0,
        dim.max(1) as f32,
    );
    EmbeddingDistributionSnapshot {
        dim,
        count,
        centroid,
        dimension_histograms,
        norm_histogram,
        created_at: now,
    }
}

fn histogram(values: impl Iterator<Item = f32>, bins: usize, min: f32, max: f32) -> Vec<f32> {
    let bins = bins.max(2);
    let mut counts = vec![0.0_f32; bins];
    let width = (max - min).max(1e-6_f32);
    let mut total = 0.0_f32;
    for value in values {
        let idx = (((value - min) / width) * bins as f32).floor() as isize;
        let idx = idx.clamp(0, bins as isize - 1) as usize;
        counts[idx] += 1.0;
        total += 1.0;
    }
    if total == 0.0 {
        return vec![1.0 / bins as f32; bins];
    }
    counts
        .iter_mut()
        .for_each(|count| *count = (*count / total).max(1e-6_f32));
    counts
}

fn average_psi(base: &[Vec<f32>], current: &[Vec<f32>]) -> f32 {
    let pairs = base.len().min(current.len());
    if pairs == 0 {
        return 0.0;
    }
    (0..pairs)
        .map(|idx| psi(&base[idx], &current[idx]))
        .sum::<f32>()
        / pairs as f32
}

fn psi(base: &[f32], current: &[f32]) -> f32 {
    base.iter()
        .zip(current)
        .map(|(expected, actual)| {
            let expected = (*expected).max(1e-6);
            let actual = (*actual).max(1e-6);
            (actual - expected) * (actual / expected).ln()
        })
        .sum::<f32>()
        .max(0.0)
}

fn symmetric_kl(a: &[f32], b: &[f32]) -> f32 {
    (kl(a, b) + kl(b, a)) * 0.5
}

fn kl(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let x = (*x).max(1e-6);
            let y = (*y).max(1e-6);
            x * (x / y).ln()
        })
        .sum::<f32>()
        .max(0.0)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dim = a.len().min(b.len());
    if dim == 0 {
        return 0.0;
    }
    let dot = (0..dim).map(|idx| a[idx] * b[idx]).sum::<f32>();
    let na = l2_norm(&a[..dim]);
    let nb = l2_norm(&b[..dim]);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (1.0 - dot / (na * nb)).clamp(0.0, 2.0) * 0.5
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn curiosity_query(entry: &ContextEntry) -> String {
    let text = entry.l0_text();
    if text.chars().count() <= 160 {
        text.to_string()
    } else {
        text.chars().take(160).collect()
    }
}

fn curiosity_tag(action: CuriosityAction) -> &'static str {
    match action {
        CuriosityAction::LocalSearch => "curiosity:local-search",
        CuriosityAction::FederatedDiscovery => "curiosity:federated-discovery",
        CuriosityAction::StormSynthesis => "curiosity:storm-synthesis",
        CuriosityAction::UserVerification => "curiosity:user-verification",
    }
}

fn dependency_counts(entries: &[&ContextEntry]) -> HashMap<ContextUri, usize> {
    let known = entries
        .iter()
        .map(|entry| entry.uri.clone())
        .collect::<HashSet<_>>();
    let mut counts = HashMap::new();
    for entry in entries {
        if let Some(meta) = &entry.metadata.consolidation {
            for uri in meta
                .evidence_uris
                .iter()
                .chain(meta.entangled_with.iter())
                .filter(|uri| known.contains(*uri))
            {
                *counts.entry(uri.clone()).or_insert(0) += 1;
            }
        }
        if let Some(validity) = &entry.metadata.validity
            && let Some(uri) = &validity.invalidated_by
            && known.contains(uri)
        {
            *counts.entry(uri.clone()).or_insert(0) += 1;
        }
    }
    counts
}

fn evidence_count(entry: &ContextEntry) -> usize {
    entry
        .metadata
        .consolidation
        .as_ref()
        .map(|meta| meta.evidence_uris.len() + meta.corroboration)
        .unwrap_or(0)
}

fn is_epistemic_claim(entry: &ContextEntry) -> bool {
    matches!(
        entry.content_type(),
        Some(ContentType::Fact | ContentType::Belief | ContentType::Hypothesis)
    )
}

fn is_currently_valid(entry: &ContextEntry, now: DateTime<Utc>) -> bool {
    entry
        .metadata
        .validity
        .as_ref()
        .is_none_or(|validity| validity.valid_until.is_none_or(|until| until > now))
}

fn same_evidence(a: &ContextEntry, b: &ContextEntry) -> bool {
    let Some(a_meta) = &a.metadata.consolidation else {
        return false;
    };
    let Some(b_meta) = &b.metadata.consolidation else {
        return false;
    };
    a_meta
        .evidence_uris
        .iter()
        .any(|uri| b_meta.evidence_uris.contains(uri))
}

fn token_overlap(a: &str, b: &str) -> f32 {
    let a_tokens = tokens(a);
    let b_tokens = tokens(b);
    if a_tokens.is_empty() || b_tokens.is_empty() {
        return 0.0;
    }
    let intersection = a_tokens.intersection(&b_tokens).count();
    let union = a_tokens.union(&b_tokens).count();
    intersection as f32 / union.max(1) as f32
}

fn tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|token| token.len() >= 3)
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn has_contradiction_marker(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    let pairs = [
        (" enabled", " disabled"),
        (" allow", " deny"),
        (" allowed", " forbidden"),
        (" true", " false"),
        (" required", " optional"),
        (" must ", " must not "),
        (" should ", " should not "),
        (" can ", " cannot "),
        (" is ", " is not "),
        ("支持", "不支持"),
        ("允许", "禁止"),
        ("必须", "不能"),
    ];
    pairs.iter().any(|(positive, negative)| {
        (a.contains(positive) && b.contains(negative))
            || (a.contains(negative) && b.contains(positive))
    })
}

fn signals_from_entry(entry: &ContextEntry, now: DateTime<Utc>) -> HorizonQualitySignals {
    let quality = entry.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
    let consolidation = entry.metadata.consolidation.as_ref();
    HorizonQualitySignals {
        adoption_rate: entry
            .metadata
            .custom_field("adoption_rate")
            .unwrap_or(quality),
        recall_rate: entry
            .metadata
            .custom_field("recall_rate")
            .unwrap_or(quality),
        downstream_success_rate: entry
            .metadata
            .custom_field("downstream_success_rate")
            .unwrap_or(0.5),
        contradiction_count: entry
            .metadata
            .custom_field("contradiction_count")
            .unwrap_or_else(|| {
                usize::from(
                    entry
                        .metadata
                        .tags
                        .iter()
                        .any(|tag| tag.contains("contradiction") || tag.contains("conflict")),
                )
            }),
        corroboration_count: consolidation.map(|meta| meta.corroboration).unwrap_or(0),
        repeated_observations: entry
            .metadata
            .custom_field("repeated_observations")
            .unwrap_or(1),
        user_confirmed: entry
            .metadata
            .custom_field("user_confirmed")
            .unwrap_or(false),
        user_corrected: entry
            .metadata
            .custom_field("user_corrected")
            .unwrap_or(false),
        retrieval_ignored_rate: entry
            .metadata
            .custom_field("retrieval_ignored_rate")
            .unwrap_or(0.0),
        info_gain: entry.metadata.custom_field("info_gain").unwrap_or(0.0),
        now,
    }
}

fn update_dimension(
    state: &mut QualityBeliefState,
    dimension: QualityBeliefDimension,
    observation: f32,
    reliability: f32,
) {
    let updated = state
        .dimension(dimension)
        .update(observation, reliability, 2.0);
    match dimension {
        QualityBeliefDimension::Intrinsic => state.intrinsic = updated,
        QualityBeliefDimension::Relevance => state.relevance = updated,
        QualityBeliefDimension::Utility => state.utility = updated,
        QualityBeliefDimension::Factuality => state.factuality = updated,
        QualityBeliefDimension::Consistency => state.consistency = updated,
        QualityBeliefDimension::Freshness => state.freshness = updated,
        QualityBeliefDimension::Stability => state.stability = updated,
        QualityBeliefDimension::Evidence => state.evidence = updated,
        QualityBeliefDimension::Trainability => state.trainability = updated,
    }
}

fn all_dimensions(state: &QualityBeliefState) -> [DimensionBelief; 9] {
    [
        state.intrinsic,
        state.relevance,
        state.utility,
        state.factuality,
        state.consistency,
        state.freshness,
        state.stability,
        state.evidence,
        state.trainability,
    ]
}

fn average_uncertainty(state: &QualityBeliefState) -> f32 {
    all_dimensions(state)
        .iter()
        .map(|belief| belief.uncertainty)
        .sum::<f32>()
        / 9.0
}

fn average_mean(state: &QualityBeliefState) -> f32 {
    all_dimensions(state)
        .iter()
        .map(|belief| belief.mean)
        .sum::<f32>()
        / 9.0
}

fn action_tag(action: ActiveLearningAction) -> &'static str {
    match action {
        ActiveLearningAction::LocalQuery => "active-learning:local-query",
        ActiveLearningAction::FederatedDiscovery => "active-learning:federated-discovery",
        ActiveLearningAction::UserVerification => "active-learning:user-verification",
        ActiveLearningAction::RehearsalProbe => "active-learning:rehearsal-probe",
    }
}

fn push_tag(tags: &mut Vec<String>, tag: &str) {
    if !tags.iter().any(|existing| existing == tag) {
        tags.push(tag.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{
        ConsolidationMeta, ConsolidationStatus, ContentPayload, ContextMeta, LineageEntry,
        MediaType, MvccVersion, TenantId,
    };
    use uuid::Uuid;

    fn entry(path: &str, text: &str, content_type: ContentType, quality: f32) -> ContextEntry {
        let uri = ContextUri::parse(path).unwrap();
        let mut entry = ContextEntry::new_text(uri, TenantId(Uuid::new_v4()), text);
        entry.media_type = MediaType::Text;
        entry.payload = ContentPayload::Text {
            sparse: text.into(),
            dense: text.into(),
            full: text.into(),
        };
        entry.metadata = ContextMeta {
            content_type: Some(content_type),
            quality_score: Some(quality),
            ..Default::default()
        };
        entry
    }

    fn consolidation(
        evidence_uris: Vec<ContextUri>,
        entangled_with: Vec<ContextUri>,
        corroboration: usize,
    ) -> ConsolidationMeta {
        ConsolidationMeta {
            source: "test".into(),
            generation: 1,
            status: ConsolidationStatus::InProgress,
            patch_count: 0,
            lineage: vec![LineageEntry {
                version: MvccVersion(0),
                timestamp: Utc::now(),
                change_summary: "test".into(),
            }],
            evidence_uris,
            corroboration,
            half_life: None,
            entangled_with,
        }
    }

    #[tokio::test]
    async fn health_diagnoses_and_marks_dangerous_bridge() -> Result<(), ContextError> {
        let mut hub = entry(
            "uwu://t/a/memory/fact/hub",
            "shared policy",
            ContentType::Fact,
            0.32,
        );
        let leaf_a = entry(
            "uwu://t/a/memory/fact/a",
            "depends a",
            ContentType::Fact,
            0.8,
        );
        let leaf_b = entry(
            "uwu://t/a/memory/fact/b",
            "depends b",
            ContentType::Fact,
            0.8,
        );
        hub.metadata.consolidation = Some(consolidation(vec![], vec![], 0));
        let hub_uri = hub.uri.clone();
        let mut leaf_a = leaf_a;
        let mut leaf_b = leaf_b;
        leaf_a.metadata.consolidation = Some(consolidation(vec![hub_uri.clone()], vec![], 1));
        leaf_b.metadata.consolidation = Some(consolidation(vec![hub_uri], vec![], 1));
        let mut entries = vec![hub, leaf_a, leaf_b];

        let diagnostician =
            KnowledgeHealthDiagnostician::new(KnowledgeHealthConfig::default()).unwrap();
        let report = diagnostician.diagnose(&entries, Utc::now()).await?;
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.issue == HealthIssueKind::DangerousBridge)
        );
        assert_eq!(diagnostician.apply_repairs(&mut entries, &report)?, 1);
        assert!(
            entries[0]
                .metadata
                .tags
                .contains(&"health:revalidate".to_string())
        );
        Ok(())
    }

    #[test]
    fn guardian_invalidates_weaker_contradiction() -> Result<(), ContextError> {
        let strong = entry(
            "uwu://t/a/memory/fact/strong",
            "cache must be enabled for writes",
            ContentType::Fact,
            0.9,
        );
        let weak = entry(
            "uwu://t/a/memory/fact/weak",
            "cache must not be enabled for writes",
            ContentType::Fact,
            0.4,
        );
        let mut entries = vec![strong, weak];
        let guardian =
            ActiveConsistencyGuardian::new(ConsistencyGuardianConfig::default()).unwrap();
        let plan = guardian.plan(&entries, Utc::now());
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(
            plan.tasks[0].action,
            ConsistencyRepairAction::InvalidateWeaker
        );
        assert_eq!(guardian.apply(&mut entries, &plan, Utc::now())?, 1);
        assert!(entries[1].metadata.validity.is_some());
        Ok(())
    }

    #[test]
    fn embedding_drift_detects_shift_and_marks_rebuild() -> Result<(), ContextError> {
        let monitor = EmbeddingDriftMonitor::new(EmbeddingDriftConfig {
            min_sample_size: 2,
            centroid_shift_threshold: 0.05,
            psi_threshold: 0.05,
            norm_kl_threshold: 0.05,
            ..Default::default()
        })
        .unwrap();
        let mut baseline_anchor =
            entry("uwu://t/a/memory/fact/base", "base", ContentType::Fact, 0.8);
        let baseline = snapshot_from_vectors(&[vec![1.0, 0.0], vec![0.9, 0.1]], 5, Utc::now());
        baseline_anchor
            .metadata
            .set_custom_field(EMBEDDING_SNAPSHOT_KEY, &baseline)
            .unwrap();
        let mut shifted_a = entry("uwu://t/a/memory/fact/a", "a", ContentType::Fact, 0.8);
        let mut shifted_b = entry("uwu://t/a/memory/fact/b", "b", ContentType::Fact, 0.8);
        shifted_a
            .metadata
            .set_custom_field(EMBEDDING_VECTOR_KEY, &vec![0.0_f32, 1.0])
            .unwrap();
        shifted_b
            .metadata
            .set_custom_field(EMBEDDING_VECTOR_KEY, &vec![0.1_f32, 0.9])
            .unwrap();
        let mut entries = vec![baseline_anchor, shifted_a, shifted_b];
        let report = monitor.analyze(&entries, Utc::now());
        assert_eq!(report.action, EmbeddingDriftAction::RebuildIndex);
        assert_eq!(monitor.apply(&mut entries, &report)?, 1);
        assert!(
            entries[0]
                .metadata
                .tags
                .contains(&"embedding-drift:rebuild-index".to_string())
        );
        Ok(())
    }

    #[test]
    fn curiosity_explorer_promotes_uncertainty_and_gaps() -> Result<(), ContextError> {
        let mut target = entry(
            "uwu://t/a/memory/fact/gap",
            "cache audit policy",
            ContentType::Fact,
            0.42,
        );
        target
            .metadata
            .set_custom_field(
                "discovered_gaps",
                &vec![CuriosityGap {
                    description: "which audit logs prove cache invalidation".into(),
                    severity: 0.8,
                    suggested_exploration: "run STORM synthesis over audit evidence".into(),
                }],
            )
            .unwrap();
        let mut entries = vec![target];
        let explorer = CuriosityExplorer::new(CuriosityExplorerConfig::default()).unwrap();
        let tasks = explorer.plan(&entries, Utc::now());
        assert!(
            tasks
                .iter()
                .any(|task| task.action == CuriosityAction::StormSynthesis)
        );
        assert_eq!(explorer.apply_tasks(&mut entries, &tasks)?, 1);
        assert!(
            entries[0]
                .metadata
                .tags
                .contains(&"curiosity:planned".to_string())
        );
        Ok(())
    }

    #[test]
    fn active_learning_plans_and_applies_observations() -> Result<(), ContextError> {
        let mut target = entry(
            "uwu://t/a/memory/fact/uncertain",
            "external vector index can support audit search",
            ContentType::Fact,
            0.48,
        );
        target.metadata.consolidation = Some(consolidation(vec![], vec![], 0));
        let mut entries = vec![target];
        let loop_ = ActiveLearningLoop::new(
            ActiveLearningPlanner::default(),
            HorizonAwareQualityScorer::default(),
            ActiveLearningExecutionConfig::default(),
        )
        .unwrap();
        let tasks = loop_.plan(&entries, Utc::now());
        assert!(!tasks.is_empty());
        assert_eq!(loop_.apply_tasks(&mut entries, &tasks)?, 1);
        assert!(
            entries[0]
                .metadata
                .tags
                .iter()
                .any(|tag| tag.starts_with("active-learning:"))
        );

        let observation = ActiveLearningObservation {
            target_uri: entries[0].uri.clone(),
            dimension: tasks[0].dimension,
            observation: 0.9,
            reliability: 0.8,
            note: "confirmed by external source".into(),
        };
        assert_eq!(
            loop_.apply_observations(&mut entries, &[observation], Utc::now())?,
            1
        );
        assert!(
            entries[0]
                .metadata
                .custom_field::<QualityBeliefState>(QUALITY_BELIEF_KEY)
                .is_some()
        );
        Ok(())
    }
}
