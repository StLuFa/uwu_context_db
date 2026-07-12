//! Validated configuration for tunable retrieval thresholds and limits.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrieveConfigError(String);

impl RetrieveConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for RetrieveConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RetrieveConfigError {}

type ConfigResult<T> = Result<T, RetrieveConfigError>;

fn finite(name: &str, values: &[f32]) -> ConfigResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(RetrieveConfigError::new(format!(
            "{name} contains a non-finite value"
        )))
    }
}

fn unit_sum(name: &str, values: &[f32]) -> ConfigResult<()> {
    finite(name, values)?;
    if values.iter().any(|value| *value < 0.0) || (values.iter().sum::<f32>() - 1.0).abs() > 1e-5 {
        Err(RetrieveConfigError::new(format!(
            "{name} weights must be non-negative and sum to 1"
        )))
    } else {
        Ok(())
    }
}

fn nonzero(name: &str, values: &[usize]) -> ConfigResult<()> {
    if values.contains(&0) {
        Err(RetrieveConfigError::new(format!(
            "{name} limits must be non-zero"
        )))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GraphRagConfig {
    pub community_summary_weight: f32,
    pub community_node_weight: f32,
    pub global_boost: f32,
    pub hit_community_weight: f32,
    pub hit_node_weight: f32,
    pub index_centrality_weight: f32,
    pub retrieval_centrality_weight: f32,
    pub degree_weight: f32,
    pub degree_saturation: f32,
    pub default_centrality: f32,
    pub lexical_weight: f32,
    pub min_hit_relevance: f32,
    pub request_hops: usize,
    pub request_communities: usize,
    pub request_nodes: usize,
    pub index_hops: usize,
    pub summary_nodes: usize,
    pub evidence_chars: usize,
    pub summary_chars: usize,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            community_summary_weight: 0.72,
            community_node_weight: 0.20,
            global_boost: 0.08,
            hit_community_weight: 0.88,
            hit_node_weight: 0.12,
            index_centrality_weight: 0.72,
            retrieval_centrality_weight: 0.75,
            degree_weight: 0.28,
            degree_saturation: 12.0,
            default_centrality: 0.5,
            lexical_weight: 0.2,
            min_hit_relevance: 0.05,
            request_hops: 2,
            request_communities: 4,
            request_nodes: 8,
            index_hops: 3,
            summary_nodes: 16,
            evidence_chars: 360,
            summary_chars: 900,
        }
    }
}

impl GraphRagConfig {
    pub fn validate(self) -> ConfigResult<Self> {
        unit_sum(
            "GraphRAG community",
            &[
                self.community_summary_weight,
                self.community_node_weight,
                self.global_boost,
            ],
        )?;
        unit_sum(
            "GraphRAG hit",
            &[self.hit_community_weight, self.hit_node_weight],
        )?;
        unit_sum(
            "GraphRAG index node",
            &[self.index_centrality_weight, self.degree_weight],
        )?;
        finite(
            "GraphRAG",
            &[
                self.retrieval_centrality_weight,
                self.degree_saturation,
                self.default_centrality,
                self.lexical_weight,
                self.min_hit_relevance,
            ],
        )?;
        if self.retrieval_centrality_weight < 0.0 || self.retrieval_centrality_weight > 1.0 {
            return Err(RetrieveConfigError::new(
                "GraphRAG retrieval centrality weight must be in [0, 1]",
            ));
        }
        if self.degree_saturation <= 0.0 {
            return Err(RetrieveConfigError::new(
                "GraphRAG degree saturation must be positive",
            ));
        }
        nonzero(
            "GraphRAG",
            &[
                self.request_hops,
                self.request_communities,
                self.request_nodes,
                self.index_hops,
                self.summary_nodes,
                self.evidence_chars,
                self.summary_chars,
            ],
        )?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RagSynthesisConfig {
    pub relevance_weight: f32,
    pub quality_weight: f32,
    pub evidence_weight: f32,
    pub coverage_weight: f32,
    pub consistency_weight: f32,
    pub contradiction_penalty: f32,
    pub max_contradiction_penalty: f32,
}

impl Default for RagSynthesisConfig {
    fn default() -> Self {
        Self {
            relevance_weight: 0.65,
            quality_weight: 0.35,
            evidence_weight: 0.45,
            coverage_weight: 0.30,
            consistency_weight: 0.25,
            contradiction_penalty: 0.50,
            max_contradiction_penalty: 0.85,
        }
    }
}

impl RagSynthesisConfig {
    pub fn validate(self) -> ConfigResult<Self> {
        unit_sum(
            "RAG citation",
            &[self.relevance_weight, self.quality_weight],
        )?;
        unit_sum(
            "RAG confidence",
            &[
                self.evidence_weight,
                self.coverage_weight,
                self.consistency_weight,
            ],
        )?;
        finite(
            "RAG penalties",
            &[self.contradiction_penalty, self.max_contradiction_penalty],
        )?;
        if self.contradiction_penalty < 0.0
            || !(0.0..=1.0).contains(&self.max_contradiction_penalty)
        {
            return Err(RetrieveConfigError::new(
                "RAG penalties are outside their valid range",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InnovationConfig {
    pub learning_rate: f32,
    pub association_increment: f32,
    pub association_initial: f32,
    pub association_decay: f32,
    pub semantic_weight: f32,
    pub uri_weight: f32,
    pub query_type_weight: f32,
    pub exploration_weight: f32,
    pub max_missing_patterns: usize,
}

impl Default for InnovationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            association_increment: 0.1,
            association_initial: 0.3,
            association_decay: 0.99,
            semantic_weight: 0.58,
            uri_weight: 0.24,
            query_type_weight: 0.14,
            exploration_weight: 0.04,
            max_missing_patterns: 128,
        }
    }
}

impl InnovationConfig {
    pub fn validate(self) -> ConfigResult<Self> {
        finite(
            "innovation",
            &[
                self.learning_rate,
                self.association_increment,
                self.association_initial,
                self.association_decay,
            ],
        )?;
        unit_sum(
            "innovation ranking",
            &[
                self.semantic_weight,
                self.uri_weight,
                self.query_type_weight,
                self.exploration_weight,
            ],
        )?;
        nonzero("innovation", &[self.max_missing_patterns])?;
        if !(0.0..=1.0).contains(&self.learning_rate)
            || !(0.0..=1.0).contains(&self.association_decay)
        {
            return Err(RetrieveConfigError::new(
                "innovation rates must be in [0, 1]",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QueryPlanConfig {
    pub default_scan_limit: usize,
    pub default_type_rows: usize,
    pub default_scope_rows: usize,
    pub default_depth: usize,
    pub max_type_limit: usize,
    pub temporal_limit: usize,
    pub preferred_temporal_limit: usize,
    pub type_scan_cost: f64,
    pub prefix_scan_cost: f64,
    pub vector_cost: f64,
    pub graph_cost: f64,
    pub filter_multiplier: f64,
    pub sort_multiplier: f64,
    pub hash_join_cost: f64,
    pub full_scan_cost: f64,
}

impl Default for QueryPlanConfig {
    fn default() -> Self {
        Self {
            default_scan_limit: 1000,
            default_type_rows: 100,
            default_scope_rows: 1000,
            default_depth: 2,
            max_type_limit: 4096,
            temporal_limit: 10,
            preferred_temporal_limit: 25,
            type_scan_cost: 0.001,
            prefix_scan_cost: 0.002,
            vector_cost: 0.01,
            graph_cost: 10.0,
            filter_multiplier: 1.1,
            sort_multiplier: 1.5,
            hash_join_cost: 5.0,
            full_scan_cost: 0.05,
        }
    }
}

impl QueryPlanConfig {
    pub fn validate(self) -> ConfigResult<Self> {
        nonzero(
            "query plan",
            &[
                self.default_scan_limit,
                self.default_type_rows,
                self.default_scope_rows,
                self.default_depth,
                self.max_type_limit,
                self.temporal_limit,
                self.preferred_temporal_limit,
            ],
        )?;
        let costs = [
            self.type_scan_cost,
            self.prefix_scan_cost,
            self.vector_cost,
            self.graph_cost,
            self.filter_multiplier,
            self.sort_multiplier,
            self.hash_join_cost,
            self.full_scan_cost,
        ];
        if costs.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(RetrieveConfigError::new(
                "query-plan costs must be finite and non-negative",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanCacheConfig {
    pub capacity: usize,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self { capacity: 256 }
    }
}

impl PlanCacheConfig {
    pub fn validate(self) -> ConfigResult<Self> {
        nonzero("plan cache", &[self.capacity])?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TokenBudgetConfig {
    pub l0_floor: usize,
    pub l1_floor: usize,
    pub l2_floor: usize,
}
impl Default for TokenBudgetConfig {
    fn default() -> Self {
        Self {
            l0_floor: 1,
            l1_floor: 50,
            l2_floor: 100,
        }
    }
}
impl TokenBudgetConfig {
    pub fn validate(self) -> ConfigResult<Self> {
        nonzero(
            "token budget",
            &[self.l0_floor, self.l1_floor, self.l2_floor],
        )?;
        Ok(self)
    }
}
