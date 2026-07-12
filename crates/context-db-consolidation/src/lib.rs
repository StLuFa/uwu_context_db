//! # context-db-consolidation
//!
//! 高级学习巩固模块：迭代自精炼 + 四操作决策 + 双时序管理 + 反向进化 + 单Agent十大创新。
//!
//! ## 核心流程
//! Generate → Critique → Revise → Resolve(四操作) → Write → Index
//! → Score → BackwardEvolve → ConsistencyCheck → Active
//!
//! ## 模块
//! - 引擎：ConsolidationEngine / MemoryResolver / DualTimeline / BackwardEvolver
//! - 单 Agent 创新：rif / halflife / associative / entanglement / opportunity / explainable

// associative 已移入 context-db-retrieve —— 检索层负责联想扩展
pub mod batch;
pub mod entanglement;
pub mod explainable;
pub mod guard;
pub mod halflife;
pub mod health;
pub mod lifecycle_execution;
pub mod lineage;
pub mod loader;
pub mod opportunity;
pub mod patcher;
pub mod prompt_budget;
pub mod quality;
pub mod relational_axis;
pub mod rif;
pub mod security;
pub mod self_consistency;
pub mod semantic_axis;
pub mod tiered_cache;

pub use loader::{BanditBudgetPolicy, LoadFeedback, LoadLevel, ProgressiveLoader};
pub use self_consistency::{
    ConsistencyVoteCluster, SelfConsistencyConfig, SelfConsistencyConsolidator,
    SelfConsistencyReport,
};

use crate::halflife::SpacedRepetitionScheduler;
use crate::health::{
    ActiveConsistencyGuardian, ActiveLearningLoop, CuriosityExplorer, EmbeddingDriftMonitor,
    KnowledgeHealthDiagnostician,
};
use crate::quality::{HorizonAwareQualityScorer, HorizonQualitySignals, QualityRoute};
use agent_context_db_core::{
    ConsolidationStatus, ContentType, ContextEntry, ContextError, ContextUri, EpistemicType,
    JsonSchema, LineageEntry, LlmClient, LlmError, LlmOpts, PageRequest, Result, StateScope,
    ValidityRecord,
};
use agent_context_db_knowledge_network::identity::IdentityRegistry;
use agent_context_db_marketplace::{
    AgentId, KnowledgeProvenance, KnowledgeProvenancePayload, PublishableProduct,
    evidence_chain_hash,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Error returned when a consolidation component receives an invalid configuration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid consolidation configuration: {0}")]
pub struct ConfigError(pub String);

impl From<ConfigError> for ContextError {
    fn from(error: ConfigError) -> Self {
        ContextError::Unsupported(error.to_string())
    }
}

fn validate_unit_f32(name: &str, value: f32) -> std::result::Result<(), ConfigError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError(format!("{name} must be finite and in [0, 1]")))
    }
}

// ===========================================================================
// ConsolidationProduct — 巩固产物
// ===========================================================================

/// 巩固产物 — 经过 Generate→Critique→Revise 收敛后的精炼记忆。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationProduct {
    pub uri: ContextUri,
    pub content_type: ContentType,
    pub epistemic_type: EpistemicType,
    pub content: String,
    pub quality_score: f32,
    pub confidence: f32,
    /// Whether this product explicitly claims that supporting evidence is required.
    pub evidence_required: bool,
    pub superseded_claim: Option<String>,
    pub evidence_uris: Vec<ContextUri>,
    pub contradiction_uris: Vec<ContextUri>,
    pub error_pattern: Option<String>,
    pub hypothesis_outcome: Option<HypothesisOutcome>,
    pub preconditions: Option<String>,
    pub expected_outcome: Option<String>,
    pub related_policy_uris: Vec<ContextUri>,
    pub provenance: Option<KnowledgeProvenance>,
    pub metadata: ConsolidationProductMeta,
}

impl ConsolidationProduct {
    pub fn provenance_payload(
        &self,
        publisher: AgentId,
        created_at: DateTime<Utc>,
    ) -> KnowledgeProvenancePayload {
        KnowledgeProvenancePayload {
            publisher,
            content: self.content.clone(),
            evidence_chain_hash: evidence_chain_hash(&self.evidence_uris),
            evidence_uris: self.evidence_uris.clone(),
            quality_score: self.quality_score,
            confidence: self.confidence,
            epistemic_type: self.epistemic_type,
            content_type: self.content_type,
            created_at,
        }
    }

    pub fn sign_provenance(
        &mut self,
        publisher: AgentId,
        identities: &IdentityRegistry,
    ) -> std::result::Result<(), agent_context_db_knowledge_network::types::KnowledgeNetworkError>
    {
        let payload = self.provenance_payload(publisher, Utc::now());
        self.provenance = Some(identities.sign_knowledge_provenance(&payload)?);
        Ok(())
    }
}

impl PublishableProduct for ConsolidationProduct {
    fn quality_score(&self) -> f32 {
        self.quality_score
    }

    fn content(&self) -> &str {
        &self.content
    }

    fn content_type(&self) -> ContentType {
        self.content_type
    }

    fn evidence_uris(&self) -> &[ContextUri] {
        &self.evidence_uris
    }

    fn confidence(&self) -> f32 {
        self.confidence
    }

    fn provenance(&self) -> Option<KnowledgeProvenance> {
        self.provenance.clone()
    }

    fn epistemic_type(&self) -> EpistemicType {
        self.epistemic_type
    }

    fn half_life(&self) -> Option<agent_context_db_core::HalfLife> {
        self.metadata.half_life
    }
}

/// 假设验证结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisOutcome {
    Confirmed,
    Falsified,
    Inconclusive,
}

/// 巩固元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationProductMeta {
    pub source_session: Option<String>,
    pub generation: usize,
    pub status: ConsolidationStatus,
    pub patch_count: usize,
    pub lineage: Vec<LineageEntry>,
    pub validity: Option<ValidityRecord>,
    pub half_life: Option<agent_context_db_core::HalfLife>,
}

// ===========================================================================
// 四操作
// ===========================================================================

/// 操作决策。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveAction {
    Add,
    Update,
    Invalidate,
    Noop,
}

/// MemoryResolver — 四操作，基于信息增益 + 边际效用 + 领域饱和检测。
pub struct MemoryResolver {
    /// 领域饱和度阈值：同领域条目超此数 → 倾向 NOOP
    domain_saturation_threshold: usize,
}

impl Default for MemoryResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryResolver {
    pub fn new() -> Self {
        Self {
            domain_saturation_threshold: 50,
        }
    }

    /// 决策 四操作（ADD / UPDATE / INVALIDATE / NOOP）。
    ///
    /// 决策流程：
    /// 1. 质量太低 → NOOP
    /// 2. 与已有 fact 矛盾 → INVALIDATE（软失效）
    /// 3. 信息增益不足（已有相似条目）→ NOOP
    /// 4. 领域饱和 → NOOP（边际效用为负）
    /// 5. 高质量 + 有证据 → ADD
    /// 6. 已有条目需要更新 → UPDATE
    pub fn resolve(
        &self,
        product: &ConsolidationProduct,
        existing: Option<&ContextEntry>,
        similar_count: usize,    // 语义相似条目的数量
        has_contradiction: bool, // 是否与已有 fact 矛盾
    ) -> ResolveAction {
        // 1. 质量太低 → 不操作
        if product.quality_score < 0.15 {
            return ResolveAction::Noop;
        }

        // 2. 矛盾 → 软失效
        if has_contradiction {
            return ResolveAction::Invalidate;
        }

        // 3. 信息增益不足（已有 ≥3 个高度相似的条目）
        if similar_count >= 3 {
            return ResolveAction::Noop;
        }

        // 4. 领域饱和（边际效用为负）
        if similar_count as f32 > self.domain_saturation_threshold as f32 * product.quality_score {
            return ResolveAction::Noop;
        }

        // 5. 高质量 + 无相似 → 新增
        if product.quality_score > 0.7 && similar_count == 0 {
            // Fact 类必须有证据
            if product.content_type == ContentType::Fact && product.evidence_uris.is_empty() {
                return ResolveAction::Noop; // Fact 无证据 → 拒绝
            }

            return ResolveAction::Add;
        }

        // 6. 中等质量 + 已有条目 → 更新
        if product.quality_score > 0.3 {
            if let Some(_existing) = existing {
                // 已有条目存在：新产品信息增益更高才更新
                let info_gain = product.quality_score - product.confidence;
                if info_gain > 0.1 {
                    return ResolveAction::Update;
                }

                return ResolveAction::Noop; // 增益不足
            }

            return ResolveAction::Add; // 无已有条目
        }

        ResolveAction::Noop
    }
}

// ===========================================================================
// 单 Agent 创新 — 认识论分类
// ===========================================================================

/// 认识论分类器 — 基于内容特征判断 knowledge type。
pub struct EpistemicTyper;

impl Default for EpistemicTyper {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicTyper {
    pub fn new() -> Self {
        Self
    }

    pub fn classify(
        &self,
        _content: &str,
        meta: &agent_context_db_core::ContextMeta,
    ) -> EpistemicType {
        meta.epistemic_type().unwrap_or(EpistemicType::Fact)
    }
}

// ===========================================================================
// 单 Agent 创新 — 元无知映射
// ===========================================================================

/// 无知地图 — 建模"缺失了什么"。
pub struct IgnoranceMap {
    blind_spots: parking_lot::RwLock<std::collections::HashMap<String, BlindSpot>>,
}

#[derive(Debug, Clone)]
pub struct BlindSpot {
    pub pattern: String,
    pub missing_count: usize,
    pub severity: f32,
}

impl Default for IgnoranceMap {
    fn default() -> Self {
        Self::new()
    }
}

impl IgnoranceMap {
    pub fn new() -> Self {
        Self {
            blind_spots: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    pub fn record_missing(&self, query: &str) {
        let mut spots = self.blind_spots.write();
        let spot = spots.entry(query.to_string()).or_insert_with(|| BlindSpot {
            pattern: query.to_string(),
            missing_count: 0,
            severity: 0.0,
        });
        spot.missing_count += 1;
        spot.severity = spot.missing_count as f32;
    }

    pub fn top_blind_spots(&self, n: usize) -> Vec<BlindSpot> {
        let mut spots: Vec<_> = self.blind_spots.read().values().cloned().collect();
        spots.sort_by(|a, b| b.severity.total_cmp(&a.severity));
        spots.truncate(n);
        spots
    }
}

trait Sigmoid {
    fn sigmoid(self) -> Self;
}

impl Sigmoid for f64 {
    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }
}

// ===========================================================================
// 单 Agent 创新 — 置信度校准（per-type 温度缩放）
// ===========================================================================

/// 校准记录 — 按认识论类型分组追踪声明 vs 实际。
#[derive(Debug, Clone, Default)]
pub struct CalibrationRecord {
    pub declared_confidences: Vec<f32>,
    pub actual_adoption_rates: Vec<f32>,
    pub temperature: f64,
}

/// 置信度校准器 — per-type 温度缩放，修复 LLM 过度自信。
pub struct ConfidenceCalibrator {
    /// 按 epistemic_type 分组的校准数据
    calibration_data:
        parking_lot::RwLock<std::collections::HashMap<EpistemicType, CalibrationRecord>>,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalibrator {
    pub fn new() -> Self {
        Self {
            calibration_data: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// 校准一个声明置信度（per-type 温度缩放）。
    pub fn calibrate(&self, declared: f32, epistemic: EpistemicType) -> f32 {
        let data = self.calibration_data.read();
        if let Some(record) = data.get(&epistemic) {
            if record.temperature <= 0.0 || declared <= 0.0 || declared >= 1.0 {
                return declared;
            }

            let logit = ((declared as f64) / (1.0 - declared as f64)).ln();
            let calibrated = (logit / record.temperature).sigmoid();
            calibrated as f32
        } else {
            declared // 冷启动阶段不校准
        }
    }

    /// Sleeptime 阶段更新校准数据。
    pub fn update(&self, epistemic: EpistemicType, declared: f32, adopted: bool) {
        let mut data = self.calibration_data.write();
        let record = data.entry(epistemic).or_default();
        record.declared_confidences.push(declared);
        record
            .actual_adoption_rates
            .push(if adopted { 1.0 } else { 0.0 });

        // 每 10 个数据点重新拟合温度
        if record.declared_confidences.len().is_multiple_of(10) {
            record.temperature = Self::fit_temperature(record);
        }
    }

    /// 线性拟合温度参数。
    fn fit_temperature(record: &CalibrationRecord) -> f64 {
        if record.declared_confidences.len() < 5 {
            return 1.0;
        }

        let n = record
            .declared_confidences
            .len()
            .min(record.actual_adoption_rates.len());
        let mut sum_ratio = 0.0;
        let mut count = 0;
        for i in 0..n {
            let declared = record.declared_confidences[i] as f64;
            let actual = record.actual_adoption_rates[i] as f64;
            if declared > 0.0 && declared < 1.0 {
                // ratio = logit(actual) / logit(declared)
                // temperature = 1/ratio
                let bounded_actual = if actual.is_nan() {
                    actual
                } else {
                    actual.clamp(0.01, 0.99)
                };
                let actual_logit = (bounded_actual / (1.0 - bounded_actual)).ln();
                let declared_logit = (declared / (1.0 - declared)).ln();
                if declared_logit != 0.0 {
                    sum_ratio += actual_logit / declared_logit;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg_ratio = sum_ratio / count as f64;
            (1.0 / avg_ratio).clamp(0.5, 3.0) // 限制在合理范围
        } else {
            1.0
        }
    }

    /// 获取所有已校准类型的温度信息。
    pub fn get_temperatures(&self) -> Vec<(EpistemicType, f64, usize)> {
        self.calibration_data
            .read()
            .iter()
            .map(|(et, rec)| (*et, rec.temperature, rec.declared_confidences.len()))
            .collect()
    }

    pub fn total_labeled_samples(&self) -> usize {
        self.calibration_data
            .read()
            .values()
            .map(|rec| rec.actual_adoption_rates.len())
            .sum()
    }
}

// ===========================================================================
// 增强版 ConsolidationEngine — Generate→Critique→Revise 收敛循环
// ===========================================================================

/// 巩固配置。
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub quality_threshold_add: f32,
    pub quality_threshold_update: f32,
    pub prompt_budget: prompt_budget::PromptBudgetConfig,
}

impl ConsolidationConfig {
    pub fn validate(&self) -> std::result::Result<(), ConfigError> {
        if self.max_iterations == 0 {
            return Err(ConfigError("max_iterations must be nonzero".into()));
        }
        validate_unit_f32("convergence_threshold", self.convergence_threshold)?;
        validate_unit_f32("quality_threshold_add", self.quality_threshold_add)?;
        validate_unit_f32("quality_threshold_update", self.quality_threshold_update)?;
        self.prompt_budget.validate()?;
        if self.quality_threshold_update > self.quality_threshold_add {
            return Err(ConfigError(
                "quality_threshold_update must not exceed quality_threshold_add".into(),
            ));
        }
        Ok(())
    }
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_threshold: 0.05,
            quality_threshold_add: 0.7,
            quality_threshold_update: 0.3,
            prompt_budget: prompt_budget::PromptBudgetConfig::default(),
        }
    }
}

/// 增强版巩固引擎 — 带迭代收敛，LLM 驱动的 Generate→Critique→Revise 循环。
pub struct ConsolidationEngine {
    config: ConsolidationConfig,
    llm: Arc<dyn LlmClient>,
    calibrator: ConfidenceCalibrator,
    ignorance: IgnoranceMap,
}

impl ConsolidationEngine {
    pub fn new(
        config: ConsolidationConfig,
        llm: Arc<dyn LlmClient>,
    ) -> std::result::Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            llm,
            calibrator: ConfidenceCalibrator::new(),
            ignorance: IgnoranceMap::new(),
        })
    }

    fn budget_prompt(&self, sections: &[prompt_budget::PromptSection<'_>]) -> Result<String> {
        prompt_budget::budget_prompt(self.config.prompt_budget, sections)
            .map(|result| result.text)
            .map_err(|error| ContextError::Unsupported(error.to_string()))
    }

    /// Generate→Critique→Revise 收敛循环。
    #[tracing::instrument(skip(self, entry))]
    pub async fn consolidate(&self, entry: &ContextEntry) -> Result<ConsolidationProduct> {
        let content = entry.l0_text().to_string();
        let ct = entry.content_type().unwrap_or(ContentType::Fact);
        let et = entry
            .metadata
            .epistemic_type()
            .unwrap_or(EpistemicType::Fact);

        // Generate (LLM-driven)
        let mut current = self.generate(entry, &content, ct, et).await?;
        let mut prev_score = 0.0;

        for iteration in 1..=self.config.max_iterations {
            // Critique
            let critique = self.critique(&current, entry).await?;
            let calibrated = self.calibrator.calibrate(critique.confidence, et);

            // 收敛检查
            let delta = (critique.quality_score - prev_score).abs();
            prev_score = critique.quality_score;

            if delta < self.config.convergence_threshold && iteration > 1 {
                current.quality_score = critique.quality_score;
                current.confidence = calibrated;
                current.metadata.status = ConsolidationStatus::Converged;
                current.metadata.generation = iteration;
                break;
            }

            // Revise
            if critique.suggestions.is_empty() {
                current.quality_score = critique.quality_score;
                current.confidence = calibrated;
                current.metadata.status = ConsolidationStatus::Converged;
                current.metadata.generation = iteration;
                break;
            }

            let revised = self.revise(&current, &critique).await?;
            current = revised;
            current.confidence = calibrated;
            current.metadata.generation = iteration;
        }

        if current.metadata.status != ConsolidationStatus::Converged {
            current.metadata.status = ConsolidationStatus::Stale;
        }

        current.metadata.patch_count = current.metadata.generation as usize;
        Ok(current)
    }

    async fn generate(
        &self,
        entry: &ContextEntry,
        content: &str,
        ct: ContentType,
        et: EpistemicType,
    ) -> Result<ConsolidationProduct> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct GeneratedInsight {
            content: String,
            confidence: f32,
            evidence_required: bool,
        }

        let instructions = format!(
            "Consolidate this memory into a concise, self-contained principle. Preserve its language and do not invent evidence. Content type: {ct:?}; epistemic type: {et:?}. Return the required JSON object."
        );
        let prompt = self.budget_prompt(&[
            prompt_budget::PromptSection {
                label: "Instructions",
                content: &instructions,
                priority: 100,
                required: true,
            },
            prompt_budget::PromptSection {
                label: "Memory",
                content,
                priority: 80,
                required: false,
            },
        ])?;
        let schema = JsonSchema::new(serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["content", "confidence", "evidence_required"],
            "properties": {
                "content": {"type": "string", "minLength": 1, "maxLength": 4000},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence_required": {"type": "boolean"}
            }
        }));
        let response = self
            .llm
            .complete_json(&prompt, &schema, &strict_llm_opts())
            .await?;
        let generated: GeneratedInsight = serde_json::from_str(&response)?;
        let consolidated = validate_llm_content(&generated.content, "generated content")?;
        let declared = validate_probability(generated.confidence, "generated confidence")?;
        let evidence_uris = entry
            .metadata
            .consolidation
            .as_ref()
            .map(|meta| meta.evidence_uris.clone())
            .unwrap_or_default();

        Ok(ConsolidationProduct {
            uri: entry.uri.clone(),
            content_type: ct,
            epistemic_type: et,
            content: consolidated,
            quality_score: entry
                .metadata
                .quality_score
                .unwrap_or(declared)
                .clamp(0.0, 1.0),
            confidence: declared,
            evidence_required: generated.evidence_required,
            superseded_claim: None,
            evidence_uris,
            contradiction_uris: vec![],
            error_pattern: None,
            hypothesis_outcome: None,
            preconditions: None,
            expected_outcome: None,
            related_policy_uris: vec![],
            provenance: None,
            metadata: ConsolidationProductMeta {
                source_session: None,
                generation: 1,
                status: ConsolidationStatus::InProgress,
                patch_count: 0,
                lineage: vec![],
                validity: entry.metadata.validity.clone(),
                half_life: entry
                    .metadata
                    .consolidation
                    .as_ref()
                    .and_then(|meta| meta.half_life),
            },
        })
    }

    async fn critique(
        &self,
        product: &ConsolidationProduct,
        _entry: &ContextEntry,
    ) -> Result<CritiqueResult> {
        let prompt = format!(
            r#"Evaluate the quality of this consolidated insight:

Content: "{}"
Type: {:?}


Epistemic type: {:?}



Score the insight on these dimensions (0.0-1.0):
- Clarity: Is it self-contained and understandable?
- Conciseness: Is it free of redundancy?
- Accuracy: Does it seem factually consistent?
- Utility: Would this be useful for future retrieval?

Return a JSON object with:
{{"quality_score": 0.0-1.0, "confidence": 0.0-1.0, "suggestions": ["..."]}}"#,
            product.content, product.content_type, product.epistemic_type
        );

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct StructuredCritique {
            quality_score: f32,
            confidence: f32,
            suggestions: Vec<String>,
        }
        let schema = JsonSchema::new(serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["quality_score", "confidence", "suggestions"],
            "properties": {
                "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "suggestions": {"type": "array", "maxItems": 8, "items": {"type": "string", "minLength": 1, "maxLength": 500}}
            }
        }));
        let response = self
            .llm
            .complete_json(&prompt, &schema, &strict_llm_opts())
            .await?;
        let parsed: StructuredCritique = serde_json::from_str(&response)?;
        let suggestions = parsed
            .suggestions
            .into_iter()
            .map(|value| validate_llm_content(&value, "critique suggestion"))
            .collect::<Result<Vec<_>>>()?;
        Ok(CritiqueResult {
            quality_score: validate_probability(parsed.quality_score, "quality score")?,
            confidence: validate_probability(parsed.confidence, "critique confidence")?,
            suggestions,
        })
    }

    async fn revise(
        &self,
        product: &ConsolidationProduct,
        critique: &CritiqueResult,
    ) -> Result<ConsolidationProduct> {
        let mut revised = product.clone();
        revised.quality_score = critique.quality_score;
        revised.confidence = critique.confidence;

        if !critique.suggestions.is_empty() {
            // Use LLM to apply suggestions
            let prompt = format!(
                r#"Revise this insight based on the critique:

Insight: "{}"
Suggestions: {:?}



Return ONLY the revised insight text (no JSON, no markup)."#,
                product.content, critique.suggestions
            );

            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Revision {
                content: String,
            }
            let schema = JsonSchema::new(serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["content"],
                "properties": {"content": {"type": "string", "minLength": 1, "maxLength": 4000}}
            }));
            let response = self
                .llm
                .complete_json(&prompt, &schema, &strict_llm_opts())
                .await?;
            let revision: Revision = serde_json::from_str(&response)?;
            revised.content = validate_llm_content(&revision.content, "revised content")?;
            revised.metadata.status = ConsolidationStatus::InProgress;
        }

        Ok(revised)
    }

    /// 批量巩固。
    pub async fn consolidate_batch(
        &self,
        entries: &[ContextEntry],
    ) -> Result<Vec<ConsolidationProduct>> {
        let mut products = Vec::new();
        for entry in entries {
            products.push(self.consolidate(entry).await?);
        }

        Ok(products)
    }
}

fn strict_llm_opts() -> LlmOpts {
    LlmOpts {
        max_tokens: Some(512),
        temperature: Some(0.0),
        ..Default::default()
    }
}

fn validate_probability(value: f32, field: &str) -> Result<f32> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(agent_context_db_core::ContextError::Llm(
            LlmError::Provider(format!("invalid {field}: expected finite value in [0, 1]")),
        ))
    }
}

fn validate_llm_content(value: &str, field: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 4000 {
        return Err(agent_context_db_core::ContextError::Llm(
            LlmError::Provider(format!("invalid {field}: expected 1..=4000 characters")),
        ));
    }
    Ok(trimmed.to_string())
}

/// 评估结果。
#[derive(Debug, Clone)]
pub struct CritiqueResult {
    pub quality_score: f32,
    pub confidence: f32,
    pub suggestions: Vec<String>,
}

/* Obsolete heuristic critique removed: structured LLM failures must remain explicit.
fn deterministic_critique(product: &ConsolidationProduct, entry: &ContextEntry) -> CritiqueResult {
    let content = product.content.trim();
    let word_count = content.split_whitespace().count();
    let char_count = content.chars().count();
    let has_sentence_shape = content.ends_with('.')
        || content.ends_with('!')
        || content.ends_with('?')
        || content.ends_with('。')
        || content.ends_with('！')
        || content.ends_with('？');
    let repeated_ratio = repeated_token_ratio(content);

    let structure = if char_count == 0 {
        0.0
    } else {
        let length_score = match char_count {
            1..=12 => 0.20,
            13..=40 => 0.55,
            41..=260 => 0.90,
            261..=600 => 0.72,
            _ => 0.45,
        };
        let sentence_bonus = if has_sentence_shape { 0.05 } else { 0.0 };
        (length_score + sentence_bonus - repeated_ratio * 0.35_f32).clamp(0.0, 1.0)
    };

    let evidence = evidence_score(product, entry);
    let validity = validity_score(
        product
            .metadata
            .validity
            .as_ref()
            .or(entry.metadata.validity.as_ref()),
    );
    let type_fit = type_fit_score(product, entry, word_count);
    let provenance = if product.provenance.is_some() {
        0.95
    } else {
        0.55
    };
    let contradiction_penalty = if product.contradiction_uris.is_empty() {
        0.0
    } else {
        (product.contradiction_uris.len() as f32 * 0.12).min(0.35)
    };

    let quality = (structure * 0.26
        + evidence * 0.22
        + validity * 0.20
        + type_fit * 0.18
        + provenance * 0.08
        + entry.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0) * 0.06
        - contradiction_penalty)
        .clamp(0.0, 1.0);

    let signal_count = [
        char_count >= 24,
        !product.evidence_uris.is_empty()
            || entry
                .metadata
                .consolidation
                .as_ref()
                .is_some_and(|meta| !meta.evidence_uris.is_empty()),
        product.metadata.validity.is_some() || entry.metadata.validity.is_some(),
        product.provenance.is_some(),
        product.metadata.half_life.is_some()
            || entry
                .metadata
                .consolidation
                .as_ref()
                .and_then(|meta| meta.half_life)
                .is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count() as f32;
    let confidence = (0.28 + signal_count * 0.10 + evidence * 0.18 + validity * 0.12)
        .min(0.82)
        .clamp(0.0, 1.0);

    let mut suggestions = Vec::new();
    if char_count < 24 {
        suggestions.push("expand the insight into a self-contained statement".into());
    }

    if repeated_ratio > 0.25 {
        suggestions.push("remove repeated wording before consolidation".into());
    }

    if product.content_type == ContentType::Fact && evidence < 0.45 {
        suggestions.push("attach evidence before treating this as a factual memory".into());
    }

    if validity < 0.45 {
        suggestions.push("revalidate or mark the claim as uncertain".into());
    }

    if !product.contradiction_uris.is_empty() {
        suggestions.push("resolve linked contradictions before promotion".into());
    }

    CritiqueResult {
        quality_score: quality,
        confidence,
        suggestions,
    }
}

fn evidence_score(product: &ConsolidationProduct, entry: &ContextEntry) -> f32 {
    let product_evidence = product.evidence_uris.len();
    let entry_meta = entry.metadata.consolidation.as_ref();
    let entry_evidence = entry_meta.map_or(0, |meta| meta.evidence_uris.len());
    let corroboration = entry_meta.map_or(0, |meta| meta.corroboration);
    let count = product_evidence + entry_evidence;
    (0.25 + (count as f32 * 0.18) + (corroboration as f32 * 0.08)).clamp(0.0, 1.0)
}

fn validity_score(validity: Option<&ValidityRecord>) -> f32 {
    let Some(validity) = validity else {
        return 0.55;
    };
    if validity
        .valid_until
        .is_some_and(|valid_until| valid_until < Utc::now())
    {
        return 0.20;
    }

    let mut score: f32 = 0.72;
    if validity.valid_from > Utc::now() {
        score -= 0.18;
    }

    if validity.invalidated_by.is_some() {
        score -= 0.25;
    }

    if validity.invalidation_reason.is_some() {
        score -= 0.12;
    }

    score.clamp(0.0, 1.0)
}

fn type_fit_score(product: &ConsolidationProduct, entry: &ContextEntry, word_count: usize) -> f32 {
    let content = product.content.to_ascii_lowercase();
    let base = match product.content_type {
        ContentType::Fact => {
            if product.evidence_uris.is_empty()
                && entry
                    .metadata
                    .consolidation
                    .as_ref()
                    .is_none_or(|meta| meta.evidence_uris.is_empty())
            {
                0.45
            } else {
                0.82
            }
        }

        ContentType::Hypothesis => {
            if product.hypothesis_outcome == Some(HypothesisOutcome::Falsified) {
                0.35
            } else if content.contains("if") || content.contains("may") || content.contains("可能")
            {
                0.80
            } else {
                0.62
            }
        }

        ContentType::Procedure => {
            if content.contains("step")
                || content.contains("then")
                || content.contains("when")
                || content.contains("先")
            {
                0.82
            } else {
                0.58
            }
        }

        ContentType::Error => {
            if product.error_pattern.is_some()
                || content.contains("error")
                || content.contains("failure")
            {
                0.82
            } else {
                0.52
            }
        }

        ContentType::Heuristic => {
            if content.contains("when")
                || content.contains("prefer")
                || content.contains("should")
                || content.contains("如果")
            {
                0.80
            } else {
                0.62
            }
        }

        _ => 0.68,
    };
    let density = if word_count == 0 {
        0.0
    } else if word_count <= 6 {
        0.58
    } else if word_count <= 80 {
        0.88
    } else {
        0.65
    };
    (base * 0.7_f32 + density * 0.3_f32).clamp(0.0, 1.0)
}

fn repeated_token_ratio(content: &str) -> f32 {
    let tokens = content
        .split_whitespace()
        .map(|token| {
            token
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|token| token.len() >= 3)
        .collect::<Vec<_>>();
    if tokens.len() < 2 {
        return 0.0;
    }

    let unique = tokens
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    1.0 - (unique as f32 / tokens.len() as f32)
}
*/

// ===========================================================================
// 双时序— valid/invalid + created/expired，持久化到 ValidityRecord
// ===========================================================================

/// 双时序管理器 — 通过 ContextMeta::validity 持久化有效期信息。
///
/// 核心语义：
/// - `valid_from` → 知识生效时间
/// - `valid_until` → None=当前有效，Some=已失效（Zep 软失效）
/// - 失效传播：invalidate 一个晶体后，其关联后代应标记受影响
pub struct DualTimeline;

impl Default for DualTimeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DualTimeline {
    pub fn new() -> Self {
        Self
    }

    /// 为条目创建有效期记录。
    pub fn mark_valid(meta: &mut agent_context_db_core::ContextMeta, from: DateTime<Utc>) {
        meta.validity = Some(ValidityRecord {
            valid_from: from,
            valid_until: None,
            invalidated_by: None,
            invalidation_reason: None,
        });
    }

    /// 软失效 — 设置 valid_until，保留数据和引用链。
    pub fn invalidate(
        meta: &mut agent_context_db_core::ContextMeta,
        invalidated_by: &ContextUri,
        reason: &str,
    ) {
        let now = Utc::now();
        if let Some(ref mut v) = meta.validity {
            v.valid_until = Some(now);
            v.invalidated_by = Some(invalidated_by.clone());
            v.invalidation_reason = Some(reason.to_string());
        } else {
            meta.validity = Some(ValidityRecord {
                valid_from: now,
                valid_until: Some(now),
                invalidated_by: Some(invalidated_by.clone()),
                invalidation_reason: Some(reason.to_string()),
            });
        }
    }

    /// 检查条目在指定时间点是否有效。
    pub fn is_valid(meta: &agent_context_db_core::ContextMeta, at: DateTime<Utc>) -> bool {
        match &meta.validity {
            Some(v) => at >= v.valid_from && v.valid_until.is_none_or(|until| at <= until),
            None => true, // 无记录默认有效
        }
    }

    /// 批量标记失效后代（失效传播）。
    /// 当某个 fact 被 invalidate 后，其 evolved_to 的晶体应标记为"可能受影响"。
    pub fn propagate_invalidation(
        parent_uri: &ContextUri,
        descendants: &[ContextUri],
        metas: &mut [&mut agent_context_db_core::ContextMeta],
    ) -> usize {
        let mut propagated = 0;
        for (_descendant, meta) in descendants.iter().zip(metas.iter_mut()) {
            if meta
                .validity
                .as_ref()
                .is_none_or(|v| v.valid_until.is_none())
            {
                let reason = format!("parent {} was invalidated", parent_uri);
                Self::invalidate(meta, parent_uri, &reason);
                propagated += 1;
            }
        }

        propagated
    }
}

// ===========================================================================
// 反向进化— 新知识追溯更新历史记忆
// ===========================================================================

/// 反向进化器 — 新晶体写入后，追溯更新语义重叠的历史条目。
///
/// 新记忆不仅是 forward 演化，还反过来更新历史记忆的 lineage 和 context version。
///
/// 触发路径：
/// 1. **文本相似**：Jaccard(new, hist) ≥ threshold → 记入受影响集合
/// 2. **图边扩展**（可选，需注入 `GraphStore`）：命中项沿 `EvolvedFrom`/`EvolvedTo` 各扩 1 跳，
///    传播 lineage 更新到间接相关的历史条目
pub struct BackwardEvolver {
    /// 语义重叠阈值（Jaccard 相似度）。
    similarity_threshold: f32,
    /// 关系图存储（可选）。注入后启用图边扩展。
    graph: Option<Arc<dyn agent_context_db_core::GraphStore>>,
}

impl Default for BackwardEvolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardEvolver {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.3,
            graph: None,
        }
    }

    /// 注入关系图存储，启用图边扩展路径。
    pub fn with_graph(mut self, graph: Arc<dyn agent_context_db_core::GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// 检查新产物是否影响历史条目，更新受影响条目的 lineage。
    ///
    /// 对每条历史 entry：
    /// 1. 计算与新产物的文本重叠度（Jaccard 相似度）
    /// 2. 相似度超过阈值 → 追溯更新 lineage（追加 LineageEntry）
    /// 3. 若注入 `graph`：沿 `EvolvedFrom`/`EvolvedTo` 扩 1 跳，间接相关条目也追加 lineage
    /// 4. 返回受影响的条目数量
    pub async fn evolve_backward(
        &self,
        new_product: &ConsolidationProduct,
        historical: &mut [ContextEntry],
    ) -> usize {
        // 提取新产物的关键词集合
        let new_keywords: std::collections::HashSet<&str> =
            new_product.content.split_whitespace().collect();

        if new_keywords.is_empty() {
            return 0;
        }

        // 阶段 1：Jaccard 命中集
        let mut affected_uris: std::collections::HashSet<agent_context_db_core::ContextUri> =
            std::collections::HashSet::new();
        for entry in historical.iter() {
            let entry_text = entry.l0_text();
            let entry_keywords: std::collections::HashSet<&str> =
                entry_text.split_whitespace().collect();
            let intersection = new_keywords.intersection(&entry_keywords).count();
            let union = new_keywords.union(&entry_keywords).count();
            let similarity = if union > 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };
            if similarity >= self.similarity_threshold {
                affected_uris.insert(entry.uri.clone());
            }
        }

        // 阶段 2：图边扩展 — 沿 EvolvedFrom / EvolvedTo 扩 1 跳
        if let Some(graph) = &self.graph {
            let seeds: Vec<agent_context_db_core::ContextUri> =
                affected_uris.iter().cloned().collect();
            if !seeds.is_empty() {
                let kinds = [
                    agent_context_db_core::GraphRelation::EvolvedFrom,
                    agent_context_db_core::GraphRelation::EvolvedTo,
                ];
                if let Ok(edges) = graph.batch_traverse(&seeds, &kinds, 1).await {
                    for (_from, to, _kind) in edges {
                        affected_uris.insert(to);
                    }
                }
            }
        }

        // 阶段 3：对所有受影响 URI 追加 lineage
        let mut updated = 0;
        for entry in historical.iter_mut() {
            if !affected_uris.contains(&entry.uri) {
                continue;
            }

            let similarity_note = if self.graph.is_some() {
                format!("backward-evolved from {} (jaccard+graph)", new_product.uri)
            } else {
                format!("backward-evolved from {}", new_product.uri)
            };
            let lineage_entry = LineageEntry {
                version: entry.mvcc_version,
                timestamp: Utc::now(),
                change_summary: similarity_note,
            };
            if let Some(ref mut consolidation) = entry.metadata.consolidation {
                consolidation.lineage.push(lineage_entry);
            } else {
                entry.metadata.consolidation = Some(agent_context_db_core::ConsolidationMeta {
                    source: "backward-evolve".to_string(),
                    generation: 1,
                    status: agent_context_db_core::ConsolidationStatus::InProgress,
                    patch_count: 1,
                    lineage: vec![lineage_entry],
                    evidence_uris: vec![],
                    corroboration: 0,
                    half_life: None,
                    entangled_with: vec![],
                });
            }

            updated += 1;
        }

        updated
    }
}

// ===========================================================================
// Sleeptime 执行器（后台）— 空闲时后台任务
// ===========================================================================

/// Sleeptime 任务类型。
#[derive(Debug, Clone)]
pub enum SleeptimeTask {
    QualityReassessment,
    ConsistencyCheck,
    EntanglementDetection,
    IgnoranceMapUpdate,
    CalibrationUpdate,
    ContextRotPrune,
    BackwardEvolve,
    SpacedRepetitionReview,
    CascadeInvalidation,
    KnowledgeHealthDiagnosis,
    ActiveConsistencyGuard,
    ActiveLearningLoop,
    EmbeddingDriftDetection,
    CuriosityExploration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SleeptimeProgress {
    pub cursor: Option<ContextUri>,
    pub watermark: Option<ContextUri>,
    pub attempt: u32,
    pub last_error: Option<String>,
}

#[async_trait::async_trait]
pub trait ProgressStore: Send + Sync {
    async fn load(&self, key: &str) -> Result<Option<SleeptimeProgress>>;
    async fn save(&self, key: &str, progress: &SleeptimeProgress) -> Result<()>;
}

#[derive(Debug, Clone, Default)]
pub struct EntrySignals {
    pub adoption_rate: Option<f32>,
    pub recall_rate: Option<f32>,
    pub downstream_success_rate: Option<f32>,
    pub contradiction_count: Option<u32>,
    pub corroboration_count: Option<u32>,
    pub repeated_observations: Option<u32>,
    pub tenant_priority: Option<f32>,
}

#[async_trait::async_trait]
pub trait SignalProvider: Send + Sync {
    async fn signals(&self, uri: &ContextUri) -> Result<EntrySignals>;
}

/// Sleeptime 执行器。构造时绑定 tenant/agent 根 scope，运行时拒绝越界目标。
pub struct SleeptimeExecutor {
    pub tasks: Vec<SleeptimeTask>,
    pub interval: std::time::Duration,
    checker: ConsistencyChecker,
    /// 纠缠检测器 —— EntanglementDetection 任务的实际执行者。
    entanglement: crate::entanglement::EntanglementDetector,
    root_scope: ContextUri,
    batch_limit: usize,
    store: Arc<dyn agent_context_db_core::ContentStore>,
    progress: Arc<dyn ProgressStore>,
    signals: Arc<dyn SignalProvider>,
    graph: Arc<dyn agent_context_db_core::GraphStore>,
    /// 可选的向量索引 —— `on_adopted()` 触发 RIF 抑制时使用。
    vector_index: Option<Arc<dyn agent_context_db_core::VectorIndex>>,
    lifecycle_executor: Option<Arc<dyn LifecycleExecutorPort>>,
}

#[async_trait::async_trait]
pub trait LifecycleExecutorPort: Send + Sync {
    async fn run_pending(&self) -> Result<Vec<crate::lifecycle_execution::LifecycleJob>>;
    async fn submit(
        &self,
        uri: ContextUri,
        action: agent_context_db_core::LifecycleAction,
    ) -> Result<crate::lifecycle_execution::LifecycleJob>;
    async fn route_metacog(
        &self,
        _entry: &ContextEntry,
    ) -> Result<Option<crate::lifecycle_execution::LifecycleJob>> {
        Ok(None)
    }
    async fn restore_metacog(&self, _uri: &ContextUri) -> Result<()> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl LifecycleExecutorPort for crate::lifecycle_execution::LifecycleActionExecutor {
    async fn run_pending(&self) -> Result<Vec<crate::lifecycle_execution::LifecycleJob>> {
        self.run_pending().await
    }

    async fn submit(
        &self,
        uri: ContextUri,
        action: agent_context_db_core::LifecycleAction,
    ) -> Result<crate::lifecycle_execution::LifecycleJob> {
        self.submit(uri, action).await
    }

    async fn route_metacog(
        &self,
        entry: &ContextEntry,
    ) -> Result<Option<crate::lifecycle_execution::LifecycleJob>> {
        self.route_metacog(entry).await
    }

    async fn restore_metacog(&self, uri: &ContextUri) -> Result<()> {
        if uri.category() != agent_context_db_core::UriCategory::Metacog {
            return Ok(());
        }
        let key = format!(
            "{}:{:?}",
            uri,
            crate::lifecycle_execution::LifecycleOperation::ColdStorage
        );
        self.restore(&key).await
    }
}

impl SleeptimeExecutor {
    pub fn new(
        root_scope: ContextUri,
        store: Arc<dyn agent_context_db_core::ContentStore>,
        graph: Arc<dyn agent_context_db_core::GraphStore>,
        progress: Arc<dyn ProgressStore>,
        signals: Arc<dyn SignalProvider>,
        llm: Arc<dyn LlmClient>,
        batch_limit: usize,
    ) -> Result<Self> {
        let segments = root_scope.segments();
        if segments.len() != 3 || segments.get(1).map(String::as_str) != Some("agent") {
            return Err(agent_context_db_core::ContextError::InvalidUri(
                "sleeptime root must be uwu://<tenant>/agent/<agent>".into(),
            ));
        }

        if batch_limit == 0 {
            return Err(agent_context_db_core::ContextError::InvalidUri(
                "sleeptime batch_limit must be positive".into(),
            ));
        }

        Ok(Self {
            tasks: vec![
                SleeptimeTask::QualityReassessment,
                SleeptimeTask::ConsistencyCheck,
                SleeptimeTask::EntanglementDetection,
                SleeptimeTask::IgnoranceMapUpdate,
                SleeptimeTask::CalibrationUpdate,
                SleeptimeTask::ContextRotPrune,
                SleeptimeTask::BackwardEvolve,
                SleeptimeTask::SpacedRepetitionReview,
                SleeptimeTask::CascadeInvalidation,
                SleeptimeTask::KnowledgeHealthDiagnosis,
                SleeptimeTask::ActiveConsistencyGuard,
                SleeptimeTask::ActiveLearningLoop,
                SleeptimeTask::EmbeddingDriftDetection,
                SleeptimeTask::CuriosityExploration,
            ],
            interval: std::time::Duration::from_secs(3600),
            checker: ConsistencyChecker::new(llm),
            entanglement: crate::entanglement::EntanglementDetector::new(
                crate::entanglement::EntanglementConfig::default(),
            )?,
            root_scope,
            batch_limit,
            store,
            progress,
            signals,
            graph,
            vector_index: None,
            lifecycle_executor: None,
        })
    }

    /// 绑定向量索引，启用 `on_adopted()` 的 RIF 邻居抑制路径。
    pub fn with_vector_index(mut self, index: Arc<dyn agent_context_db_core::VectorIndex>) -> Self {
        self.vector_index = Some(index);
        self
    }

    /// 绑定持久生命周期执行器；Sleeptime 会先恢复未完成任务，再执行本轮决策。
    pub fn with_lifecycle_executor(mut self, executor: Arc<dyn LifecycleExecutorPort>) -> Self {
        self.lifecycle_executor = Some(executor);
        self
    }

    /// 记录一次跨条目 patch 共现事件（供 EntanglementDetection 汇总）。
    pub fn record_co_patch(&self, a: &ContextUri, b: &ContextUri) {
        self.entanglement.record_co_patch(a, b);
    }

    /// 采纳事件钩子 —— 触发 RIF 抑制 + 校准器更新。
    ///
    /// 调用方在 `MemoryResolver::resolve` 决策为 `ADD` 且被上层实际写入后调用。
    /// 返回被判定为"冗余邻居"的 URI 列表（上层应用可据此下调其质量分）。
    ///
    /// - 若未注入 `vector_index` → 返回空列表（RIF 静默禁用）
    /// - 若命中邻居 → 同时调用 `engine.calibrator.update(epistemic, declared, adopted=false)`
    ///   把"被抑制"当作"未被采纳"喂给校准器，收紧过度自信
    pub async fn on_adopted(
        &self,
        engine: &ConsolidationEngine,
        adopted: &ConsolidationProduct,
        embedding: &[f32],
    ) -> Vec<ContextUri> {
        let index = match &self.vector_index {
            Some(i) => i.clone(),
            None => return vec![],
        };
        let rif = crate::rif::RifSuppressor::new(index, 0.15);
        let suppressed = rif
            .on_adopted(&adopted.uri, embedding)
            .await
            .unwrap_or_default();

        // 采纳自身：喂给校准器（adopted=true）
        engine
            .calibrator
            .update(adopted.epistemic_type, adopted.confidence, true);
        // 被抑制邻居：视作"未采纳"，帮助收紧同类型温度
        for _ in &suppressed {
            engine
                .calibrator
                .update(adopted.epistemic_type, adopted.confidence, false);
        }

        suppressed
    }

    /// 执行一轮 Sleeptime 后台整理，从存储加载数据并运行全部检查。
    pub async fn run_once(
        &self,
        engine: &ConsolidationEngine,
        scope: &ContextUri,
    ) -> Result<SleeptimeReport> {
        let root = self.root_scope.to_string();
        let target = scope.to_string();
        if target != root && !target.starts_with(&format!("{root}/")) {
            return Err(agent_context_db_core::ContextError::PermissionDenied(
                format!("sleeptime scope {target} is outside bound root {root}"),
            ));
        }

        let progress_key = format!("sleeptime:{target}");
        let mut state = self.progress.load(&progress_key).await?.unwrap_or_default();
        state.attempt = state.attempt.saturating_add(1);
        let page_request = state.cursor.as_ref().map_or_else(
            || PageRequest::new(self.batch_limit),
            |cursor| PageRequest::new(self.batch_limit).after(cursor.to_string()),
        );
        let scanned = match self.store.scan_by_prefix(&target, page_request).await {
            Ok(page) => page.items,
            Err(error) => {
                state.last_error = Some(error.to_string());
                self.progress.save(&progress_key, &state).await?;
                return Err(error);
            }
        };
        let entries: Vec<ContextEntry> = scanned
            .into_iter()
            .filter(|entry| {
                state
                    .cursor
                    .as_ref()
                    .is_none_or(|cursor| entry.uri > *cursor)
            })
            .filter(|entry| {
                state
                    .watermark
                    .as_ref()
                    .is_none_or(|watermark| entry.uri <= *watermark)
            })
            .take(self.batch_limit)
            .collect();
        if entries.is_empty() && state.cursor.is_some() {
            state.cursor = None;
            state.watermark = None;
            state.attempt = 0;
        }

        let mut report = SleeptimeReport::default();
        if !entries.is_empty() {
            let mut known = 0usize;
            const SIGNALS_PER_ENTRY: usize = 8;
            for entry in &entries {
                self.graph.centrality(&entry.uri).await?;
                known += 1;
                let observed = self.signals.signals(&entry.uri).await?;
                known += [
                    observed.adoption_rate.is_some(),
                    observed.recall_rate.is_some(),
                    observed.downstream_success_rate.is_some(),
                    observed.contradiction_count.is_some(),
                    observed.corroboration_count.is_some(),
                    observed.repeated_observations.is_some(),
                    observed.tenant_priority.is_some(),
                ]
                .into_iter()
                .filter(|present| *present)
                .count();
            }

            report.signal_completeness = known as f32 / (entries.len() * SIGNALS_PER_ENTRY) as f32;
        }

        if let Some(executor) = &self.lifecycle_executor {
            let recovered = executor.run_pending().await?;
            report.lifecycle_succeeded += recovered
                .iter()
                .filter(|j| {
                    matches!(
                        j.state,
                        crate::lifecycle_execution::LifecycleJobState::Succeeded
                    )
                })
                .count();
            report.lifecycle_failed += recovered.len().saturating_sub(report.lifecycle_succeeded);
        }

        for task in &self.tasks {
            match task {
                SleeptimeTask::QualityReassessment => {
                    let scorer = HorizonAwareQualityScorer::default();
                    let store = &self.store;

                    let mut reassessed = 0usize;
                    let mut promoted_mid = 0usize;
                    let mut promoted_long = 0usize;
                    let mut archived = 0usize;
                    let mut training_candidates = 0usize;

                    for entry in &entries {
                        let observed = self.signals.signals(&entry.uri).await?;
                        let signals = HorizonQualitySignals {
                            adoption_rate: observed.adoption_rate.unwrap_or(0.0),
                            recall_rate: observed.recall_rate.unwrap_or(0.0),
                            downstream_success_rate: observed
                                .downstream_success_rate
                                .unwrap_or(0.0),
                            contradiction_count: observed.contradiction_count.unwrap_or(0) as usize,
                            corroboration_count: observed.corroboration_count.unwrap_or(0) as usize,
                            repeated_observations: observed.repeated_observations.unwrap_or(0)
                                as usize,
                            user_confirmed: false,
                            user_corrected: false,
                            retrieval_ignored_rate: 0.0,
                            info_gain: 0.0,
                            now: Utc::now(),
                        };
                        let outcome = scorer.reassess(entry, signals);
                        if !outcome.should_writeback {
                            continue;
                        }

                        let mut updated = entry.clone();
                        updated.metadata.quality_score = Some(outcome.posterior.overall);
                        let mut route_promoted_mid = false;
                        let mut route_promoted_long = false;
                        let mut route_archived = false;
                        let mut route_training_candidate = false;
                        match outcome.route {
                            QualityRoute::CompressToMidTerm => {
                                updated.metadata.state_scope = Some(StateScope::Mid);
                                route_promoted_mid = true;
                            }

                            QualityRoute::PromoteToLongTerm | QualityRoute::IncludeInTraining => {
                                updated.metadata.state_scope = Some(StateScope::Long);
                                route_promoted_long = true;
                                route_training_candidate =
                                    matches!(outcome.route, QualityRoute::IncludeInTraining);
                            }

                            QualityRoute::Archive
                            | QualityRoute::ForgetCandidate
                            | QualityRoute::ForgetShortTerm => {
                                updated
                                    .metadata
                                    .tags
                                    .push("quality:archive-candidate".into());
                                route_archived = true;
                                if let Some(executor) = &self.lifecycle_executor {
                                    match executor
                                        .submit(
                                            entry.uri.clone(),
                                            agent_context_db_core::LifecycleAction::Archive,
                                        )
                                        .await
                                    {
                                        Ok(_) => report.lifecycle_submitted += 1,
                                        Err(_) => report.lifecycle_failed += 1,
                                    }
                                }
                            }

                            QualityRoute::Rehearse => {
                                updated.metadata.tags.push("quality:rehearse".into());
                            }

                            QualityRoute::Revalidate => {
                                updated.metadata.tags.push("quality:revalidate".into());
                            }

                            QualityRoute::ExcludeFromTraining => {
                                updated
                                    .metadata
                                    .tags
                                    .push("quality:exclude-training".into());
                            }

                            _ => {}
                        }

                        store.write(updated).await?;
                        reassessed += 1;
                        promoted_mid += usize::from(route_promoted_mid);
                        promoted_long += usize::from(route_promoted_long);
                        archived += usize::from(route_archived);
                        training_candidates += usize::from(route_training_candidate);
                    }

                    report.quality_reassessed = reassessed;
                    report.quality_promoted_mid = promoted_mid;
                    report.quality_promoted_long = promoted_long;
                    report.quality_archived = archived;
                    report.training_candidates = training_candidates;
                    tracing::info!(
                        reassessed,
                        promoted_mid,
                        promoted_long,
                        archived,
                        training_candidates,
                        "sleeptime: horizon-aware quality reassessment"
                    );
                    report.tasks_executed += 1;
                }

                SleeptimeTask::ConsistencyCheck => {
                    let products: Vec<ConsolidationProduct> = entries
                        .iter()
                        .map(|e| ConsolidationProduct {
                            uri: e.uri.clone(),
                            content_type: e.content_type().unwrap_or(ContentType::Fact),
                            epistemic_type: e
                                .metadata
                                .epistemic_type()
                                .unwrap_or(EpistemicType::Fact),
                            content: e.l0_text().to_string(),
                            quality_score: e.metadata.quality_score.unwrap_or(0.5),
                            confidence: 0.5,
                            evidence_required: false,
                            superseded_claim: None,
                            evidence_uris: e
                                .metadata
                                .consolidation
                                .as_ref()
                                .map(|meta| meta.evidence_uris.clone())
                                .unwrap_or_default(),
                            contradiction_uris: vec![],
                            error_pattern: None,
                            hypothesis_outcome: None,
                            preconditions: None,
                            expected_outcome: None,
                            related_policy_uris: vec![],
                            provenance: None,
                            metadata: ConsolidationProductMeta {
                                source_session: None,
                                generation: 0,
                                status: ConsolidationStatus::Pending,
                                patch_count: 0,
                                lineage: vec![],
                                validity: e.metadata.validity.clone(),
                                half_life: None,
                            },
                        })
                        .collect();
                    let violations = self.checker.check(&products).await?;
                    report.contradictions_found = violations.len();
                    report.tasks_executed += 1;
                }

                SleeptimeTask::EntanglementDetection => {
                    // 应用衰减，让长期不再共现的纠缠自然消退
                    self.entanglement.decay(0.05);
                    // 统计当前仍高于阈值的纠缠对数
                    let count = entries
                        .iter()
                        .map(|e| self.entanglement.get_entangled(&e.uri).len())
                        .sum::<usize>();
                    report.entanglements_detected = count;
                    report.tasks_executed += 1;
                }

                SleeptimeTask::IgnoranceMapUpdate => {
                    let spots = engine.ignorance.top_blind_spots(10);
                    report.blind_spots_updated = spots.len();
                    report.tasks_executed += 1;
                }

                SleeptimeTask::CalibrationUpdate => {
                    // 校准器只接受真实采纳/抑制反馈；批量巡检不能用 quality_score 反推标签。
                    // 真实反馈由 `on_adopted` 写入，避免把模型自评分训练成“采纳事实”。
                    report.calibration_samples = engine.calibrator.total_labeled_samples();
                    report.tasks_executed += 1;
                }

                SleeptimeTask::ContextRotPrune => {
                    let pruned = entries
                        .iter()
                        .filter(|e| e.metadata.quality_score.unwrap_or(0.5) < 0.1)
                        .count();
                    report.pruned = pruned;
                    report.tasks_executed += 1;
                }

                SleeptimeTask::BackwardEvolve => {
                    let products: Vec<ConsolidationProduct> = entries
                        .iter()
                        .map(|e| ConsolidationProduct {
                            uri: e.uri.clone(),
                            content_type: e.content_type().unwrap_or(ContentType::Fact),
                            epistemic_type: e
                                .metadata
                                .epistemic_type()
                                .unwrap_or(EpistemicType::Fact),
                            content: e.l0_text().to_string(),
                            quality_score: e.metadata.quality_score.unwrap_or(0.5),
                            confidence: 0.5,
                            evidence_required: false,
                            superseded_claim: None,
                            evidence_uris: e
                                .metadata
                                .consolidation
                                .as_ref()
                                .map(|meta| meta.evidence_uris.clone())
                                .unwrap_or_default(),
                            contradiction_uris: vec![],
                            error_pattern: None,
                            hypothesis_outcome: None,
                            preconditions: None,
                            expected_outcome: None,
                            related_policy_uris: vec![],
                            provenance: None,
                            metadata: ConsolidationProductMeta {
                                source_session: None,
                                generation: 0,
                                status: ConsolidationStatus::Pending,
                                patch_count: 0,
                                lineage: vec![],
                                validity: e.metadata.validity.clone(),
                                half_life: None,
                            },
                        })
                        .collect();
                    if let Some(last) = products.last() {
                        let mut evolver = BackwardEvolver::new();
                        evolver = evolver.with_graph(self.graph.clone());
                        let mut mutable = entries.clone();
                        let _updated = evolver.evolve_backward(last, &mut mutable).await;
                    }

                    report.tasks_executed += 1;
                }

                SleeptimeTask::SpacedRepetitionReview => {
                    let scheduler = SpacedRepetitionScheduler::new(
                        crate::halflife::SpacedRepetitionConfig::default(),
                    )?;
                    let tasks = scheduler.plan(&entries, Utc::now());
                    {
                        let store = &self.store;
                        let by_uri = tasks
                            .iter()
                            .map(|task| (task.uri.clone(), task))
                            .collect::<std::collections::HashMap<_, _>>();
                        for entry in &entries {
                            let Some(task) = by_uri.get(&entry.uri) else {
                                continue;
                            };
                            let mut updated = entry.clone();
                            updated.metadata.tags.push(format!(
                                "quality:review:{}",
                                match task.action {
                                    crate::halflife::ReviewAction::Rehearse => "rehearse",
                                    crate::halflife::ReviewAction::Revalidate => "revalidate",
                                    crate::halflife::ReviewAction::ForgetCandidate =>
                                        "forget-candidate",
                                }
                            ));
                            store.write(updated).await?;
                            report.review_tasks_written += 1;
                        }
                    }

                    report.review_tasks = tasks.len();
                    report.tasks_executed += 1;
                }

                SleeptimeTask::CascadeInvalidation => {
                    let invalidated = entries
                        .iter()
                        .filter(|entry| {
                            entry
                                .metadata
                                .validity
                                .as_ref()
                                .and_then(|validity| validity.valid_until)
                                .is_some()
                        })
                        .collect::<Vec<_>>();
                    let detector = Arc::new(crate::entanglement::EntanglementDetector::new(
                        crate::entanglement::EntanglementConfig::default(),
                    )?);
                    for entry in &entries {
                        if let Some(meta) = &entry.metadata.consolidation {
                            for partner in &meta.entangled_with {
                                detector.record_co_patch(&entry.uri, partner);
                            }
                        }
                    }

                    let mut invalidator = crate::entanglement::CascadeInvalidator::new(
                        detector,
                        crate::entanglement::CascadeInvalidationConfig::default(),
                    )?;
                    invalidator = invalidator.with_graph(self.graph.clone());
                    let mut mutable = entries.clone();
                    for entry in invalidated {
                        let plan = invalidator.plan_from_invalidated(&entry.uri).await;
                        report.cascade_revalidation_tasks += plan
                            .tasks
                            .iter()
                            .filter(|task| {
                                matches!(
                                    task.action,
                                    crate::entanglement::CascadeInvalidationAction::Revalidate
                                )
                            })
                            .count();
                        report.cascade_invalidations += plan
                            .tasks
                            .iter()
                            .filter(|task| {
                                matches!(
                                    task.action,
                                    crate::entanglement::CascadeInvalidationAction::Invalidate
                                )
                            })
                            .count();
                        invalidator.apply_to_entries(&mut mutable, &plan);
                    }

                    {
                        let store = &self.store;
                        for updated in mutable {
                            if updated
                                .metadata
                                .tags
                                .iter()
                                .any(|tag| tag.starts_with("cascade:"))
                            {
                                store.write(updated).await?;
                            }
                        }
                    }

                    report.tasks_executed += 1;
                }

                SleeptimeTask::KnowledgeHealthDiagnosis => {
                    let mut diagnostician = KnowledgeHealthDiagnostician::new(
                        crate::health::KnowledgeHealthConfig::default(),
                    )?;
                    diagnostician = diagnostician.with_graph(self.graph.clone());
                    let report_health = diagnostician.diagnose(&entries, Utc::now()).await?;
                    report.health_issues = report_health.issues.len();
                    let mut mutable = entries.clone();
                    diagnostician.apply_repairs(&mut mutable, &report_health)?;
                    {
                        let store = &self.store;
                        for updated in mutable {
                            if updated
                                .metadata
                                .tags
                                .iter()
                                .any(|tag| tag.starts_with("health:"))
                            {
                                store.write(updated).await?;
                                report.health_repairs_written += 1;
                            }
                        }
                    }

                    report.tasks_executed += 1;
                }

                SleeptimeTask::ActiveConsistencyGuard => {
                    let guardian = ActiveConsistencyGuardian::new(
                        crate::health::ConsistencyGuardianConfig::default(),
                    )?;
                    let plan = guardian.plan(&entries, Utc::now());
                    report.consistency_guard_tasks = plan.tasks.len();
                    let mut mutable = entries.clone();
                    report.consistency_guard_repairs =
                        guardian.apply(&mut mutable, &plan, Utc::now())?;
                    {
                        let store = &self.store;
                        for updated in mutable {
                            if updated.metadata.tags.iter().any(|tag| {
                                tag.starts_with("consistency:")
                                    || tag.starts_with("cascade:")
                                    || tag.starts_with("health:")
                            }) || updated.metadata.validity.is_some()
                            {
                                store.write(updated).await?;
                            }
                        }
                    }

                    report.tasks_executed += 1;
                }

                SleeptimeTask::ActiveLearningLoop => {
                    let active_learning = ActiveLearningLoop::new(
                        crate::quality::ActiveLearningPlanner::default(),
                        HorizonAwareQualityScorer::default(),
                        crate::health::ActiveLearningExecutionConfig::default(),
                    )?;
                    let tasks = active_learning.plan(&entries, Utc::now());
                    report.active_learning_tasks = tasks.len();
                    let mut mutable = entries.clone();
                    report.active_learning_tasks_written =
                        active_learning.apply_tasks(&mut mutable, &tasks)?;
                    {
                        let store = &self.store;
                        for updated in mutable {
                            if updated
                                .metadata
                                .tags
                                .iter()
                                .any(|tag| tag.starts_with("active-learning:"))
                            {
                                store.write(updated).await?;
                            }
                        }
                    }

                    report.tasks_executed += 1;
                }

                SleeptimeTask::EmbeddingDriftDetection => {
                    let monitor =
                        EmbeddingDriftMonitor::new(crate::health::EmbeddingDriftConfig::default())?;
                    let drift = monitor.analyze(&entries, Utc::now());
                    report.embedding_drift_samples = drift.sample_size;
                    report.embedding_drift_score =
                        drift.psi.max(drift.centroid_shift).max(drift.norm_kl);
                    let mut mutable = entries.clone();
                    report.embedding_drift_updates = monitor.apply(&mut mutable, &drift)?;
                    {
                        let store = &self.store;
                        for updated in mutable {
                            if updated
                                .metadata
                                .tags
                                .iter()
                                .any(|tag| tag.starts_with("embedding-drift:"))
                            {
                                store.write(updated).await?;
                            }
                        }
                    }

                    report.tasks_executed += 1;
                }

                SleeptimeTask::CuriosityExploration => {
                    let explorer =
                        CuriosityExplorer::new(crate::health::CuriosityExplorerConfig::default())?;
                    let tasks = explorer.plan(&entries, Utc::now());
                    report.curiosity_tasks = tasks.len();
                    let mut mutable = entries.clone();
                    report.curiosity_tasks_written = explorer.apply_tasks(&mut mutable, &tasks)?;
                    {
                        let store = &self.store;
                        for updated in mutable {
                            if updated
                                .metadata
                                .tags
                                .iter()
                                .any(|tag| tag.starts_with("curiosity:"))
                            {
                                store.write(updated).await?;
                            }
                        }
                    }

                    report.tasks_executed += 1;
                }
            }
        }

        if let Some(executor) = &self.lifecycle_executor {
            let completed = executor.run_pending().await?;
            report.lifecycle_succeeded += completed
                .iter()
                .filter(|j| {
                    matches!(
                        j.state,
                        crate::lifecycle_execution::LifecycleJobState::Succeeded
                    )
                })
                .count();
            report.lifecycle_failed += completed.len().saturating_sub(
                completed
                    .iter()
                    .filter(|j| {
                        matches!(
                            j.state,
                            crate::lifecycle_execution::LifecycleJobState::Succeeded
                        )
                    })
                    .count(),
            );
        }

        if let Some(last) = entries.last() {
            state.cursor = Some(last.uri.clone());
        }

        state.last_error = None;
        self.progress.save(&progress_key, &state).await?;
        report.cursor = state.cursor.clone();
        report.watermark = state.watermark.clone();
        report.attempt = state.attempt;
        report.completeness = if entries.len() < self.batch_limit {
            1.0
        } else {
            0.0
        };
        Ok(report)
    }
}

/// Sleeptime 报告。
#[derive(Debug, Clone, Default)]
pub struct SleeptimeReport {
    pub cursor: Option<ContextUri>,
    pub watermark: Option<ContextUri>,
    pub attempt: u32,
    pub completeness: f32,
    pub signal_completeness: f32,
    pub tasks_executed: usize,
    pub contradictions_found: usize,
    pub entanglements_detected: usize,
    pub pruned: usize,
    pub blind_spots_updated: usize,
    pub calibration_samples: usize,
    pub quality_reassessed: usize,
    pub quality_promoted_mid: usize,
    pub quality_promoted_long: usize,
    pub quality_archived: usize,
    pub training_candidates: usize,
    pub review_tasks: usize,
    pub review_tasks_written: usize,
    pub cascade_revalidation_tasks: usize,
    pub cascade_invalidations: usize,
    pub health_issues: usize,
    pub health_repairs_written: usize,
    pub consistency_guard_tasks: usize,
    pub consistency_guard_repairs: usize,
    pub active_learning_tasks: usize,
    pub active_learning_tasks_written: usize,
    pub embedding_drift_samples: usize,
    pub embedding_drift_score: f32,
    pub embedding_drift_updates: usize,
    pub curiosity_tasks: usize,
    pub curiosity_tasks_written: usize,
    pub lifecycle_submitted: usize,
    pub lifecycle_succeeded: usize,
    pub lifecycle_failed: usize,
}

// ===========================================================================
// 一致性约束检查（embedding 驱动）
// ===========================================================================

/// 一致性约束规则。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraint {
    /// 同一 agent 的两条 fact 不能语义矛盾。
    NoContradiction,
    /// Fact 必须有 ≥1 证据 URI。
    FactRequiresEvidence,
    /// 被 invalidate 的晶体的后代应标记受影响。
    InvalidationPropagation,
    /// Profile/Preference 不能有冲突值。
    NoProfileConflict,
}

/// 一致性约束检查器 — Sleeptime 阶段通过严格结构化 LLM 批量裁决事实矛盾。
pub struct ConsistencyChecker {
    llm: Arc<dyn LlmClient>,
}

#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub uri: ContextUri,
    pub constraint: Constraint,
    pub description: String,
}

impl ConsistencyChecker {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// 检查所有一致性约束。
    ///
    /// 支持：
    /// - 空内容检查
    /// - Fact 必须有证据
    /// - Fact vs Fact 语义矛盾（Jaccard + 否定词检测）
    /// - 被 invalidate 后代标记
    pub async fn check(
        &self,
        products: &[ConsolidationProduct],
    ) -> Result<Vec<ConstraintViolation>> {
        let mut violations = Vec::new();

        for p in products {
            // 空内容
            if p.content.is_empty() {
                violations.push(ConstraintViolation {
                    uri: p.uri.clone(),
                    constraint: Constraint::NoContradiction,
                    description: "empty content".into(),
                });
            }

            // Fact 必须有证据
            if p.content_type == ContentType::Fact
                && p.evidence_required
                && p.evidence_uris.is_empty()
            {
                violations.push(ConstraintViolation {
                    uri: p.uri.clone(),
                    constraint: Constraint::FactRequiresEvidence,
                    description: format!("Fact '{}' has no evidence URIs", p.uri),
                });
            }
        }

        // 语义矛盾检测（fact vs fact）
        let facts: Vec<_> = products
            .iter()
            .filter(|p| p.content_type == ContentType::Fact)
            .collect();

        if facts.len() > 1 {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Classification {
                left: usize,
                right: usize,
                contradiction: bool,
                confidence: f32,
                rationale: String,
            }
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct BatchClassification {
                classifications: Vec<Classification>,
            }
            let claims = facts
                .iter()
                .enumerate()
                .map(|(index, fact)| serde_json::json!({"index": index, "claim": fact.content}))
                .collect::<Vec<_>>();
            let expected_pairs = facts.len() * (facts.len() - 1) / 2;
            let prompt = format!(
                "Classify every unordered pair of claims as logically contradictory or not. Judge meaning across languages; lexical similarity, cosine similarity, and mere topic overlap are not contradiction. Return exactly one classification for every pair. Claims: {}",
                serde_json::to_string(&claims)?
            );
            let schema = JsonSchema::new(serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["classifications"],
                "properties": {"classifications": {"type": "array", "minItems": expected_pairs, "maxItems": expected_pairs, "items": {
                    "type": "object", "additionalProperties": false,
                    "required": ["left", "right", "contradiction", "confidence", "rationale"],
                    "properties": {
                        "left": {"type": "integer", "minimum": 0}, "right": {"type": "integer", "minimum": 0},
                        "contradiction": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string", "minLength": 1, "maxLength": 500}
                    }
                }}}
            }));
            let response = self
                .llm
                .complete_json(&prompt, &schema, &strict_llm_opts())
                .await?;
            let batch: BatchClassification = serde_json::from_str(&response)?;
            if batch.classifications.len() != expected_pairs {
                return Err(agent_context_db_core::ContextError::Llm(
                    LlmError::Provider(format!(
                        "contradiction classifier returned {} pairs; expected {expected_pairs}",
                        batch.classifications.len()
                    )),
                ));
            }
            let mut seen = std::collections::HashSet::new();
            for item in batch.classifications {
                let confidence = validate_probability(item.confidence, "contradiction confidence")?;
                if item.left >= item.right
                    || item.right >= facts.len()
                    || !seen.insert((item.left, item.right))
                {
                    return Err(agent_context_db_core::ContextError::Llm(
                        LlmError::Provider(
                            "contradiction classifier returned invalid or duplicate pair".into(),
                        ),
                    ));
                }
                let rationale = validate_llm_content(&item.rationale, "contradiction rationale")?;
                if item.contradiction {
                    violations.push(ConstraintViolation {
                        uri: facts[item.left].uri.clone(),
                        constraint: Constraint::NoContradiction,
                        description: format!("contradiction between {} and {} (confidence={confidence:.2}): {rationale}", facts[item.left].uri, facts[item.right].uri),
                    });
                }
            }
        }

        // 被 invalidate 的传播检查
        for p in products {
            if p.metadata.status == ConsolidationStatus::Stale
                && p.contradiction_uris.iter().any(|c| c == &p.uri)
            {
                violations.push(ConstraintViolation {
                    uri: p.uri.clone(),
                    constraint: Constraint::InvalidationPropagation,
                    description: format!(
                        "product {} is stale and may be based on invalidated evidence",
                        p.uri
                    ),
                });
            }
        }

        Ok(violations)
    }

    /// 检查 Profile 冲突。
    pub fn check_profile_conflicts(
        &self,
        products: &[ConsolidationProduct],
    ) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();
        let profiles: Vec<_> = products
            .iter()
            .filter(|p| p.content_type == ContentType::Profile)
            .collect();

        for i in 0..profiles.len() {
            for j in (i + 1)..profiles.len() {
                if profiles[i].content == profiles[j].content {
                    violations.push(ConstraintViolation {
                        uri: profiles[i].uri.clone(),
                        constraint: Constraint::NoProfileConflict,
                        description: format!("duplicate profile {}", profiles[j].uri),
                    });
                }
            }
        }

        violations
    }
}

#[cfg(test)]
mod sleeptime_tests {
    use super::*;
    use agent_context_db_core::{
        ContentLevel, ContentPayload, ContentStore, ContextError, GraphRelation, GraphStore,
        JsonSchema, LlmError, MvccVersion, Page, TenantId,
    };
    use agent_context_db_testkit::MemoryContextStore;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    #[derive(Default)]
    struct MemoryProgress(Mutex<HashMap<String, SleeptimeProgress>>);
    #[async_trait]
    impl ProgressStore for MemoryProgress {
        async fn load(&self, key: &str) -> Result<Option<SleeptimeProgress>> {
            Ok(self.0.lock().get(key).cloned())
        }
        async fn save(&self, key: &str, progress: &SleeptimeProgress) -> Result<()> {
            self.0.lock().insert(key.into(), progress.clone());
            Ok(())
        }
    }

    #[derive(Default)]
    struct CountingSignals {
        calls: AtomicUsize,
        value: EntrySignals,
    }
    #[async_trait]
    impl SignalProvider for CountingSignals {
        async fn signals(&self, _: &ContextUri) -> Result<EntrySignals> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.value.clone())
        }
    }

    struct CountingGraph {
        calls: AtomicUsize,
    }
    #[async_trait]
    impl GraphStore for CountingGraph {
        async fn add_edge(&self, _: &ContextUri, _: &ContextUri, _: GraphRelation) -> Result<()> {
            Ok(())
        }
        async fn remove_edge(&self, _: &ContextUri, _: &ContextUri) -> Result<()> {
            Ok(())
        }
        async fn outgoing_neighbors(
            &self,
            _: &ContextUri,
            _: Option<GraphRelation>,
        ) -> Result<Vec<ContextUri>> {
            Ok(vec![])
        }
        async fn batch_traverse(
            &self,
            _: &[ContextUri],
            _: &[GraphRelation],
            _: usize,
        ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
            Ok(vec![])
        }
        async fn centrality(&self, _: &ContextUri) -> Result<f32> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(0.8)
        }
    }

    struct NoopLlm;
    #[async_trait]
    impl LlmClient for NoopLlm {
        async fn complete(&self, _: &str, _: &LlmOpts) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }
        async fn complete_json(
            &self,
            _: &str,
            _: &JsonSchema,
            _: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            Ok("{}".into())
        }
    }

    fn uri(value: &str) -> ContextUri {
        ContextUri::parse(value).unwrap()
    }
    fn entry(value: &str) -> ContextEntry {
        ContextEntry::new_text(uri(value), TenantId(Uuid::nil()), "memory")
    }
    async fn seeded_store(count: usize) -> Arc<MemoryContextStore> {
        let store = Arc::new(MemoryContextStore::new());
        for index in 0..count {
            ContentStore::write(
                &*store,
                entry(&format!("uwu://tenant/agent/a/memories/{index:02}")),
            )
            .await
            .unwrap();
        }
        store
    }
    fn executor(
        store: Arc<dyn ContentStore>,
        progress: Arc<dyn ProgressStore>,
        signals: Arc<dyn SignalProvider>,
        graph: Arc<dyn GraphStore>,
        limit: usize,
    ) -> SleeptimeExecutor {
        let mut executor = SleeptimeExecutor::new(
            uri("uwu://tenant/agent/a"),
            store,
            graph,
            progress,
            signals,
            Arc::new(NoopLlm),
            limit,
        )
        .unwrap();
        executor.tasks = vec![];
        executor
    }
    fn engine() -> ConsolidationEngine {
        ConsolidationEngine::new(ConsolidationConfig::default(), Arc::new(NoopLlm)).unwrap()
    }

    #[tokio::test]
    async fn rejects_scope_outside_bound_agent() {
        let store = seeded_store(1).await;
        let executor = executor(
            store.clone(),
            Arc::new(MemoryProgress::default()),
            Arc::new(CountingSignals::default()),
            Arc::new(CountingGraph {
                calls: AtomicUsize::new(0),
            }),
            1,
        );
        let error = executor
            .run_once(&engine(), &uri("uwu://tenant/agent/b"))
            .await
            .unwrap_err();
        assert!(matches!(error, ContextError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn second_batch_is_not_starved_and_restart_resumes_progress() {
        let store = seeded_store(3).await;
        let progress = Arc::new(MemoryProgress::default());
        let signals = Arc::new(CountingSignals::default());
        let graph = Arc::new(CountingGraph {
            calls: AtomicUsize::new(0),
        });
        let first = executor(
            store.clone(),
            progress.clone(),
            signals.clone(),
            graph.clone(),
            2,
        )
        .run_once(&engine(), &uri("uwu://tenant/agent/a"))
        .await
        .unwrap();
        assert_eq!(first.cursor, Some(uri("uwu://tenant/agent/a/memories/01")));
        let second = executor(store, progress, signals, graph, 2)
            .run_once(&engine(), &uri("uwu://tenant/agent/a"))
            .await
            .unwrap();
        assert_eq!(second.cursor, Some(uri("uwu://tenant/agent/a/memories/02")));
        assert_eq!(second.completeness, 1.0);
    }

    struct WriteFailingStore {
        inner: Arc<MemoryContextStore>,
        writes: AtomicUsize,
    }
    #[async_trait]
    impl ContentStore for WriteFailingStore {
        async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload> {
            self.inner.read(uri, level).await
        }
        async fn write(&self, _: ContextEntry) -> Result<MvccVersion> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            Err(ContextError::Storage("injected write failure".into()))
        }
        async fn delete(&self, uri: &ContextUri) -> Result<()> {
            self.inner.delete(uri).await
        }
        async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()> {
            self.inner.rename(from, to).await
        }
        async fn batch_write(&self, _: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
            Err(ContextError::Storage("injected write failure".into()))
        }
        async fn scan_by_prefix(
            &self,
            prefix: &str,
            page: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            self.inner.scan_by_prefix(prefix, page).await
        }
        async fn scan_by_type(
            &self,
            prefix: &str,
            content_type: ContentType,
            page: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            self.inner.scan_by_type(prefix, content_type, page).await
        }
    }

    async fn write_failing_executor(
        mut source: ContextEntry,
        task: SleeptimeTask,
    ) -> (SleeptimeExecutor, Arc<WriteFailingStore>) {
        source.updated_at = Utc::now() - chrono::Duration::days(365);
        if matches!(task, SleeptimeTask::SpacedRepetitionReview) {
            source.metadata.state_scope = Some(StateScope::Long);
        }
        let inner = Arc::new(MemoryContextStore::new());
        inner.write(source).await.unwrap();
        let store = Arc::new(WriteFailingStore {
            inner,
            writes: AtomicUsize::new(0),
        });
        let mut executor = executor(
            store.clone(),
            Arc::new(MemoryProgress::default()),
            Arc::new(CountingSignals::default()),
            Arc::new(CountingGraph {
                calls: AtomicUsize::new(0),
            }),
            10,
        );
        executor.tasks = vec![task];
        (executor, store)
    }

    #[tokio::test]
    async fn quality_write_failure_is_propagated_without_completing_task() {
        let mut source = entry("uwu://tenant/agent/a/memories/quality");
        source.metadata.quality_score = Some(0.0);
        let (executor, store) =
            write_failing_executor(source, SleeptimeTask::QualityReassessment).await;

        let error = executor
            .run_once(&engine(), &uri("uwu://tenant/agent/a"))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "storage: injected write failure");
        assert_eq!(store.writes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn review_write_failure_is_propagated_without_counting_written_review() {
        let mut source = entry("uwu://tenant/agent/a/memories/review");
        source.metadata.quality_score = Some(0.0);
        let (executor, store) =
            write_failing_executor(source, SleeptimeTask::SpacedRepetitionReview).await;

        let error = executor
            .run_once(&engine(), &uri("uwu://tenant/agent/a"))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "storage: injected write failure");
        assert_eq!(store.writes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn health_family_write_failure_is_propagated() {
        let mut source = entry("uwu://tenant/agent/a/fact/health");
        source.metadata.content_type = Some(ContentType::Fact);
        source.metadata.quality_score = Some(0.0);
        let (executor, store) =
            write_failing_executor(source, SleeptimeTask::KnowledgeHealthDiagnosis).await;

        let error = executor
            .run_once(&engine(), &uri("uwu://tenant/agent/a"))
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "storage: injected write failure");
        assert_eq!(store.writes.load(Ordering::SeqCst), 1);
    }

    struct FailingStore;
    #[async_trait]
    impl ContentStore for FailingStore {
        async fn read(&self, _: &ContextUri, _: ContentLevel) -> Result<ContentPayload> {
            Err(ContextError::Storage("scan failed".into()))
        }
        async fn write(&self, _: ContextEntry) -> Result<MvccVersion> {
            Err(ContextError::Storage("scan failed".into()))
        }
        async fn delete(&self, _: &ContextUri) -> Result<()> {
            Ok(())
        }
        async fn rename(&self, _: &ContextUri, _: &ContextUri) -> Result<()> {
            Ok(())
        }
        async fn batch_write(&self, _: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
            Ok(vec![])
        }
        async fn scan_by_prefix(&self, _: &str, _: PageRequest) -> Result<Page<ContextEntry>> {
            Err(ContextError::Storage("scan failed".into()))
        }
        async fn scan_by_type(
            &self,
            _: &str,
            _: ContentType,
            _: PageRequest,
        ) -> Result<Page<ContextEntry>> {
            Err(ContextError::Storage("scan failed".into()))
        }
    }

    #[tokio::test]
    async fn scan_error_is_propagated_and_persisted() {
        let progress = Arc::new(MemoryProgress::default());
        let executor = executor(
            Arc::new(FailingStore),
            progress.clone(),
            Arc::new(CountingSignals::default()),
            Arc::new(CountingGraph {
                calls: AtomicUsize::new(0),
            }),
            2,
        );
        let error = executor
            .run_once(&engine(), &uri("uwu://tenant/agent/a"))
            .await
            .unwrap_err();
        assert!(matches!(error, ContextError::Storage(_)));
        let saved = progress
            .load("sleeptime:uwu://tenant/agent/a")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(saved.last_error.as_deref(), Some("storage: scan failed"));
        assert_eq!(saved.attempt, 1);
    }

    #[tokio::test]
    async fn missing_signals_reduce_completeness_and_real_providers_are_called() {
        let store = seeded_store(1).await;
        let signals = Arc::new(CountingSignals {
            calls: AtomicUsize::new(0),
            value: EntrySignals {
                downstream_success_rate: Some(0.9),
                contradiction_count: Some(2),
                corroboration_count: Some(3),
                repeated_observations: Some(4),
                tenant_priority: Some(0.8),
                ..Default::default()
            },
        });
        let graph = Arc::new(CountingGraph {
            calls: AtomicUsize::new(0),
        });
        let report = executor(
            store,
            Arc::new(MemoryProgress::default()),
            signals.clone(),
            graph.clone(),
            2,
        )
        .run_once(&engine(), &uri("uwu://tenant/agent/a"))
        .await
        .unwrap();
        assert!(report.signal_completeness < 1.0);
        assert_eq!(signals.calls.load(Ordering::SeqCst), 1);
        assert_eq!(graph.calls.load(Ordering::SeqCst), 1);
    }

    #[derive(Default)]
    struct CountingLifecycle(AtomicUsize);
    #[async_trait]
    impl LifecycleExecutorPort for CountingLifecycle {
        async fn run_pending(&self) -> Result<Vec<crate::lifecycle_execution::LifecycleJob>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![])
        }
        async fn submit(
            &self,
            _: ContextUri,
            _: agent_context_db_core::LifecycleAction,
        ) -> Result<crate::lifecycle_execution::LifecycleJob> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn lifecycle_executor_is_wired_before_and_after_run() {
        let store = seeded_store(0).await;
        let lifecycle = Arc::new(CountingLifecycle::default());
        let executor = executor(
            store,
            Arc::new(MemoryProgress::default()),
            Arc::new(CountingSignals::default()),
            Arc::new(CountingGraph {
                calls: AtomicUsize::new(0),
            }),
            2,
        )
        .with_lifecycle_executor(lifecycle.clone());
        executor
            .run_once(&engine(), &uri("uwu://tenant/agent/a"))
            .await
            .unwrap();
        assert_eq!(lifecycle.0.load(Ordering::SeqCst), 2);
    }
}
