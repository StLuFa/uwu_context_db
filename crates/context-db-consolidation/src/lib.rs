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
pub mod lineage;
pub mod loader;
pub mod marketplace;
pub mod opportunity;
pub mod patcher;
pub mod quality;
pub mod relational_axis;
pub mod rif;
pub mod security;
pub mod semantic_axis;
pub mod tiered_cache;

use crate::quality::{HorizonAwareQualityScorer, HorizonQualitySignals, QualityRoute};
use agent_context_db_core::{
    ConsolidationStatus, ContentType, ContextEntry, ContextUri, EpistemicType, LineageEntry,
    LlmClient, LlmOpts, Result, StateScope, ValidityRecord,
};
use agent_context_db_knowledge_network::identity::IdentityRegistry;
use agent_context_db_marketplace_types::{
    AgentId, KnowledgeProvenance, KnowledgeProvenancePayload, evidence_chain_hash,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    pub superseded_claim: Option<String>,
    pub evidence_uris: Vec<ContextUri>,
    pub contradiction_uris: Vec<ContextUri>,
    pub error_pattern: Option<String>,
    pub hypothesis_outcome: Option<HypothesisOutcome>,
    pub preconditions: Option<String>,
    pub expected_outcome: Option<String>,
    pub related_policy_uris: Vec<ContextUri>,
    pub provenance: Option<KnowledgeProvenance>,
    pub metadata: ConsolidationMeta,
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

/// 假设验证结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisOutcome {
    Confirmed,
    Falsified,
    Inconclusive,
}

/// 巩固元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationMeta {
    pub source_session: Option<String>,
    pub generation: usize,
    pub status: ConsolidationStatus,
    pub patch_count: usize,
    pub lineage: Vec<LineageEntry>,
    pub validity: Option<ValidityRecord>,
    pub half_life_days: Option<f64>,
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
        spots.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());
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
        if record.declared_confidences.len() % 10 == 0 {
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
                let actual_logit =
                    (actual.max(0.01).min(0.99) / (1.0 - actual.max(0.01).min(0.99))).ln();
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
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_threshold: 0.05,
            quality_threshold_add: 0.7,
            quality_threshold_update: 0.3,
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
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self {
            config: ConsolidationConfig::default(),
            llm,
            calibrator: ConfidenceCalibrator::new(),
            ignorance: IgnoranceMap::new(),
        }
    }

    pub fn with_config(config: ConsolidationConfig, llm: Arc<dyn LlmClient>) -> Self {
        Self {
            config,
            llm,
            calibrator: ConfidenceCalibrator::new(),
            ignorance: IgnoranceMap::new(),
        }
    }

    /// Generate→Critique→Revise 收敛循环。
    #[tracing::instrument(skip(self, entry), fields(uri = %entry.uri.as_str()))]
    pub async fn consolidate(&self, entry: &ContextEntry) -> Result<ConsolidationProduct> {
        let content = entry.l0_text().to_string();
        let ct = entry.content_type().unwrap_or(ContentType::Fact);
        let et = entry
            .metadata
            .epistemic_type()
            .unwrap_or(EpistemicType::Fact);

        // Generate (LLM-driven)
        let mut current = self.generate(entry, &content, ct, et).await;
        let mut prev_score = 0.0;

        for iteration in 1..=self.config.max_iterations {
            // Critique
            let critique = self.critique(&current, entry).await;
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

            let revised = self.revise(&current, &critique).await;
            current = revised;
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
    ) -> ConsolidationProduct {
        let prompt = format!(
            r#"Consolidate the following memory into a structured insight.

Memory content: "{content}"
Content type: {ct:?}
Epistemic type: {et:?}

Produce a concise, self-contained principle that captures the key knowledge.
Return ONLY the consolidated principle text (1-3 sentences, no JSON, no markup)."#
        );

        let consolidated = match self
            .llm
            .complete(
                &prompt,
                &LlmOpts {
                    max_tokens: Some(512),
                    temperature: Some(0.1),
                    ..Default::default()
                },
            )
            .await
        {
            Ok(text) => text.trim().to_string(),
            Err(_) => content.to_string(),
        };

        ConsolidationProduct {
            uri: entry.uri.clone(),
            content_type: ct,
            epistemic_type: et,
            content: consolidated,
            quality_score: entry.metadata.quality_score.unwrap_or(0.5),
            confidence: 0.5,
            superseded_claim: None,
            evidence_uris: vec![],
            contradiction_uris: vec![],
            error_pattern: None,
            hypothesis_outcome: None,
            preconditions: None,
            expected_outcome: None,
            related_policy_uris: vec![],
            provenance: None,
            metadata: ConsolidationMeta {
                source_session: None,
                generation: 1,
                status: ConsolidationStatus::InProgress,
                patch_count: 0,
                lineage: vec![],
                validity: entry.metadata.validity.clone(),
                half_life_days: None,
            },
        }
    }

    async fn critique(
        &self,
        product: &ConsolidationProduct,
        _entry: &ContextEntry,
    ) -> CritiqueResult {
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

        let opts = LlmOpts {
            max_tokens: Some(512),
            temperature: Some(0.0),
            ..Default::default()
        };

        match self.llm.complete(&prompt, &opts).await {
            Ok(response) => {
                let parsed: Option<serde_json::Value> = serde_json::from_str(&response).ok();
                match parsed {
                    Some(ref v) => CritiqueResult {
                        quality_score: v["quality_score"].as_f64().unwrap_or(0.7) as f32,
                        confidence: v["confidence"].as_f64().unwrap_or(0.7) as f32,
                        suggestions: v["suggestions"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|s| s.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    },
                    None => {
                        // Parse failed — fall back to heuristic
                        let score = if product.content.len() >= 10 {
                            0.8
                        } else {
                            0.3
                        };
                        CritiqueResult {
                            quality_score: score,
                            confidence: score,
                            suggestions: vec![],
                        }
                    }
                }
            }
            Err(_) => {
                // LLM unavailable — fall back to heuristic
                let score = if !product.content.is_empty() && product.content.len() >= 10 {
                    0.8
                } else {
                    0.3
                };
                CritiqueResult {
                    quality_score: score,
                    confidence: score,
                    suggestions: vec![],
                }
            }
        }
    }

    async fn revise(
        &self,
        product: &ConsolidationProduct,
        critique: &CritiqueResult,
    ) -> ConsolidationProduct {
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

            if let Ok(response) = self
                .llm
                .complete(
                    &prompt,
                    &LlmOpts {
                        max_tokens: Some(512),
                        temperature: Some(0.0),
                        ..Default::default()
                    },
                )
                .await
            {
                revised.content = response.trim().to_string();
            }
            revised.metadata.status = ConsolidationStatus::InProgress;
        }
        revised
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

/// 评估结果。
#[derive(Debug, Clone)]
pub struct CritiqueResult {
    pub quality_score: f32,
    pub confidence: f32,
    pub suggestions: Vec<String>,
}

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
            Some(v) => at >= v.valid_from && v.valid_until.map_or(true, |until| at <= until),
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
                .map_or(true, |v| v.valid_until.is_none())
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
                    half_life_days: None,
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
}

/// Sleeptime 执行器 — 空闲时执行后台整理，支持从存储加载数据。
pub struct SleeptimeExecutor {
    pub tasks: Vec<SleeptimeTask>,
    pub interval: std::time::Duration,
    checker: ConsistencyChecker,
    /// 纠缠检测器 —— EntanglementDetection 任务的实际执行者。
    entanglement: crate::entanglement::EntanglementDetector,
    /// 可选的存储读取器（Sleeptime 加载条目用）。
    store: Option<Arc<dyn agent_context_db_core::ContentStore>>,
    /// 可选的关系图存储（BackwardEvolve 用于图边扩展）。
    graph: Option<Arc<dyn agent_context_db_core::GraphStore>>,
    /// 可选的向量索引 —— `on_adopted()` 触发 RIF 抑制时使用。
    vector_index: Option<Arc<dyn agent_context_db_core::VectorIndex>>,
}

impl SleeptimeExecutor {
    pub fn new() -> Self {
        Self {
            tasks: vec![
                SleeptimeTask::QualityReassessment,
                SleeptimeTask::ConsistencyCheck,
                SleeptimeTask::EntanglementDetection,
                SleeptimeTask::IgnoranceMapUpdate,
                SleeptimeTask::CalibrationUpdate,
                SleeptimeTask::ContextRotPrune,
                SleeptimeTask::BackwardEvolve,
            ],
            interval: std::time::Duration::from_secs(3600),
            checker: ConsistencyChecker::new(),
            entanglement: crate::entanglement::EntanglementDetector::new(0.3),
            store: None,
            graph: None,
            vector_index: None,
        }
    }

    /// 绑定存储读取器，使 Sleeptime 可以从存储中加载条目。
    pub fn with_store(mut self, store: Arc<dyn agent_context_db_core::ContentStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// 绑定关系图存储，启用 BackwardEvolve 的图边扩展路径。
    pub fn with_graph(mut self, graph: Arc<dyn agent_context_db_core::GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// 绑定向量索引，启用 `on_adopted()` 的 RIF 邻居抑制路径。
    pub fn with_vector_index(mut self, index: Arc<dyn agent_context_db_core::VectorIndex>) -> Self {
        self.vector_index = Some(index);
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
    ) -> SleeptimeReport {
        let mut report = SleeptimeReport::default();

        // 从存储中加载 scope 下的条目
        let entries: Vec<ContextEntry> = if let Some(ref store) = self.store {
            store
                .scan_by_prefix(&scope.to_string(), 100)
                .await
                .unwrap_or_default()
        } else {
            vec![]
        };

        for task in &self.tasks {
            match task {
                SleeptimeTask::QualityReassessment => {
                    let scorer = HorizonAwareQualityScorer::default();
                    let Some(store) = self.store.as_ref() else {
                        tracing::info!(scope=%scope, count=%entries.len(), "sleeptime: quality reassessment skipped without store");
                        report.tasks_executed += 1;
                        continue;
                    };

                    let mut reassessed = 0usize;
                    let mut promoted_mid = 0usize;
                    let mut promoted_long = 0usize;
                    let mut archived = 0usize;
                    let mut training_candidates = 0usize;

                    for entry in &entries {
                        let signals = HorizonQualitySignals {
                            adoption_rate: entry
                                .metadata
                                .quality_score
                                .unwrap_or(0.5)
                                .clamp(0.0, 1.0),
                            recall_rate: entry
                                .metadata
                                .quality_score
                                .unwrap_or(0.5)
                                .clamp(0.0, 1.0),
                            downstream_success_rate: 0.5,
                            contradiction_count: 0,
                            corroboration_count: 0,
                            repeated_observations: 1,
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
                        match outcome.route {
                            QualityRoute::CompressToMidTerm => {
                                updated.metadata.state_scope = Some(StateScope::Mid);
                                promoted_mid += 1;
                            }
                            QualityRoute::PromoteToLongTerm | QualityRoute::IncludeInTraining => {
                                updated.metadata.state_scope = Some(StateScope::Long);
                                promoted_long += 1;
                                if matches!(outcome.route, QualityRoute::IncludeInTraining) {
                                    training_candidates += 1;
                                }
                            }
                            QualityRoute::Archive
                            | QualityRoute::ForgetCandidate
                            | QualityRoute::ForgetShortTerm => {
                                updated
                                    .metadata
                                    .tags
                                    .push("quality:archive-candidate".into());
                                archived += 1;
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
                        if store.write(updated).await.is_ok() {
                            reassessed += 1;
                        }
                    }

                    report.quality_reassessed = reassessed;
                    report.quality_promoted_mid = promoted_mid;
                    report.quality_promoted_long = promoted_long;
                    report.quality_archived = archived;
                    report.training_candidates = training_candidates;
                    tracing::info!(scope=%scope, reassessed, promoted_mid, promoted_long, archived, training_candidates, "sleeptime: horizon-aware quality reassessment");
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
                            superseded_claim: None,
                            evidence_uris: vec![],
                            contradiction_uris: vec![],
                            error_pattern: None,
                            hypothesis_outcome: None,
                            preconditions: None,
                            expected_outcome: None,
                            related_policy_uris: vec![],
                            provenance: None,
                            metadata: ConsolidationMeta {
                                source_session: None,
                                generation: 0,
                                status: ConsolidationStatus::Pending,
                                patch_count: 0,
                                lineage: vec![],
                                validity: e.metadata.validity.clone(),
                                half_life_days: None,
                            },
                        })
                        .collect();
                    let violations = self.checker.check(&products);
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
                    // 用当前条目样本喂校准器 —— 声明置信度 = quality_score，
                    // 采纳与否用 quality_score ≥ 0.7 作为启发式判据
                    let mut samples = 0usize;
                    for entry in &entries {
                        let epistemic = entry
                            .metadata
                            .epistemic_type()
                            .unwrap_or(EpistemicType::Fact);
                        let declared = entry.metadata.quality_score.unwrap_or(0.5);
                        let adopted = declared >= 0.7;
                        engine.calibrator.update(epistemic, declared, adopted);
                        samples += 1;
                    }
                    report.calibration_samples = samples;
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
                            superseded_claim: None,
                            evidence_uris: vec![],
                            contradiction_uris: vec![],
                            error_pattern: None,
                            hypothesis_outcome: None,
                            preconditions: None,
                            expected_outcome: None,
                            related_policy_uris: vec![],
                            provenance: None,
                            metadata: ConsolidationMeta {
                                source_session: None,
                                generation: 0,
                                status: ConsolidationStatus::Pending,
                                patch_count: 0,
                                lineage: vec![],
                                validity: e.metadata.validity.clone(),
                                half_life_days: None,
                            },
                        })
                        .collect();
                    if let Some(last) = products.last() {
                        let mut evolver = BackwardEvolver::new();
                        if let Some(ref g) = self.graph {
                            evolver = evolver.with_graph(g.clone());
                        }
                        let mut mutable = entries.clone();
                        let _updated = evolver.evolve_backward(last, &mut mutable).await;
                    }
                    report.tasks_executed += 1;
                }
            }
        }
        report
    }
}

/// Sleeptime 报告。
#[derive(Debug, Clone, Default)]
pub struct SleeptimeReport {
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

/// 一致性约束检查器 — Sleeptime 阶段运行。
pub struct ConsistencyChecker {
    /// 语义矛盾阈值（余弦距离 < 此值 → 可能矛盾，需 LLM 确认）。
    contradiction_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub uri: ContextUri,
    pub constraint: Constraint,
    pub description: String,
}

impl ConsistencyChecker {
    pub fn new() -> Self {
        Self {
            contradiction_threshold: 0.15,
        } // 余弦距离 < 0.15 = 高度矛盾
    }

    /// 检查所有一致性约束。
    ///
    /// 支持：
    /// - 空内容检查
    /// - Fact 必须有证据
    /// - Fact vs Fact 语义矛盾（Jaccard + 否定词检测）
    /// - 被 invalidate 后代标记
    pub fn check(&self, products: &[ConsolidationProduct]) -> Vec<ConstraintViolation> {
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
            if p.content_type == ContentType::Fact && p.evidence_uris.is_empty() {
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

        for i in 0..facts.len() {
            for j in (i + 1)..facts.len() {
                let a = &facts[i];
                let b = &facts[j];

                // Jaccard 相似度 + 否定词检测
                let a_words: std::collections::HashSet<&str> =
                    a.content.split_whitespace().collect();
                let b_words: std::collections::HashSet<&str> =
                    b.content.split_whitespace().collect();
                let intersection = a_words.intersection(&b_words).count();
                let union = a_words.union(&b_words).count();
                let jaccard = if union > 0 {
                    intersection as f32 / union as f32
                } else {
                    0.0
                };

                // 高重叠 + 语义对立 = 可能矛盾
                let has_contradiction_marker =
                    Self::detect_contradiction_marker(&a.content, &b.content);

                if jaccard > 0.5 && has_contradiction_marker {
                    violations.push(ConstraintViolation {
                        uri: a.uri.clone(),
                        constraint: Constraint::NoContradiction,
                        description: format!(
                            "possible contradiction between {} and {} (jaccard={:.2})",
                            a.uri, b.uri, jaccard
                        ),
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

        violations
    }

    /// 检测两个文本中是否存在语义矛盾标记（否定词 + 相同主题）。
    fn detect_contradiction_marker(a: &str, b: &str) -> bool {
        let negation_words = [
            "not",
            "no",
            "never",
            "impossible",
            "cannot",
            "can't",
            "don't",
            "doesn't",
            "false",
            "wrong",
            "incorrect",
            "不",
            "没有",
            "错误",
            "无法",
            "不能",
            "不可",
        ];
        let a_has_negation = negation_words.iter().any(|n| a.contains(n));
        let b_has_negation = negation_words.iter().any(|n| b.contains(n));
        // 一个肯定一个否定 = 可能矛盾
        a_has_negation != b_has_negation
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
