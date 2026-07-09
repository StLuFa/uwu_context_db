//! 创新功能扩展（F16 预测性预加载 + F28 增量检索学习）。

use agent_context_db_core::{ContentLevel, ContentPayload, ContentType, ContextUri, FsOps};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// F16 预测性预加载
// ═══════════════════════════════════════════════════════════════════════════

/// 上下文访问记录。
#[derive(Debug, Clone)]
pub struct AccessRecord {
    pub uri: ContextUri,
    pub timestamp: Instant,
    pub access_count: u64,
}

/// 访问模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Random,
    Scan,
    Burst,
}

/// 预测结果。
#[derive(Debug, Clone)]
pub struct PrefetchPrediction {
    pub uri: ContextUri,
    pub probability: f32,
    pub pattern: AccessPattern,
    pub prefetch_level: ContentLevel,
}

/// 预测性预加载器。
///
/// 基于访问频率和时序关联性预测下一步可能需要的上下文，
/// 异步预加载到热缓存。预测模型可替换（JEPA / 统计 / 规则）。
pub struct PredictivePrefetcher {
    fs: Arc<dyn FsOps>,
    /// URI → 访问记录
    access_history: parking_lot::Mutex<HashMap<String, AccessRecord>>,
    /// URI → 关联 URI（通过检索轨迹学习）
    associations: parking_lot::Mutex<HashMap<String, Vec<(String, f32)>>>,
    /// 预取大小
    prefetch_size: usize,
}

impl PredictivePrefetcher {
    pub fn new(fs: Arc<dyn FsOps>, prefetch_size: usize) -> Self {
        Self {
            fs,
            access_history: parking_lot::Mutex::new(HashMap::new()),
            associations: parking_lot::Mutex::new(HashMap::new()),
            prefetch_size,
        }
    }

    /// 记录一次上下文访问。
    pub fn record_access(&self, uri: &ContextUri, _pattern: AccessPattern) {
        let mut history = self.access_history.lock();
        let entry = history
            .entry(uri.to_string().clone())
            .or_insert_with(|| AccessRecord {
                uri: uri.clone(),
                timestamp: Instant::now(),
                access_count: 0,
            });
        entry.access_count += 1;
    }

    /// 记录检索轨迹中的关联（URI A → URI B）。
    pub fn learn_association(&self, from: &ContextUri, to: &ContextUri) {
        let mut assoc = self.associations.lock();
        let list = assoc.entry(from.to_string()).or_default();
        let to_str = to.to_string();
        if let Some(entry) = list.iter_mut().find(|(u, _)| u == &to_str) {
            entry.1 += 0.1; // 强化
        } else {
            list.push((to_str, 0.3));
        }
        // 衰减旧关联
        for (_, score) in list.iter_mut() {
            *score *= 0.99;
        }
    }

    /// 预测下一步可能需要的上下文 URI。
    pub fn predict(&self, current: &ContextUri) -> Vec<PrefetchPrediction> {
        let mut predictions = Vec::new();

        // 策略1：关联图预测
        if let Some(associations) = self.associations.lock().get(&current.to_string()) {
            for (uri, prob) in associations.iter().take(self.prefetch_size) {
                if let Ok(parsed) = ContextUri::parse(uri.clone()) {
                    predictions.push(PrefetchPrediction {
                        uri: parsed,
                        probability: *prob,
                        pattern: AccessPattern::Sequential,
                        prefetch_level: if *prob > 0.7 {
                            ContentLevel::L1
                        } else {
                            ContentLevel::L0
                        },
                    });
                }
            }
        }

        // 策略2：高频访问排序
        if predictions.is_empty() {
            let history = self.access_history.lock();
            let mut sorted: Vec<&AccessRecord> = history.values().collect();
            sorted.sort_by_key(|r| -(r.access_count as i64));
            for record in sorted.iter().take(self.prefetch_size) {
                if &record.uri != current {
                    predictions.push(PrefetchPrediction {
                        uri: record.uri.clone(),
                        probability: 0.4,
                        pattern: AccessPattern::Burst,
                        prefetch_level: ContentLevel::L0,
                    });
                }
            }
        }

        predictions
    }

    /// 执行预取（加载到调用方缓存）。
    pub async fn prefetch(
        &self,
        predictions: &[PrefetchPrediction],
    ) -> Vec<(ContextUri, ContentPayload)> {
        let mut loaded = Vec::new();
        for pred in predictions.iter().take(self.prefetch_size) {
            if let Ok(content) = self.fs.read(&pred.uri, pred.prefetch_level).await {
                loaded.push((pred.uri.clone(), content));
            }
        }
        loaded
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F28 增量检索学习
// ═══════════════════════════════════════════════════════════════════════════

/// 用户反馈信号。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelevanceFeedback {
    /// 命中相关
    Relevant { uri: ContextUri, score: f32 },
    /// 命中不相关
    NotRelevant { uri: ContextUri },
    /// 缺少的信息描述
    Missing { description: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalLearningSignal {
    pub uri: ContextUri,
    pub query: String,
    pub content_type: Option<ContentType>,
    pub base_relevance: f32,
    pub clicked: bool,
    pub adopted: bool,
    pub dwell_seconds: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LearnedRankingWeights {
    pub semantic: f32,
    pub uri_feedback: f32,
    pub query_type: f32,
    pub exploration: f32,
}

impl Default for LearnedRankingWeights {
    fn default() -> Self {
        Self {
            semantic: 0.58,
            uri_feedback: 0.24,
            query_type: 0.14,
            exploration: 0.04,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingKnowledgeSignal {
    pub query: String,
    pub description: String,
    pub expansion_terms: Vec<String>,
    pub weight: f32,
}

/// 增量学习器 —— 基于用户反馈持续调整检索权重。
pub struct IncrementalRetrievalLearner {
    /// URI → 累计反馈分数
    uri_scores: parking_lot::Mutex<HashMap<String, f32>>,
    /// 查询模式 → 最佳 target_dir 偏好
    dir_preferences: parking_lot::Mutex<HashMap<String, Vec<(String, f32)>>>,
    /// 查询 token/type → 内容类型偏好
    query_type_weights: parking_lot::Mutex<HashMap<String, HashMap<ContentType, f32>>>,
    /// 缺失知识模式，用于触发扩召回和主动探索
    missing_patterns: parking_lot::Mutex<Vec<MissingKnowledgeSignal>>,
    weights: parking_lot::RwLock<LearnedRankingWeights>,
    /// 学习率
    learning_rate: f32,
}

impl IncrementalRetrievalLearner {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            uri_scores: parking_lot::Mutex::new(HashMap::new()),
            dir_preferences: parking_lot::Mutex::new(HashMap::new()),
            query_type_weights: parking_lot::Mutex::new(HashMap::new()),
            missing_patterns: parking_lot::Mutex::new(Vec::new()),
            weights: parking_lot::RwLock::new(LearnedRankingWeights::default()),
            learning_rate,
        }
    }

    /// 接受一批反馈，更新内部权重。
    pub fn apply_feedback(&self, query: &str, feedbacks: &[RelevanceFeedback]) {
        let mut scores = self.uri_scores.lock();
        let mut prefs = self.dir_preferences.lock();

        for fb in feedbacks {
            match fb {
                RelevanceFeedback::Relevant { uri, score } => {
                    let entry = scores.entry(uri.to_string().clone()).or_insert(0.5);
                    *entry = (*entry + self.learning_rate * (score.clamp(0.0, 1.0) - *entry))
                        .clamp(0.0, 1.0);
                    reinforce_dir(&mut prefs, query, uri, self.learning_rate, *score);
                }
                RelevanceFeedback::NotRelevant { uri } => {
                    let entry = scores.entry(uri.to_string().clone()).or_insert(0.5);
                    *entry = (*entry * (1.0 - self.learning_rate)).max(0.05);
                    reinforce_dir(&mut prefs, query, uri, self.learning_rate, 0.1);
                }
                RelevanceFeedback::Missing { description } => {
                    self.record_missing(query, description);
                    let normalized = normalize_query(query);
                    let dirs = prefs.entry(normalized).or_default();
                    if let Some(last) = dirs.last_mut() {
                        last.1 = (last.1 * 0.8).max(0.1);
                    }
                }
            }
        }
        self.rebalance_weights();
    }

    pub fn apply_learning_signals(&self, signals: &[RetrievalLearningSignal]) {
        let mut grouped: HashMap<String, Vec<RelevanceFeedback>> = HashMap::new();
        for signal in signals {
            let reward = signal_reward(signal);
            let feedback = if reward >= 0.5 {
                RelevanceFeedback::Relevant {
                    uri: signal.uri.clone(),
                    score: reward,
                }
            } else {
                RelevanceFeedback::NotRelevant {
                    uri: signal.uri.clone(),
                }
            };
            grouped
                .entry(signal.query.clone())
                .or_default()
                .push(feedback);
            if let Some(content_type) = signal.content_type {
                self.update_query_type(&signal.query, content_type, reward);
            }
        }
        for (query, feedbacks) in grouped {
            self.apply_feedback(&query, &feedbacks);
        }
    }

    /// 获取 URI 的学习后相关度分数。
    pub fn learned_score(&self, uri: &ContextUri) -> f32 {
        self.uri_scores
            .lock()
            .get(&uri.to_string())
            .copied()
            .unwrap_or(0.5)
    }

    pub fn learned_ranking_score(
        &self,
        query: &str,
        uri: &ContextUri,
        content_type: Option<ContentType>,
        base_relevance: f32,
    ) -> f32 {
        let weights = *self.weights.read();
        let semantic = base_relevance.clamp(0.0, 1.0);
        let uri_feedback = self.learned_score(uri);
        let query_type = content_type
            .and_then(|ty| self.query_type_score(query, ty))
            .unwrap_or(0.5);
        let exploration = self.exploration_bonus(query, uri);
        (semantic * weights.semantic
            + uri_feedback * weights.uri_feedback
            + query_type * weights.query_type
            + exploration * weights.exploration)
            .clamp(0.0, 1.0)
    }

    pub fn rerank_hits(
        &self,
        query: &str,
        mut hits: Vec<crate::RetrievalHit>,
    ) -> Vec<crate::RetrievalHit> {
        for hit in &mut hits {
            hit.relevance =
                self.learned_ranking_score(query, &hit.uri, hit.content_type, hit.relevance);
        }
        hits.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(Ordering::Equal)
        });
        hits
    }

    pub fn missing_patterns(&self) -> Vec<MissingKnowledgeSignal> {
        self.missing_patterns.lock().clone()
    }

    /// 为查询调整 target_dir 偏好。
    pub fn adjust_dir_scores(
        &self,
        query: &str,
        dirs: Vec<(ContextUri, f32)>,
    ) -> Vec<(ContextUri, f32)> {
        let prefs = self.dir_preferences.lock();
        let normalized = normalize_query(query);

        if let Some(adjustments) = prefs.get(&normalized) {
            dirs.into_iter()
                .map(|(uri, base_score)| {
                    let adjusted = adjustments
                        .iter()
                        .find(|(d, _)| uri.to_string().contains(d))
                        .map(|(_, w)| base_score * w)
                        .unwrap_or(base_score);
                    (uri, adjusted.clamp(0.0, 1.0))
                })
                .collect()
        } else {
            dirs
        }
    }

    fn record_missing(&self, query: &str, description: &str) {
        let mut missing = self.missing_patterns.lock();
        let terms = important_terms(description);
        let normalized = normalize_query(query);
        if let Some(existing) = missing
            .iter_mut()
            .find(|item| item.query == normalized && item.description == description)
        {
            existing.weight = (existing.weight + self.learning_rate).min(1.0);
            return;
        }
        missing.push(MissingKnowledgeSignal {
            query: normalized,
            description: description.to_string(),
            expansion_terms: terms,
            weight: 0.5,
        });
        missing.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));
        missing.truncate(128);
    }

    fn update_query_type(&self, query: &str, content_type: ContentType, reward: f32) {
        let mut map = self.query_type_weights.lock();
        let entry = map.entry(query_signature(query)).or_default();
        let weight = entry.entry(content_type).or_insert(0.5);
        *weight =
            (*weight + self.learning_rate * (reward.clamp(0.0, 1.0) - *weight)).clamp(0.0, 1.0);
    }

    fn query_type_score(&self, query: &str, content_type: ContentType) -> Option<f32> {
        self.query_type_weights
            .lock()
            .get(&query_signature(query))
            .and_then(|weights| weights.get(&content_type).copied())
    }

    fn exploration_bonus(&self, query: &str, uri: &ContextUri) -> f32 {
        let uri_text = uri.to_string().to_ascii_lowercase();
        let missing = self.missing_patterns.lock();
        missing
            .iter()
            .filter(|pattern| pattern.query == normalize_query(query))
            .flat_map(|pattern| {
                pattern
                    .expansion_terms
                    .iter()
                    .map(move |term| (term, pattern.weight))
            })
            .filter(|(term, _)| uri_text.contains(term.as_str()))
            .map(|(_, weight)| weight)
            .fold(0.0, f32::max)
    }

    fn rebalance_weights(&self) {
        let missing_pressure = self
            .missing_patterns
            .lock()
            .iter()
            .map(|pattern| pattern.weight)
            .fold(0.0, f32::max)
            .clamp(0.0, 1.0);
        let mut weights = self.weights.write();
        weights.exploration = (0.04 + missing_pressure * 0.08).clamp(0.04, 0.16);
        weights.uri_feedback = (0.26 - weights.exploration * 0.35).clamp(0.18, 0.30);
        weights.query_type = 0.14;
        weights.semantic = (1.0 - weights.uri_feedback - weights.query_type - weights.exploration)
            .clamp(0.45, 0.65);
    }
}

fn signal_reward(signal: &RetrievalLearningSignal) -> f32 {
    (signal.base_relevance.clamp(0.0, 1.0) * 0.35
        + if signal.clicked { 0.20 } else { 0.0 }
        + if signal.adopted { 0.35 } else { 0.0 }
        + (signal.dwell_seconds / 90.0).clamp(0.0, 1.0) * 0.10)
        .clamp(0.0, 1.0)
}

fn reinforce_dir(
    prefs: &mut HashMap<String, Vec<(String, f32)>>,
    query: &str,
    uri: &ContextUri,
    learning_rate: f32,
    reward: f32,
) {
    let normalized = normalize_query(query);
    let dir = uri
        .segments()
        .iter()
        .take(5)
        .cloned()
        .collect::<Vec<_>>()
        .join("/");
    if dir.is_empty() {
        return;
    }
    let list = prefs.entry(normalized).or_default();
    if let Some((_, weight)) = list.iter_mut().find(|(existing, _)| existing == &dir) {
        *weight = (*weight + learning_rate * (reward.clamp(0.0, 1.0) - *weight)).clamp(0.05, 1.4);
    } else {
        list.push((dir, (0.5 + reward * 0.5).clamp(0.05, 1.4)));
    }
    list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    list.truncate(16);
}

fn normalize_query(query: &str) -> String {
    important_terms(query).join(" ")
}

fn query_signature(query: &str) -> String {
    let mut terms = important_terms(query);
    terms.truncate(4);
    terms.join(" ")
}

fn important_terms(text: &str) -> Vec<String> {
    let stop = [
        "the", "and", "for", "with", "that", "this", "what", "when", "where", "how", "why", "一个",
        "这个", "那个", "什么", "如何", "怎么", "以及", "或者",
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    let mut terms = text
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| token.len() >= 3 && !stop.contains(token.as_str()))
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{DirEntry, FindPattern, GrepHit, Result as CoreResult, TreeNode};

    struct NoopFs;
    #[async_trait::async_trait]
    impl FsOps for NoopFs {
        async fn ls(&self, _: &ContextUri) -> CoreResult<Vec<DirEntry>> {
            Ok(vec![])
        }
        async fn find(&self, _: &FindPattern) -> CoreResult<Vec<ContextUri>> {
            Ok(vec![])
        }
        async fn grep(&self, _: &str, _: &ContextUri) -> CoreResult<Vec<GrepHit>> {
            Ok(vec![])
        }
        async fn tree(&self, r: &ContextUri, _: usize) -> CoreResult<TreeNode> {
            Ok(TreeNode {
                uri: r.clone(),
                is_dir: true,
                children: vec![],
            })
        }
        async fn read(&self, _: &ContextUri, _: ContentLevel) -> CoreResult<ContentPayload> {
            Ok(ContentPayload::Text {
                sparse: String::new(),
                dense: String::new(),
                full: String::new(),
            })
        }
    }

    #[test]
    fn prefetcher_predicts_from_associations() {
        let prefetcher = PredictivePrefetcher::new(Arc::new(NoopFs), 3);
        let from = ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap();
        let to = ContextUri::parse("uwu://t/agent/a/memories/tools/t1").unwrap();

        prefetcher.learn_association(&from, &to);
        let preds = prefetcher.predict(&from);
        assert!(!preds.is_empty());
    }

    #[test]
    fn incremental_learner_applies_feedback() {
        let learner = IncrementalRetrievalLearner::new(0.1);
        let uri = ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap();

        learner.apply_feedback(
            "bug fix",
            &[RelevanceFeedback::Relevant {
                uri: uri.clone(),
                score: 0.9,
            }],
        );

        assert!(learner.learned_score(&uri) > 0.5);
    }

    #[test]
    fn incremental_learner_penalizes_irrelevant() {
        let learner = IncrementalRetrievalLearner::new(0.1);
        let uri = ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap();

        // 初始：默认分 + 强化
        learner.apply_feedback(
            "query",
            &[RelevanceFeedback::Relevant {
                uri: uri.clone(),
                score: 0.9,
            }],
        );
        let after_relevant = learner.learned_score(&uri);

        // 然后：衰减
        learner.apply_feedback(
            "query",
            &[RelevanceFeedback::NotRelevant { uri: uri.clone() }],
        );
        let after_penalty = learner.learned_score(&uri);

        assert!(after_penalty < after_relevant);
    }

    #[test]
    fn learner_reranks_hits_with_feedback_and_content_type() {
        let learner = IncrementalRetrievalLearner::new(0.4);
        let good = ContextUri::parse("uwu://t/a/memory/fact/good").unwrap();
        let weak = ContextUri::parse("uwu://t/a/memory/fact/weak").unwrap();
        learner.apply_feedback(
            "cache audit",
            &[RelevanceFeedback::Relevant {
                uri: good.clone(),
                score: 1.0,
            }],
        );
        learner.apply_learning_signals(&[RetrievalLearningSignal {
            uri: good.clone(),
            query: "cache audit".into(),
            content_type: Some(ContentType::Fact),
            base_relevance: 0.8,
            clicked: true,
            adopted: true,
            dwell_seconds: 30.0,
        }]);
        assert!(
            learner.learned_ranking_score("cache audit", &good, Some(ContentType::Fact), 0.6)
                > learner.learned_ranking_score("cache audit", &weak, Some(ContentType::Fact), 0.6)
        );
    }

    #[test]
    fn missing_feedback_records_exploration_terms() {
        let learner = IncrementalRetrievalLearner::new(0.2);
        learner.apply_feedback(
            "cache audit",
            &[RelevanceFeedback::Missing {
                description: "durable invalidation proof".into(),
            }],
        );
        let missing = learner.missing_patterns();
        assert_eq!(missing.len(), 1);
        assert!(missing[0].expansion_terms.contains(&"durable".to_string()));
    }
}
