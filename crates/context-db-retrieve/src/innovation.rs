//! 创新功能扩展（F16 预测性预加载 + F28 增量检索学习）。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, FsOps};
use std::collections::HashMap;
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
        let entry = history.entry(uri.to_string().clone()).or_insert_with(|| AccessRecord {
            uri: uri.clone(),
            timestamp: Instant::now(),
            access_count: 0,
        });
        entry.access_count += 1;
    }

    /// 记录检索轨迹中的关联（URI A → URI B）。
    pub fn learn_association(&self, from: &ContextUri, to: &ContextUri) {
        let mut assoc = self.associations.lock();
        let list = assoc.entry(from.0.clone()).or_default();
        if let Some(entry) = list.iter_mut().find(|(u, _)| u == &to.0) {
            entry.1 += 0.1; // 强化
        } else {
            list.push((to.0.clone(), 0.3));
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
        if let Some(associations) = self.associations.lock().get(&current.0) {
            for (uri, prob) in associations.iter().take(self.prefetch_size) {
                predictions.push(PrefetchPrediction {
                    uri: ContextUri(uri.clone()),
                    probability: *prob,
                    pattern: AccessPattern::Sequential,
                    prefetch_level: if *prob > 0.7 { ContentLevel::L1 } else { ContentLevel::L0 },
                });
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
    pub async fn prefetch(&self, predictions: &[PrefetchPrediction]) -> Vec<(ContextUri, ContentPayload)> {
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
#[derive(Debug, Clone)]
pub enum RelevanceFeedback {
    /// 命中相关
    Relevant { uri: ContextUri, score: f32 },
    /// 命中不相关
    NotRelevant { uri: ContextUri },
    /// 缺少的信息描述
    Missing { description: String },
}

/// 增量学习器 —— 基于用户反馈持续调整检索权重。
pub struct IncrementalRetrievalLearner {
    /// URI → 累计反馈分数
    uri_scores: parking_lot::Mutex<HashMap<String, f32>>,
    /// 查询模式 → 最佳 target_dir 偏好
    dir_preferences: parking_lot::Mutex<HashMap<String, Vec<(String, f32)>>>,
    /// 学习率
    learning_rate: f32,
}

impl IncrementalRetrievalLearner {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            uri_scores: parking_lot::Mutex::new(HashMap::new()),
            dir_preferences: parking_lot::Mutex::new(HashMap::new()),
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
                    *entry = *entry + self.learning_rate * (score - *entry);
                }
                RelevanceFeedback::NotRelevant { uri } => {
                    let entry = scores.entry(uri.to_string().clone()).or_insert(0.5);
                    *entry = (*entry * 0.9).max(0.05); // 衰减
                }
                RelevanceFeedback::Missing { description: _ } => {
                    // 记录缺失模式，影响未来检索的 target_dir 选择
                    let normalized = query.to_lowercase();
                    let dirs = prefs.entry(normalized).or_default();
                    // 标记需要扩大搜索范围
                    if let Some(last) = dirs.last_mut() {
                        last.1 = (last.1 * 0.8).max(0.1);
                    }
                }
            }
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

    /// 为查询调整 target_dir 偏好。
    pub fn adjust_dir_scores(
        &self,
        query: &str,
        dirs: Vec<(ContextUri, f32)>,
    ) -> Vec<(ContextUri, f32)> {
        let prefs = self.dir_preferences.lock();
        let normalized = query.to_lowercase();

        if let Some(adjustments) = prefs.get(&normalized) {
            dirs.into_iter()
                .map(|(uri, base_score)| {
                    let adjusted = adjustments
                        .iter()
                        .find(|(d, _)| uri.to_string().contains(d))
                        .map(|(_, w)| base_score * w)
                        .unwrap_or(base_score);
                    (uri, adjusted)
                })
                .collect()
        } else {
            dirs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{DirEntry, FindPattern, GrepHit, Result as CoreResult, TreeNode};

    struct NoopFs;
    #[async_trait::async_trait]
    impl FsOps for NoopFs {
        async fn ls(&self, _: &ContextUri) -> CoreResult<Vec<DirEntry>> { Ok(vec![]) }
        async fn find(&self, _: &FindPattern) -> CoreResult<Vec<ContextUri>> { Ok(vec![]) }
        async fn grep(&self, _: &str, _: &ContextUri) -> CoreResult<Vec<GrepHit>> { Ok(vec![]) }
        async fn tree(&self, r: &ContextUri, _: usize) -> CoreResult<TreeNode> { Ok(TreeNode { uri: r.clone(), is_dir: true, children: vec![] }) }
        async fn read(&self, _: &ContextUri, _: ContentLevel) -> CoreResult<ContentPayload> { Ok(ContentPayload::Abstract(String::new())) }
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

        learner.apply_feedback("bug fix", &[
            RelevanceFeedback::Relevant { uri: uri.clone(), score: 0.9 },
        ]);

        assert!(learner.learned_score(&uri) > 0.5);
    }

    #[test]
    fn incremental_learner_penalizes_irrelevant() {
        let learner = IncrementalRetrievalLearner::new(0.1);
        let uri = ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap();

        // 初始：默认分 + 强化
        learner.apply_feedback("query", &[
            RelevanceFeedback::Relevant { uri: uri.clone(), score: 0.9 },
        ]);
        let after_relevant = learner.learned_score(&uri);

        // 然后：衰减
        learner.apply_feedback("query", &[
            RelevanceFeedback::NotRelevant { uri: uri.clone() },
        ]);
        let after_penalty = learner.learned_score(&uri);

        assert!(after_penalty < after_relevant);
    }
}
