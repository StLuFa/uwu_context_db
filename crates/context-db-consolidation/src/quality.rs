//! QualityScorer — 七维质量分 + SGD 调权。

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDim { Adoption, Consistency, Recall, Freshness, Downstream, InfoGain, Consensus }

pub struct QualityScorer { weights: HashMap<QualityDim, f32> }

impl QualityScorer {
    pub fn new() -> Self {
        let mut w = HashMap::new();
        w.insert(QualityDim::Adoption, 0.20); w.insert(QualityDim::Consistency, 0.15);
        w.insert(QualityDim::Recall, 0.10); w.insert(QualityDim::Freshness, 0.10);
        w.insert(QualityDim::Downstream, 0.15); w.insert(QualityDim::InfoGain, 0.10);
        w.insert(QualityDim::Consensus, 0.20);
        Self { weights: w }
    }
    pub fn score(&self, entry: &agent_context_db_core::ContextEntry, fb: &QualityFeedback) -> f32 {
        let w = &self.weights;
        let base = entry.metadata.quality_score.unwrap_or(0.5);
        w[&QualityDim::Adoption] * fb.adoption_rate
            + w[&QualityDim::Consistency] * if fb.contradictions == 0 { 1.0 } else { 0.5 }
            + w[&QualityDim::Recall] * fb.recall_rate
            + w[&QualityDim::Freshness] * base
            + w[&QualityDim::Downstream] * if fb.downstream_positive { 1.0 } else { 0.3 }
            + w[&QualityDim::InfoGain] * fb.info_gain.clamp(0.0, 1.0)
            + w[&QualityDim::Consensus] * (fb.corroboration.min(3) as f32 / 3.0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct QualityFeedback {
    pub adopted: bool, pub adoption_rate: f32, pub contradictions: usize,
    pub recall_rate: f32, pub downstream_positive: bool,
    pub info_gain: f32, pub corroboration: usize,
}
