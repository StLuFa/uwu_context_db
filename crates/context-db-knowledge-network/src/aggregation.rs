use agent_context_db_marketplace_types::FederatedDiscoveryHit;

#[derive(Debug, Clone)]
pub struct StreamingTopKAggregator {
    pub final_top_k: usize,
    pub min_confidence: f32,
    pub early_return: bool,
}

impl Default for StreamingTopKAggregator {
    fn default() -> Self {
        Self {
            final_top_k: 20,
            min_confidence: 0.65,
            early_return: true,
        }
    }
}

impl StreamingTopKAggregator {
    pub fn merge(
        &self,
        mut hits: Vec<FederatedDiscoveryHit>,
        final_top_k: usize,
    ) -> Vec<FederatedDiscoveryHit> {
        for hit in &mut hits {
            hit.final_score = (hit.noisy_score * 0.35
                + hit.publication.quality_score * 0.20
                + hit.reputation_bonus * 0.15
                + hit.corroboration_bonus * 0.15
                + hit.lineage_independence * 0.10
                + if hit.publication.license.derivative_allowed {
                    0.05
                } else {
                    0.0
                })
            .clamp(0.0, 1.0);
        }
        hits.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.dedup_by_key(|hit| hit.publication.id);
        hits.truncate(final_top_k.min(self.final_top_k.max(final_top_k)));
        hits
    }
}
