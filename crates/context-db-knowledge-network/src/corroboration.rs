use agent_context_db_marketplace_types::{AgentId, FederatedDiscoveryHit, MarketId};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default)]
pub struct CorroborationRecord {
    pub publication: MarketId,
    pub sources: HashSet<AgentId>,
    pub independent_domains: HashSet<String>,
    pub contradictions: u32,
}

#[derive(Debug, Default)]
pub struct CorroborationGraph {
    records: RwLock<HashMap<MarketId, CorroborationRecord>>,
}

impl CorroborationGraph {
    pub fn observe_hits(&self, hits: &[FederatedDiscoveryHit]) {
        let mut records = self.records.write();
        for hit in hits {
            let id = hit.publication.id;
            let record = records.entry(id).or_insert_with(|| CorroborationRecord {
                publication: id,
                ..Default::default()
            });
            if let Some(peer) = &hit.source_peer {
                record.sources.insert(peer.clone());
            }
            record
                .independent_domains
                .insert(hit.publication.domain.clone());
        }
    }

    pub fn record_contradiction(&self, publication: MarketId) {
        let mut records = self.records.write();
        let record = records
            .entry(publication)
            .or_insert_with(|| CorroborationRecord {
                publication,
                ..Default::default()
            });
        record.contradictions = record.contradictions.saturating_add(1);
    }

    pub fn enrich_scores(&self, hits: &mut [FederatedDiscoveryHit]) {
        let records = self.records.read();
        for hit in hits {
            if let Some(record) = records.get(&hit.publication.id) {
                let source_bonus = (record.sources.len() as f32 / 4.0).clamp(0.0, 0.2);
                let domain_bonus = (record.independent_domains.len() as f32 / 3.0).clamp(0.0, 0.2);
                let penalty = (record.contradictions as f32 * 0.08).clamp(0.0, 0.3);
                hit.corroboration_bonus = (hit.corroboration_bonus + source_bonus + domain_bonus
                    - penalty)
                    .clamp(0.0, 1.0);
                hit.lineage_independence =
                    (hit.lineage_independence + domain_bonus).clamp(0.0, 1.0);
            }
        }
    }
}
