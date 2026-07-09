use crate::types::{FederatedQueryIntent, FederationReturnMode, MeshDiscoveryOpts};
use agent_context_db_marketplace::DiscoveryQuery;

#[derive(Debug, Clone)]
pub struct QueryIntentClassifier {
    pub privacy_domain_threshold: usize,
    pub high_recall_domain_threshold: usize,
    pub high_precision_quality_threshold: f32,
}

impl Default for QueryIntentClassifier {
    fn default() -> Self {
        Self {
            privacy_domain_threshold: 1,
            high_recall_domain_threshold: 3,
            high_precision_quality_threshold: 0.82,
        }
    }
}

impl QueryIntentClassifier {
    pub fn classify(
        &self,
        query: &DiscoveryQuery,
        requested: FederatedQueryIntent,
    ) -> FederatedQueryIntent {
        if matches!(
            requested,
            FederatedQueryIntent::PrivacyCritical
                | FederatedQueryIntent::TrainingCandidate
                | FederatedQueryIntent::CorroborationCheck
                | FederatedQueryIntent::CrossDomainSynthesis
        ) {
            return requested;
        }

        if query.license_compatible
            && query.domains.len() <= self.privacy_domain_threshold
            && query.min_quality >= 0.9
        {
            return FederatedQueryIntent::PrivacyCritical;
        }

        if query.domains.len() >= self.high_recall_domain_threshold {
            return FederatedQueryIntent::CrossDomainSynthesis;
        }

        if query.min_quality >= self.high_precision_quality_threshold {
            return FederatedQueryIntent::HighPrecision;
        }

        requested
    }

    pub fn tune_opts(
        &self,
        query: &DiscoveryQuery,
        mut opts: MeshDiscoveryOpts,
    ) -> MeshDiscoveryOpts {
        opts.intent = self.classify(query, opts.intent);
        match opts.intent {
            FederatedQueryIntent::FastLookup => {
                opts.return_mode = FederationReturnMode::FastPartial;
                opts.probe_peers = opts.probe_peers.max(3);
                opts.fetch_peers = opts.fetch_peers.max(2);
            }
            FederatedQueryIntent::HighPrecision | FederatedQueryIntent::PrivacyCritical => {
                opts.return_mode = FederationReturnMode::Balanced;
                opts.probe_peers = opts.probe_peers.max(5);
                opts.fetch_peers = opts.fetch_peers.max(3);
            }
            FederatedQueryIntent::HighRecall | FederatedQueryIntent::CrossDomainSynthesis => {
                opts.return_mode = FederationReturnMode::Exhaustive;
                opts.max_peers = opts.max_peers.max(24);
                opts.probe_peers = opts.probe_peers.max(12);
                opts.fetch_peers = opts.fetch_peers.max(8);
            }
            FederatedQueryIntent::TrainingCandidate | FederatedQueryIntent::CorroborationCheck => {
                opts.return_mode = FederationReturnMode::Balanced;
                opts.max_peers = opts.max_peers.max(16);
                opts.probe_peers = opts.probe_peers.max(8);
                opts.fetch_peers = opts.fetch_peers.max(5);
            }
        }
        opts
    }
}
