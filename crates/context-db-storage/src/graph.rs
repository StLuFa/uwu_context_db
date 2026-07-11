use agent_context_db_core::{ContextError, Result};
use std::collections::HashMap;
use std::num::NonZeroUsize;

/// Validated limits and convergence parameters for bounded graph centrality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphCentralityConfig {
    max_nodes: usize,
    max_hops: usize,
    max_iterations: usize,
    epsilon: f32,
    damping: f32,
}

impl GraphCentralityConfig {
    pub fn new(
        max_nodes: usize,
        max_hops: usize,
        max_iterations: usize,
        epsilon: f32,
        damping: f32,
    ) -> Result<Self> {
        let nonzero = |name, value| {
            NonZeroUsize::new(value).ok_or_else(|| {
                ContextError::Storage(format!(
                    "invalid graph centrality configuration: {name} must be greater than zero"
                ))
            })
        };
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ContextError::Storage(
                "invalid graph centrality configuration: epsilon must be finite and positive"
                    .into(),
            ));
        }
        if !damping.is_finite() || damping <= 0.0 || damping >= 1.0 {
            return Err(ContextError::Storage(
                "invalid graph centrality configuration: damping must be finite and between zero and one".into(),
            ));
        }
        Ok(Self {
            max_nodes: nonzero("max_nodes", max_nodes)?.get(),
            max_hops: nonzero("max_hops", max_hops)?.get(),
            max_iterations: nonzero("max_iterations", max_iterations)?.get(),
            epsilon,
            damping,
        })
    }

    pub fn max_nodes(self) -> usize {
        self.max_nodes
    }
    pub fn max_hops(self) -> usize {
        self.max_hops
    }
    pub fn max_iterations(self) -> usize {
        self.max_iterations
    }
    pub fn epsilon(self) -> f32 {
        self.epsilon
    }
    pub fn damping(self) -> f32 {
        self.damping
    }
}

impl Default for GraphCentralityConfig {
    fn default() -> Self {
        Self {
            max_nodes: 256,
            max_hops: 3,
            max_iterations: 32,
            epsilon: 1e-5,
            damping: 0.85,
        }
    }
}

pub(crate) fn pagerank_score(
    target: &str,
    nodes: &[String],
    edges: &[(String, String)],
    config: GraphCentralityConfig,
) -> f32 {
    let n = nodes.len() as f32;
    let mut outgoing = HashMap::<&str, Vec<&str>>::new();
    for (from, to) in edges {
        outgoing.entry(from).or_default().push(to);
    }
    let mut ranks = nodes
        .iter()
        .map(|node| (node.as_str(), 1.0 / n))
        .collect::<HashMap<_, _>>();
    for _ in 0..config.max_iterations() {
        let dangling = nodes
            .iter()
            .filter(|node| outgoing.get(node.as_str()).is_none_or(Vec::is_empty))
            .map(|node| ranks.get(node.as_str()).copied().unwrap_or(0.0))
            .sum::<f32>()
            / n;
        let mut next = nodes
            .iter()
            .map(|node| {
                (
                    node.as_str(),
                    (1.0 - config.damping()) / n + config.damping() * dangling,
                )
            })
            .collect::<HashMap<_, _>>();
        for (from, targets) in &outgoing {
            let contribution =
                config.damping() * ranks.get(from).copied().unwrap_or(0.0) / targets.len() as f32;
            for target in targets {
                if let Some(value) = next.get_mut(target) {
                    *value += contribution;
                }
            }
        }
        let delta = nodes
            .iter()
            .map(|node| {
                (next.get(node.as_str()).copied().unwrap_or(0.0)
                    - ranks.get(node.as_str()).copied().unwrap_or(0.0))
                .abs()
            })
            .sum::<f32>();
        ranks = next;
        if delta < config.epsilon() {
            break;
        }
    }
    (ranks.get(target).copied().unwrap_or(0.0) * n).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_configuration() {
        for values in [
            (0, 1, 1, 1e-5, 0.85),
            (1, 0, 1, 1e-5, 0.85),
            (1, 1, 0, 1e-5, 0.85),
            (1, 1, 1, 0.0, 0.85),
            (1, 1, 1, f32::NAN, 0.85),
            (1, 1, 1, 1e-5, 0.0),
            (1, 1, 1, 1e-5, 1.0),
            (1, 1, 1, 1e-5, f32::INFINITY),
        ] {
            assert!(
                GraphCentralityConfig::new(values.0, values.1, values.2, values.3, values.4)
                    .is_err()
            );
        }
    }
}
