use agent_context_db_core::{
    ContentType, ContextEntry, ContextError, MvccVersion, Result, StateScope,
    sanitize_entry_for_write,
};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, serde::Serialize)]
pub(crate) struct PreparedBatchRow {
    pub ordinal: usize,
    pub uri: String,
    pub tenant_id: String,
    pub l0: String,
    pub l1: Option<String>,
    pub l2: String,
    pub content_type: &'static str,
    pub state_scope: Option<&'static str>,
    pub tags: serde_json::Value,
    pub custom: serde_json::Value,
    pub entry: serde_json::Value,
    pub version: i64,
    pub created_at: String,
    pub updated_at: String,
    pub is_current: bool,
    pub outbox_id: Option<String>,
    pub mutation: Option<serde_json::Value>,
}

/// Expands a chunk into immutable snapshots after the caller has locked and loaded each URI's
/// current version. Duplicate URIs receive consecutive versions in input order; only their last
/// occurrence is marked as the current row.
pub(crate) fn prepare_batch_chunk(
    entries: &[ContextEntry],
    base_versions: &HashMap<String, i64>,
) -> Result<(Vec<PreparedBatchRow>, Vec<MvccVersion>)> {
    let mut next = base_versions.clone();
    let mut last = HashMap::new();
    for (ordinal, entry) in entries.iter().enumerate() {
        last.insert(entry.uri.to_string(), ordinal);
    }
    let mut rows = Vec::with_capacity(entries.len());
    let mut versions = Vec::with_capacity(entries.len());
    for (ordinal, input) in entries.iter().enumerate() {
        input.validate_tenant_binding()?;
        let mut entry = sanitize_entry_for_write(input)?;
        let uri = entry.uri.to_string();
        let version = next
            .get(&uri)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| {
                ContextError::VersionConflict(format!("MVCC version overflow for {uri}"))
            })?;
        next.insert(uri.clone(), version);
        let public_version = u64::try_from(version)
            .map(MvccVersion)
            .map_err(|_| ContextError::Storage(format!("invalid MVCC version {version}")))?;
        let projection = entry.payload.index_projection();
        let now = chrono::Utc::now();
        entry.updated_at = now;
        entry.mvcc_version = public_version;
        let mutation = crate::outbox::upsert_mutation(&entry, public_version)?;
        rows.push(PreparedBatchRow {
            ordinal,
            uri: uri.clone(),
            tenant_id: entry.tenant.0.to_string(),
            l0: projection.l0,
            l1: projection.l1,
            l2: projection.l2,
            content_type: entry
                .metadata
                .content_type
                .unwrap_or(ContentType::Evidence)
                .as_path_segment(),
            state_scope: entry.metadata.state_scope.map(|scope| match scope {
                StateScope::Short => "short",
                StateScope::Mid => "mid",
                StateScope::Long => "long",
            }),
            tags: serde_json::to_value(&entry.metadata.tags)?,
            custom: entry.metadata.custom.clone(),
            entry: serde_json::to_value(&entry)?,
            version,
            created_at: entry.created_at.to_rfc3339(),
            updated_at: now.to_rfc3339(),
            is_current: last.get(&uri) == Some(&ordinal),
            outbox_id: mutation.as_ref().map(|_| uuid::Uuid::now_v7().to_string()),
            mutation: mutation.map(serde_json::to_value).transpose()?,
        });
        versions.push(public_version);
    }
    Ok((rows, versions))
}
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchWriteConfig {
    pub max_rows_per_chunk: usize,
    pub max_bytes_per_chunk: usize,
}

impl BatchWriteConfig {
    pub fn validate(self) -> Result<Self> {
        if self.max_rows_per_chunk == 0 || self.max_bytes_per_chunk == 0 {
            return Err(ContextError::Storage(
                "batch write chunk limits must be greater than zero".into(),
            ));
        }
        Ok(self)
    }
}

impl BatchWriteConfig {
    pub(crate) fn chunks(
        self,
        entries: &[agent_context_db_core::ContextEntry],
    ) -> Result<Vec<&[agent_context_db_core::ContextEntry]>> {
        self.validate()?;
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < entries.len() {
            let mut end = start;
            let mut bytes = 0usize;
            while end < entries.len() && end - start < self.max_rows_per_chunk {
                let size = serde_json::to_vec(&entries[end])?.len();
                if end > start && bytes.saturating_add(size) > self.max_bytes_per_chunk {
                    break;
                }
                bytes = bytes.saturating_add(size);
                end += 1;
                if bytes >= self.max_bytes_per_chunk {
                    break;
                }
            }
            chunks.push(&entries[start..end]);
            start = end;
        }
        Ok(chunks)
    }
}

impl Default for BatchWriteConfig {
    fn default() -> Self {
        Self {
            max_rows_per_chunk: 256,
            max_bytes_per_chunk: 4 * 1024 * 1024,
        }
    }
}

/// Validated limits and convergence parameters for bounded graph centrality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphCentralityConfig {
    max_nodes: usize,
    max_hops: usize,
    max_iterations: usize,
    epsilon: f32,
    damping: f32,
    incremental_threshold: usize,
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
            incremental_threshold: max_nodes / 2,
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
    pub fn incremental_threshold(self) -> usize {
        self.incremental_threshold
    }
    pub fn with_incremental_threshold(mut self, threshold: usize) -> Result<Self> {
        self.incremental_threshold = NonZeroUsize::new(threshold)
            .ok_or_else(|| ContextError::Storage("incremental threshold must be positive".into()))?
            .get();
        Ok(self)
    }
    pub(crate) fn cache_key(self) -> String {
        format!(
            "pagerank:v1:max_nodes={}:max_hops={}:max_iterations={}:epsilon={:08x}:damping={:08x}:incremental_threshold={}",
            self.max_nodes,
            self.max_hops,
            self.max_iterations,
            self.epsilon.to_bits(),
            self.damping.to_bits(),
            self.incremental_threshold
        )
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
            incremental_threshold: 128,
        }
    }
}

pub(crate) fn pagerank_scores(
    nodes: &[String],
    edges: &[(String, String)],
    config: GraphCentralityConfig,
) -> HashMap<String, f32> {
    if nodes.is_empty() {
        return HashMap::new();
    }
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
    ranks
        .into_iter()
        .map(|(uri, score)| (uri.to_owned(), (score * n).clamp(0.0, 1.0)))
        .collect()
}

/// Recomputes PageRank from the previous snapshot after a small set of graph mutations.
///
/// The dirty region is used to decide whether incremental work is safe. PageRank has global
/// influence, so the convergent solve still includes every node; warm-starting from the previous
/// snapshot avoids discarding the stable portion while preserving full-recompute semantics.
pub(crate) fn incremental_pagerank_scores(
    nodes: &[String],
    edges: &[(String, String)],
    previous: &HashMap<String, f32>,
    dirty_seeds: &HashSet<String>,
    config: GraphCentralityConfig,
) -> Option<HashMap<String, f32>> {
    if previous.is_empty() || dirty_seeds.is_empty() {
        return None;
    }
    let node_set = nodes.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut adjacency = HashMap::<&str, Vec<&str>>::new();
    for (from, to) in edges {
        adjacency.entry(from).or_default().push(to);
        adjacency.entry(to).or_default().push(from);
    }
    let mut dirty = HashSet::new();
    let mut queue = dirty_seeds
        .iter()
        .filter(|seed| node_set.contains(seed.as_str()))
        .map(|seed| (seed.as_str(), 0usize))
        .collect::<VecDeque<_>>();
    while let Some((node, depth)) = queue.pop_front() {
        if !dirty.insert(node) {
            continue;
        }
        if dirty.len() > config.incremental_threshold() {
            return None;
        }
        if depth < config.max_hops() {
            for neighbor in adjacency.get(node).into_iter().flatten() {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }

    // Stored scores are normalized by node count. Convert them back before warm-starting.
    let n = nodes.len() as f32;
    let mut ranks = nodes
        .iter()
        .map(|node| {
            (
                node.as_str(),
                previous.get(node).copied().unwrap_or(1.0) / n,
            )
        })
        .collect::<HashMap<_, _>>();
    let total = ranks.values().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    for rank in ranks.values_mut() {
        *rank /= total;
    }
    solve_pagerank(nodes, edges, ranks, config)
}

fn solve_pagerank<'a>(
    nodes: &'a [String],
    edges: &'a [(String, String)],
    mut ranks: HashMap<&'a str, f32>,
    config: GraphCentralityConfig,
) -> Option<HashMap<String, f32>> {
    if nodes.is_empty() {
        return Some(HashMap::new());
    }
    let n = nodes.len() as f32;
    let mut outgoing = HashMap::<&str, Vec<&str>>::new();
    for (from, to) in edges {
        outgoing.entry(from).or_default().push(to);
    }
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
    Some(
        ranks
            .into_iter()
            .map(|(uri, score)| (uri.to_owned(), (score * n).clamp(0.0, 1.0)))
            .collect(),
    )
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
