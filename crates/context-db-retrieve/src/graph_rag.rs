//! GraphRAG — 社区摘要索引 + 图增强检索。

use crate::{
    GraphRagConfig, RetrievalHit, RetrievalResult, RetrievalTrace, RetrieveConfigError,
    RetrieveContext, TraceStep,
};
use agent_context_db_core::{
    ContentLevel, ContentPayload, ContextUri, FsOps, GraphRelation, GraphStore, LlmClient, LlmOpts,
    Result, count_tokens,
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::SystemTime;

#[derive(Debug, Clone)]
pub struct GraphRagRequest {
    pub query: String,
    pub seed_uris: Vec<ContextUri>,
    pub relations: Vec<GraphRelation>,
    pub max_hops: usize,
    pub max_communities: usize,
    pub max_nodes_per_community: usize,
}

impl GraphRagRequest {
    pub fn new(
        query: impl Into<String>,
        seed_uris: Vec<ContextUri>,
        config: GraphRagConfig,
    ) -> std::result::Result<Self, RetrieveConfigError> {
        let config = config.validate()?;
        Ok(Self {
            query: query.into(),
            seed_uris,
            relations: default_graph_rag_relations(),
            max_hops: config.request_hops,
            max_communities: config.request_communities,
            max_nodes_per_community: config.request_nodes,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GraphRagIndexConfig {
    pub seed_uris: Vec<ContextUri>,
    pub relations: Vec<GraphRelation>,
    pub max_hops: usize,
    pub max_summary_nodes: usize,
}

impl From<&GraphRagRequest> for GraphRagIndexConfig {
    fn from(request: &GraphRagRequest) -> Self {
        Self {
            seed_uris: request.seed_uris.clone(),
            relations: request.relations.clone(),
            max_hops: request.max_hops,
            max_summary_nodes: request.max_nodes_per_community.max(8),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphRagCommunity {
    pub id: String,
    pub nodes: Vec<ContextUri>,
    pub summary: String,
    pub score: f32,
    pub parent: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GraphRagIndexStats {
    pub communities: usize,
    pub nodes: usize,
    pub edges: usize,
    pub built_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct GraphRagIndex {
    communities: Vec<GraphRagCommunity>,
    summary_terms: Vec<HashSet<String>>,
    node_to_community: HashMap<ContextUri, Vec<usize>>,
    node_centrality: HashMap<ContextUri, f32>,
    adjacency: HashMap<ContextUri, HashSet<ContextUri>>,
    stats: GraphRagIndexStats,
    config: GraphRagConfig,
}

impl GraphRagIndex {
    pub fn communities(&self) -> &[GraphRagCommunity] {
        &self.communities
    }

    pub fn stats(&self) -> &GraphRagIndexStats {
        &self.stats
    }

    pub fn community_for_node(&self, uri: &ContextUri) -> Option<&GraphRagCommunity> {
        self.node_to_community
            .get(uri)
            .and_then(|ids| ids.first())
            .and_then(|idx| self.communities.get(*idx))
    }

    pub async fn retrieve(
        &self,
        fs: Arc<dyn FsOps>,
        request: &GraphRagRequest,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        let selected = self.rank_communities(&request.query, request.max_communities);
        let mut hits = Vec::new();
        for (idx, score) in selected {
            let Some(community) = self.communities.get(idx) else {
                continue;
            };
            hits.push(community_summary_hit(community, score)?);
            for node in self
                .rank_nodes_in_community(community, &request.query)
                .into_iter()
                .take(request.max_nodes_per_community)
            {
                let content = fs
                    .read(&node, ctx.prefer_level)
                    .await
                    .unwrap_or_else(|_| empty_text_payload());
                hits.push(RetrievalHit {
                    uri: node.clone(),
                    level: ctx.prefer_level,
                    content,
                    relevance: (score * self.config.hit_community_weight
                        + self.node_score(&node) * self.config.hit_node_weight)
                        .clamp(self.config.min_hit_relevance, 1.0),
                    parent_chain: vec![ContextUri::parse(format!(
                        "uwu://graph-rag/community/{}",
                        community.id
                    ))?],
                    content_type: None,
                    metadata: Default::default(),
                    created_at: None,
                    updated_at: None,
                });
            }
        }
        let tokens_used = hits
            .iter()
            .map(|hit| count_tokens(hit.content.sparse_text()))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum();
        Ok(RetrievalResult {
            hits,
            trace: RetrievalTrace {
                steps: vec![TraceStep::PlanOptimized {
                    logical: format!(
                        "GraphRAGIndex(query={}, indexed_communities={}, indexed_nodes={})",
                        request.query, self.stats.communities, self.stats.nodes
                    ),
                    physical: "SummaryIndexLookup -> CommunityDrillDown".into(),
                }],
            },
            tokens_used,
        })
    }

    fn rank_communities(&self, query: &str, max_communities: usize) -> Vec<(usize, f32)> {
        let query_terms = terms(query);
        let mut scored = self
            .communities
            .iter()
            .enumerate()
            .map(|(idx, community)| {
                let summary_score = term_overlap(&query_terms, &self.summary_terms[idx]);
                let node_score = community.score;
                let global_boost = if community.parent.is_none() {
                    self.config.global_boost
                } else {
                    0.0
                };
                (
                    idx,
                    (summary_score * self.config.community_summary_weight
                        + node_score * self.config.community_node_weight
                        + global_boost)
                        .clamp(0.0, 1.0),
                )
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(max_communities.max(1));
        scored
    }

    fn rank_nodes_in_community(
        &self,
        community: &GraphRagCommunity,
        query: &str,
    ) -> Vec<ContextUri> {
        let query_terms = terms(query);
        let mut nodes = community.nodes.clone();
        nodes.sort_by(|left, right| {
            let left_score = self.node_score(left) + self.node_lexical_score(left, &query_terms);
            let right_score = self.node_score(right) + self.node_lexical_score(right, &query_terms);
            right_score
                .partial_cmp(&left_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.as_str().cmp(right.as_str()))
        });
        nodes
    }

    fn node_score(&self, uri: &ContextUri) -> f32 {
        let centrality = self
            .node_centrality
            .get(uri)
            .copied()
            .unwrap_or(self.config.default_centrality);
        let degree = self.adjacency.get(uri).map(|n| n.len()).unwrap_or(0) as f32;
        let degree_weight = 1.0 - self.config.retrieval_centrality_weight;
        (centrality * self.config.retrieval_centrality_weight
            + (degree / self.config.degree_saturation).min(1.0) * degree_weight)
            .clamp(0.0, 1.0)
    }

    fn node_lexical_score(&self, uri: &ContextUri, query_terms: &HashSet<String>) -> f32 {
        let uri_terms = terms(uri.as_str());
        term_overlap(query_terms, &uri_terms) * self.config.lexical_weight
    }
}

pub struct GraphRagIndexer {
    fs: Arc<dyn FsOps>,
    graph: Arc<dyn GraphStore>,
    llm: Option<Arc<dyn LlmClient>>,
    config: GraphRagConfig,
}

impl GraphRagIndexer {
    pub fn new(
        fs: Arc<dyn FsOps>,
        graph: Arc<dyn GraphStore>,
        config: GraphRagConfig,
    ) -> std::result::Result<Self, RetrieveConfigError> {
        Ok(Self {
            fs,
            graph,
            llm: None,
            config: config.validate()?,
        })
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn build_index(&self, config: &GraphRagIndexConfig) -> Result<GraphRagIndex> {
        let expanded = self.expand_graph(config).await?;
        let mut communities = detect_communities(&expanded.adjacency);
        for community in &mut communities {
            community.nodes.sort_by(|left, right| {
                let left_score =
                    node_rank(left, &expanded.adjacency, &expanded.centrality, self.config);
                let right_score = node_rank(
                    right,
                    &expanded.adjacency,
                    &expanded.centrality,
                    self.config,
                );
                right_score
                    .partial_cmp(&left_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.as_str().cmp(right.as_str()))
            });
            community.score = average_node_rank(
                &community.nodes,
                &expanded.adjacency,
                &expanded.centrality,
                self.config,
            );
            community.summary = self
                .summarize_community(&community.nodes, config.max_summary_nodes)
                .await;
        }
        communities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        let communities = build_hierarchy(communities);
        let mut node_to_community: HashMap<ContextUri, Vec<usize>> = HashMap::new();
        let summary_terms = communities
            .iter()
            .enumerate()
            .map(|(idx, community)| {
                for node in &community.nodes {
                    node_to_community.entry(node.clone()).or_default().push(idx);
                }
                terms(&community.summary)
            })
            .collect::<Vec<_>>();
        let stats = GraphRagIndexStats {
            communities: communities.len(),
            nodes: expanded.adjacency.len(),
            edges: expanded
                .adjacency
                .values()
                .map(|neighbors| neighbors.len())
                .sum::<usize>()
                / 2,
            built_at: SystemTime::now(),
        };
        Ok(GraphRagIndex {
            communities,
            summary_terms,
            node_to_community,
            node_centrality: expanded.centrality,
            adjacency: expanded.adjacency,
            stats,
            config: self.config,
        })
    }

    async fn expand_graph(&self, config: &GraphRagIndexConfig) -> Result<ExpandedGraph> {
        let mut adjacency: HashMap<ContextUri, HashSet<ContextUri>> = HashMap::new();
        let mut centrality = HashMap::new();
        let mut neighbor_cache: HashMap<(ContextUri, GraphRelation), Vec<ContextUri>> =
            HashMap::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        for seed in &config.seed_uris {
            queue.push_back((seed.clone(), 0usize));
            visited.insert(seed.clone());
            adjacency.entry(seed.clone()).or_default();
        }
        while let Some((uri, depth)) = queue.pop_front() {
            centrality.entry(uri.clone()).or_insert(
                self.graph
                    .centrality(&uri)
                    .await
                    .unwrap_or(self.config.default_centrality)
                    .clamp(0.0, 1.0),
            );
            if depth >= config.max_hops {
                continue;
            }
            for relation in &config.relations {
                let key = (uri.clone(), *relation);
                let neighbors = match neighbor_cache.get(&key) {
                    Some(cached) => cached.clone(),
                    None => {
                        let loaded = self.graph.outgoing_neighbors(&uri, Some(*relation)).await?;
                        neighbor_cache.insert(key, loaded.clone());
                        loaded
                    }
                };
                for neighbor in neighbors {
                    adjacency
                        .entry(uri.clone())
                        .or_default()
                        .insert(neighbor.clone());
                    adjacency
                        .entry(neighbor.clone())
                        .or_default()
                        .insert(uri.clone());
                    centrality.entry(neighbor.clone()).or_insert(
                        self.graph
                            .centrality(&neighbor)
                            .await
                            .unwrap_or(self.config.default_centrality)
                            .clamp(0.0, 1.0),
                    );
                    if visited.insert(neighbor.clone()) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        Ok(ExpandedGraph {
            adjacency,
            centrality,
        })
    }

    async fn summarize_community(&self, nodes: &[ContextUri], max_summary_nodes: usize) -> String {
        let evidence = self.community_evidence(nodes, max_summary_nodes).await;
        if let Some(llm) = &self.llm {
            let prompt = format!(
                "You are building a persistent GraphRAG community index.\n\
                 Community evidence:\n{evidence}\n\n\
                 Produce a compact hierarchical summary that captures:\n\
                 1. the community theme,\n\
                 2. important architecture/knowledge transitions,\n\
                 3. representative nodes for later drill-down."
            );
            if let Ok(summary) = llm
                .complete(
                    &prompt,
                    &LlmOpts {
                        max_tokens: Some(512),
                        temperature: Some(0.1),
                        ..Default::default()
                    },
                )
                .await
                && !summary.trim().is_empty()
            {
                return summary;
            }
        }
        if evidence.is_empty() {
            format!("Community with {} related knowledge nodes.", nodes.len())
        } else {
            format!(
                "Community theme from {} nodes: {}",
                nodes.len(),
                truncate(&evidence.replace('\n', " "), self.config.summary_chars)
            )
        }
    }

    async fn community_evidence(&self, nodes: &[ContextUri], max_summary_nodes: usize) -> String {
        let mut evidence = Vec::new();
        for uri in nodes.iter().take(max_summary_nodes) {
            if let Ok(payload) = self.fs.read(uri, ContentLevel::L0).await {
                let text = payload.sparse_text();
                if !text.is_empty() {
                    evidence.push(format!(
                        "- {}: {}",
                        uri,
                        truncate(text, self.config.evidence_chars)
                    ));
                }
            }
        }
        evidence.join("\n")
    }
}

pub struct GraphRagEngine {
    fs: Arc<dyn FsOps>,
    graph: Arc<dyn GraphStore>,
    llm: Option<Arc<dyn LlmClient>>,
    config: GraphRagConfig,
}

impl GraphRagEngine {
    pub fn new(
        fs: Arc<dyn FsOps>,
        graph: Arc<dyn GraphStore>,
        config: GraphRagConfig,
    ) -> std::result::Result<Self, RetrieveConfigError> {
        Ok(Self {
            fs,
            graph,
            llm: None,
            config: config.validate()?,
        })
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn build_index(&self, config: &GraphRagIndexConfig) -> Result<GraphRagIndex> {
        let mut indexer = GraphRagIndexer {
            fs: self.fs.clone(),
            graph: self.graph.clone(),
            llm: None,
            config: self.config,
        };
        if let Some(llm) = &self.llm {
            indexer = indexer.with_llm(llm.clone());
        }
        indexer.build_index(config).await
    }

    pub async fn retrieve(
        &self,
        request: &GraphRagRequest,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        let index = self
            .build_index(&GraphRagIndexConfig::from(request))
            .await?;
        index.retrieve(self.fs.clone(), request, ctx).await
    }
}

#[derive(Debug)]
struct ExpandedGraph {
    adjacency: HashMap<ContextUri, HashSet<ContextUri>>,
    centrality: HashMap<ContextUri, f32>,
}

fn detect_communities(graph: &HashMap<ContextUri, HashSet<ContextUri>>) -> Vec<GraphRagCommunity> {
    let mut visited = HashSet::new();
    let mut communities = Vec::new();
    for node in graph.keys() {
        if !visited.insert(node.clone()) {
            continue;
        }
        let mut queue = VecDeque::from([node.clone()]);
        let mut nodes = Vec::new();
        while let Some(current) = queue.pop_front() {
            nodes.push(current.clone());
            for neighbor in graph.get(&current).into_iter().flatten() {
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor.clone());
                }
            }
        }
        nodes.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        let id = community_id(&nodes);
        communities.push(GraphRagCommunity {
            id,
            nodes,
            summary: String::new(),
            score: 0.0,
            parent: None,
        });
    }
    communities
}

fn build_hierarchy(mut communities: Vec<GraphRagCommunity>) -> Vec<GraphRagCommunity> {
    if communities.len() <= 1 {
        return communities;
    }
    let root_id = "community-root".to_string();
    let root_nodes = communities
        .iter()
        .flat_map(|community| community.nodes.iter().cloned())
        .collect::<Vec<_>>();
    for community in &mut communities {
        community.parent = Some(root_id.clone());
    }
    let root_score = communities
        .iter()
        .map(|community| community.score)
        .fold(0.0_f32, f32::max);
    let root_summary = format!(
        "Global graph community over {} sub-communities and {} knowledge nodes.",
        communities.len(),
        root_nodes.len()
    );
    let mut out = vec![GraphRagCommunity {
        id: root_id,
        nodes: root_nodes,
        summary: root_summary,
        score: root_score,
        parent: None,
    }];
    out.extend(communities);
    out
}

fn community_summary_hit(community: &GraphRagCommunity, score: f32) -> Result<RetrievalHit> {
    Ok(RetrievalHit {
        uri: ContextUri::parse(format!("uwu://graph-rag/community/{}", community.id))?,
        level: ContentLevel::L1,
        content: ContentPayload::Text {
            sparse: community.summary.clone(),
            dense: community.summary.clone(),
            full: community.summary.clone(),
        },
        relevance: score.max(community.score).min(1.0),
        parent_chain: community.nodes.clone(),
        content_type: None,
        metadata: Default::default(),
        created_at: None,
        updated_at: None,
    })
}

fn empty_text_payload() -> ContentPayload {
    ContentPayload::Text {
        sparse: String::new(),
        dense: String::new(),
        full: String::new(),
    }
}

fn default_graph_rag_relations() -> Vec<GraphRelation> {
    vec![
        GraphRelation::EvolvedFrom,
        GraphRelation::EvolvedTo,
        GraphRelation::DerivedFrom,
        GraphRelation::Supersedes,
        GraphRelation::EvidenceOf,
        GraphRelation::EntangledWith,
        GraphRelation::Corroborates,
        GraphRelation::DrivesPolicy,
    ]
}

fn node_rank(
    uri: &ContextUri,
    adjacency: &HashMap<ContextUri, HashSet<ContextUri>>,
    centrality: &HashMap<ContextUri, f32>,
    config: GraphRagConfig,
) -> f32 {
    let degree = adjacency.get(uri).map(|n| n.len()).unwrap_or(0) as f32;
    let centrality = centrality
        .get(uri)
        .copied()
        .unwrap_or(config.default_centrality);
    (centrality * config.index_centrality_weight
        + (degree / config.degree_saturation).min(1.0) * config.degree_weight)
        .clamp(0.0, 1.0)
}

fn average_node_rank(
    nodes: &[ContextUri],
    adjacency: &HashMap<ContextUri, HashSet<ContextUri>>,
    centrality: &HashMap<ContextUri, f32>,
    config: GraphRagConfig,
) -> f32 {
    if nodes.is_empty() {
        return 0.0;
    }
    nodes
        .iter()
        .map(|node| node_rank(node, adjacency, centrality, config))
        .sum::<f32>()
        / nodes.len() as f32
}

fn terms(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|part| part.len() >= 2)
        .map(|part| part.to_lowercase())
        .collect()
}

fn term_overlap(query_terms: &HashSet<String>, text_terms: &HashSet<String>) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let matched = query_terms
        .iter()
        .filter(|term| text_terms.contains(*term))
        .count();
    matched as f32 / query_terms.len() as f32
}

fn community_id(nodes: &[ContextUri]) -> String {
    let joined = nodes
        .iter()
        .map(|n| n.as_str())
        .collect::<Vec<_>>()
        .join("|");
    blake3::hash(joined.as_bytes())
        .to_hex()
        .chars()
        .take(12)
        .collect()
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let end = text
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    let prefix = text.get(..end).unwrap_or(text);
    format!("{prefix}...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{
        DirEntry, FindPattern, GraphRelation, GrepHit, Page, PageRequest, TreeNode,
    };
    use async_trait::async_trait;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    fn truncate_is_unicode_safe_and_handles_zero_limit() {
        assert_eq!(truncate("你好世界", 2), "你好...");
        assert_eq!(truncate("éclair", 1), "é...");
        assert_eq!(truncate("坏", 0), "...");
        assert_eq!(truncate("short", 10), "short");
    }

    #[derive(Default)]
    struct MockFs {
        docs: HashMap<ContextUri, String>,
    }

    #[async_trait]
    impl FsOps for MockFs {
        async fn ls(&self, _dir: &ContextUri, _page: PageRequest) -> Result<Page<DirEntry>> {
            Ok(Page::new(vec![], None))
        }

        async fn find(
            &self,
            _pattern: &FindPattern,
            _page: PageRequest,
        ) -> Result<Page<ContextUri>> {
            Ok(Page::new(self.docs.keys().cloned().collect(), None))
        }

        async fn grep(&self, _regex: &str, _scope: &ContextUri) -> Result<Vec<GrepHit>> {
            Ok(vec![])
        }

        async fn tree(
            &self,
            root: &ContextUri,
            _depth: usize,
            _page: PageRequest,
        ) -> Result<Page<TreeNode>> {
            Ok(Page::new(
                vec![TreeNode {
                    uri: root.clone(),
                    is_dir: true,
                    children: vec![],
                }],
                None,
            ))
        }

        async fn read(&self, uri: &ContextUri, _level: ContentLevel) -> Result<ContentPayload> {
            Ok(ContentPayload::Text {
                sparse: self.docs.get(uri).cloned().unwrap_or_default(),
                dense: String::new(),
                full: String::new(),
            })
        }
    }

    #[derive(Default)]
    struct MockGraph {
        edges: Mutex<HashMap<ContextUri, Vec<ContextUri>>>,
        neighbor_calls: AtomicUsize,
    }

    impl MockGraph {
        fn neighbor_calls(&self) -> usize {
            self.neighbor_calls.load(AtomicOrdering::SeqCst)
        }
    }

    #[async_trait]
    impl GraphStore for MockGraph {
        async fn add_edge(
            &self,
            from: &ContextUri,
            to: &ContextUri,
            _kind: GraphRelation,
        ) -> Result<()> {
            self.edges
                .lock()
                .unwrap()
                .entry(from.clone())
                .or_default()
                .push(to.clone());
            Ok(())
        }

        async fn remove_edge(&self, _from: &ContextUri, _to: &ContextUri) -> Result<()> {
            Ok(())
        }

        async fn outgoing_neighbors(
            &self,
            uri: &ContextUri,
            _kind: Option<GraphRelation>,
        ) -> Result<Vec<ContextUri>> {
            self.neighbor_calls.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(self
                .edges
                .lock()
                .unwrap()
                .get(uri)
                .cloned()
                .unwrap_or_default())
        }

        async fn batch_traverse(
            &self,
            seeds: &[ContextUri],
            kinds: &[GraphRelation],
            _max_hops: usize,
        ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>> {
            let mut out = Vec::new();
            for seed in seeds {
                for neighbor in self.outgoing_neighbors(seed, None).await? {
                    out.push((seed.clone(), neighbor, kinds[0]));
                }
            }
            Ok(out)
        }

        async fn centrality(&self, uri: &ContextUri) -> Result<f32> {
            Ok(if self.edges.lock().unwrap().contains_key(uri) {
                0.9
            } else {
                0.4
            })
        }
    }

    fn graph_fixture() -> (
        Arc<MockFs>,
        Arc<MockGraph>,
        ContextUri,
        ContextUri,
        ContextUri,
    ) {
        let a = ContextUri::parse("uwu://t/a/architecture/start").unwrap();
        let b = ContextUri::parse("uwu://t/a/architecture/memory").unwrap();
        let c = ContextUri::parse("uwu://t/a/architecture/retrieval").unwrap();
        let mut fs = MockFs::default();
        fs.docs
            .insert(a.clone(), "architecture started with memory storage".into());
        fs.docs.insert(
            b.clone(),
            "memory layer evolved into graph relations".into(),
        );
        fs.docs
            .insert(c.clone(), "retrieval layer added GraphRAG summaries".into());
        (Arc::new(fs), Arc::new(MockGraph::default()), a, b, c)
    }

    #[tokio::test]
    async fn graph_rag_returns_community_summary_then_drill_down_nodes() {
        let (fs, graph, a, b, c) = graph_fixture();
        graph
            .add_edge(&a, &b, GraphRelation::EvolvedTo)
            .await
            .unwrap();
        graph
            .add_edge(&b, &c, GraphRelation::EvolvedTo)
            .await
            .unwrap();
        let engine = GraphRagEngine::new(fs, graph, GraphRagConfig::default()).unwrap();
        let result = engine
            .retrieve(
                &GraphRagRequest::new(
                    "overall architecture evolution",
                    vec![a],
                    GraphRagConfig::default(),
                )
                .unwrap(),
                &RetrieveContext {
                    prefer_level: ContentLevel::L0,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(!result.hits.is_empty());
        assert_eq!(result.hits[0].level, ContentLevel::L1);
        assert!(result.hits[0].uri.as_str().contains("graph-rag/community"));
        assert!(result.hits.iter().any(|hit| hit.uri == b));
        assert!(result.hits.iter().any(|hit| hit.uri == c));
    }

    #[tokio::test]
    async fn graph_rag_index_reuses_graph_expansion_for_hot_queries() {
        let (fs, graph, a, b, c) = graph_fixture();
        graph
            .add_edge(&a, &b, GraphRelation::EvolvedTo)
            .await
            .unwrap();
        graph
            .add_edge(&b, &c, GraphRelation::EvolvedTo)
            .await
            .unwrap();
        let indexer =
            GraphRagIndexer::new(fs.clone(), graph.clone(), GraphRagConfig::default()).unwrap();
        let index = indexer
            .build_index(&GraphRagIndexConfig {
                seed_uris: vec![a.clone()],
                relations: vec![GraphRelation::EvolvedTo],
                max_hops: 2,
                max_summary_nodes: 8,
            })
            .await
            .unwrap();
        let calls_after_build = graph.neighbor_calls();
        let result = index
            .retrieve(
                fs,
                &GraphRagRequest::new("architecture retrieval", vec![a], GraphRagConfig::default())
                    .unwrap(),
                &RetrieveContext {
                    prefer_level: ContentLevel::L0,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(graph.neighbor_calls(), calls_after_build);
        assert!(index.stats().communities >= 1);
        assert!(result.hits.iter().any(|hit| hit.uri == c));
    }
}
