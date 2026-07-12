//! 意图智能层：将自然语言查询编译为可解释执行图。
//!
//! - [`RuleBasedIntentAnalyzer`]：外置 policy pack + 编译索引 + 规则评分。

use agent_context_db_core::{ContextUri, Envelope, EventMesh, EventTypeId, Result, Topic};
use aho_corasick::AhoCorasick;
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use regex::RegexSet;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use crate::RetrieveContext;

const DEFAULT_INTENT_POLICY_TOML: &str = include_str!("default_intent_policy.toml");

pub const INTENT_ANALYSIS_STARTED_TOPIC: &str = "intent.analysis.started";
pub const INTENT_ANALYSIS_COMPLETED_TOPIC: &str = "intent.analysis.completed";
pub const INTENT_POLICY_RELOADED_TOPIC: &str = "intent.policy.reloaded";
pub const INTENT_POLICY_REJECTED_TOPIC: &str = "intent.policy.rejected";
pub const INTENT_FEEDBACK_RECORDED_TOPIC: &str = "intent.feedback.recorded";

// ==========================================================================
// Intent Intelligence Fabric 数据模型
// ==========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentKind {
    SemanticSearch,
    EntityLookup,
    EventRecall,
    SkillReuse,
    PatternMatch,
    StateSnapshot,
    PersonaRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentRoute {
    LocalExact,
    LocalVector,
    LocalHybrid,
    GraphTraversal,
    TemporalIndex,
    KnowledgeNetwork,
    MultiStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentCaller {
    Retrieve,
    Sleeptime,
    KnowledgeNetwork,
    Version,
    Wiki,
    Cli,
    Api,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentInput {
    pub query: String,
    pub caller: IntentCaller,
    pub active_uri: Option<ContextUri>,
    pub recent_uris: Vec<ContextUri>,
    pub tenant: Option<String>,
    pub agent: Option<String>,
    pub language_hint: Option<String>,
}

impl IntentInput {
    pub fn from_retrieve(
        query: &str,
        ctx: &RetrieveContext,
        default_user: &str,
        default_agent: &str,
    ) -> Self {
        Self {
            query: query.to_string(),
            caller: IntentCaller::Retrieve,
            active_uri: None,
            recent_uris: Vec::new(),
            tenant: Some(
                ctx.user_id
                    .clone()
                    .unwrap_or_else(|| default_user.to_string()),
            ),
            agent: Some(
                ctx.agent_id
                    .clone()
                    .unwrap_or_else(|| default_agent.to_string()),
            ),
            language_hint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExecutionPlan {
    #[serde(default)]
    pub prefer_exact: bool,
    #[serde(default)]
    pub prefer_vector: bool,
    #[serde(default)]
    pub prefer_graph: bool,
    #[serde(default)]
    pub prefer_temporal: bool,
    #[serde(default)]
    pub allow_federation: bool,
    #[serde(default)]
    pub require_high_precision: bool,
    #[serde(default)]
    pub require_corroboration: bool,
    #[serde(default)]
    pub max_expansion_depth: usize,
    #[serde(default = "default_top_k_multiplier")]
    pub top_k_multiplier: f32,
}

impl Default for IntentExecutionPlan {
    fn default() -> Self {
        Self {
            prefer_exact: false,
            prefer_vector: false,
            prefer_graph: false,
            prefer_temporal: false,
            allow_federation: false,
            require_high_precision: false,
            require_corroboration: false,
            max_expansion_depth: 0,
            top_k_multiplier: default_top_k_multiplier(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExecutionGraph {
    pub nodes: Vec<IntentExecutionNode>,
    pub edges: Vec<IntentExecutionEdge>,
    pub budget: IntentExecutionBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExecutionNode {
    pub id: String,
    pub kind: IntentExecutionNodeKind,
    pub route: IntentRoute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentExecutionNodeKind {
    ExactLookup,
    VectorRetrieve,
    GraphTraversal,
    TemporalReplay,
    KnowledgeNetworkProbe,
    KnowledgeNetworkFetch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExecutionEdge {
    pub from: String,
    pub to: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExecutionBudget {
    #[serde(default = "default_max_latency_ms")]
    pub max_latency_ms: u64,
    #[serde(default)]
    pub max_llm_calls: usize,
    #[serde(default)]
    pub max_federated_peers: usize,
    #[serde(default = "default_budget_graph_depth")]
    pub max_graph_depth: usize,
}

impl Default for IntentExecutionBudget {
    fn default() -> Self {
        Self {
            max_latency_ms: default_max_latency_ms(),
            max_llm_calls: 0,
            max_federated_peers: 0,
            max_graph_depth: default_budget_graph_depth(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDecision {
    pub primary: IntentKind,
    pub secondary: Vec<IntentKind>,
    pub route: IntentRoute,
    pub confidence: f32,
    pub ambiguity: f32,
    pub candidates: Vec<IntentCandidate>,
    pub execution_graph: IntentExecutionGraph,
    pub explanation: IntentExplanation,
    pub policy: IntentPolicyRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicyRef {
    pub id: String,
    pub version: String,
    pub engine_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCandidate {
    pub intent: IntentKind,
    pub route: IntentRoute,
    pub score: f32,
    pub priority: u32,
    pub matched_terms: Vec<String>,
    pub matched_patterns: Vec<String>,
    pub breakdown: IntentScoreBreakdown,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntentScoreBreakdown {
    pub keyword: f32,
    pub phrase: f32,
    pub regex: f32,
    pub context: f32,
    pub caller: f32,
    pub feedback: f32,
    pub negative: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentFeedbackEvent {
    pub query: String,
    pub predicted: IntentKind,
    pub accepted: Option<IntentKind>,
    pub route: IntentRoute,
    pub success: bool,
    pub reward: f32,
}

#[derive(Debug, Default)]
pub struct IntentFeedbackLearning {
    bias: RwLock<HashMap<IntentKind, f32>>,
}

impl IntentFeedbackLearning {
    pub fn record(&self, event: &IntentFeedbackEvent, max_abs_bias: f32) {
        let target = event.accepted.unwrap_or(event.predicted);
        let direction = if event.success { 1.0 } else { -1.0 };
        let delta = (event.reward.clamp(0.0, 1.0) * 0.03 * direction).clamp(-0.05, 0.05);
        let mut bias = self.bias.write();
        for value in bias.values_mut() {
            *value *= 0.995;
        }
        let entry = bias.entry(target).or_insert(0.0);
        *entry = (*entry + delta).clamp(-max_abs_bias, max_abs_bias);
    }

    pub fn bias_for(&self, intent: IntentKind) -> f32 {
        self.bias.read().get(&intent).copied().unwrap_or(0.0)
    }

    pub fn snapshot(&self) -> HashMap<IntentKind, f32> {
        self.bias.read().clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentExplanation {
    pub policy_pack: String,
    pub policy_version: String,
    pub matched_rule_ids: Vec<String>,
    pub notes: Vec<String>,
}

// ==========================================================================
// Policy Pack: TOML + JSON 双格式
// ==========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicyPack {
    pub pack: IntentPolicyPackMetadata,
    #[serde(default)]
    pub governance: IntentGovernancePolicy,
    #[serde(default)]
    pub intent: Vec<IntentRule>,
    #[serde(default)]
    pub caller_boost: BTreeMap<String, BTreeMap<String, f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicyPackMetadata {
    pub id: String,
    pub version: String,
    #[serde(default = "default_engine_version")]
    pub engine_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentGovernancePolicy {
    #[serde(default = "default_max_rules")]
    pub max_rules: usize,
    #[serde(default = "default_max_regex_per_rule")]
    pub max_regex_per_rule: usize,
    #[serde(default = "default_max_pattern_len")]
    pub max_pattern_len: usize,
    #[serde(default = "default_true")]
    pub allow_execution_graph: bool,
    #[serde(default = "default_true")]
    pub allow_regex: bool,
    #[serde(default = "default_max_policy_layers")]
    pub max_policy_layers: usize,
    #[serde(default = "default_max_feedback_bias")]
    pub max_feedback_bias: f32,
}

impl Default for IntentGovernancePolicy {
    fn default() -> Self {
        Self {
            max_rules: default_max_rules(),
            max_regex_per_rule: default_max_regex_per_rule(),
            max_pattern_len: default_max_pattern_len(),
            allow_execution_graph: default_true(),
            allow_regex: default_true(),
            max_policy_layers: default_max_policy_layers(),
            max_feedback_bias: default_max_feedback_bias(),
        }
    }
}

impl IntentGovernancePolicy {
    fn merge_strict(&self, other: &Self) -> Self {
        Self {
            max_rules: self.max_rules.min(other.max_rules),
            max_regex_per_rule: self.max_regex_per_rule.min(other.max_regex_per_rule),
            max_pattern_len: self.max_pattern_len.min(other.max_pattern_len),
            allow_execution_graph: self.allow_execution_graph && other.allow_execution_graph,
            allow_regex: self.allow_regex && other.allow_regex,
            max_policy_layers: self.max_policy_layers.min(other.max_policy_layers),
            max_feedback_bias: self.max_feedback_bias.min(other.max_feedback_bias),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRule {
    pub id: String,
    pub kind: IntentKind,
    #[serde(default)]
    pub priority: u32,
    pub route: IntentRoute,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub phrases: Vec<String>,
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub negative_keywords: Vec<String>,
    #[serde(default)]
    pub target_dirs: Vec<String>,
    #[serde(default)]
    pub expected_type: Option<String>,
    #[serde(default)]
    pub plan: IntentExecutionPlan,
}

impl IntentPolicyPack {
    pub fn default_builtin() -> Result<Self> {
        Self::from_toml_str(DEFAULT_INTENT_POLICY_TOML)
    }

    pub fn merge_layers(mut layers: Vec<Self>) -> Result<Self> {
        if layers.is_empty() {
            return Self::default_builtin();
        }
        let max_layers = layers
            .iter()
            .map(|layer| layer.governance.max_policy_layers)
            .min()
            .unwrap_or_else(default_max_policy_layers);
        if layers.len() > max_layers {
            return Err(agent_context_db_core::ContextError::Unsupported(format!(
                "intent policy has {} layers, max is {}",
                layers.len(),
                max_layers
            )));
        }

        let mut merged = layers.remove(0);
        for layer in layers {
            merged.pack.version = layer.pack.version.clone();
            merged.pack.engine_version = merged.pack.engine_version.max(layer.pack.engine_version);
            merged.governance = merged.governance.merge_strict(&layer.governance);

            let mut rule_pos = merged
                .intent
                .iter()
                .enumerate()
                .map(|(idx, rule)| (rule.id.clone(), idx))
                .collect::<HashMap<_, _>>();
            for rule in layer.intent {
                if let Some(idx) = rule_pos.get(&rule.id).copied() {
                    merged.intent[idx] = rule;
                } else {
                    rule_pos.insert(rule.id.clone(), merged.intent.len());
                    merged.intent.push(rule);
                }
            }

            for (caller, boosts) in layer.caller_boost {
                merged
                    .caller_boost
                    .entry(caller)
                    .or_default()
                    .extend(boosts);
            }
        }
        merged.validate()?;
        Ok(merged)
    }

    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let pack: Self = toml::from_str(raw).map_err(|err| {
            agent_context_db_core::ContextError::Unsupported(format!(
                "intent TOML parse failed: {err}"
            ))
        })?;
        pack.validate()?;
        Ok(pack)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let pack: Self = serde_json::from_str(raw).map_err(|err| {
            agent_context_db_core::ContextError::Unsupported(format!(
                "intent JSON parse failed: {err}"
            ))
        })?;
        pack.validate()?;
        Ok(pack)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|err| {
            agent_context_db_core::ContextError::Unsupported(format!(
                "intent policy read failed {}: {err}",
                path.display()
            ))
        })?;
        match path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
        {
            "json" => Self::from_json_str(&raw),
            "toml" | "tml" => Self::from_toml_str(&raw),
            other => Err(agent_context_db_core::ContextError::Unsupported(format!(
                "unsupported intent policy extension `{other}`; expected .toml or .json"
            ))),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.intent.len() > self.governance.max_rules {
            return Err(agent_context_db_core::ContextError::Unsupported(format!(
                "intent policy has {} rules, max is {}",
                self.intent.len(),
                self.governance.max_rules
            )));
        }
        for rule in &self.intent {
            if !self.governance.allow_regex && !rule.patterns.is_empty() {
                return Err(agent_context_db_core::ContextError::Unsupported(format!(
                    "intent rule {} uses regex but regex is disabled by governance",
                    rule.id
                )));
            }
            if rule.patterns.len() > self.governance.max_regex_per_rule {
                return Err(agent_context_db_core::ContextError::Unsupported(format!(
                    "intent rule {} has too many regex patterns",
                    rule.id
                )));
            }
            for pattern in &rule.patterns {
                if pattern.len() > self.governance.max_pattern_len {
                    return Err(agent_context_db_core::ContextError::Unsupported(format!(
                        "intent rule {} has too long regex pattern",
                        rule.id
                    )));
                }
            }
        }
        Ok(())
    }
}

fn default_top_k_multiplier() -> f32 {
    1.0
}
fn default_max_latency_ms() -> u64 {
    250
}
fn default_budget_graph_depth() -> usize {
    2
}
fn default_engine_version() -> u32 {
    1
}
fn default_max_rules() -> usize {
    256
}
fn default_max_regex_per_rule() -> usize {
    16
}
fn default_max_pattern_len() -> usize {
    512
}
fn default_max_policy_layers() -> usize {
    8
}
fn default_max_feedback_bias() -> f32 {
    0.2
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentPolicyLayerKind {
    Builtin,
    Deployment,
    Workspace,
    Tenant,
    Experiment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicyLayer {
    pub kind: IntentPolicyLayerKind,
    pub source: String,
    pub pack: IntentPolicyPack,
    pub modified_at: Option<SystemTime>,
    #[serde(default)]
    pub signature: Option<IntentPolicySignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicySignature {
    pub signer: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedIntentPolicyPack {
    pub pack: IntentPolicyPack,
    pub signer: String,
    pub signature: String,
}

#[derive(Debug, Default)]
pub struct IntentPolicySignatureVerifier {
    shared_keys: RwLock<HashMap<String, String>>,
    require_signatures: bool,
}

impl IntentPolicySignatureVerifier {
    pub fn new(require_signatures: bool) -> Self {
        Self {
            shared_keys: RwLock::new(HashMap::new()),
            require_signatures,
        }
    }

    pub fn upsert_shared_key(&self, signer: impl Into<String>, key: impl Into<String>) {
        self.shared_keys.write().insert(signer.into(), key.into());
    }

    pub fn sign(&self, signer: &str, pack: &IntentPolicyPack) -> Result<SignedIntentPolicyPack> {
        let keys = self.shared_keys.read();
        let Some(key) = keys.get(signer) else {
            return Err(agent_context_db_core::ContextError::PermissionDenied(
                format!("unknown intent policy signer `{signer}`"),
            ));
        };
        Ok(SignedIntentPolicyPack {
            pack: pack.clone(),
            signer: signer.to_string(),
            signature: compute_policy_signature(pack, signer, key)?,
        })
    }

    pub fn verify_layer(&self, layer: &IntentPolicyLayer) -> Result<()> {
        let Some(signature) = &layer.signature else {
            if self.require_signatures {
                return Err(agent_context_db_core::ContextError::PermissionDenied(
                    format!("intent policy layer `{}` is unsigned", layer.source),
                ));
            }
            return Ok(());
        };
        let keys = self.shared_keys.read();
        let Some(key) = keys.get(&signature.signer) else {
            return Err(agent_context_db_core::ContextError::PermissionDenied(
                format!("unknown intent policy signer `{}`", signature.signer),
            ));
        };
        let expected = compute_policy_signature(&layer.pack, &signature.signer, key)?;
        if expected != signature.signature {
            return Err(agent_context_db_core::ContextError::PermissionDenied(
                format!(
                    "invalid signature for intent policy layer `{}`",
                    layer.source
                ),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicySnapshot {
    pub layers: Vec<IntentPolicyLayer>,
}

impl IntentPolicySnapshot {
    pub fn merged_pack(self) -> Result<IntentPolicyPack> {
        IntentPolicyPack::merge_layers(self.layers.into_iter().map(|layer| layer.pack).collect())
    }
}

pub trait IntentPolicyProvider: Send + Sync {
    fn load(&self) -> Result<IntentPolicySnapshot>;
}

#[derive(Debug, Clone, Default)]
pub struct BuiltinIntentPolicyProvider;

impl IntentPolicyProvider for BuiltinIntentPolicyProvider {
    fn load(&self) -> Result<IntentPolicySnapshot> {
        Ok(IntentPolicySnapshot {
            layers: vec![IntentPolicyLayer {
                kind: IntentPolicyLayerKind::Builtin,
                source: "builtin:default_intent_policy.toml".into(),
                pack: IntentPolicyPack::default_builtin()?,
                modified_at: None,
                signature: None,
            }],
        })
    }
}

#[derive(Debug, Clone)]
pub struct FileIntentPolicyProvider {
    path: PathBuf,
    kind: IntentPolicyLayerKind,
}

impl FileIntentPolicyProvider {
    pub fn new(path: impl Into<PathBuf>, kind: IntentPolicyLayerKind) -> Self {
        Self {
            path: path.into(),
            kind,
        }
    }
}

impl IntentPolicyProvider for FileIntentPolicyProvider {
    fn load(&self) -> Result<IntentPolicySnapshot> {
        let modified_at = std::fs::metadata(&self.path)
            .ok()
            .and_then(|metadata| metadata.modified().ok());
        Ok(IntentPolicySnapshot {
            layers: vec![IntentPolicyLayer {
                kind: self.kind,
                source: self.path.display().to_string(),
                pack: IntentPolicyPack::from_path(&self.path)?,
                modified_at,
                signature: None,
            }],
        })
    }
}

#[derive(Clone)]
pub struct LayeredIntentPolicyProvider {
    providers: Vec<Arc<dyn IntentPolicyProvider>>,
}

impl LayeredIntentPolicyProvider {
    pub fn new(providers: Vec<Arc<dyn IntentPolicyProvider>>) -> Self {
        Self { providers }
    }
}

impl IntentPolicyProvider for LayeredIntentPolicyProvider {
    fn load(&self) -> Result<IntentPolicySnapshot> {
        let mut layers = Vec::new();
        for provider in &self.providers {
            layers.extend(provider.load()?.layers);
        }
        Ok(IntentPolicySnapshot { layers })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentPolicyReloadStatus {
    Reloaded,
    Unchanged,
    Rejected { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentPolicyReloadReport {
    pub status: IntentPolicyReloadStatus,
    pub active_policy: IntentPolicyRef,
    pub attempted_sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IntentTraceEvent {
    AnalysisStarted {
        query_len: usize,
    },
    AnalysisCompleted {
        decision: IntentDecision,
    },
    PolicyReloaded {
        report: IntentPolicyReloadReport,
    },
    PolicyRejected {
        report: IntentPolicyReloadReport,
    },
    FeedbackRecorded {
        event: IntentFeedbackEvent,
        bias: HashMap<IntentKind, f32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentTraceDiagnostic {
    Accepted,
    InvalidTopic(String),
    SerializationFailed(String),
    PublishFailed(String),
}

pub trait IntentTraceSink: Send + Sync {
    fn emit(&self, topic: &'static str, event: &IntentTraceEvent) -> IntentTraceDiagnostic;
}

#[derive(Debug, Default)]
pub struct TracingIntentTraceSink;

impl IntentTraceSink for TracingIntentTraceSink {
    fn emit(&self, topic: &'static str, event: &IntentTraceEvent) -> IntentTraceDiagnostic {
        tracing::debug!(target: "context_db::intent", topic, ?event);
        IntentTraceDiagnostic::Accepted
    }
}

#[derive(Clone)]
pub struct EventMeshIntentTraceSink {
    mesh: EventMesh,
    source: String,
}

impl EventMeshIntentTraceSink {
    pub fn new(mesh: EventMesh, source: impl Into<String>) -> Self {
        Self {
            mesh,
            source: source.into(),
        }
    }
}

impl IntentTraceSink for EventMeshIntentTraceSink {
    fn emit(&self, topic: &'static str, event: &IntentTraceEvent) -> IntentTraceDiagnostic {
        let topic_id = match Topic::new(topic) {
            Ok(topic_id) => topic_id,
            Err(error) => return IntentTraceDiagnostic::InvalidTopic(error.to_string()),
        };
        let payload = match serde_json::to_value(event) {
            Ok(payload) => payload,
            Err(error) => return IntentTraceDiagnostic::SerializationFailed(error.to_string()),
        };
        let mut env = Envelope::new(&topic_id, payload);
        env.type_id = Some(EventTypeId::new("intent", intent_event_type_name(topic)));
        env.source = Some(self.source.clone());
        let mesh = self.mesh.clone();
        tokio::spawn(async move {
            if let Err(error) = mesh.publish(env).await {
                tracing::error!(error = ?agent_context_db_core::ErrorReport::from_error(&error), "intent trace EventMesh publish failed");
            }
        });
        IntentTraceDiagnostic::Accepted
    }
}

// ==========================================================================
// 编译索引 + 评分
// ==========================================================================

#[derive(Debug)]
pub struct CompiledIntentPolicy {
    pack: IntentPolicyPack,
    keyword_index: Option<AhoCorasick>,
    keyword_patterns: Vec<String>,
    keyword_to_rules: Vec<SmallVec<[usize; 4]>>,
    negative_index: Option<AhoCorasick>,
    negative_to_rules: Vec<SmallVec<[usize; 4]>>,
    regex_set: Option<RegexSet>,
    regex_patterns: Vec<String>,
    regex_to_rules: Vec<SmallVec<[usize; 2]>>,
}

impl CompiledIntentPolicy {
    pub fn policy_ref(&self) -> IntentPolicyRef {
        IntentPolicyRef {
            id: self.pack.pack.id.clone(),
            version: self.pack.pack.version.clone(),
            engine_version: self.pack.pack.engine_version,
        }
    }

    pub fn governance(&self) -> &IntentGovernancePolicy {
        &self.pack.governance
    }

    pub fn compile(pack: IntentPolicyPack) -> Result<Self> {
        let mut keyword_patterns = Vec::<String>::new();
        let mut keyword_to_rules = Vec::<SmallVec<[usize; 4]>>::new();
        let mut keyword_seen = HashMap::<String, usize>::new();
        let mut negative_patterns = Vec::<String>::new();
        let mut negative_to_rules = Vec::<SmallVec<[usize; 4]>>::new();
        let mut negative_seen = HashMap::<String, usize>::new();
        let mut regex_patterns = Vec::<String>::new();
        let mut regex_to_rules = Vec::<SmallVec<[usize; 2]>>::new();

        for (rule_idx, rule) in pack.intent.iter().enumerate() {
            for term in rule.keywords.iter().chain(rule.phrases.iter()) {
                let key = term.to_lowercase();
                let idx = *keyword_seen.entry(key.clone()).or_insert_with(|| {
                    keyword_patterns.push(key);
                    keyword_to_rules.push(SmallVec::new());
                    keyword_patterns.len() - 1
                });
                keyword_to_rules[idx].push(rule_idx);
            }
            for term in &rule.negative_keywords {
                let key = term.to_lowercase();
                let idx = *negative_seen.entry(key.clone()).or_insert_with(|| {
                    negative_patterns.push(key);
                    negative_to_rules.push(SmallVec::new());
                    negative_patterns.len() - 1
                });
                negative_to_rules[idx].push(rule_idx);
            }
            for pattern in &rule.patterns {
                regex_patterns.push(pattern.clone());
                let mut rules = SmallVec::new();
                rules.push(rule_idx);
                regex_to_rules.push(rules);
            }
        }

        let keyword_index = if keyword_patterns.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&keyword_patterns).map_err(|err| {
                agent_context_db_core::ContextError::Unsupported(format!(
                    "intent keyword index compile failed: {err}"
                ))
            })?)
        };
        let negative_index = if negative_patterns.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&negative_patterns).map_err(|err| {
                agent_context_db_core::ContextError::Unsupported(format!(
                    "intent negative index compile failed: {err}"
                ))
            })?)
        };
        let regex_set = if regex_patterns.is_empty() {
            None
        } else {
            Some(RegexSet::new(&regex_patterns).map_err(|err| {
                agent_context_db_core::ContextError::Unsupported(format!(
                    "intent regex set compile failed: {err}"
                ))
            })?)
        };

        Ok(Self {
            pack,
            keyword_index,
            keyword_patterns,
            keyword_to_rules,
            negative_index,
            negative_to_rules,
            regex_set,
            regex_patterns,
            regex_to_rules,
        })
    }

    pub fn decide(&self, input: &IntentInput) -> IntentDecision {
        self.decide_with_feedback(input, |_| 0.0)
    }

    pub fn decide_with_feedback(
        &self,
        input: &IntentInput,
        feedback_bias: impl Fn(IntentKind) -> f32,
    ) -> IntentDecision {
        let normalized = input.query.to_lowercase();
        let mut states = self
            .pack
            .intent
            .iter()
            .map(RuleScoreState::new)
            .collect::<Vec<_>>();
        let mut candidate_rules = BTreeSet::<usize>::new();

        if let Some(index) = &self.keyword_index {
            for mat in index.find_iter(&normalized) {
                let pid = mat.pattern().as_usize();
                for rule_idx in &self.keyword_to_rules[pid] {
                    candidate_rules.insert(*rule_idx);
                    states[*rule_idx]
                        .matched_terms
                        .insert(self.keyword_patterns[pid].clone());
                    states[*rule_idx].breakdown.keyword += 0.28;
                }
            }
        }
        if let Some(index) = &self.negative_index {
            for mat in index.find_iter(&normalized) {
                let pid = mat.pattern().as_usize();
                for rule_idx in &self.negative_to_rules[pid] {
                    candidate_rules.insert(*rule_idx);
                    states[*rule_idx].breakdown.negative += 0.35;
                }
            }
        }
        if let Some(regex_set) = &self.regex_set {
            for pid in regex_set.matches(&input.query).iter() {
                for rule_idx in &self.regex_to_rules[pid] {
                    candidate_rules.insert(*rule_idx);
                    states[*rule_idx]
                        .matched_patterns
                        .push(self.regex_patterns[pid].clone());
                    states[*rule_idx].breakdown.regex += 0.32;
                }
            }
        }

        if candidate_rules.is_empty()
            && let Some(default_idx) = self
                .pack
                .intent
                .iter()
                .position(|rule| rule.kind == IntentKind::SemanticSearch)
        {
            candidate_rules.insert(default_idx);
            states[default_idx].breakdown.context += 0.2;
        }

        for idx in &candidate_rules {
            let rule = &self.pack.intent[*idx];
            let caller_boost = self.caller_boost(input.caller, rule.kind);
            states[*idx].breakdown.caller += caller_boost;
            states[*idx].breakdown.context += self.context_boost(input, rule);
        }

        let mut candidates = candidate_rules
            .into_iter()
            .map(|idx| {
                let rule = &self.pack.intent[idx];
                let mut state = states[idx].clone();
                state.breakdown.feedback = feedback_bias(rule.kind);
                state.breakdown.final_score = (0.08
                    + state.breakdown.keyword
                    + state.breakdown.phrase
                    + state.breakdown.regex
                    + state.breakdown.context
                    + state.breakdown.caller
                    + state.breakdown.feedback
                    - state.breakdown.negative)
                    .clamp(0.0, 1.0);
                IntentCandidate {
                    intent: rule.kind,
                    route: rule.route,
                    score: state.breakdown.final_score,
                    priority: rule.priority,
                    matched_terms: state.matched_terms.into_iter().collect(),
                    matched_patterns: state.matched_patterns,
                    breakdown: state.breakdown,
                }
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.priority.cmp(&a.priority))
        });

        let primary_candidate = candidates
            .first()
            .cloned()
            .unwrap_or_else(|| IntentCandidate {
                intent: IntentKind::SemanticSearch,
                route: IntentRoute::LocalVector,
                score: 0.2,
                priority: 0,
                matched_terms: Vec::new(),
                matched_patterns: Vec::new(),
                breakdown: IntentScoreBreakdown {
                    final_score: 0.2,
                    ..Default::default()
                },
            });
        let secondary = candidates
            .iter()
            .skip(1)
            .take(3)
            .map(|c| c.intent)
            .collect::<Vec<_>>();
        let second_score = candidates.get(1).map(|c| c.score).unwrap_or(0.0);
        let ambiguity = (second_score / primary_candidate.score.max(0.01)).clamp(0.0, 1.0);
        let matched_rule_ids = candidates
            .iter()
            .filter_map(|candidate| {
                self.pack
                    .intent
                    .iter()
                    .find(|rule| rule.kind == candidate.intent && rule.route == candidate.route)
                    .map(|rule| rule.id.clone())
            })
            .collect::<Vec<_>>();
        let rule = self.pack.intent.iter().find(|rule| {
            rule.kind == primary_candidate.intent && rule.route == primary_candidate.route
        });
        let execution_graph = build_execution_graph(
            primary_candidate.intent,
            primary_candidate.route,
            rule.map(|r| &r.plan)
                .unwrap_or(&IntentExecutionPlan::default()),
        );

        IntentDecision {
            primary: primary_candidate.intent,
            secondary,
            route: primary_candidate.route,
            confidence: primary_candidate.score,
            ambiguity,
            candidates,
            execution_graph,
            explanation: IntentExplanation {
                policy_pack: self.pack.pack.id.clone(),
                policy_version: self.pack.pack.version.clone(),
                matched_rule_ids,
                notes: explanation_reasons(&primary_candidate),
            },
            policy: self.policy_ref(),
        }
    }

    fn caller_boost(&self, caller: IntentCaller, intent: IntentKind) -> f32 {
        let caller_key = format!("{:?}", caller).to_lowercase();
        let intent_key = format!("{:?}", intent).to_lowercase();
        self.pack
            .caller_boost
            .get(&caller_key)
            .and_then(|table| table.get(&intent_key))
            .copied()
            .unwrap_or(0.0)
    }

    fn context_boost(&self, input: &IntentInput, rule: &IntentRule) -> f32 {
        let mut boost: f32 = 0.0;
        if matches!(input.caller, IntentCaller::Retrieve) {
            boost += 0.04;
        }
        if let Some(active) = &input.active_uri {
            let uri = active.as_str().to_lowercase();
            if uri.contains("version") && matches!(rule.route, IntentRoute::TemporalIndex) {
                boost += 0.08;
            }
            if uri.contains("knowledge") && matches!(rule.route, IntentRoute::KnowledgeNetwork) {
                boost += 0.08;
            }
        }
        boost.clamp(0.0, 0.25)
    }
}

#[derive(Debug, Clone)]
struct RuleScoreState {
    matched_terms: BTreeSet<String>,
    matched_patterns: Vec<String>,
    breakdown: IntentScoreBreakdown,
}

impl RuleScoreState {
    fn new(_rule: &IntentRule) -> Self {
        Self {
            matched_terms: BTreeSet::new(),
            matched_patterns: Vec::new(),
            breakdown: IntentScoreBreakdown::default(),
        }
    }
}

// ==========================================================================
// RuleBasedIntentAnalyzer
// ==========================================================================

pub struct RuleBasedIntentAnalyzer {
    default_user_id: String,
    default_agent_id: String,
    policy: ArcSwap<CompiledIntentPolicy>,
    provider: RwLock<Option<Arc<dyn IntentPolicyProvider>>>,
    signature_verifier: RwLock<Option<Arc<IntentPolicySignatureVerifier>>>,
    last_policy_signature: RwLock<Option<String>>,
    feedback: Arc<IntentFeedbackLearning>,
    trace_sink: RwLock<Arc<dyn IntentTraceSink>>,
}

impl RuleBasedIntentAnalyzer {
    pub fn new(
        default_user_id: impl Into<String>,
        default_agent_id: impl Into<String>,
    ) -> Result<Self> {
        let pack = IntentPolicyPack::default_builtin()?;
        let compiled = CompiledIntentPolicy::compile(pack)?;
        Ok(Self::with_compiled(
            default_user_id,
            default_agent_id,
            compiled,
        ))
    }

    pub fn from_policy_pack(
        default_user_id: impl Into<String>,
        default_agent_id: impl Into<String>,
        pack: IntentPolicyPack,
    ) -> Result<Self> {
        let compiled = CompiledIntentPolicy::compile(pack)?;
        Ok(Self::with_compiled(
            default_user_id,
            default_agent_id,
            compiled,
        ))
    }

    pub fn from_policy_path(
        default_user_id: impl Into<String>,
        default_agent_id: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::from_policy_pack(
            default_user_id,
            default_agent_id,
            IntentPolicyPack::from_path(path)?,
        )
    }

    fn with_compiled(
        default_user_id: impl Into<String>,
        default_agent_id: impl Into<String>,
        compiled: CompiledIntentPolicy,
    ) -> Self {
        let signature = policy_signature(&compiled.pack, &[]);
        Self {
            default_user_id: default_user_id.into(),
            default_agent_id: default_agent_id.into(),
            policy: ArcSwap::from_pointee(compiled),
            provider: RwLock::new(None),
            signature_verifier: RwLock::new(None),
            last_policy_signature: RwLock::new(Some(signature)),
            feedback: Arc::new(IntentFeedbackLearning::default()),
            trace_sink: RwLock::new(Arc::new(TracingIntentTraceSink)),
        }
    }

    pub fn with_policy_provider(self, provider: Arc<dyn IntentPolicyProvider>) -> Self {
        *self.provider.write() = Some(provider);
        self
    }

    pub fn with_trace_sink(self, trace_sink: Arc<dyn IntentTraceSink>) -> Self {
        *self.trace_sink.write() = trace_sink;
        self
    }

    pub fn with_signature_verifier(self, verifier: Arc<IntentPolicySignatureVerifier>) -> Self {
        *self.signature_verifier.write() = Some(verifier);
        self
    }

    pub fn feedback_learning(&self) -> Arc<IntentFeedbackLearning> {
        self.feedback.clone()
    }

    pub fn active_policy(&self) -> IntentPolicyRef {
        self.policy.load().policy_ref()
    }

    pub fn reload_policy_pack(&self, pack: IntentPolicyPack) -> Result<()> {
        let compiled = CompiledIntentPolicy::compile(pack)?;
        *self.last_policy_signature.write() = Some(policy_signature(&compiled.pack, &[]));
        self.policy.store(Arc::new(compiled));
        Ok(())
    }

    pub fn reload_from_provider(&self) -> IntentPolicyReloadReport {
        let Some(provider) = self.provider.read().clone() else {
            return IntentPolicyReloadReport {
                status: IntentPolicyReloadStatus::Unchanged,
                active_policy: self.active_policy(),
                attempted_sources: Vec::new(),
            };
        };

        let report = match provider.load().and_then(|snapshot| {
            if let Some(verifier) = self.signature_verifier.read().clone() {
                for layer in &snapshot.layers {
                    verifier.verify_layer(layer)?;
                }
            }
            let sources = snapshot
                .layers
                .iter()
                .map(|layer| layer.source.clone())
                .collect::<Vec<_>>();
            let pack = snapshot.clone().merged_pack()?;
            let signature = policy_signature(&pack, &snapshot.layers);
            if self.last_policy_signature.read().as_deref() == Some(signature.as_str()) {
                return Ok(IntentPolicyReloadReport {
                    status: IntentPolicyReloadStatus::Unchanged,
                    active_policy: self.active_policy(),
                    attempted_sources: sources,
                });
            }
            let compiled = CompiledIntentPolicy::compile(pack)?;
            let active_policy = compiled.policy_ref();
            self.policy.store(Arc::new(compiled));
            *self.last_policy_signature.write() = Some(signature);
            Ok(IntentPolicyReloadReport {
                status: IntentPolicyReloadStatus::Reloaded,
                active_policy,
                attempted_sources: sources,
            })
        }) {
            Ok(report) => report,
            Err(err) => IntentPolicyReloadReport {
                status: IntentPolicyReloadStatus::Rejected {
                    reason: err.to_string(),
                },
                active_policy: self.active_policy(),
                attempted_sources: Vec::new(),
            },
        };

        let event = match report.status {
            IntentPolicyReloadStatus::Rejected { .. } => IntentTraceEvent::PolicyRejected {
                report: report.clone(),
            },
            IntentPolicyReloadStatus::Reloaded | IntentPolicyReloadStatus::Unchanged => {
                IntentTraceEvent::PolicyReloaded {
                    report: report.clone(),
                }
            }
        };
        let topic = match report.status {
            IntentPolicyReloadStatus::Rejected { .. } => INTENT_POLICY_REJECTED_TOPIC,
            IntentPolicyReloadStatus::Reloaded | IntentPolicyReloadStatus::Unchanged => {
                INTENT_POLICY_RELOADED_TOPIC
            }
        };
        self.emit_trace(topic, event);
        report
    }

    pub fn record_feedback(&self, event: IntentFeedbackEvent) {
        let max_bias = self.policy.load().governance().max_feedback_bias;
        self.feedback.record(&event, max_bias);
        self.emit_trace(
            INTENT_FEEDBACK_RECORDED_TOPIC,
            IntentTraceEvent::FeedbackRecorded {
                event,
                bias: self.feedback.snapshot(),
            },
        );
    }

    pub fn decide(&self, query: &str, ctx: &RetrieveContext) -> IntentDecision {
        let input =
            IntentInput::from_retrieve(query, ctx, &self.default_user_id, &self.default_agent_id);
        self.emit_trace(
            INTENT_ANALYSIS_STARTED_TOPIC,
            IntentTraceEvent::AnalysisStarted {
                query_len: query.len(),
            },
        );
        let decision = self
            .policy
            .load()
            .decide_with_feedback(&input, |kind| self.feedback.bias_for(kind));
        self.emit_trace(
            INTENT_ANALYSIS_COMPLETED_TOPIC,
            IntentTraceEvent::AnalysisCompleted {
                decision: decision.clone(),
            },
        );
        decision
    }

    fn emit_trace(&self, topic: &'static str, event: IntentTraceEvent) {
        self.trace_sink.read().emit(topic, &event);
    }
}

// ==========================================================================
// 辅助
// ==========================================================================

fn compute_policy_signature(pack: &IntentPolicyPack, signer: &str, key: &str) -> Result<String> {
    let canonical = serde_json::to_string(pack).map_err(|err| {
        agent_context_db_core::ContextError::Unsupported(format!(
            "intent policy signature serialization failed: {err}"
        ))
    })?;
    let mut hasher = DefaultHasher::new();
    signer.hash(&mut hasher);
    key.hash(&mut hasher);
    canonical.hash(&mut hasher);
    Ok(format!("intent-sha256:{:016x}", hasher.finish()))
}

fn policy_signature(pack: &IntentPolicyPack, layers: &[IntentPolicyLayer]) -> String {
    let mut signature = format!(
        "{}:{}:{}:{}",
        pack.pack.id,
        pack.pack.version,
        pack.pack.engine_version,
        pack.intent.len()
    );
    for layer in layers {
        signature.push('|');
        signature.push_str(&layer.source);
        signature.push(':');
        signature.push_str(&format!("{:?}", layer.modified_at));
    }
    signature
}

fn intent_event_type_name(topic: &str) -> &'static str {
    match topic {
        INTENT_ANALYSIS_STARTED_TOPIC => "analysis_started",
        INTENT_ANALYSIS_COMPLETED_TOPIC => "analysis_completed",
        INTENT_POLICY_RELOADED_TOPIC => "policy_reloaded",
        INTENT_POLICY_REJECTED_TOPIC => "policy_rejected",
        INTENT_FEEDBACK_RECORDED_TOPIC => "feedback_recorded",
        _ => "intent_event",
    }
}

fn explanation_reasons(candidate: &IntentCandidate) -> Vec<String> {
    let mut notes = Vec::new();
    if !candidate.matched_terms.is_empty() {
        notes.push(format!(
            "matched terms: {}",
            candidate.matched_terms.join(", ")
        ));
    }
    if !candidate.matched_patterns.is_empty() {
        notes.push(format!(
            "matched regex patterns: {}",
            candidate.matched_patterns.join(", ")
        ));
    }
    if candidate.breakdown.feedback.abs() > f32::EPSILON {
        notes.push(format!(
            "feedback bias: {:.3}",
            candidate.breakdown.feedback
        ));
    }
    notes.push(format!(
        "route {:?} selected with score {:.3}",
        candidate.route, candidate.score
    ));
    notes
}

fn build_execution_graph(
    intent: IntentKind,
    route: IntentRoute,
    plan: &IntentExecutionPlan,
) -> IntentExecutionGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    match route {
        IntentRoute::LocalExact => {
            nodes.push(node("exact", IntentExecutionNodeKind::ExactLookup, route));
            if plan.prefer_vector {
                nodes.push(node(
                    "vector",
                    IntentExecutionNodeKind::VectorRetrieve,
                    route,
                ));
                edges.push(edge("exact", "vector", "fallback"));
            }
        }
        IntentRoute::LocalVector | IntentRoute::LocalHybrid => {
            if plan.prefer_exact {
                nodes.push(node("exact", IntentExecutionNodeKind::ExactLookup, route));
            }
            nodes.push(node(
                "vector",
                IntentExecutionNodeKind::VectorRetrieve,
                route,
            ));
            if plan.prefer_exact {
                edges.push(edge("exact", "vector", "expand"));
            }
        }
        IntentRoute::GraphTraversal => {
            nodes.push(node(
                "graph",
                IntentExecutionNodeKind::GraphTraversal,
                route,
            ));
        }
        IntentRoute::TemporalIndex => {
            nodes.push(node(
                "temporal",
                IntentExecutionNodeKind::TemporalReplay,
                route,
            ));
        }
        IntentRoute::KnowledgeNetwork => {
            nodes.push(node(
                "probe",
                IntentExecutionNodeKind::KnowledgeNetworkProbe,
                route,
            ));
            nodes.push(node(
                "fetch",
                IntentExecutionNodeKind::KnowledgeNetworkFetch,
                route,
            ));
            edges.push(edge("probe", "fetch", "responsive_peer"));
        }
        IntentRoute::MultiStage => {
            nodes.push(node(
                "vector",
                IntentExecutionNodeKind::VectorRetrieve,
                route,
            ));
            nodes.push(node(
                "graph",
                IntentExecutionNodeKind::GraphTraversal,
                route,
            ));
            edges.push(edge("vector", "graph", "refine"));
        }
    }
    if nodes.is_empty() {
        nodes.push(node(
            "vector",
            IntentExecutionNodeKind::VectorRetrieve,
            IntentRoute::LocalVector,
        ));
    }
    IntentExecutionGraph {
        nodes,
        edges,
        budget: IntentExecutionBudget {
            max_latency_ms: match intent {
                IntentKind::SemanticSearch => 500,
                IntentKind::EventRecall
                | IntentKind::PatternMatch
                | IntentKind::PersonaRelation => 900,
                _ => 650,
            },
            max_llm_calls: 0,
            max_federated_peers: if plan.allow_federation { 8 } else { 0 },
            max_graph_depth: plan.max_expansion_depth.max(1),
        },
    }
}

fn node(id: &str, kind: IntentExecutionNodeKind, route: IntentRoute) -> IntentExecutionNode {
    IntentExecutionNode {
        id: id.to_string(),
        kind,
        route,
    }
}

fn edge(from: &str, to: &str, relation: &str) -> IntentExecutionEdge {
    IntentExecutionEdge {
        from: from.to_string(),
        to: to.to_string(),
        relation: relation.to_string(),
    }
}

// ==========================================================================
// 测试
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RetrieveContext {
        RetrieveContext {
            user_id: Some("u1".into()),
            agent_id: Some("a1".into()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn event_query_routes_to_event_recall() {
        let ia =
            RuleBasedIntentAnalyzer::new("u1", "a1").expect("default intent policy must compile");
        let decision = ia.decide("when did that migration happen?", &ctx());
        assert_eq!(decision.primary, IntentKind::EventRecall);
    }

    #[tokio::test]
    async fn pattern_query_gets_pattern_match() {
        let ia =
            RuleBasedIntentAnalyzer::new("u1", "a1").expect("default intent policy must compile");
        let decision = ia.decide("rust async patterns", &ctx());
        assert!(
            decision
                .candidates
                .iter()
                .any(|candidate| candidate.intent == IntentKind::PatternMatch)
        );
    }

    #[tokio::test]
    async fn preference_query_falls_back_to_semantic() {
        let ia =
            RuleBasedIntentAnalyzer::new("u1", "a1").expect("default intent policy must compile");
        let decision = ia.decide("what does the user like?", &ctx());
        assert_eq!(decision.primary, IntentKind::SemanticSearch);
    }

    #[test]
    fn policy_pack_supports_json() {
        let json = r#"{
            "pack": {"id": "test.intent", "version": "1.0.0", "engine_version": 1},
            "governance": {"max_rules": 16, "max_regex_per_rule": 4, "max_pattern_len": 128, "allow_execution_graph": true},
            "intent": [{
                "id": "test.entity",
                "kind": "entity_lookup",
                "priority": 99,
                "route": "local_exact",
                "keywords": ["owner"],
                "target_dirs": ["memories/entities"],
                "expected_type": "fact",
                "plan": {"prefer_exact": true, "top_k_multiplier": 1.0, "max_expansion_depth": 1}
            }]
        }"#;
        let pack = IntentPolicyPack::from_json_str(json).unwrap();
        assert_eq!(pack.intent[0].kind, IntentKind::EntityLookup);
    }

    #[test]
    fn decision_contains_execution_graph() {
        let ia =
            RuleBasedIntentAnalyzer::new("u1", "a1").expect("default intent policy must compile");
        let decision = ia.decide("who owns the project?", &ctx());
        assert_eq!(decision.primary, IntentKind::EntityLookup);
        assert_eq!(decision.policy.id, "context-db.default.intent");
        assert!(!decision.execution_graph.nodes.is_empty());
        assert!(!decision.explanation.matched_rule_ids.is_empty());
    }

    #[test]
    fn layered_policy_overrides_rule_by_id() {
        let builtin = IntentPolicyPack::default_builtin().unwrap();
        let mut overlay = builtin.clone();
        overlay.pack.version = "1.0.1".into();
        overlay.intent = vec![IntentRule {
            id: "retrieve.entity_lookup".into(),
            kind: IntentKind::EntityLookup,
            priority: 120,
            route: IntentRoute::LocalHybrid,
            keywords: vec!["owner".into()],
            phrases: Vec::new(),
            patterns: Vec::new(),
            negative_keywords: Vec::new(),
            target_dirs: vec!["memories/entities".into()],
            expected_type: Some("fact".into()),
            plan: IntentExecutionPlan {
                prefer_exact: true,
                prefer_vector: true,
                max_expansion_depth: 1,
                ..Default::default()
            },
        }];
        let merged = IntentPolicyPack::merge_layers(vec![builtin, overlay]).unwrap();
        let rule = merged
            .intent
            .iter()
            .find(|rule| rule.id == "retrieve.entity_lookup")
            .unwrap();
        assert_eq!(merged.pack.version, "1.0.1");
        assert_eq!(rule.route, IntentRoute::LocalHybrid);
        assert_eq!(rule.priority, 120);
    }

    #[test]
    fn file_provider_loads_builtin_toml_file() {
        let provider = FileIntentPolicyProvider::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/default_intent_policy.toml"),
            IntentPolicyLayerKind::Deployment,
        );
        let snapshot = provider.load().unwrap();
        let pack = snapshot.merged_pack().unwrap();
        assert_eq!(pack.pack.id, "context-db.default.intent");
        assert!(
            pack.intent
                .iter()
                .any(|rule| rule.id == "retrieve.event_recall")
        );
    }

    #[test]
    fn reload_from_provider_keeps_last_good_on_rejection() {
        #[derive(Clone)]
        struct BadProvider;
        impl IntentPolicyProvider for BadProvider {
            fn load(&self) -> Result<IntentPolicySnapshot> {
                Err(agent_context_db_core::ContextError::Unsupported(
                    "bad policy".into(),
                ))
            }
        }

        let ia = RuleBasedIntentAnalyzer::new("u1", "a1")
            .expect("default intent policy must compile")
            .with_policy_provider(Arc::new(BadProvider));
        let before = ia.active_policy();
        let report = ia.reload_from_provider();
        assert!(matches!(
            report.status,
            IntentPolicyReloadStatus::Rejected { .. }
        ));
        assert_eq!(ia.active_policy().id, before.id);
    }

    #[test]
    fn feedback_bias_changes_candidate_breakdown() {
        let ia =
            RuleBasedIntentAnalyzer::new("u1", "a1").expect("default intent policy must compile");
        ia.record_feedback(IntentFeedbackEvent {
            query: "pattern".into(),
            predicted: IntentKind::PatternMatch,
            accepted: Some(IntentKind::PatternMatch),
            route: IntentRoute::GraphTraversal,
            success: true,
            reward: 1.0,
        });
        let decision = ia.decide("pattern", &ctx());
        let pattern = decision
            .candidates
            .iter()
            .find(|candidate| candidate.intent == IntentKind::PatternMatch)
            .unwrap();
        assert!(pattern.breakdown.feedback > 0.0);
    }

    #[test]
    fn signature_verifier_rejects_bad_policy_layer() {
        #[derive(Clone)]
        struct SignedProvider(IntentPolicyLayer);
        impl IntentPolicyProvider for SignedProvider {
            fn load(&self) -> Result<IntentPolicySnapshot> {
                Ok(IntentPolicySnapshot {
                    layers: vec![self.0.clone()],
                })
            }
        }

        let verifier = Arc::new(IntentPolicySignatureVerifier::new(true));
        verifier.upsert_shared_key("ops", "secret");
        let pack = IntentPolicyPack::default_builtin().unwrap();
        let layer = IntentPolicyLayer {
            kind: IntentPolicyLayerKind::Deployment,
            source: "signed:test".into(),
            pack,
            modified_at: None,
            signature: Some(IntentPolicySignature {
                signer: "ops".into(),
                signature: "bad".into(),
            }),
        };
        let ia = RuleBasedIntentAnalyzer::new("u1", "a1")
            .expect("default intent policy must compile")
            .with_policy_provider(Arc::new(SignedProvider(layer)))
            .with_signature_verifier(verifier);
        let report = ia.reload_from_provider();
        assert!(matches!(
            report.status,
            IntentPolicyReloadStatus::Rejected { .. }
        ));
        assert_eq!(ia.active_policy().id, "context-db.default.intent");
    }
}
