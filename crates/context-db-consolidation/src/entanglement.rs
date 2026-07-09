//! 知识纠缠检测 — patch 共现率 > 阈值 → 标记 entangled。

use agent_context_db_core::{ContextEntry, ContextUri, GraphRelation, GraphStore, ValidityRecord};
use chrono::Utc;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// 纠缠检测器 — Sleeptime 阶段分析 patch 共现。
pub struct EntanglementDetector {
    entanglements: parking_lot::RwLock<HashMap<String, Vec<Entanglement>>>,
    co_occurrence_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct Entanglement {
    pub partner_uri: ContextUri,
    pub co_occurrence: f32,
    pub direction: EntangleDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntangleDirection {
    Symmetric,
    OneWay,
}

#[derive(Debug, Clone, Copy)]
pub struct CascadeInvalidationConfig {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub min_co_occurrence: f32,
    pub graph_expansion: bool,
}

impl Default for CascadeInvalidationConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_nodes: 128,
            min_co_occurrence: 0.35,
            graph_expansion: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeInvalidationAction {
    Revalidate,
    Invalidate,
}

#[derive(Debug, Clone)]
pub struct CascadeInvalidationTask {
    pub uri: ContextUri,
    pub trigger_uri: ContextUri,
    pub depth: usize,
    pub action: CascadeInvalidationAction,
    pub reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct CascadeInvalidationPlan {
    pub tasks: Vec<CascadeInvalidationTask>,
}

pub struct CascadeInvalidator {
    detector: Arc<EntanglementDetector>,
    graph: Option<Arc<dyn GraphStore>>,
    config: CascadeInvalidationConfig,
}

impl EntanglementDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            entanglements: parking_lot::RwLock::new(HashMap::new()),
            co_occurrence_threshold: threshold,
        }
    }

    /// 记录一次 patch 共现。
    pub fn record_co_patch(&self, uri_a: &ContextUri, uri_b: &ContextUri) {
        self.record_directed(uri_a, uri_b, EntangleDirection::Symmetric);
        self.record_directed(uri_b, uri_a, EntangleDirection::Symmetric);
    }

    pub fn record_dependency(&self, source: &ContextUri, dependent: &ContextUri) {
        self.record_directed(source, dependent, EntangleDirection::OneWay);
    }

    fn record_directed(
        &self,
        uri_a: &ContextUri,
        uri_b: &ContextUri,
        direction: EntangleDirection,
    ) {
        let mut ents = self.entanglements.write();
        let entry = ents.entry(uri_a.to_string()).or_default();

        if let Some(existing) = entry.iter_mut().find(|e| e.partner_uri == *uri_b) {
            existing.co_occurrence = (existing.co_occurrence * 0.8 + 0.2).min(1.0);
            existing.direction = direction;
        } else {
            entry.push(Entanglement {
                partner_uri: uri_b.clone(),
                co_occurrence: 0.2,
                direction,
            });
        }
    }

    /// 检测 A 的所有纠缠伙伴（共现率 > 阈值）。
    pub fn get_entangled(&self, uri: &ContextUri) -> Vec<ContextUri> {
        self.entanglements_for(uri, self.co_occurrence_threshold)
            .into_iter()
            .map(|e| e.partner_uri)
            .collect()
    }

    pub fn entanglements_for(&self, uri: &ContextUri, threshold: f32) -> Vec<Entanglement> {
        self.entanglements
            .read()
            .get(&uri.to_string())
            .map(|list| {
                list.iter()
                    .filter(|e| e.co_occurrence >= threshold)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Sleeptime 阶段：清理低共现率的纠缠（衰减）。
    pub fn decay(&self, decay_rate: f32) {
        let mut ents = self.entanglements.write();
        for list in ents.values_mut() {
            for e in list.iter_mut() {
                e.co_occurrence *= 1.0 - decay_rate;
            }
            list.retain(|e| e.co_occurrence > 0.05);
        }
        ents.retain(|_, list| !list.is_empty());
    }
}

impl CascadeInvalidator {
    pub fn new(detector: Arc<EntanglementDetector>) -> Self {
        Self {
            detector,
            graph: None,
            config: CascadeInvalidationConfig::default(),
        }
    }

    pub fn with_graph(mut self, graph: Arc<dyn GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    pub fn with_config(mut self, config: CascadeInvalidationConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn plan_from_invalidated(&self, trigger_uri: &ContextUri) -> CascadeInvalidationPlan {
        let mut tasks = Vec::new();
        let mut seen = HashSet::from([trigger_uri.clone()]);
        let mut queue = VecDeque::from([(trigger_uri.clone(), 0usize)]);

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= self.config.max_depth || tasks.len() >= self.config.max_nodes {
                continue;
            }

            let next_depth = depth + 1;
            for edge in self
                .detector
                .entanglements_for(&current, self.config.min_co_occurrence)
            {
                if !seen.insert(edge.partner_uri.clone()) {
                    continue;
                }
                let action = if next_depth == 1 && edge.co_occurrence >= 0.80 {
                    CascadeInvalidationAction::Invalidate
                } else {
                    CascadeInvalidationAction::Revalidate
                };
                tasks.push(CascadeInvalidationTask {
                    uri: edge.partner_uri.clone(),
                    trigger_uri: trigger_uri.clone(),
                    depth: next_depth,
                    action,
                    reason: format!(
                        "entangled with {} at co-occurrence {:.2}",
                        current, edge.co_occurrence
                    ),
                });
                queue.push_back((edge.partner_uri, next_depth));
                if tasks.len() >= self.config.max_nodes {
                    break;
                }
            }

            if self.config.graph_expansion {
                self.expand_graph(trigger_uri, &current, next_depth, &mut seen, &mut tasks)
                    .await;
            }
        }

        CascadeInvalidationPlan { tasks }
    }

    pub fn apply_to_entries(
        &self,
        entries: &mut [ContextEntry],
        plan: &CascadeInvalidationPlan,
    ) -> usize {
        let mut task_by_uri = HashMap::new();
        for task in &plan.tasks {
            task_by_uri.insert(task.uri.clone(), task);
        }

        let mut updated = 0usize;
        for entry in entries {
            let Some(task) = task_by_uri.get(&entry.uri) else {
                continue;
            };
            match task.action {
                CascadeInvalidationAction::Invalidate => {
                    entry.metadata.validity = Some(ValidityRecord {
                        valid_from: entry.created_at,
                        valid_until: Some(Utc::now()),
                        invalidated_by: Some(task.trigger_uri.clone()),
                        invalidation_reason: Some(task.reason.clone()),
                    });
                    entry.metadata.tags.push("cascade:invalidated".into());
                }
                CascadeInvalidationAction::Revalidate => {
                    entry.metadata.tags.push("cascade:revalidate".into());
                }
            }
            updated += 1;
        }
        updated
    }

    async fn expand_graph(
        &self,
        trigger_uri: &ContextUri,
        current: &ContextUri,
        depth: usize,
        seen: &mut HashSet<ContextUri>,
        tasks: &mut Vec<CascadeInvalidationTask>,
    ) {
        let Some(graph) = &self.graph else {
            return;
        };
        if tasks.len() >= self.config.max_nodes {
            return;
        }
        let kinds = [
            GraphRelation::EntangledWith,
            GraphRelation::DerivedFrom,
            GraphRelation::Supersedes,
        ];
        let Ok(edges) = graph
            .batch_traverse(std::slice::from_ref(current), &kinds, 1)
            .await
        else {
            return;
        };
        for (_from, to, kind) in edges {
            if !seen.insert(to.clone()) {
                continue;
            }
            tasks.push(CascadeInvalidationTask {
                uri: to,
                trigger_uri: trigger_uri.clone(),
                depth,
                action: CascadeInvalidationAction::Revalidate,
                reason: format!("graph relation {:?} from {}", kind, current),
            });
            if tasks.len() >= self.config.max_nodes {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, StateScope, TenantId};
    use uuid::Uuid;

    fn uri(id: &str) -> ContextUri {
        ContextUri::parse(&format!("uwu://t/a/memory/fact/{id}")).unwrap()
    }

    #[tokio::test]
    async fn cascade_follows_entanglement_with_depth_limit() {
        let detector = Arc::new(EntanglementDetector::new(0.3));
        let a = uri("a");
        let b = uri("b");
        let c = uri("c");
        for _ in 0..5 {
            detector.record_co_patch(&a, &b);
            detector.record_co_patch(&b, &c);
        }
        let invalidator =
            CascadeInvalidator::new(detector).with_config(CascadeInvalidationConfig {
                max_depth: 1,
                max_nodes: 16,
                min_co_occurrence: 0.3,
                graph_expansion: false,
            });

        let plan = invalidator.plan_from_invalidated(&a).await;

        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].uri, b);
        assert_eq!(plan.tasks[0].depth, 1);
    }

    #[tokio::test]
    async fn cascade_applies_invalidation_and_revalidation_tags() {
        let detector = Arc::new(EntanglementDetector::new(0.3));
        let a = uri("a");
        let b = uri("b");
        for _ in 0..8 {
            detector.record_co_patch(&a, &b);
        }
        let invalidator =
            CascadeInvalidator::new(detector).with_config(CascadeInvalidationConfig {
                max_depth: 2,
                max_nodes: 16,
                min_co_occurrence: 0.3,
                graph_expansion: false,
            });
        let mut entry = ContextEntry::new_text(b.clone(), TenantId(Uuid::nil()), "dependent fact");
        entry.metadata.content_type = Some(ContentType::Fact);
        entry.metadata.state_scope = Some(StateScope::Long);
        let mut entries = vec![entry];
        let plan = invalidator.plan_from_invalidated(&a).await;

        let updated = invalidator.apply_to_entries(&mut entries, &plan);

        assert_eq!(updated, 1);
        assert!(entries[0].metadata.validity.is_some());
        assert!(
            entries[0]
                .metadata
                .tags
                .contains(&"cascade:invalidated".to_string())
        );
    }
}
