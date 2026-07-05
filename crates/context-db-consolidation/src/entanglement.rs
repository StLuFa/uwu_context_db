//! 知识纠缠检测 — patch 共现率 > 阈值 → 标记 entangled。

use agent_context_db_core::ContextUri;
use std::collections::HashMap;

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

impl EntanglementDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            entanglements: parking_lot::RwLock::new(HashMap::new()),
            co_occurrence_threshold: threshold,
        }
    }

    /// 记录一次 patch 共现。
    pub fn record_co_patch(&self, uri_a: &ContextUri, uri_b: &ContextUri) {
        let mut ents = self.entanglements.write();
        let entry = ents.entry(uri_a.to_string()).or_default();

        if let Some(existing) = entry.iter_mut().find(|e| e.partner_uri == *uri_b) {
            existing.co_occurrence = (existing.co_occurrence * 0.8 + 0.2).min(1.0);
        } else {
            entry.push(Entanglement {
                partner_uri: uri_b.clone(),
                co_occurrence: 0.2,
                direction: EntangleDirection::Symmetric,
            });
        }
    }

    /// 检测 A 的所有纠缠伙伴（共现率 > 阈值）。
    pub fn get_entangled(&self, uri: &ContextUri) -> Vec<ContextUri> {
        self.entanglements
            .read()
            .get(&uri.to_string())
            .map(|list| {
                list.iter()
                    .filter(|e| e.co_occurrence >= self.co_occurrence_threshold)
                    .map(|e| e.partner_uri.clone())
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
    }
}
