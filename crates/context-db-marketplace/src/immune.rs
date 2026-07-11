//! Immune Memory Protocol — 只共享攻击签名，不共享原始数据。

use crate::types::*;
use agent_context_db_core::{EventMesh, Topic};

/// 抗体 — 攻击模式的特征签名（不是原始 prompt）。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Antibody {
    pub id: MarketId,
    pub pattern_signature: Vec<f32>,
    pub threat_type: ThreatType,
    pub severity: ThreatSeverity,
    pub detected_by: AgentId,
    pub detected_at: chrono::DateTime<chrono::Utc>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ThreatType {
    PromptInjection,
    DataExfiltration,
    Hallucination,
    Misinformation,
    Jailbreak,
    Other,
}

/// 威胁检查结果。
#[derive(Debug, Clone)]
pub enum ThreatCheck {
    Clean,
    Suspicious {
        matched_antibodies: Vec<MarketId>,
        risk_score: f32,
    },
    Blocked {
        matched_antibodies: Vec<MarketId>,
        severity: ThreatSeverity,
    },
}

/// 免疫协议 — 一个 Agent 踩坑，全员免疫。
pub struct ImmuneProtocol {
    antibodies: parking_lot::RwLock<Vec<Antibody>>,
    event_mesh: Option<EventMesh>,
    similarity_threshold: f32,
}

impl Default for ImmuneProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl ImmuneProtocol {
    pub fn new() -> Self {
        Self {
            antibodies: parking_lot::RwLock::new(Vec::new()),
            event_mesh: None,
            similarity_threshold: 0.85,
        }
    }

    pub fn with_event_mesh(mut self, mesh: EventMesh) -> Self {
        self.event_mesh = Some(mesh);
        self
    }

    /// 发布抗体 — 从攻击中提取签名，广播全网。
    pub async fn publish_antibody(
        &self,
        pattern_signature: Vec<f32>,
        threat_type: ThreatType,
        severity: ThreatSeverity,
        detected_by: AgentId,
    ) -> Antibody {
        let antibody = Antibody {
            id: MarketId::new(),
            pattern_signature,
            threat_type,
            severity,
            detected_by: detected_by.clone(),
            detected_at: chrono::Utc::now(),
            confidence: 0.9,
        };

        self.antibodies.write().push(antibody.clone());

        // 广播到 EventMesh → 所有订阅 Agent 自动加载
        if let Some(ref mesh) = self.event_mesh
            && let Ok(payload) = serde_json::to_value(&antibody)
        {
            let mesh = mesh.clone();
            if let Ok(topic) = Topic::new("immune.broadcast") {
                tokio::spawn(async move {
                    let _ = mesh.emit(&topic, payload).await;
                });
            }
        }

        antibody
    }

    /// 加载外部抗体（从 EventMesh 收到广播时调用）。
    pub fn load_antibody(&self, antibody: Antibody) {
        // 去重
        if !self.antibodies.read().iter().any(|a| a.id == antibody.id) {
            self.antibodies.write().push(antibody);
        }
    }

    /// 检查内容是否与已知抗体匹配。
    ///
    /// 用余弦相似度比较输入签名与所有抗体的模式签名。
    pub fn check(&self, content_signature: &[f32]) -> ThreatCheck {
        let antibodies = self.antibodies.read();
        if antibodies.is_empty() {
            return ThreatCheck::Clean;
        }

        let mut matched = Vec::new();
        let mut max_severity = ThreatSeverity::Low;
        let mut total_risk = 0.0;

        for antibody in antibodies.iter() {
            let similarity = cosine_similarity(content_signature, &antibody.pattern_signature);
            if similarity >= self.similarity_threshold {
                matched.push(antibody.id);
                total_risk += similarity * antibody.confidence;
                if antibody.severity > max_severity {
                    max_severity = antibody.severity;
                }
            }
        }

        if matched.is_empty() {
            ThreatCheck::Clean
        } else {
            let risk_score = (total_risk / matched.len() as f32).min(1.0);
            if max_severity >= ThreatSeverity::High {
                ThreatCheck::Blocked {
                    matched_antibodies: matched,
                    severity: max_severity,
                }
            } else {
                ThreatCheck::Suspicious {
                    matched_antibodies: matched,
                    risk_score,
                }
            }
        }
    }

    /// 已加载的抗体数量。
    pub fn antibody_count(&self) -> usize {
        self.antibodies.read().len()
    }

    /// 按威胁类型分组的抗体统计。
    pub fn stats_by_threat(&self) -> std::collections::HashMap<ThreatType, usize> {
        let mut stats = std::collections::HashMap::new();
        for ab in self.antibodies.read().iter() {
            *stats.entry(ab.threat_type).or_insert(0) += 1;
        }
        stats
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..len {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f64::EPSILON);
    (dot / denom) as f32
}
