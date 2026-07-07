//! SecurityGate + ImmuneMemory — 写入前安全检查 + 抗体免疫。

use agent_context_db_core::{ContentType, ContextEntry, ContextUri};

/// 注入扫描结果。
#[derive(Debug, Clone)]
pub enum ThreatVerdict {
    Safe,
    Suspicious {
        reason: String,
    },
    Blocked {
        threat_type: ThreatType,
        confidence: f32,
    },
}

#[derive(Debug, Clone)]
pub enum ThreatType {
    Injection,
    Hallucination,
    Misinformation,
    PromptLeak,
}

/// 写入安全检查门。
pub struct SecurityGate {
    blocked_patterns: Vec<String>,
}

impl SecurityGate {
    pub fn new() -> Self {
        Self {
            blocked_patterns: vec![
                "ignore previous instructions".into(),
                "system prompt:".into(),
                "<|im_start|>".into(),
            ],
        }
    }

    /// 写入前扫描 — 检测注入/有害内容。
    pub fn scan(&self, entry: &ContextEntry) -> ThreatVerdict {
        let text = entry.l0_text().to_lowercase();

        for pattern in &self.blocked_patterns {
            if text.contains(&pattern.to_lowercase()) {
                return ThreatVerdict::Blocked {
                    threat_type: ThreatType::Injection,
                    confidence: 0.9,
                };
            }
        }

        // 空内容检查
        if text.trim().is_empty() {
            return ThreatVerdict::Suspicious {
                reason: "empty content".into(),
            };
        }

        ThreatVerdict::Safe
    }
}

/// 免疫记忆 — 已知有害模式抗体库。
pub struct ImmuneMemory {
    antibodies: parking_lot::RwLock<Vec<Antibody>>,
}

#[derive(Debug, Clone)]
pub struct Antibody {
    pub pattern_signature: String,
    pub threat_type: ThreatType,
    pub confidence: f32,
    pub detected_count: u32,
}

impl ImmuneMemory {
    pub fn new() -> Self {
        Self {
            antibodies: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// 记录一次检测到的威胁 → 生成抗体。
    pub fn record_threat(&self, content: &str, threat_type: ThreatType) {
        let mut antibodies = self.antibodies.write();
        if let Some(existing) = antibodies
            .iter_mut()
            .find(|a| content.contains(&a.pattern_signature))
        {
            existing.detected_count += 1;
            return;
        }
        antibodies.push(Antibody {
            pattern_signature: content.chars().take(100).collect(),
            threat_type,
            confidence: 0.8,
            detected_count: 1,
        });
    }

    /// 检查内容是否匹配已知抗体。
    pub fn check(&self, content: &str) -> ThreatVerdict {
        let antibodies = self.antibodies.read();
        for ab in antibodies.iter() {
            if ab.detected_count >= 2 && content.contains(&ab.pattern_signature) {
                return ThreatVerdict::Blocked {
                    threat_type: ab.threat_type.clone(),
                    confidence: ab.confidence,
                };
            }
        }
        ThreatVerdict::Safe
    }
}
