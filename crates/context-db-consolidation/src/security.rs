//! SecurityGate + ImmuneMemory — 写入前安全检查 + 抗体免疫。

use agent_context_db_core::{
    ContextEntry, SensitiveFinding, redact_sensitive_entry, scan_sensitive_entry,
};

/// 注入扫描结果。
#[derive(Debug, Clone)]
pub enum ThreatVerdict {
    Safe,
    Redacted {
        findings: Vec<SensitiveFinding>,
    },
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

impl Default for SecurityGate {
    fn default() -> Self {
        Self::new()
    }
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

    /// 写入前扫描 — 检测注入/有害内容与敏感信息。
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

        if text.trim().is_empty() {
            return ThreatVerdict::Suspicious {
                reason: "empty content".into(),
            };
        }

        let findings = scan_sensitive_entry(entry);
        if findings.is_empty() {
            ThreatVerdict::Safe
        } else {
            ThreatVerdict::Redacted { findings }
        }
    }

    pub fn redact_entry(
        &self,
        entry: &ContextEntry,
    ) -> agent_context_db_core::Result<(ContextEntry, Vec<SensitiveFinding>)> {
        redact_sensitive_entry(entry)
    }

    pub fn sanitize_for_write(
        &self,
        entry: &ContextEntry,
    ) -> agent_context_db_core::Result<Result<ContextEntry, ThreatVerdict>> {
        match self.scan(entry) {
            ThreatVerdict::Safe | ThreatVerdict::Suspicious { .. } => Ok(Ok(entry.clone())),
            ThreatVerdict::Redacted { .. } => Ok(Ok(self.redact_entry(entry)?.0)),
            blocked @ ThreatVerdict::Blocked { .. } => Ok(Err(blocked)),
        }
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

impl Default for ImmuneMemory {
    fn default() -> Self {
        Self::new()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentPayload, ContextEntry, ContextUri, TenantId};

    #[test]
    fn security_gate_redacts_pii_across_text_levels() -> agent_context_db_core::Result<()> {
        let gate = SecurityGate::new();
        let mut entry = ContextEntry::new_text(
            ContextUri::parse("uwu://t/agent/a/memories/evidence/e1").unwrap(),
            TenantId(uuid::Uuid::nil()),
            "email me at user@example.com",
        );
        entry.payload = ContentPayload::Text {
            sparse: "email me at user@example.com".into(),
            dense: "phone +1-555-123-4567".into(),
            full: "token sk-secret12345678901234567890".into(),
        };

        assert!(matches!(gate.scan(&entry), ThreatVerdict::Redacted { .. }));
        let sanitized = gate.sanitize_for_write(&entry)?.map_err(|verdict| {
            agent_context_db_core::ContextError::TrustPolicy(format!(
                "entry unexpectedly blocked during sanitization: {verdict:?}"
            ))
        })?;
        let ContentPayload::Text {
            sparse,
            dense,
            full,
        } = sanitized.payload
        else {
            panic!("expected text payload");
        };
        assert!(sparse.contains("[REDACTED_EMAIL]"));
        assert!(dense.contains("[REDACTED_ID]"));
        assert!(full.contains("[REDACTED_SECRET]"));
        assert!(
            sanitized
                .metadata
                .custom
                .get("security_redactions")
                .is_some()
        );
        Ok(())
    }
}
