//! Write-path security helpers shared by storage adapters.

use crate::{ContentPart, ContentPayload, ContextEntry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SensitiveFinding {
    pub kind: SensitiveKind,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensitiveKind {
    Email,
    PhoneOrLongId,
    ApiKey,
    PrivateKey,
}

pub fn scan_sensitive_entry(entry: &ContextEntry) -> Vec<SensitiveFinding> {
    let mut findings = Vec::new();
    scan_payload(&entry.payload, &mut findings);
    findings
}

pub fn sanitize_entry_for_write(entry: &ContextEntry) -> ContextEntry {
    redact_sensitive_entry(entry).0
}

pub fn redact_sensitive_entry(entry: &ContextEntry) -> (ContextEntry, Vec<SensitiveFinding>) {
    let mut redacted = entry.clone();
    let mut findings = Vec::new();
    redact_payload(&mut redacted.payload, &mut findings);
    if !findings.is_empty() {
        let _ = redacted
            .metadata
            .set_custom_field("security_redactions", &findings_to_json(&findings));
    }
    (redacted, findings)
}

fn scan_payload(payload: &ContentPayload, findings: &mut Vec<SensitiveFinding>) {
    match payload {
        ContentPayload::Text {
            sparse,
            dense,
            full,
        } => {
            let _ = redact_text(sparse, findings);
            let _ = redact_text(dense, findings);
            let _ = redact_text(full, findings);
        }
        ContentPayload::Audio { transcript, .. } => {
            let _ = redact_text(transcript, findings);
        }
        ContentPayload::Structured { summary, data, .. } => {
            let _ = redact_text(summary, findings);
            scan_json(data, findings);
        }
        ContentPayload::Composite { summary, parts } => {
            let _ = redact_text(summary, findings);
            for part in parts {
                match part {
                    ContentPart::Text(payload)
                    | ContentPart::Image(payload)
                    | ContentPart::Audio(payload) => scan_payload(payload, findings),
                    ContentPart::Reference(_) => {}
                }
            }
        }
        ContentPayload::Image { .. } => {}
    }
}

fn scan_json(value: &serde_json::Value, findings: &mut Vec<SensitiveFinding>) {
    match value {
        serde_json::Value::String(s) => {
            let _ = redact_text(s, findings);
        }
        serde_json::Value::Array(items) => {
            for item in items {
                scan_json(item, findings);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                scan_json(value, findings);
            }
        }
        _ => {}
    }
}

fn redact_payload(payload: &mut ContentPayload, findings: &mut Vec<SensitiveFinding>) {
    match payload {
        ContentPayload::Text {
            sparse,
            dense,
            full,
        } => {
            *sparse = redact_text(sparse, findings);
            *dense = redact_text(dense, findings);
            *full = redact_text(full, findings);
        }
        ContentPayload::Audio { transcript, .. } => {
            *transcript = redact_text(transcript, findings);
        }
        ContentPayload::Structured { summary, data, .. } => {
            *summary = redact_text(summary, findings);
            redact_json(data, findings);
        }
        ContentPayload::Composite { summary, parts } => {
            *summary = redact_text(summary, findings);
            for part in parts {
                match part {
                    ContentPart::Text(payload)
                    | ContentPart::Image(payload)
                    | ContentPart::Audio(payload) => redact_payload(payload, findings),
                    ContentPart::Reference(_) => {}
                }
            }
        }
        ContentPayload::Image { .. } => {}
    }
}

fn redact_json(value: &mut serde_json::Value, findings: &mut Vec<SensitiveFinding>) {
    match value {
        serde_json::Value::String(s) => *s = redact_text(s, findings),
        serde_json::Value::Array(items) => {
            for item in items {
                redact_json(item, findings);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values_mut() {
                redact_json(value, findings);
            }
        }
        _ => {}
    }
}

fn redact_text(text: &str, findings: &mut Vec<SensitiveFinding>) -> String {
    let mut out = Vec::new();
    for token in text.split_whitespace() {
        out.push(redact_token(token, findings));
    }
    let joined = out.join(" ");
    redact_private_key_blocks(&joined, findings)
}

fn redact_token(token: &str, findings: &mut Vec<SensitiveFinding>) -> String {
    let trimmed = token.trim_matches(|c: char| c.is_ascii_punctuation());
    if looks_like_email(trimmed) {
        push_finding(findings, SensitiveKind::Email);
        return token.replace(trimmed, "[REDACTED_EMAIL]");
    }
    if looks_like_api_key(trimmed) {
        push_finding(findings, SensitiveKind::ApiKey);
        return token.replace(trimmed, "[REDACTED_SECRET]");
    }
    if looks_like_phone_or_long_id(trimmed) {
        push_finding(findings, SensitiveKind::PhoneOrLongId);
        return token.replace(trimmed, "[REDACTED_ID]");
    }
    token.to_string()
}

fn redact_private_key_blocks(text: &str, findings: &mut Vec<SensitiveFinding>) -> String {
    if !(text.contains("-----BEGIN") && text.contains("PRIVATE KEY-----")) {
        return text.to_string();
    }
    push_finding(findings, SensitiveKind::PrivateKey);
    let mut output = String::new();
    let mut in_key = false;
    for line in text.lines() {
        if line.contains("-----BEGIN") && line.contains("PRIVATE KEY-----") {
            in_key = true;
            output.push_str("[REDACTED_PRIVATE_KEY]\n");
            continue;
        }
        if in_key && line.contains("-----END") && line.contains("PRIVATE KEY-----") {
            in_key = false;
            continue;
        }
        if !in_key {
            output.push_str(line);
            output.push('\n');
        }
    }
    output.trim_end().to_string()
}

fn looks_like_email(token: &str) -> bool {
    let Some((local, domain)) = token.split_once('@') else {
        return false;
    };
    !local.is_empty() && domain.contains('.') && domain.len() >= 3
}

fn looks_like_api_key(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    lower.starts_with("sk-")
        || lower.starts_with("pk-")
        || lower.starts_with("ghp_")
        || lower.starts_with("github_pat_")
        || (token.len() >= 24
            && token
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
}

fn looks_like_phone_or_long_id(token: &str) -> bool {
    let digits = token.chars().filter(|c| c.is_ascii_digit()).count();
    digits >= 11
        && token
            .chars()
            .all(|c| c.is_ascii_digit() || "+-() ".contains(c))
}

fn push_finding(findings: &mut Vec<SensitiveFinding>, kind: SensitiveKind) {
    if let Some(existing) = findings.iter_mut().find(|f| f.kind == kind) {
        existing.count += 1;
    } else {
        findings.push(SensitiveFinding { kind, count: 1 });
    }
}

fn findings_to_json(findings: &[SensitiveFinding]) -> Vec<serde_json::Value> {
    findings
        .iter()
        .map(|finding| {
            serde_json::json!({
                "kind": format!("{:?}", finding.kind),
                "count": finding.count,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentPayload, ContextUri, TenantId};

    #[test]
    fn sanitize_entry_redacts_all_text_levels() {
        let mut entry = ContextEntry::new_text(
            ContextUri::parse("uwu://t/agent/a/memories/evidence/e1").unwrap(),
            TenantId(uuid::Uuid::nil()),
            "email user@example.com",
        );
        entry.payload = ContentPayload::Text {
            sparse: "email user@example.com".into(),
            dense: "phone +1-555-123-4567".into(),
            full: "token sk-secret12345678901234567890".into(),
        };

        let sanitized = sanitize_entry_for_write(&entry);
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
    }
}
