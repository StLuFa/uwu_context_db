use crate::{Result, VersionError};
use agent_context_db_core::{ContentPayload, ContextEntry, ContextUri};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Backend-neutral semantic contradiction service used by every VersionStore.
/// `threshold` is the minimum detector confidence in the inclusive range [0, 1].
#[async_trait]
pub trait ContradictionDetector: Send + Sync {
    async fn contradiction_confidence(&self, left: &str, right: &str) -> Result<f32>;
}

pub async fn detect_snapshot_contradictions(
    detector: &Arc<dyn ContradictionDetector>,
    from: &HashMap<String, String>,
    into: &HashMap<String, String>,
    threshold: f32,
) -> Result<Vec<ContextUri>> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(VersionError::MergeConflict(format!(
            "contradiction threshold must be finite and within [0, 1], got {threshold}"
        )));
    }
    let mut conflicts = Vec::new();
    for (uri_text, left_json) in from {
        let Some(right_json) = into.get(uri_text) else {
            continue;
        };
        if left_json == right_json {
            continue;
        }
        let left = canonical_payload_semantic_text(left_json)?;
        let right = canonical_payload_semantic_text(right_json)?;
        let confidence = detector.contradiction_confidence(&left, &right).await?;
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(VersionError::MergeConflict(format!(
                "contradiction detector returned invalid confidence {confidence} for {uri_text}"
            )));
        }
        if confidence >= threshold {
            conflicts.push(ContextUri::parse(uri_text).map_err(|error| {
                VersionError::Storage(format!(
                    "snapshot contains invalid URI {uri_text:?}: {error}"
                ))
            })?);
        }
    }
    conflicts.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    Ok(conflicts)
}

fn canonical_payload_semantic_text(entry_json: &str) -> Result<String> {
    let entry: ContextEntry = serde_json::from_str(entry_json).map_err(|error| {
        VersionError::Storage(format!(
            "deserialize snapshot entry for contradiction detection: {error}"
        ))
    })?;
    semantic_text(&entry.payload)
}

fn semantic_text(payload: &ContentPayload) -> Result<String> {
    let projection = payload.index_projection();
    let text = if projection.l2.trim().is_empty() {
        projection.l1.unwrap_or(projection.l0)
    } else {
        projection.l2
    };
    let canonical = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if canonical.is_empty() {
        return Err(VersionError::MergeConflict(
            "contradiction detection requires non-empty canonical semantic text".into(),
        ));
    }
    Ok(canonical)
}
