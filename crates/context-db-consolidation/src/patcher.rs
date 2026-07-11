//! SkillPatcher — 运行时运行时 patch。

use crate::ConsolidationProduct;
use agent_context_db_core::{ContextEntry, ContextUri};

pub struct SkillPatcher;

#[derive(Debug, Clone)]
pub struct PatchResult {
    pub applied: bool,
    pub patch_summary: String,
    pub affected_uris: Vec<ContextUri>,
}

impl Default for SkillPatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillPatcher {
    pub fn new() -> Self {
        Self
    }
    pub fn patch(
        &self,
        product: &ConsolidationProduct,
        existing: Option<&ContextEntry>,
    ) -> PatchResult {
        let existing = match existing {
            Some(e) => e,
            None => {
                return PatchResult {
                    applied: false,
                    patch_summary: "no existing entry".into(),
                    affected_uris: vec![],
                };
            }
        };
        if existing.content_type() != Some(product.content_type) {
            return PatchResult {
                applied: false,
                patch_summary: "type mismatch".into(),
                affected_uris: vec![],
            };
        }
        let existing_q = existing.metadata.quality_score.unwrap_or(0.5);
        if product.quality_score > existing_q + 0.1 {
            PatchResult {
                applied: true,
                patch_summary: format!("quality {existing_q:.2} → {:.2}", product.quality_score),
                affected_uris: vec![existing.uri.clone()],
            }
        } else {
            PatchResult {
                applied: false,
                patch_summary: "quality not improved enough".into(),
                affected_uris: vec![],
            }
        }
    }
}
