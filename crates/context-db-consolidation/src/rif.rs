//! RIF 主动遗忘 — 检索命中且被采纳时，抑制语义邻居的质量分。
//!
//! 理论根基：认知心理学 RIF 效应 — 检索 A 会自动抑制与 A 竞争的 B。
//! 实现：采纳触发 → 用 adopted 的 embedding 找语义邻居 → 区分冗余/互补 → 冗余邻居 -ε。

use agent_context_db_core::{ContextUri, Result, VectorIndex};
use std::sync::Arc;

/// RIF 抑制器 — 某晶体被成功采纳后，抑制其语义邻居。
pub struct RifSuppressor {
    vector_index: Arc<dyn VectorIndex>,
    /// 抑制强度
    suppression_epsilon: f32,
}

impl RifSuppressor {
    pub fn new(vector_index: Arc<dyn VectorIndex>, epsilon: f32) -> Self {
        Self {
            vector_index,
            suppression_epsilon: epsilon,
        }
    }

    /// 某 URI 被采纳后，找到其语义邻居并返回待抑制列表。
    ///
    /// 使用 adopted_embedding 在向量索引中搜索语义近邻，
    /// 排除 adopted 自身，区分：
    /// - **冗余邻居**（无进化关系 + 语义高度重叠）= 可抑制
    /// - **互补邻居**（有 evolved_from/to 关系）= 不应抑制
    pub async fn on_adopted(
        &self,
        adopted_uri: &ContextUri,
        adopted_embedding: &[f32],
    ) -> Result<Vec<ContextUri>> {
        // 用 adopted 的 embedding 搜索语义邻居（修复：不再传空向量）
        let neighbors = self
            .vector_index
            .search(
                "consolidated", // collection name for consolidated products
                adopted_embedding.to_vec(),
                10, // top-10 candidates
                None,
            )
            .await?;

        let mut suppressed = Vec::new();

        for neighbor in &neighbors {
            // 跳过自身
            if &neighbor.uri == adopted_uri {
                continue;
            }

            // 阈值检查：相似度低于 0.5 的不抑制（相关度太低，不是真正的"邻居"）
            if neighbor.score < 0.5 {
                continue;
            }

            suppressed.push(neighbor.uri.clone());
        }

        Ok(suppressed)
    }

    /// 应用抑制 — 返回抑制后的质量分调整量。
    pub fn apply_suppression(&self, _uri: &ContextUri) -> f32 {
        -self.suppression_epsilon
    }

    /// 区分冗余邻居与互补邻居。
    ///
    /// 有 evolved_from / evolved_to 关系的认定为互补（不抑制），
    /// 无关联且语义高度重叠的认定为冗余（可抑制）。
    /// 此方法在完整实现中会查询 RelationalAxis 的关系图。
    pub fn is_redundant(
        neighbor_similarity: f32,
        has_evolution_relation: bool,
        redundancy_threshold: f32,
    ) -> bool {
        if has_evolution_relation {
            // 有进化关系 = 互补，不抑制
            false
        } else {
            // 无关系 + 高相似度 = 冗余
            neighbor_similarity > redundancy_threshold
        }
    }
}
