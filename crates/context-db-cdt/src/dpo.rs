//! KnowledgeConstrainedDPO — 偏好loss + 矛盾 penalty。

/// 知识约束 DPO 损失计算器。
pub struct KnowledgeConstrainedDPO {
    pub beta: f64,                  // KL 约束强度
    pub contradiction_penalty: f64, // 矛盾惩罚系数
}

impl KnowledgeConstrainedDPO {
    pub fn new(beta: f64, contradiction_penalty: f64) -> Self {
        Self {
            beta,
            contradiction_penalty,
        }
    }

    /// 偏好loss + 矛盾 penalty。
    /// chosen 矛盾更少 → penalty 为负（奖励)；rejected 矛盾更多 → penalty 为正（惩罚）。
    pub fn loss(
        &self,
        policy_logp_chosen: f64,
        policy_logp_rejected: f64,
        ref_logp_chosen: f64,
        ref_logp_rejected: f64,
        contradiction_delta: i32,
    ) -> f64 {
        // 标准 偏好loss
        let pi_ratio_chosen = (policy_logp_chosen - ref_logp_chosen).exp();
        let pi_ratio_rejected = (policy_logp_rejected - ref_logp_rejected).exp();
        let dpo_loss = -(pi_ratio_chosen / (pi_ratio_chosen + pi_ratio_rejected)).ln();

        // 矛盾 penalty
        let penalty = self.contradiction_penalty * contradiction_delta as f64;

        dpo_loss + penalty
    }

    /// 简化版：直接计算偏好对的 loss。
    pub fn pair_loss(
        &self,
        chosen_confidence: f32,
        rejected_confidence: f32,
        chosen_contradictions: usize,
        rejected_contradictions: usize,
    ) -> f64 {
        let logp_c = (chosen_confidence as f64).ln();
        let logp_r = (rejected_confidence as f64).ln();
        let ref_logp = 0.5_f64.ln(); // reference model 均匀分布
        self.loss(
            logp_c,
            logp_r,
            ref_logp,
            ref_logp,
            rejected_contradictions as i32 - chosen_contradictions as i32,
        )
    }
}
