//! KnowledgeConstrainedDPO — 标准 DPO loss + 认知矛盾 penalty。

use crate::TrajectorySummary;

/// 一组显式 log-prob 输入。
#[derive(Debug, Clone, Copy)]
pub struct PreferenceLogProbs {
    pub policy_chosen: f64,
    pub policy_rejected: f64,
    pub reference_chosen: f64,
    pub reference_rejected: f64,
}

/// 轨迹级 reward/logit 分数，不伪装成模型 log-prob。
#[derive(Debug, Clone, Copy)]
pub struct PreferenceScores {
    pub chosen: f64,
    pub rejected: f64,
    pub reference_chosen: f64,
    pub reference_rejected: f64,
}

/// 知识约束 DPO 损失计算器。
pub struct KnowledgeConstrainedDPO {
    pub beta: f64,
    pub contradiction_penalty: f64,
}

impl KnowledgeConstrainedDPO {
    pub fn new(beta: f64, contradiction_penalty: f64) -> Self {
        Self {
            beta,
            contradiction_penalty,
        }
    }

    /// 标准 DPO: -log sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))。
    pub fn loss_from_log_probs(
        &self,
        log_probs: PreferenceLogProbs,
        contradiction_delta: i32,
    ) -> f64 {
        let policy_margin = log_probs.policy_chosen - log_probs.policy_rejected;
        let reference_margin = log_probs.reference_chosen - log_probs.reference_rejected;
        self.margin_loss(policy_margin, reference_margin, contradiction_delta)
    }

    /// 兼容底层调用：参数必须是真实 log-prob，不接受 confidence 代替。
    pub fn loss(
        &self,
        policy_logp_chosen: f64,
        policy_logp_rejected: f64,
        ref_logp_chosen: f64,
        ref_logp_rejected: f64,
        contradiction_delta: i32,
    ) -> f64 {
        self.loss_from_log_probs(
            PreferenceLogProbs {
                policy_chosen: policy_logp_chosen,
                policy_rejected: policy_logp_rejected,
                reference_chosen: ref_logp_chosen,
                reference_rejected: ref_logp_rejected,
            },
            contradiction_delta,
        )
    }

    /// 用轨迹质量 reward/logit 计算 DPO 形式的 pair loss。
    ///
    /// 这是没有模型 token log-prob 时的认知偏好优化入口；它使用显式 reward score，
    /// 不把 confidence 取对数伪装成 log-prob。
    pub fn loss_from_scores(&self, scores: PreferenceScores, contradiction_delta: i32) -> f64 {
        let policy_margin = scores.chosen - scores.rejected;
        let reference_margin = scores.reference_chosen - scores.reference_rejected;
        self.margin_loss(policy_margin, reference_margin, contradiction_delta)
    }

    pub fn trajectory_pair_loss(
        &self,
        chosen: &TrajectorySummary,
        rejected: &TrajectorySummary,
    ) -> f64 {
        let scores = PreferenceScores {
            chosen: trajectory_reward(chosen),
            rejected: trajectory_reward(rejected),
            reference_chosen: reference_reward(chosen),
            reference_rejected: reference_reward(rejected),
        };
        self.loss_from_scores(
            scores,
            rejected.contradictions as i32 - chosen.contradictions as i32,
        )
    }

    /// 旧调用方的轻量入口：输入解释为 trajectory quality signal，不再取 ln。
    pub fn pair_loss(
        &self,
        chosen_confidence: f32,
        rejected_confidence: f32,
        chosen_contradictions: usize,
        rejected_contradictions: usize,
    ) -> f64 {
        let chosen = reward_from_parts(chosen_confidence, chosen_contradictions, true, 1);
        let rejected = reward_from_parts(rejected_confidence, rejected_contradictions, false, 1);
        let reference_chosen =
            reference_reward_from_parts(chosen_confidence, chosen_contradictions, true, 1);
        let reference_rejected =
            reference_reward_from_parts(rejected_confidence, rejected_contradictions, false, 1);
        self.loss_from_scores(
            PreferenceScores {
                chosen,
                rejected,
                reference_chosen,
                reference_rejected,
            },
            rejected_contradictions as i32 - chosen_contradictions as i32,
        )
    }

    fn margin_loss(
        &self,
        policy_margin: f64,
        reference_margin: f64,
        contradiction_delta: i32,
    ) -> f64 {
        let preference_margin = self.beta * (policy_margin - reference_margin);
        let dpo_loss = -log_sigmoid(preference_margin);
        let penalty = self.contradiction_penalty * contradiction_delta as f64;
        dpo_loss + penalty
    }
}

fn trajectory_reward(summary: &TrajectorySummary) -> f64 {
    reward_from_parts(
        summary.avg_confidence,
        summary.contradictions,
        summary.success,
        summary.steps,
    )
}

fn reference_reward(summary: &TrajectorySummary) -> f64 {
    reference_reward_from_parts(
        summary.avg_confidence,
        summary.contradictions,
        summary.success,
        summary.steps,
    )
}

fn reward_from_parts(confidence: f32, contradictions: usize, success: bool, steps: usize) -> f64 {
    let confidence = confidence.clamp(0.0, 1.0) as f64;
    let success_bonus = if success { 0.35 } else { -0.15 };
    let contradiction_penalty = contradictions as f64 * 0.08;
    let efficiency = 1.0 / (steps.max(1) as f64).ln().max(1.0);
    confidence * 1.2 + success_bonus + efficiency * 0.15 - contradiction_penalty
}

fn reference_reward_from_parts(
    confidence: f32,
    contradictions: usize,
    success: bool,
    steps: usize,
) -> f64 {
    let confidence = confidence.clamp(0.0, 1.0) as f64;
    let success_bonus = if success { 0.15 } else { 0.0 };
    let contradiction_penalty = contradictions as f64 * 0.04;
    let efficiency = 1.0 / (steps.max(1) as f64).ln().max(1.0);
    confidence * 0.6 + success_bonus + efficiency * 0.05 - contradiction_penalty
}

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_prob_loss_prefers_chosen_margin() {
        let dpo = KnowledgeConstrainedDPO::new(1.0, 0.0);
        let good = dpo.loss_from_log_probs(
            PreferenceLogProbs {
                policy_chosen: -0.1,
                policy_rejected: -2.0,
                reference_chosen: -0.5,
                reference_rejected: -1.0,
            },
            0,
        );
        let bad = dpo.loss_from_log_probs(
            PreferenceLogProbs {
                policy_chosen: -2.0,
                policy_rejected: -0.1,
                reference_chosen: -0.5,
                reference_rejected: -1.0,
            },
            0,
        );
        assert!(good < bad);
    }

    #[test]
    fn trajectory_pair_loss_does_not_require_confidence_log() {
        let dpo = KnowledgeConstrainedDPO::new(1.0, 0.0);
        let chosen = TrajectorySummary {
            task_id: "chosen".into(),
            task_description: "good".into(),
            success: true,
            steps: 2,
            contradictions: 0,
            avg_confidence: 0.9,
        };
        let rejected = TrajectorySummary {
            task_id: "rejected".into(),
            task_description: "bad".into(),
            success: false,
            steps: 5,
            contradictions: 3,
            avg_confidence: 0.2,
        };
        assert!(dpo.trajectory_pair_loss(&chosen, &rejected).is_finite());
        assert!(
            dpo.trajectory_pair_loss(&chosen, &rejected)
                < dpo.trajectory_pair_loss(&rejected, &chosen)
        );
    }
}
