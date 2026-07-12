use crate::execution::{ExecutionContext, ExecutionRequest, ExecutionResponse};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyEffect {
    Allow,
    Deny,
    Require,
    RewriteRequest,
    RewriteResponse,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyStrength {
    Soft,
    Hard,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicySource {
    Persona,
    System,
    ExplicitUser,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PolicyPriority(pub i32);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    pub effect: PolicyEffect,
    pub strength: PolicyStrength,
    pub source: PolicySource,
    pub priority: PolicyPriority,
    pub operation: Option<String>,
    pub content_contains: Option<String>,
    pub replacement: Option<String>,
    pub reason: String,
}

impl PolicyRule {
    fn matches(&self, request: &ExecutionRequest) -> bool {
        self.operation
            .as_ref()
            .is_none_or(|v| v == &request.operation)
            && self
                .content_contains
                .as_ref()
                .is_none_or(|v| request.content.contains(v))
    }
    fn rank(&self) -> (u8, i32, &str) {
        let precedence = match (self.strength, self.effect, self.source) {
            (PolicyStrength::Hard, PolicyEffect::Deny, _) => 4,
            (PolicyStrength::Hard, PolicyEffect::Require, _) => 3,
            (_, _, PolicySource::ExplicitUser) => 2,
            (_, _, PolicySource::Persona) => 0,
            _ => 1,
        };
        (precedence, self.priority.0, self.id.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyAudit {
    pub rule_id: String,
    pub matched: bool,
    pub selected: bool,
    pub reason: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub allowed: bool,
    pub required: bool,
    pub policy_version: u64,
    pub request: ExecutionRequest,
    pub response: Option<ExecutionResponse>,
    pub selected_rule: Option<String>,
    pub audit: Vec<PolicyAudit>,
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("execution denied by policy: {0}")]
    Denied(String),
    #[error("policy requirement not satisfied: {0}")]
    Requirement(String),
    #[error("policy evaluation failed: {0}")]
    Evaluation(String),
}

pub trait ExecutionGate: Send + Sync {
    fn version(&self) -> u64;
    fn preflight(
        &self,
        context: &ExecutionContext,
        request: ExecutionRequest,
    ) -> Result<PolicyDecision, PolicyError>;
    fn postflight(
        &self,
        context: &ExecutionContext,
        decision: &PolicyDecision,
        response: ExecutionResponse,
    ) -> Result<ExecutionResponse, PolicyError>;
}

#[derive(Debug, Clone)]
pub struct RuleExecutionGate {
    version: u64,
    rules: Vec<PolicyRule>,
}
impl RuleExecutionGate {
    pub fn new(version: u64, rules: Vec<PolicyRule>) -> Self {
        Self { version, rules }
    }
}
impl ExecutionGate for RuleExecutionGate {
    fn version(&self) -> u64 {
        self.version
    }
    fn preflight(
        &self,
        _context: &ExecutionContext,
        mut request: ExecutionRequest,
    ) -> Result<PolicyDecision, PolicyError> {
        let mut matching: Vec<&PolicyRule> =
            self.rules.iter().filter(|r| r.matches(&request)).collect();
        matching.sort_by(|a, b| {
            let ord = b.rank().cmp(&a.rank());
            if ord == Ordering::Equal {
                a.id.cmp(&b.id)
            } else {
                ord
            }
        });
        let selected = matching.first().copied();
        if let Some(rule) = selected.filter(|r| r.effect == PolicyEffect::RewriteRequest)
            && let (Some(needle), Some(replacement)) = (&rule.content_contains, &rule.replacement)
        {
            request.content = request.content.replace(needle, replacement);
        }
        let allowed = !matches!(selected.map(|r| r.effect), Some(PolicyEffect::Deny));
        let required = matches!(selected.map(|r| r.effect), Some(PolicyEffect::Require));
        let selected_id = selected.map(|r| r.id.clone());
        let audit = self
            .rules
            .iter()
            .map(|r| {
                let matched = r.matches(&request);
                PolicyAudit {
                    rule_id: r.id.clone(),
                    matched,
                    selected: selected_id.as_ref() == Some(&r.id),
                    reason: r.reason.clone(),
                }
            })
            .collect();
        Ok(PolicyDecision {
            allowed,
            required,
            policy_version: self.version,
            request,
            response: None,
            selected_rule: selected_id,
            audit,
        })
    }
    fn postflight(
        &self,
        _context: &ExecutionContext,
        decision: &PolicyDecision,
        mut response: ExecutionResponse,
    ) -> Result<ExecutionResponse, PolicyError> {
        if !decision.allowed {
            return Err(PolicyError::Denied(
                decision.selected_rule.clone().unwrap_or_default(),
            ));
        }
        let mut rules: Vec<_> = self
            .rules
            .iter()
            .filter(|r| r.effect == PolicyEffect::RewriteResponse && r.matches(&decision.request))
            .collect();
        rules.sort_by(|a, b| b.rank().cmp(&a.rank()));
        if let Some(rule) = rules.first()
            && let (Some(needle), Some(replacement)) = (&rule.content_contains, &rule.replacement)
        {
            response.content = response.content.replace(needle, replacement);
        }
        Ok(response)
    }
}

#[derive(Clone)]
pub struct GatedExecutor {
    gate: Arc<dyn ExecutionGate>,
}
impl GatedExecutor {
    pub fn new(gate: Arc<dyn ExecutionGate>) -> Self {
        Self { gate }
    }
    pub fn authorize(
        &self,
        context: &ExecutionContext,
        request: ExecutionRequest,
    ) -> Result<PolicyDecision, PolicyError> {
        let decision = self.gate.preflight(context, request)?;
        if decision.allowed {
            Ok(decision)
        } else {
            Err(PolicyError::Denied(
                decision.selected_rule.unwrap_or_default(),
            ))
        }
    }
    pub fn finish(
        &self,
        context: &ExecutionContext,
        decision: &PolicyDecision,
        response: ExecutionResponse,
    ) -> Result<ExecutionResponse, PolicyError> {
        self.gate.postflight(context, decision, response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionKind;
    use std::collections::BTreeMap;
    fn ctx() -> ExecutionContext {
        ExecutionContext {
            tenant_id: "t".into(),
            actor_id: "a".into(),
            session_id: None,
            request_id: "r".into(),
            kind: ExecutionKind::Tool { name: "x".into() },
            attributes: BTreeMap::new(),
        }
    }
    fn rule(
        id: &str,
        effect: PolicyEffect,
        strength: PolicyStrength,
        source: PolicySource,
    ) -> PolicyRule {
        PolicyRule {
            id: id.into(),
            effect,
            strength,
            source,
            priority: PolicyPriority(0),
            operation: None,
            content_contains: None,
            replacement: None,
            reason: id.into(),
        }
    }
    #[test]
    fn deterministic_precedence() {
        let g = RuleExecutionGate::new(
            1,
            vec![
                rule(
                    "user",
                    PolicyEffect::Allow,
                    PolicyStrength::Soft,
                    PolicySource::ExplicitUser,
                ),
                rule(
                    "deny",
                    PolicyEffect::Deny,
                    PolicyStrength::Hard,
                    PolicySource::System,
                ),
            ],
        );
        let d = g
            .preflight(&ctx(), ExecutionRequest::new("run", "x"))
            .unwrap();
        assert!(!d.allowed);
        assert_eq!(d.selected_rule.as_deref(), Some("deny"));
    }
}
