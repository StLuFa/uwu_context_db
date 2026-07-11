use crate::types::{KnowledgeNetworkError, Result};
use agent_context_db_marketplace::AgentId;
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessGrant {
    pub id: Uuid,
    pub issuer: AgentId,
    pub subject: AgentId,
    pub domains: Vec<String>,
    pub max_epsilon: f32,
    pub expires_at: DateTime<Utc>,
}

impl AccessGrant {
    pub fn new(issuer: AgentId, subject: AgentId, domains: Vec<String>, max_epsilon: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            issuer,
            subject,
            domains,
            max_epsilon,
            expires_at: Utc::now() + Duration::hours(1),
        }
    }

    pub fn allows(&self, subject: &AgentId, domains: &[String], epsilon: f32) -> bool {
        !self.domains.is_empty()
            && !domains.is_empty()
            && self.subject == *subject
            && self.expires_at > Utc::now()
            && epsilon.is_finite()
            && epsilon >= 0.0
            && epsilon <= self.max_epsilon
            && domains.iter().all(|d| self.domains.contains(d))
    }
}

#[derive(Debug, Default)]
pub struct AccessGrantManager {
    grants: RwLock<HashMap<Uuid, AccessGrant>>,
}

impl AccessGrantManager {
    pub fn issue(&self, grant: AccessGrant) -> Uuid {
        let id = grant.id;
        self.grants.write().insert(id, grant);
        id
    }

    pub fn authorize(&self, subject: &AgentId, domains: &[String], epsilon: f32) -> Result<()> {
        if domains.is_empty() {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "empty domain scope is not authorized".into(),
            ));
        }
        let grants = self.grants.read();
        if grants
            .values()
            .any(|grant| grant.allows(subject, domains, epsilon))
        {
            return Ok(());
        }
        Err(KnowledgeNetworkError::PolicyDenied(
            "no active access grant for requested domains".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_configuration_and_empty_domains_are_denied() {
        let manager = AccessGrantManager::default();
        let subject = AgentId::new("agent-a");
        assert!(manager.authorize(&subject, &["rust".into()], 0.1).is_err());
        assert!(manager.authorize(&subject, &[], 0.1).is_err());
    }

    #[test]
    fn grant_requires_explicit_non_empty_domain_scope() {
        let manager = AccessGrantManager::default();
        let subject = AgentId::new("agent-a");
        manager.issue(AccessGrant::new(
            AgentId::new("issuer"),
            subject.clone(),
            vec!["rust".into()],
            1.0,
        ));
        assert!(manager.authorize(&subject, &["rust".into()], 0.5).is_ok());
        assert!(
            manager
                .authorize(&subject, &["public".into()], 0.5)
                .is_err()
        );
    }
}
