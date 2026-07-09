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
        self.subject == *subject
            && self.expires_at > Utc::now()
            && epsilon <= self.max_epsilon
            && (self.domains.is_empty() || domains.iter().all(|d| self.domains.contains(d)))
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
            return Ok(());
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

    pub fn has_grants(&self) -> bool {
        !self.grants.read().is_empty()
    }
}
