use crate::types::{KnowledgeNetworkError, Result};
use agent_context_db_marketplace_types::AgentId;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMeshPayload<T> {
    pub signer: AgentId,
    pub issued_at: DateTime<Utc>,
    pub payload: T,
    pub signature: String,
}

#[derive(Debug, Default)]
pub struct IdentityRegistry {
    shared_keys: RwLock<HashMap<AgentId, String>>,
}

impl IdentityRegistry {
    pub fn upsert_shared_key(&self, agent: AgentId, key: impl Into<String>) {
        self.shared_keys.write().insert(agent, key.into());
    }

    pub fn sign<T: Serialize>(&self, signer: AgentId, payload: T) -> Result<SignedMeshPayload<T>> {
        let issued_at = Utc::now();
        let signature = self.signature_for(&signer, &payload, issued_at)?;
        Ok(SignedMeshPayload {
            signer,
            issued_at,
            payload,
            signature,
        })
    }

    pub fn verify<T: Serialize>(&self, signed: &SignedMeshPayload<T>) -> Result<()> {
        let expected = self.signature_for(&signed.signer, &signed.payload, signed.issued_at)?;
        if expected == signed.signature {
            Ok(())
        } else {
            Err(KnowledgeNetworkError::PolicyDenied(
                "mesh payload signature mismatch".into(),
            ))
        }
    }

    fn signature_for<T: Serialize>(
        &self,
        signer: &AgentId,
        payload: &T,
        issued_at: DateTime<Utc>,
    ) -> Result<String> {
        let keys = self.shared_keys.read();
        let key = keys.get(signer).ok_or_else(|| {
            KnowledgeNetworkError::PolicyDenied(format!("unknown mesh identity: {signer}"))
        })?;
        let bytes = serde_json::to_vec(payload).map_err(|err| {
            KnowledgeNetworkError::PolicyDenied(format!("payload signing failed: {err}"))
        })?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        signer.hash(&mut hasher);
        key.hash(&mut hasher);
        issued_at.timestamp_millis().hash(&mut hasher);
        bytes.hash(&mut hasher);
        Ok(format!("kn-sig:{:016x}", hasher.finish()))
    }
}
