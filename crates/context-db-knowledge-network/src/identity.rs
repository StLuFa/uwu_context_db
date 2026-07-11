use crate::types::{KnowledgeNetworkError, Result};
use agent_context_db_marketplace::{
    AgentId, KnowledgeProvenance, KnowledgeProvenancePayload, KnowledgeSigner,
    provenance_payload_hash,
};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMeshPayload<T> {
    pub signer: AgentId,
    pub issued_at: DateTime<Utc>,
    pub payload: T,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Default)]
pub struct IdentityRegistry {
    signing_keys: RwLock<HashMap<AgentId, SigningKey>>,
    verifying_keys: RwLock<HashMap<AgentId, VerifyingKey>>,
}

impl IdentityRegistry {
    pub fn upsert_signing_key(&self, agent: AgentId, secret_key: [u8; 32]) {
        let signing_key = SigningKey::from_bytes(&secret_key);
        self.verifying_keys
            .write()
            .insert(agent.clone(), signing_key.verifying_key());
        self.signing_keys.write().insert(agent, signing_key);
    }

    /// Explicitly registers a trust anchor. Verification never learns keys from messages.
    pub fn register_public_key(&self, agent: AgentId, public_key_hex: &str) -> Result<()> {
        let verifying_key = verifying_key_from_hex(public_key_hex)?;
        let mut keys = self.verifying_keys.write();
        if let Some(existing) = keys.get(&agent)
            && existing.to_bytes() != verifying_key.to_bytes()
        {
            return Err(KnowledgeNetworkError::PolicyDenied(format!(
                "refusing to replace trust anchor for mesh identity: {agent}"
            )));
        }
        keys.insert(agent, verifying_key);
        Ok(())
    }

    pub fn public_key_hex(&self, agent: &AgentId) -> Result<String> {
        let keys = self.verifying_keys.read();
        let key = keys.get(agent).ok_or_else(|| {
            KnowledgeNetworkError::PolicyDenied(format!("unknown mesh identity: {agent}"))
        })?;
        Ok(hex::encode(key.to_bytes()))
    }

    pub fn sign<T: Serialize>(&self, signer: AgentId, payload: T) -> Result<SignedMeshPayload<T>> {
        let issued_at = Utc::now();
        let signing_key = self.signing_key(&signer)?;
        let bytes = mesh_signing_bytes(&signer, issued_at, &payload)?;
        let signature = signing_key.sign(&bytes);
        Ok(SignedMeshPayload {
            signer,
            issued_at,
            payload,
            public_key: hex::encode(signing_key.verifying_key().to_bytes()),
            signature: format!("ed25519:{}", hex::encode(signature.to_bytes())),
        })
    }

    pub fn verify<T: Serialize>(&self, signed: &SignedMeshPayload<T>) -> Result<()> {
        let bytes = mesh_signing_bytes(&signed.signer, signed.issued_at, &signed.payload)?;
        let verifying_key = self.verifying_key(&signed.signer, &signed.public_key)?;
        verify_ed25519(&verifying_key, &bytes, &signed.signature)
    }

    pub fn sign_knowledge_provenance(
        &self,
        payload: &KnowledgeProvenancePayload,
    ) -> Result<KnowledgeProvenance> {
        let signing_key = self.signing_key(&payload.publisher)?;
        let bytes = knowledge_signing_bytes(payload)?;
        let signature = signing_key.sign(&bytes);
        Ok(KnowledgeProvenance {
            publisher: payload.publisher.clone(),
            public_key: hex::encode(signing_key.verifying_key().to_bytes()),
            signature: format!("ed25519:{}", hex::encode(signature.to_bytes())),
            evidence_chain_hash: payload.evidence_chain_hash.clone(),
            signed_at: payload.created_at,
        })
    }

    pub fn verify_knowledge_provenance(
        &self,
        payload: &KnowledgeProvenancePayload,
        provenance: &KnowledgeProvenance,
    ) -> Result<()> {
        if provenance.publisher != payload.publisher {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "knowledge provenance publisher mismatch".into(),
            ));
        }
        if provenance.evidence_chain_hash != payload.evidence_chain_hash {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "knowledge provenance evidence chain mismatch".into(),
            ));
        }
        let bytes = knowledge_signing_bytes(payload)?;
        let verifying_key = self.verifying_key(&payload.publisher, &provenance.public_key)?;
        verify_ed25519(&verifying_key, &bytes, &provenance.signature)
    }

    fn signing_key(&self, signer: &AgentId) -> Result<SigningKey> {
        let keys = self.signing_keys.read();
        keys.get(signer).cloned().ok_or_else(|| {
            KnowledgeNetworkError::PolicyDenied(format!(
                "missing private key for mesh identity: {signer}"
            ))
        })
    }

    fn verifying_key(&self, signer: &AgentId, provided_public_key: &str) -> Result<VerifyingKey> {
        let provided = verifying_key_from_hex(provided_public_key)?;
        let registered = self
            .verifying_keys
            .read()
            .get(signer)
            .copied()
            .ok_or_else(|| {
                KnowledgeNetworkError::PolicyDenied(format!(
                    "unknown mesh identity; explicit registration required: {signer}"
                ))
            })?;
        if registered.to_bytes() != provided.to_bytes() {
            return Err(KnowledgeNetworkError::PolicyDenied(format!(
                "public key mismatch for mesh identity: {signer}"
            )));
        }
        Ok(registered)
    }
}

impl KnowledgeSigner for IdentityRegistry {
    fn sign_knowledge_provenance(
        &self,
        payload: &KnowledgeProvenancePayload,
    ) -> std::result::Result<KnowledgeProvenance, String> {
        IdentityRegistry::sign_knowledge_provenance(self, payload).map_err(|err| err.to_string())
    }
}

fn mesh_signing_bytes<T: Serialize>(
    signer: &AgentId,
    issued_at: DateTime<Utc>,
    payload: &T,
) -> Result<Vec<u8>> {
    serde_json::to_vec(&("mesh-payload-v1", signer, issued_at, payload)).map_err(|err| {
        KnowledgeNetworkError::PolicyDenied(format!("payload signing failed: {err}"))
    })
}

fn knowledge_signing_bytes(payload: &KnowledgeProvenancePayload) -> Result<Vec<u8>> {
    let hash = provenance_payload_hash(payload).map_err(|err| {
        KnowledgeNetworkError::PolicyDenied(format!("knowledge provenance hashing failed: {err}"))
    })?;
    Ok(format!("knowledge-provenance-v1:{hash}").into_bytes())
}

fn verifying_key_from_hex(public_key_hex: &str) -> Result<VerifyingKey> {
    let bytes = hex::decode(public_key_hex).map_err(|err| {
        KnowledgeNetworkError::PolicyDenied(format!("invalid ed25519 public key: {err}"))
    })?;
    let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
        KnowledgeNetworkError::PolicyDenied("invalid ed25519 public key length".into())
    })?;
    VerifyingKey::from_bytes(&bytes).map_err(|err| {
        KnowledgeNetworkError::PolicyDenied(format!("invalid ed25519 public key: {err}"))
    })
}

fn verify_ed25519(verifying_key: &VerifyingKey, bytes: &[u8], signature: &str) -> Result<()> {
    let signature_hex = signature.strip_prefix("ed25519:").ok_or_else(|| {
        KnowledgeNetworkError::PolicyDenied("unsupported signature algorithm".into())
    })?;
    let signature_bytes = hex::decode(signature_hex).map_err(|err| {
        KnowledgeNetworkError::PolicyDenied(format!("invalid ed25519 signature: {err}"))
    })?;
    let signature = Signature::from_slice(&signature_bytes).map_err(|err| {
        KnowledgeNetworkError::PolicyDenied(format!("invalid ed25519 signature: {err}"))
    })?;
    verifying_key
        .verify(bytes, &signature)
        .map_err(|_| KnowledgeNetworkError::PolicyDenied("mesh payload signature mismatch".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, ContextUri, EpistemicType};
    use agent_context_db_marketplace::evidence_chain_hash;

    #[test]
    fn ed25519_mesh_payload_rejects_tampering() {
        let registry = IdentityRegistry::default();
        let agent = AgentId::new("agent-a");
        registry.upsert_signing_key(agent.clone(), [7u8; 32]);

        let signed = registry.sign(agent, "original payload").unwrap();
        registry.verify(&signed).unwrap();

        let mut tampered = signed.clone();
        tampered.payload = "tampered payload";
        assert!(registry.verify(&tampered).is_err());
    }

    #[test]
    fn knowledge_provenance_requires_registration_and_rejects_content_tamper() {
        let registry = IdentityRegistry::default();
        let agent = AgentId::new("agent-a");
        registry.upsert_signing_key(agent.clone(), [9u8; 32]);
        let evidence_uris = vec![ContextUri::parse("uwu://tenant/evidence/1").unwrap()];
        let payload = KnowledgeProvenancePayload {
            publisher: agent,
            content: "verified principle".into(),
            evidence_chain_hash: evidence_chain_hash(&evidence_uris),
            evidence_uris,
            quality_score: 0.91,
            confidence: 0.84,
            epistemic_type: EpistemicType::Fact,
            content_type: ContentType::Fact,
            created_at: Utc::now(),
        };

        let provenance = registry.sign_knowledge_provenance(&payload).unwrap();
        registry
            .verify_knowledge_provenance(&payload, &provenance)
            .unwrap();

        let untrusted = IdentityRegistry::default();
        assert!(
            untrusted
                .verify_knowledge_provenance(&payload, &provenance)
                .is_err()
        );

        let mut tampered = payload.clone();
        tampered.content = "rewritten principle".into();
        assert!(
            registry
                .verify_knowledge_provenance(&tampered, &provenance)
                .is_err()
        );
    }
}
