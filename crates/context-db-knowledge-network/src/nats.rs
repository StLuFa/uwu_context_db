//! NATS production bridge notes for KnowledgeNetwork.
//!
//! KnowledgeNetwork emits standard `uwu_event_mesh` topics through
//! [`crate::EventMeshMeshTransport`]. Cross-process delivery should be wired by
//! attaching `agent-context-db-nats` / `uwu_nats_bridge` to the same `EventMesh`.
//! This keeps NATS connection lifecycle outside the federated discovery core.

pub use crate::event_mesh_transport::{KN_FETCH_TOPIC, KN_GOSSIP_CAPABILITY_TOPIC, KN_PROBE_TOPIC};
