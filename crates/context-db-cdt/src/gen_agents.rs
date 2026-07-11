//! GenAgents — 长期 agent 记忆、反思和行为模拟。
//!
//! 该模块把 CDT 中的轨迹、insight 和记忆条目组织成 agent profile，
//! 生成下一步行为预测，并输出可进入 consolidation 的长期记忆信号。

use crate::consolidation::{CdtConsolidationSignal, CdtSignalSource};
use crate::trajectory_encoder::Trajectory;
use crate::voting::EvolvableInsight;
use agent_context_db_core::{ContentType, ContextEntry, ContextError, ContextUri, EpistemicType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub agent_uri: ContextUri,
    pub name: String,
    pub goals: Vec<String>,
    pub traits: Vec<String>,
    pub preferences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEpisode {
    pub uri: ContextUri,
    pub description: String,
    pub outcome: EpisodeOutcome,
    pub salient_memories: Vec<ContextUri>,
    pub lessons: Vec<String>,
    pub importance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodeOutcome {
    Success,
    Failure,
    Ambiguous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorForecast {
    pub agent_uri: ContextUri,
    pub next_action: String,
    pub rationale: String,
    pub confidence: f32,
    pub supporting_episodes: Vec<ContextUri>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SocialMemoryGraph {
    edges: HashMap<ContextUri, Vec<ContextUri>>,
}

impl SocialMemoryGraph {
    pub fn link(&mut self, from: ContextUri, to: ContextUri) {
        self.edges.entry(from).or_default().push(to);
    }

    pub fn neighbors(&self, uri: &ContextUri) -> Vec<ContextUri> {
        self.edges.get(uri).cloned().unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct GenAgentMemory {
    pub profile: AgentProfile,
    pub episodes: Vec<AgentEpisode>,
    pub social_graph: SocialMemoryGraph,
}

impl GenAgentMemory {
    pub fn new(profile: AgentProfile) -> Self {
        Self {
            profile,
            episodes: Vec::new(),
            social_graph: SocialMemoryGraph::default(),
        }
    }

    pub fn ingest_trajectory(
        &mut self,
        trajectory: &Trajectory,
    ) -> Result<AgentEpisode, ContextError> {
        let uri = make_agent_uri(
            &self.profile.agent_uri,
            "episode",
            self.episodes.len(),
            &trajectory.task_description,
        )?;
        let outcome = match (trajectory.success, trajectory.error_message.is_some()) {
            (true, _) => EpisodeOutcome::Success,
            (false, true) => EpisodeOutcome::Failure,
            (false, false) => EpisodeOutcome::Ambiguous,
        };
        let lessons = if trajectory.success {
            vec![format!(
                "Repeat strategy for `{}` when similar preconditions appear",
                trajectory.task_description
            )]
        } else {
            vec![format!(
                "Avoid repeating `{}` without addressing `{}`",
                trajectory.task_description,
                trajectory
                    .error_message
                    .as_deref()
                    .unwrap_or("unknown failure")
            )]
        };
        let importance = compute_importance(
            outcome,
            trajectory.steps.len(),
            trajectory.relevant_knowledge.len(),
        );
        let episode = AgentEpisode {
            uri,
            description: trajectory.task_description.clone(),
            outcome,
            salient_memories: vec![],
            lessons,
            importance,
        };
        self.episodes.push(episode.clone());
        Ok(episode)
    }

    pub fn ingest_insights(&mut self, insights: &[EvolvableInsight]) -> Result<(), ContextError> {
        for insight in insights {
            let episode_uri = make_agent_uri(
                &self.profile.agent_uri,
                "episode",
                self.episodes.len(),
                &insight.content,
            )?;
            self.episodes.push(AgentEpisode {
                uri: episode_uri,
                description: insight.content.clone(),
                outcome: if insight.votes.net_score > 0.0 {
                    EpisodeOutcome::Success
                } else {
                    EpisodeOutcome::Ambiguous
                },
                salient_memories: vec![insight.uri.clone()],
                lessons: vec![insight.content.clone()],
                importance: insight.votes.net_score.clamp(0.1, 1.0),
            });
        }
        Ok(())
    }

    pub fn forecast_behavior(&self, context: &[ContextEntry]) -> BehaviorForecast {
        let best_episode = self.episodes.iter().max_by(|a, b| {
            a.importance
                .partial_cmp(&b.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let context_hint = context
            .iter()
            .find_map(|entry| (!entry.l0_text().is_empty()).then(|| entry.l0_text().to_string()))
            .unwrap_or_else(|| "current context".into());
        let next_action = best_episode
            .and_then(|episode| episode.lessons.first().cloned())
            .unwrap_or_else(|| format!("Collect more evidence before acting on {context_hint}"));
        let confidence = best_episode
            .map(|episode| (episode.importance * 0.7 + context.len() as f32 * 0.05).clamp(0.0, 1.0))
            .unwrap_or(0.25);

        BehaviorForecast {
            agent_uri: self.profile.agent_uri.clone(),
            next_action,
            rationale: format!(
                "forecast from {} episodes and {} context memories",
                self.episodes.len(),
                context.len()
            ),
            confidence,
            supporting_episodes: best_episode
                .map(|episode| vec![episode.uri.clone()])
                .unwrap_or_default(),
        }
    }

    pub fn consolidation_signals(
        &self,
        forecast: &BehaviorForecast,
    ) -> Result<Vec<CdtConsolidationSignal>, ContextError> {
        let mut signals = Vec::new();
        signals.push(CdtConsolidationSignal {
            uri: make_agent_uri(&self.profile.agent_uri, "profile", 0, &self.profile.name)?,
            content_type: ContentType::Profile,
            epistemic_type: EpistemicType::Belief,
            content: format!(
                "Agent {} goals: {}; traits: {}; preferences: {}",
                self.profile.name,
                self.profile.goals.join(", "),
                self.profile.traits.join(", "),
                self.profile.preferences.join(", ")
            ),
            quality_score: 0.75,
            confidence: 0.75,
            evidence_uris: vec![],
            contradiction_uris: vec![],
            source: CdtSignalSource::GenAgent,
            tags: vec!["gen-agent".into(), "profile".into()],
            hypothesis_outcome: None,
        });

        signals.push(CdtConsolidationSignal {
            uri: make_agent_uri(
                &self.profile.agent_uri,
                "forecast",
                0,
                &forecast.next_action,
            )?,
            content_type: ContentType::Reflection,
            epistemic_type: EpistemicType::Heuristic,
            content: format!(
                "NEXT_ACTION: {}\nRATIONALE: {}",
                forecast.next_action, forecast.rationale
            ),
            quality_score: forecast.confidence,
            confidence: forecast.confidence,
            evidence_uris: forecast.supporting_episodes.clone(),
            contradiction_uris: vec![],
            source: CdtSignalSource::GenAgent,
            tags: vec!["gen-agent".into(), "behavior-forecast".into()],
            hypothesis_outcome: None,
        });

        signals.extend(self.episodes.iter().map(|episode| CdtConsolidationSignal {
            uri: episode.uri.clone(),
            content_type: match episode.outcome {
                EpisodeOutcome::Success => ContentType::Skill,
                EpisodeOutcome::Failure => ContentType::Error,
                EpisodeOutcome::Ambiguous => ContentType::Reflection,
            },
            epistemic_type: match episode.outcome {
                EpisodeOutcome::Success => EpistemicType::Procedure,
                EpisodeOutcome::Failure | EpisodeOutcome::Ambiguous => EpistemicType::Heuristic,
            },
            content: format!(
                "EPISODE: {}\nOUTCOME: {:?}\nLESSONS: {}",
                episode.description,
                episode.outcome,
                episode.lessons.join("; ")
            ),
            quality_score: episode.importance,
            confidence: episode.importance,
            evidence_uris: episode.salient_memories.clone(),
            contradiction_uris: vec![],
            source: CdtSignalSource::GenAgent,
            tags: vec!["gen-agent".into(), "episode".into()],
            hypothesis_outcome: None,
        }));

        Ok(signals)
    }
}

fn compute_importance(outcome: EpisodeOutcome, steps: usize, knowledge_count: usize) -> f32 {
    let outcome_weight = match outcome {
        EpisodeOutcome::Success => 0.55,
        EpisodeOutcome::Failure => 0.75,
        EpisodeOutcome::Ambiguous => 0.35,
    };
    (outcome_weight + steps as f32 * 0.03 + knowledge_count as f32 * 0.05).clamp(0.0, 1.0)
}

fn make_agent_uri(
    base: &ContextUri,
    kind: &str,
    index: usize,
    content: &str,
) -> Result<ContextUri, ContextError> {
    let hash = blake3::hash(content.as_bytes());
    let short = hash.to_hex().chars().take(8).collect::<String>();
    ContextUri::parse(format!("{}/{}/{:02}-{}", base.as_str(), kind, index, short))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trajectory(success: bool) -> Trajectory {
        Trajectory {
            task_id: "deploy".into(),
            task_description: "deploy app".into(),
            steps: vec!["build".into(), "apply".into()],
            error_message: if success {
                None
            } else {
                Some("timeout".into())
            },
            success,
            relevant_knowledge: vec!["kubeconfig required".into()],
        }
    }

    #[test]
    fn gen_agent_forecast_materializes_signals() {
        let profile = AgentProfile {
            agent_uri: ContextUri::parse("uwu://t/agent/gen").unwrap(),
            name: "builder".into(),
            goals: vec!["ship reliable changes".into()],
            traits: vec!["cautious".into()],
            preferences: vec!["verify before deploy".into()],
        };
        let mut memory = GenAgentMemory::new(profile);
        memory.ingest_trajectory(&trajectory(false)).unwrap();
        let forecast = memory.forecast_behavior(&[]);
        let signals = memory.consolidation_signals(&forecast).unwrap();
        assert!(forecast.confidence > 0.0);
        assert!(signals.len() >= 3);
        assert!(
            signals
                .iter()
                .any(|s| s.source == CdtSignalSource::GenAgent)
        );
    }
}
