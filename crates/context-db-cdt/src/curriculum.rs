//! CurriculumGenerator — 主动课程：知识图谱拓扑前沿 + ZPD 排序。

use crate::TrainingGoal;
use agent_context_db_core::{ContentType, ContextUri, Result};
use std::collections::HashMap;

/// 课程生成器。
pub struct CurriculumGenerator {
    pub exploration_ratio: f32,
    pub zpd_difficulty: f32,
}

/// 知识图谱前沿节点。
#[derive(Debug, Clone)]
pub struct FrontierNode {
    pub uri: ContextUri,
    pub difficulty: f32,
    pub prerequisite_count: usize,
    pub expected_knowledge: String,
    pub content_type: Option<ContentType>,
}

impl CurriculumGenerator {
    pub fn new(exploration_ratio: f32) -> Self {
        Self {
            exploration_ratio,
            zpd_difficulty: 0.6,
        }
    }

    /// 生成下一个训练目标。
    pub async fn next_goal(&self, _known_uris: &[ContextUri]) -> Result<TrainingGoal> {
        // 简化：基于已知 URI 找未学邻居（前沿节点）。
        // 完整实现需要 GraphStore 查询。
        let node = FrontierNode {
            uri: ContextUri::parse("uwu://t/a/x/skill/next")?,
            difficulty: 0.5,
            prerequisite_count: 1,
            expected_knowledge: String::new(),
            content_type: Some(ContentType::Skill),
        };

        Ok(TrainingGoal {
            target_node: node.uri,
            difficulty: node.difficulty,
            prerequisite_skills: vec![],
            expected_new_knowledge: node.expected_knowledge,
        })
    }

    /// 找前沿节点：已巩固知识的相邻未学节点。
    pub fn find_frontier(
        &self,
        known: &HashMap<String, f32>, // uri → confidence
    ) -> Vec<FrontierNode> {
        let mut frontier = Vec::new();
        // 按最近发展区排序：难度在 zpd_difficulty ± 0.2 的最优
        for (uri_str, confidence) in known {
            if *confidence > 0.7 {
                // 已掌握 → 找相邻未学
                frontier.push(FrontierNode {
                    uri: ContextUri::parse(uri_str).unwrap(),
                    difficulty: 1.0 - confidence,
                    prerequisite_count: 1,
                    expected_knowledge: format!("next level of {uri_str}"),
                    content_type: None,
                });
            }
        }
        // ZPD 排序：接近 zpd_difficulty 的优先
        frontier.sort_by(|a, b| {
            let za = (a.difficulty - self.zpd_difficulty).abs();
            let zb = (b.difficulty - self.zpd_difficulty).abs();
            za.partial_cmp(&zb).unwrap()
        });
        frontier
    }
}
