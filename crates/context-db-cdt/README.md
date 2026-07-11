# agent-context-db-cdt

认知驱动训练（Cognitive-Driven Training）：把 `context-db-consolidation` 产生的巩固产物转化为可验证的 Skill 库，形成"轨迹 → 编码 → 巩固 → 认知梯度 → 偏好优化 → Skill 写回 → 反馈回流"的闭环。

## 定位

- 承接 `consolidation` 的输出：`ConsolidationProduct` 是训练信号的第一来源。
- 承接 `retrieve` 的轨迹/上下文能力，做训练时的记忆检索。
- 承接 `version` 的状态机，用于 Skill 的生命周期（Hypothesized → Validating → Validated / Falsified / Deprecated）。
- 是训练侧的终点 crate：目前没有其他 crate 依赖它，只被应用层直接使用。

## 主要模块

| 模块 | 作用 |
| --- | --- |
| `pipeline.rs` | `CognitiveTrainingPipeline`：完整的 CDT 闭环执行器 |
| `preference.rs` | `CognitivePreferenceExtractor`：从对比轨迹提取三层偏好（TaskOutcome / KnowledgeConsistency / EpistemicConfidence） |
| `dpo.rs` | `KnowledgeConstrainedDPO`：知识约束偏好学习，含矛盾惩罚 |
| `curriculum.rs` | `CurriculumGenerator`：基于 ZPD 与知识图拓扑的主动课程 |
| `skill_library.rs` | `SkillLibrary`：Skill 生命周期管理与向量检索 |
| `metric.rs` | `CognitiveMetric`：一致性 / 置信度 / 完成度 / 效率的复合评分 |
| `policy_value.rs` | `PolicyGate`：新策略需胜率过阈才能替换，AlphaZero 风格 |
| `tree_search.rs` / `self_play.rs` / `self_verify.rs` | 树搜索、自我对弈、自验证 |
| `multi_perspective.rs` / `voting.rs` / `reflection.rs` | 多视角巩固、投票演化、反思 |
| `trajectory_encoder.rs` / `hybrid_retrieval.rs` | 轨迹编码与混合检索 |

## 关键导出

- `CognitiveTrainingPipeline`、`TrainingConfig`、`TrainingReport`、`EpochResult`
- `CognitiveGradient`、`GradientType`：策略改进信号（FactCorrection / AvoidanceRule / ValidationRule / SkillExtraction / PreferenceUpdate / MetaCognitive）
- `CognitivePreferencePair`：偏好对（含置信度与认知差异）
- `Skill`、`SkillValidationStatus`：Skill 状态机
- `CognitiveMetric`、`PolicyGate`、`GateDecision`
- 辅助函数：`extract_gradients_from_products`、`feedback_evaluation_to_memories`

## 依赖

- `agent-context-db-core` — 基础类型 / URI
- `agent-context-db-consolidation` — `ConsolidationProduct`、`ConsolidationEngine`
- `agent-context-db-retrieve` — 训练时上下文检索
- `agent-context-db-version` — Skill 版本状态
- `tokio` / `parking_lot` / `chrono` / `uuid` / `blake3` / `rand` / `tracing`

## 用法

```rust
use agent_context_db_cdt::{
    CognitiveTrainingPipeline, TrainingConfig,
    extract_gradients_from_products,
    Skill,
};

// 1) 从巩固产物直接提取认知梯度（供离线分析或增量学习）
let gradients = extract_gradients_from_products(&products, /* min_confidence= */ 0.6);

// 2) 完整训练闭环
let pipeline = CognitiveTrainingPipeline::new(
    consolidation_engine,
    lifecycle_engine,
    llm_client,
    graph_store,
    signal_provider,
);
let report = pipeline.train(&TrainingConfig::default(), &trajectories).await?;

// 3) Skill 生命周期
let mut skill = Skill::new_hypothesis(uri, procedure, precondition, outcome);
skill.start_validating(epoch);
skill.record_trial(success);
skill.evaluate(/* success_threshold */ 0.8, /* failure_threshold */ 0.3);
```

## 与其他 crate 的关系

- `consolidation` → **cdt**：巩固产物驱动梯度和 Skill 抽取
- **cdt** → `consolidation`：训练后 Skill / 修正回灌为新的记忆
- **cdt** → `retrieve`：在训练循环里做上下文召回
- **cdt** → `version`：Skill 状态迁移使用统一版本模型
