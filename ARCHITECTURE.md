# uwu-context-db 架构总览

以文件系统范式统一管理 Memory / Resource / Skill / Wiki 的 Agent 上下文数据库。

## 设计原则

1. **FS 范式统一性**：一切上下文皆 URI（`uwu://`），可 `ls`/`find`/`grep`/`tree`/`read`
2. **双层存储单一数据源**：内容层（PostgreSQL）= 唯一真相源；索引层（Qdrant）只存 URI + 向量指针
3. **通用核心与专有扩展分离**：core/retrieve/version/session/parse/compressor/storage/wiki 零 uwu 依赖
4. **事实层 / 派生层分离**：内存热态是派生层真值源，context-db 是其冷归档与重算来源
5. **未发布产品不保留旧 API**：核心模型只暴露 `ContentType`，不保留 `MemoryClass` / `memory_class` / `class` 等历史兼容层

## 核心数据模型

- `ContextEntry` 是唯一内容条目：`ContextUri` + `TenantId` + `ContentPayload` + `ContextMeta` + MVCC 时间戳。
- `ContentPayload` 统一承载 L0/L1/L2：文本为 `sparse` / `dense` / `full`，多模态内容通过摘要、特征或 blob 引用进入同一模型。
- `ContentType` 是唯一内容分类，共 13 类：`Fact`、`Belief`、`Hypothesis`、`Heuristic`、`Procedure`、`Preference`、`Profile`、`Goal`、`Skill`、`Reflection`、`Evidence`、`Error`、`Meta`。
- `ContextMeta.content_type` 是查询、模板、CRDT 合并、PG 存储和解析输出的统一分类字段；旧 `MemoryClass`、`memory_class`、`FindPattern.class` 已移除。
- `ContextMeta.validity` / `ValidityRecord` 表示现实有效时间（valid-time），独立于 `created_at` / `updated_at` 事务时间（transaction-time），WQL 和版本层双时态查询都会使用这两条时间轴。
- PG schema 使用 `content_type TEXT` 存储 `ContentType::as_path_segment()`，不再创建或写入 `memory_class` 列。
- `FindPattern` 只通过 `content_type` 过滤，`DirEntry` 只返回 `content_type`。

## crate 清单

workspace 内 15 个 crate，全部命名 `agent-context-db-*`（目录 `crates/context-db-*/`）。

| crate | 职责 |
|-------|------|
| `context-db-core` | URI + 三层模型 + 窄端口（FsOps/ContentRepo/VersionOps/TenantOps）+ LlmClient/VectorIndex/GraphStore/BlobStore/ReadCache/RateLimiter/生命周期/ACL/Pack/血缘/模板/继承链/去重聚类/UwuConfig/observability metrics + WatchHub/WatchableStore + SemanticWriteDedupStore + prompt 优化 + LLM cache/cascade |
| `context-db-retrieve` | Query DSL + LogicalPlan + CBO 优化器 + PhysicalPlan 算子 + WQL 预算/谓词下推 + GraphTraverse 批量图遍历 + 双时态谓词过滤 + Theory-of-Mind persona/relations 结构化模型 + 分层检索 + GraphRAG 社区摘要/图增强检索 + 意图分析(Rule/LLM) + Rerank + 幻觉检测 + 分层背包预算装载 + 压缩感知加载 + 预测性预加载 + 增量检索学习 + 联想扩展 |
| `context-db-version` | DAG 版本管理（CommitOps/BranchOps/TagOps/MergeOps/HistoryOps 五窄 trait） + ContentType 驱动 CRDT 合并 + 语义 diff + 时态推理 + TemporalIndex 双时态查询(valid-time/transaction-time) + CausalDag + PC/GES 因果发现 + do-calculus/反事实干预推断 + Neural-Symbolic AGM 信念修正 + 知识晶体 + 自修复 + DreamConsolidator 睡眠期经验重放/技能候选合成 + cherry-pick/rebase (ConflictStrategy + 可选持久化交互式 ConflictSession) + CEL 语义标签 |
| `context-db-session` | 两阶段 commit 会话压缩 |
| `context-db-parse` | SemanticProcessor + 内容哈希/子摘要指纹增量缓存 + MemoryExtractor(ContentType 分类) + TrajectoryExtractor(两层归纳) |
| `context-db-compressor` | tokio mpsc 异步语义处理队列 + RedisSemanticQueue |
| `context-db-storage` | PgContextStore + PgVersionStore（Git 风格差量 + L1 内存 + L2 checkpoint 三级快照缓存 + 可选持久化 version_conflict_sessions）+ UwuVectorIndex + UwuCacheAdapter + ContextDbService（ACL → 写入前语义去重 → WatchableStore 主路径） |
| `context-db-wiki` | wiki-core → context-db 存储桥接（WikiVectorStoreAdapter） |
| `context-db-testkit` | MemoryContextStore + MemoryVersionStore |
| `context-db-nats` | EventSystem 装配壳：EventMesh + NatsBridge(按事件类型路由 Main/Consolidation/Monitoring) + NatsIngestor 跨进程事件桥接 |
| `context-db-llm-provider` | LlmClient provider：OpenAI / Anthropic / 自建 OpenAI-compatible HTTP 后端 + config-driven factory；默认组装 PromptOptimizingLlmClient → CascadeLlmClient → CachingLlmClient，并下发 provider prompt-cache 标记 |
| `context-db-marketplace` | Agent-to-Agent 知识市场：MarketId/AgentId/PublicationMetadata/FederatedDiscoveryResult 等 DTO + PublishGate + DiscoveryEngine + FederatedRegistry + ReputationEngine + SocialVoter + DP + secret sharing 安全聚合 + CRDT merge + ConflictResolver 多智能体 debate 仲裁 + immune/influence/phylogeny/community + SpeciationTracker 特化 fork + MemeticEvolutionEngine 选择/变异/交叉/淘汰 |
| `context-db-knowledge-network` | 独立 KnowledgeNetwork：真 Laplace/Gaussian DP + RDP privacy budget + capability index + trust routing + probe/fetch + streaming top-k 聚合 + P4 语义能力图/佐证图/路由学习/拓扑优化 + P5 治理策略/ed25519 身份签名/AccessGrant/持久化接口/EventMesh 传输；实现 marketplace 的联邦发现/签名端口；NATS 继续通过 `context-db-nats` + `uwu_nats_bridge` 桥接 |
| `context-db-consolidation` | 单 Agent 十大创新 + 七维质量分 + 短/中/长期 horizon-aware QualityReassessment + uncertainty-driven ActiveLearningPlanner + BanditBudgetPolicy 渐进加载预算分配 + SelfConsistencyConsolidator 多采样投票巩固 + 预测驱动 TieredCache hot/warm promotion + 可验证发布/证据链签名 + Sleeptime 执行器 + 巩固状态机；巩固产物通过 marketplace 的 PublishableProduct 端口发布 |
| `context-db-cdt` | 认知驱动训练：认知梯度 + CDT Pipeline + 生命周期门控训练样本 + Skill 系统 + Dream Replay skill candidates 入库 + DPO 融合 + LATS/MCTS 自我对弈 + ValueModule rollout/backprop + Voyager/AlphaZero/DSPy 基础组件 + STORM 多视角批量 LLM 分析/合成 |

## 功能矩阵

已落地能力（按域组织）：

| 域 | 能力 |
|----|------|
| 存储 | PG 双层存储 · ContentType-only schema · 内存实现 · 路径 ACL · 写入前语义去重 · WatchableStore CDC/订阅流 · ContextPack 导出导入 · 批量写入 · pg_trgm/tags GIN · BlobStore |
| 检索 | 意图分析(规则+LLM) · 向量召回 · 目录递归 · GraphRAG 社区摘要 + 图增强检索 · WQL Query DSL → LogicalPlan → CBO → PhysicalPlan · Scan/VectorSearch 预算与谓词下推 · GraphTraverse 批量图遍历 · 双时态谓词过滤 · Theory-of-Mind persona/relations 模型 · Rerank · 幻觉检测 · cl100k tokenizer 计数 · 分层背包预算装载 · BanditBudgetPolicy 渐进加载 · 压缩感知加载 · 预测性预加载 · 增量检索学习 · 检索管线 6 阶段 tracing span |
| 版本 | Commit DAG · Branch/Tag(含 CEL Semantic) · merge · 时间旅行 · ContentType 驱动 CRDT 合并 · 语义差异推理 · 时态推理 · TemporalIndex 双时态 as-of/between 查询 · PC/GES 因果图 · do-calculus/反事实干预推断 · Neural-Symbolic AGM 信念修正 · 睡眠期经验重放/梦境技能候选 · cherry-pick/rebase (ConflictStrategy Fail/Ours/Theirs) · squash · gc · Git 风格差量存储 · 三级快照缓存 |
| 语义 | L0/L1 生成 · 摘要增量缓存（内容哈希/子摘要指纹）· ContentType 记忆提取+去重 · 轨迹/经验两层归纳 · 两阶段 commit |
| Agent | State fork/promote · 联邦发现/查询 · DP 隐私预算组合追踪 · 可验证 provenance + DP clipping/noise + additive secret sharing 安全聚合 |
| 创新 | 遗忘曲线(Ebbinghaus/SM-2/Bayesian) · Token 经济 · Bandit/RL 渐进加载预算分配 · 自洽性投票巩固 · 多智能体辩论式仲裁 · Neural-Symbolic AGM 信念修正 · 知识晶体蒸馏 · Dream Replay 睡眠期技能合成 · Theory-of-Mind 交互对象建模 · 知识物种分化 fork + phylogeny + Memetic 演化选择 · 自修复 · 梦境巩固(batch_complete 批量洞察) · STORM 多视角批量分析/合成 · 因果推断 · 事件流因果链(CausalDag) · 血缘图 · 跨 Agent 去重 · 上下文模板 · 继承链 |
| LLM | complete · embed/embed_batch · complete_json · stream · batch · speculative · PromptOptimizingLlmClient · CascadeLlmClient(cheap/strong 模型路由) · CachingLlmClient · provider 原生 prompt caching 标记 · OpenAI / Anthropic / 自建 OpenAI-compatible HTTP provider |
| 事件 | WatchHub/WatchSource CDC · EventMeshBridge → uwu_event_mesh → NatsBridge 按事件类型分流 → uwu_nats_bridge(跨进程) |
| 缓存 | L1 moka async cache + L2 快照 checkpoint（热度统计/容量上限/冷数据驱逐）· PredictivePrefetcher → TieredCache hot/warm promotion · completion/embed_batch 缓存 · 摘要增量缓存 · provider prompt cache 标记 · TTL ±10% 抖动（雪崩防护）· 负缓存 marker（穿透防护）· DedupStore 可选持久化 |
| 配置 | UwuConfig 5 子模块 · Arc<ArcSwap<UwuConfig>> 无锁热更新 |
| 可观测 | metrics crate 8 类计数器 · Prometheus exporter/recorder · tracing::instrument 全线关键路径 · 检索管线 6 阶段 span |

## 尚未落地能力

以下 ARCHITECTURE 早期设想的能力**尚未实现**，属外围应用层：

| 缺项 | 说明 |
|------|------|
| Character 约束 | 无 Character/Persona/Constraint 结构 |
| Reaction 学习 | 无反应学习闭环 |
| WASM 沙箱 | 无 wasm/wasmtime 集成 |
| Metacog 冷热归档 | URI 分类到位，但缺 hot↔cold 迁移策略 |
| 安全沙箱（capability） | 只有 TokenBudget，缺 URI 级权限/资源隔离 |
| 多模态对齐 | 只有 `multimodal_to_text`，缺跨模态 embedding 对齐 |
| 联邦跨进程生产化 | DP/RDP 会计与联邦发现模型已落地，仍需生产网络拓扑、配额治理和端到端跨进程压测 |

## 解耦约束

- 每层只依赖下层的**窄 trait**，禁止依赖具体 struct
- `ContextStore` 聚合 trait 仅应用层使用；库内部只用 `FsOps`/`ContentRepo`/`VersionOps` 窄端口
- `VersionStore` 内部拆为 5 个窄 trait：`CommitOps`/`BranchOps`/`TagOps`/`MergeOps`/`HistoryOps`（blanket impl 自动派生）
- 后端可替换：Memory（testkit）↔ PG+Qdrant（生产）

## URI 结构

```
uwu://{tenant}/agent/{id}/memory/{content_type}/{semantic_path}/{entry}
uwu://{tenant}/agent/{id}/state/{short|mid|long}/
uwu://{tenant}/agent/{id}/persona/relations/
uwu://{tenant}/agent/{id}/metacog/pred_errors/
uwu://{tenant}/wiki/{space}/{doc}/
uwu://{tenant}/sessions/{id}/archive/{n}/
uwu://market/{publisher}/{entry_id}
```

`ContextUri` 内部 `Arc<UriInner>`，支持结构化查询参数（`?as_of=commit-abc&level=L1`）。`memory` 是 agent-scoped context entries 的命名空间段，放在 agent scope 与 `content_type` 之间，用于让类型轴过滤稳定匹配 `/memory/{content_type}/`；它不是 `ContextUri` 解析器的保留字，解析器只保留路径段。

## 存储与索引

- PostgreSQL 是内容事实源，`context_entries` 当前字段为 `uri`、`tenant_id`、`l0_abstract`、`l1_overview`、`l2_detail_ref`、`content_type`、`state_scope`、`tags`、`custom`、`mvcc_version`、`created_at`、`updated_at`。
- `context_versions` 记录 MVCC 版本快照，`entry_json` 保存完整 `ContextEntry`，用于 rollback/time travel。
- `content_type` 存储 `ContentType::as_path_segment()`，例如 `fact`、`preference`、`error`；不再存在 `memory_class` 列。
- 向量索引只保存 URI 与 embedding 指针/向量，内容解码仍回到 `ContentRepo` / `FsOps`。
- embedding cache 按 `blake3(content)` 去重，可使用内存或 Redis 后端；provider 主路径支持 `embed_batch`，批量迁移和检索生成会按请求内去重后批量调用。
- LSH 近邻索引用表级随机投影初始化 bucket，作为跨 Agent 去重/相似召回的轻量本地候选源。

## 数据流

```
写入: ContentRepo::write() → AclProtectedStore → SemanticWriteDedupStore → WatchableStore → PgContextStore → PostgreSQL(content_type-only schema) → context_versions 快照 → ChangeEvent
语义: SessionCompressor → MemoryExtractor(ContentType) → SemanticProcessor(摘要增量缓存) → SemanticQueue
检索: QueryPlanner → LogicalPlan → CboOptimizer(预算/谓词下推) → PhysicalPlan(TypeScan/VectorSearch/GraphTraverse/Parallel) → GraphRAG 社区摘要/图下钻 → 6 阶段执行 (parse → optimize → execute → rerank → expand → knapsack budget.load)
LLM: 调用方 LlmOpts(task/prompt) → PromptOptimizingLlmClient → CascadeLlmClient(cheap/strong 路由) → CachingLlmClient → provider(prompt-cache 标记 / batch embedding)
版本: VersionStore::commit() → DAG → branch/tag → merge/cherry_pick/rebase → time travel / TemporalIndex 双时态查询 → CRDT merge(ContentType) → PC/GES 因果发现 → do(A)/remove(A) 反事实下游影响推断 → Neural-Symbolic AGM 信念修正 → Dream Replay 睡眠期经验重放/技能候选
事件: WatchHub/WatchSource → 内部 uwu_event_mesh → NatsBridge(type_id 分流 Main/Consolidation/Monitoring) → uwu_nats_bridge → 跨进程节点
巩固: Sleeptime → 十大单 Agent 创新 + Marketplace 三级发现/多智能体 debate 仲裁/安全聚合 → Bandit 渐进加载预算 → Self-Consistency 多采样投票/Dream Replay/STORM 批量 LLM 合成 → KnowledgeCrystal/ReplaySkillCandidate 写回 → ed25519 发布签名 + 证据链 hash → 七维质量分 → 主动学习任务 → 巩固状态(Pending/InProgress/Converged/Stale)
训练: CDT Pipeline → LifecycleEngine 过滤冻结/归档/删除样本 → 认知梯度 → LATS/MCTS 自我对弈 → DPO preference pair → Skill 库更新
```
