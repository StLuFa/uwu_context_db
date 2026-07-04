# uwu-context-db 架构总览

以文件系统范式统一管理 Memory / Resource / Skill / Wiki 的 Agent 上下文数据库。

## 设计原则

1. **FS 范式统一性**：一切上下文皆 URI（`uwu://`），可 `ls`/`find`/`grep`/`tree`/`read`
2. **双层存储单一数据源**：内容层（PostgreSQL）= 唯一真相源；索引层（Qdrant）只存 URI + 向量指针
3. **通用核心与专有扩展分离**：core/retrieve/version/session/parse/compressor/storage/wiki 零 uwu 依赖
4. **事实层 / 派生层分离**：内存热态是派生层真值源，context-db 是其冷归档与重算来源

## crate 清单

| crate | 职责 |
|-------|------|
| `context-db-core` | URI + 三层模型 + 窄端口（FsOps/ContentRepo/VersionOps/TenantOps）+ LlmClient/VectorIndex/生命周期/ACL/Pack/订阅/血缘/模板/事件流/去重聚类 |
| `context-db-retrieve` | 分层检索 + 意图分析(Rule/LLM) + 幻觉检测 + 压缩感知加载 + 预测性预加载 + 增量检索学习 |
| `context-db-version` | DAG 版本管理 + CRDT 合并 + 语义差异推理 + 时态推理 + 知识晶体 + 自修复 + 梦境巩固 + 因果推断 |
| `agent-context-db` | StateBridge + MetacogBridge + CharacterConstraint + 安全沙箱 + 联邦协议 + 多模态对齐 + Reaction 学习 + EventMesh 桥接 + WASM 沙箱 + HttpLlmClient |
| `context-db-session` | 两阶段 commit 会话压缩 |
| `context-db-parse` | SemanticProcessor + MemoryExtractor(8类) + TrajectoryExtractor(两层归纳) |
| `context-db-compressor` | tokio mpsc 异步语义处理队列 |
| `context-db-storage` | PgContextStore + UwuVectorIndex + ContextDbService |
| `context-db-wiki` | wiki-core → context-db 存储桥接 |
| `context-db-testkit` | MemoryContextStore + MemoryVersionStore |

## 功能矩阵

| 域 | 能力 |
|----|------|
| 存储 | PG 双层存储 · 内存实现 · 路径 ACL · ContextPack 导出导入 · 订阅推送 |
| 检索 | 意图分析(规则+LLM) · 向量召回 · 目录递归 · Rerank · 幻觉检测 · 压缩感知加载 · 预测性预加载 · 增量检索学习 |
| 版本 | Commit DAG · Branch/Tag · merge · 时间旅行 · CRDT 合并 · 语义差异推理 · 时态推理 · cherry-pick/rebase/squash/gc |
| 语义 | L0/L1 生成 · 8 类记忆提取+去重 · 轨迹/经验两层归纳 · 两阶段 commit |
| Agent | State fork/promote · Metacog 冷热归档 · Character 约束 · 安全沙箱 · 联邦协议 · 多模态对齐 · Reaction 学习 · WASM 沙箱 |
| 创新 | 遗忘曲线 · Token 经济 · 知识晶体蒸馏 · 自修复 · 梦境巩固 · 因果推断 · 事件流因果链 · 血缘图 · 跨 Agent 去重 · 上下文模板 · 继承链 |
| LLM | complete/embed/complete_json/stream/batch/speculative · HttpLlmClient + MockLlmClient |
| 事件 | EventMeshBridge → uwu_event_mesh → uwu_nats_bridge(跨进程) |

## 解耦约束

- 每层只依赖下层的**窄 trait**，禁止依赖具体 struct
- `ContextStore` 聚合 trait 仅应用层使用；库内部只用 `FsOps`/`ContentRepo`/`VersionOps` 窄端口
- 后端可替换：Memory（测试）↔ PG+Qdrant（生产）

## URI 结构

```
uwu://{tenant}/agent/{id}/memories/{class}/{entry}
uwu://{tenant}/agent/{id}/state/{short|mid|long}/
uwu://{tenant}/agent/{id}/persona/relations/
uwu://{tenant}/agent/{id}/metacog/pred_errors/
uwu://{tenant}/wiki/{space}/{doc}/
uwu://{tenant}/sessions/{id}/archive/{n}/
```

## 数据流

```
写入: ContentRepo::write() → PgContextStore → PostgreSQL
语义: SessionCompressor → MemoryExtractor → SemanticProcessor → SemanticQueue
检索: IntentAnalyzer → VectorIndex(embed+search) → FsOps(ls/grep/read) → Rerank → HallucinationDetector
版本: VersionStore::commit() → DAG → branch/tag → merge → time travel → CRDT merge
事件: EventMeshBridge → uwu_event_mesh → uwu_nats_bridge(跨进程)
```
