# uwu-context-db

`uwu-context-db` 是面向 Agent 的多租户上下文数据库。它以 `uwu://` URI 和文件系统操作统一管理 Memory、State、Resource、Skill、Wiki、Session 与 Metacognition，并提供关系型事实存储、向量检索、MVCC、版本 DAG、生命周期、联邦查询、策略门禁、WASM 沙箱、Reaction 学习和审计能力。

根包 `agent-context-db` 是推荐的应用门面。`crates/context-db-*` 仍是公开的低层领域 crate，适合框架扩展和自定义装配；直接使用低层 API 时，调用方需要自行承担租户隔离、deadline、策略、审计和资源治理。

## 核心能力

- PostgreSQL 与 SQLite 关系型事实源，事务内维护当前记录、MVCC 历史和向量索引 outbox。
- `uwu://{tenant}/...` 统一寻址，支持 `read`、`list`、`find`、`grep`、`tree`、批量写入和类型扫描。
- 向量召回、WQL、GraphRAG、重排、预算装载和增量检索学习。
- Commit DAG、Branch、Tag、merge、time travel、cherry-pick、rebase、squash、GC 和交互式冲突会话。
- Watch checkpoint replay、lag recovery 和租户范围过滤。
- Metacog 冷热路由、分阶段恢复 checkpoint、跨进程 lease/CAS。
- LLM、工具、技能、联邦和 WASM 操作统一经过 `ExecutionGate`。
- capability token、tenant 隔离、fuel/deadline、内存、并发、输出和 trace 配额。
- 结构化审计、Reaction attribution、取消、绝对 deadline 和有界资源。

完整架构与数据流参见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## Workspace

| 包 | 职责 |
|---|---|
| `agent-context-db` | 根应用门面、请求边界、统一门禁和依赖装配 |
| `context-db-core` | URI、内容模型、存储端口、LLM/向量接口、Watch、策略与 Reaction |
| `context-db-storage` | PostgreSQL/SQLite、MVCC、outbox、版本持久化与向量适配 |
| `context-db-retrieve` | Query DSL、规划、GraphRAG、检索和预算装载 |
| `context-db-version` | 版本 DAG、合并、时间旅行、因果与冲突会话 |
| `context-db-consolidation` | 生命周期、冷热归档、质量评估和睡眠期巩固 |
| `context-db-knowledge-network` | 联邦发现、隐私预算、拓扑和配额治理 |
| `context-db-wasm` | `uwu_wasm` 的 context-db 安全执行适配 |
| `context-db-nats` | EventMesh/NATS 跨进程事件桥接与运行证据探针 |
| `context-db-testkit` | 内存存储和确定性测试实现 |

其他领域 crate 位于 `crates/`，每个 crate 可包含自己的 README。

## 使用根门面

### 1. 装配端口

根门面不静默创建假实现。必需端口必须显式提供，可选能力未配置时返回 `FacadeError::NotConfigured`。

```rust,no_run
use agent_context_db::{
    AuditSink, ContextDbBuilder, ContextDbParts, ContextDbConfig,
};
use std::sync::Arc;

# async fn build(parts: ContextDbParts) -> agent_context_db::Result<()> {
let db = ContextDbBuilder::injected(parts)
    .config(ContextDbConfig::default())
    .build()?;
# let _ = db;
# Ok(())
# }
```

`ContextDbParts` 的必需端口包括：

- `FsOps`、`ContentRepo`、`ContentStore`
- `VectorIndex`、`WatchSource`
- `VersionStore`、`InteractiveVersionStore`
- `ExecutionGate`
- `AuditSink`

生命周期、工具、技能、LLM、Reaction、WASM、联邦和 runtime guard 是可选端口。测试可使用 `agent-context-db-testkit` 的内存实现；生产环境可使用 `context-db-storage` 提供的 PostgreSQL/SQLite 装配能力。

### 2. 创建请求上下文

每次调用必须携带不可变的租户、actor、request ID、绝对 deadline 和取消令牌。门面不保存全局“当前租户”。

```rust,no_run
use agent_context_db::{CancellationToken, RequestContext, TenantIdentity};
use agent_context_db_core::TenantId;
use std::time::{Duration, Instant};
use uuid::Uuid;

# fn request() -> agent_context_db::Result<RequestContext> {
let tenant = TenantIdentity::new(
    TenantId(Uuid::parse_str("11111111-1111-1111-1111-111111111111").unwrap()),
    "acme",
)?;

RequestContext::new(
    tenant,
    "agent-runtime",
    "request-01",
    Instant::now() + Duration::from_secs(10),
    CancellationToken::default(),
)
# }
```

URI authority 与 `TenantIdentity::name()` 必须一致；`ContextEntry.tenant` 与 `TenantIdentity::id()` 必须一致。

### 3. 内容操作

```rust,no_run
use agent_context_db::{ContentLevel, ContextDb, ContextUri, RequestContext};

# async fn read(db: &ContextDb, ctx: &RequestContext) -> agent_context_db::Result<()> {
let uri = ContextUri::parse(
    "uwu://acme/agent/planner/memory/fact/project/status"
)?;
let payload = db.read(ctx, &uri, ContentLevel::L2).await?;
# let _ = payload;
# Ok(())
# }
```

门面还提供：

- `write`、`batch_write`、`delete`、`rename`
- `list`、`find`、`grep`、`tree`
- `scan`、`scan_by_type`
- `retrieve`、`watch`

写入成功后才触发生命周期路由。若数据已持久化，但生命周期或审计等必要后置步骤失败，返回 `CommittedWithPostCommitFailure`，调用方不能把它当作未提交错误盲目重试。

## 版本管理

```rust,no_run
# use agent_context_db::{ContextDb, ContextUri, RequestContext};
# use agent_context_db::version::LogOpts;
# async fn versions(db: &ContextDb, ctx: &RequestContext, scope: &ContextUri) -> agent_context_db::Result<()> {
let commits = db.versions(ctx).log(scope, &LogOpts::default()).await?;
# let _ = commits;
# Ok(())
# }
```

`Versions` 覆盖 `VersionStore` 的提交、分支、标签、读取、时间旅行、merge、diff、cherry-pick、rebase、squash、GC、语义标签、provenance、impact 和 evolution 操作。

`InteractiveVersions` 提供交互式 cherry-pick/rebase 冲突会话，并在门面内维护有界、带 TTL 的租户所有权绑定。

## LLM、工具与技能

根门面中的 LLM、工具和技能调用均经过策略门禁、deadline、取消、审计和 Reaction：

```rust,no_run
# use agent_context_db::{ContextDb, RequestContext};
# use agent_context_db_core::{ExecutionRequest, LlmOpts};
# async fn execute(db: &ContextDb, ctx: &RequestContext) -> agent_context_db::Result<()> {
let response = db
    .execute_tool(
        ctx,
        "search",
        ExecutionRequest::new("tool.search", "query"),
    )
    .await?;
# let _ = response;
# Ok(())
# }
```

可用入口包括：

- `llm_complete`、`llm_complete_json`、`llm_embed`
- `execute_tool`、`execute_skill`
- `federate`

未配置对应 provider/gateway 时会明确返回 `NotConfigured`。

## WASM 沙箱

根门面不会暴露原始 `uwu_wasm` registry。调用方使用租户绑定的安全入口：

- `wasm_register_tenant`
- `wasm_install`
- `wasm_invoke`

这些入口执行 `ExecutionGate`、module digest admission、capability token 校验、deadline/cancellation，并限制模块、请求和响应大小。底层 `context-db-wasm` 使用远端 `uwu_wasm` 实现 fuel、epoch deadline、内存、table、实例并发、WASI 和运行收据。

## 审计与 Reaction

`AuditSink` 是必需端口，不存在隐式 no-op。仓库提供 `BoundedAuditSink` 作为本地和测试用途的有界实现：

```rust
use agent_context_db::BoundedAuditSink;

let audit = BoundedAuditSink::new(1024)?;
# Ok::<(), agent_context_db::FacadeError>(())
```

生产环境应注入持久化或事件化审计实现。Reaction 会使用真实 request ID、actor 和操作名生成 attribution，而不是匿名随机关联。

## 错误语义

根门面统一返回 `FacadeError`：

- `TenantViolation`：URI 或条目越过请求租户边界。
- `Cancelled` / `DeadlineExceeded` / `Timeout`：请求取消或超过时间预算。
- `NotConfigured`：可选能力未装配。
- `PolicyDenied`：执行门禁拒绝请求。
- `CommittedWithPostCommitFailure`：持久化已提交，但必要后置步骤失败。
- `Context` / `Version` / `Llm` / `Wasm` / `Federation` / `Audit`：对应子系统错误。

## 关闭

```rust,no_run
# use agent_context_db::ContextDb;
# async fn stop(db: &ContextDb) -> agent_context_db::Result<()> {
db.shutdown().await?;
# Ok(())
# }
```

关闭会停止新请求准入、等待在途操作结束，并依次关闭 runtime guard。多个 `ContextDb` clone 共享关闭状态；失败的 guard 不会被错误标记为已成功关闭。

## 开发与质量门禁

```bash
cargo test --workspace --lib --tests --bins --examples
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --lib --bins --examples -- \
  -D warnings \
  -D clippy::unwrap_used \
  -D clippy::expect_used \
  -D clippy::panic \
  -D clippy::todo \
  -D clippy::unimplemented
cargo fmt --all -- --check
git diff --check
```

根门面测试覆盖租户隔离、取消/deadline、生命周期顺序、审计失败、Reaction attribution、工具/技能门禁、联邦上下文、WASM 边界和幂等关闭。

## 外部集成证据

源码、契约测试和 PostgreSQL/NATS 证据采集工具已经具备。真实连接失败、断线、重连和服务重启证据必须在配置了 `DATABASE_URL`、`NATS_URL` 的 CI 或集成环境运行，不能用本地 synthetic 记录冒充外部证据。

## License

MIT
