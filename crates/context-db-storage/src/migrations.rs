//! context-db PG schema migrations，通过 uwu_database::SqlMigration 注册。
//!
//! 每个 migration 只包含纯 SQL 字符串，在调用方注入到 `uwu_database::Migrator`。

use uwu_database::SqlMigration;

/// 返回 context-db 所需的全部 migration（按版本号排序）。
pub fn context_db_migrations() -> Vec<SqlMigration> {
    vec![
        SqlMigration::new(
            1,
            "create_context_entries",
            // ································································
            // context_entries：URI 为 PK 的内容表，承载 L0/L1/L2 三层模型。
            // ································································
            r#"
            CREATE TABLE IF NOT EXISTS context_entries (
                uri             TEXT PRIMARY KEY,
                tenant_id       UUID NOT NULL,
                l0_abstract     TEXT NOT NULL,
                l1_overview     TEXT,
                l2_detail_ref   UUID,
                content_type    TEXT NOT NULL DEFAULT 'evidence',
                state_scope     TEXT,
                tags            JSONB NOT NULL DEFAULT '[]',
                custom          JSONB NOT NULL DEFAULT '{}',
                mvcc_version    BIGINT NOT NULL DEFAULT 0,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_ctx_tenant ON context_entries (tenant_id);
            CREATE INDEX IF NOT EXISTS idx_ctx_parent ON context_entries
                USING btree (uri text_pattern_ops);
            "#,
            None::<&str>,
        ),
        SqlMigration::new(
            2,
            "create_context_versions",
            // ································································
            // context_versions：MVCC 版本历史，每个 (uri, version) 存储完整条目快照。
            // ································································
            r#"
            CREATE TABLE IF NOT EXISTS context_versions (
                uri             TEXT NOT NULL,
                mvcc_version    BIGINT NOT NULL,
                tenant_id       UUID NOT NULL,
                l0_abstract     TEXT NOT NULL,
                l1_overview     TEXT,
                l2_detail_ref   UUID,
                content_type    TEXT NOT NULL DEFAULT 'evidence',
                state_scope     TEXT,
                tags            JSONB NOT NULL DEFAULT '[]',
                custom          JSONB NOT NULL DEFAULT '{}',
                entry_json      JSONB NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (uri, mvcc_version)
            );

            CREATE INDEX IF NOT EXISTS idx_ctx_ver_uri ON context_versions (uri, mvcc_version DESC);
            "#,
            None::<&str>,
        ),
        // D.3: GIN trigram 索引 — 加速 grep/ILIKE 全表扫描。
        SqlMigration::new(
            3,
            "create_grep_gin_index",
            r#"
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
            CREATE INDEX IF NOT EXISTS idx_ctx_l0_trgm ON context_entries
                USING gin (l0_abstract gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS idx_ctx_l1_trgm ON context_entries
                USING gin (l1_overview gin_trgm_ops);
            "#,
            None::<&str>,
        ),
        // D.6: tags GIN 索引 — 加速按标签过滤。
        SqlMigration::new(
            4,
            "create_tags_gin_index",
            r#"
            CREATE INDEX IF NOT EXISTS idx_ctx_tags_gin ON context_entries
                USING gin (tags jsonb_path_ops);
            "#,
            None::<&str>,
        ),
        // 版本 DAG（PgVersionStore）—— Git 风格差量存储：
        // - version_commits：每个 commit 的元数据 + 差量指向 parent[0]
        // - version_commit_parents：多父关系（merge commit 支持）
        // - version_branches / version_tags：命名引用
        // - version_entry_deltas：commit 相对 parent[0] 的变更（add/update/delete/rename）
        // - version_heads：scope → HEAD commit（未处于任何分支时用）
        SqlMigration::new(
            5,
            "create_version_dag",
            r#"
            CREATE TABLE IF NOT EXISTS version_commits (
                id              UUID PRIMARY KEY,
                scope           TEXT NOT NULL,
                tree_hash       TEXT NOT NULL,
                author_json     JSONB NOT NULL,
                message         TEXT NOT NULL,
                timestamp       TIMESTAMPTZ NOT NULL,
                metadata_json   JSONB NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_version_commits_scope_time
                ON version_commits (scope, timestamp DESC);

            CREATE TABLE IF NOT EXISTS version_commit_parents (
                commit_id       UUID NOT NULL REFERENCES version_commits(id) ON DELETE CASCADE,
                parent_id       UUID NOT NULL,
                ordinal         SMALLINT NOT NULL,
                PRIMARY KEY (commit_id, parent_id)
            );
            CREATE INDEX IF NOT EXISTS idx_version_parents_child
                ON version_commit_parents (parent_id);

            CREATE TABLE IF NOT EXISTS version_branches (
                scope           TEXT NOT NULL,
                name            TEXT NOT NULL,
                head            UUID NOT NULL,
                branch_type     TEXT NOT NULL,
                lifecycle_json  JSONB NOT NULL DEFAULT '{}',
                created_from    UUID NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (scope, name)
            );

            CREATE TABLE IF NOT EXISTS version_tags (
                scope           TEXT NOT NULL,
                name            TEXT NOT NULL,
                target          UUID NOT NULL,
                tag_type        TEXT NOT NULL,
                message         TEXT,
                timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (scope, name)
            );

            CREATE TABLE IF NOT EXISTS version_entry_deltas (
                commit_id       UUID NOT NULL REFERENCES version_commits(id) ON DELETE CASCADE,
                uri             TEXT NOT NULL,
                op              TEXT NOT NULL,     -- 'add' | 'update' | 'delete' | 'rename'
                entry_json      JSONB,             -- new value (NULL for 'delete')
                rename_from     TEXT,              -- for 'rename' ops
                PRIMARY KEY (commit_id, uri)
            );
            CREATE INDEX IF NOT EXISTS idx_version_deltas_uri
                ON version_entry_deltas (uri, commit_id);

            CREATE TABLE IF NOT EXISTS version_heads (
                scope           TEXT PRIMARY KEY,
                commit_id       UUID NOT NULL
            );
            "#,
            None::<&str>,
        ),
        // 语义标签 CEL 表达式列 —— SemanticCondition.expr 存于此
        SqlMigration::new(
            6,
            "add_version_tags_condition_expr",
            r#"
            ALTER TABLE version_tags
                ADD COLUMN IF NOT EXISTS condition_expr TEXT;
            "#,
            None::<&str>,
        ),
        // L2 快照检查点 —— 深链场景第一次读的兜底
        //
        // reconstruct_snapshot 查找顺序：L1 内存缓存 → L2 checkpoint 行 → 沿 first_parent 链遍历。
        // 遍历超过阈值时（默认 32）自动写入 checkpoint。
        // ON DELETE CASCADE 保证 gc 删 commit 时 checkpoint 一并清理。
        SqlMigration::new(
            7,
            "create_version_commit_checkpoints",
            r#"
            CREATE TABLE IF NOT EXISTS version_commit_checkpoints (
                commit_id       UUID PRIMARY KEY REFERENCES version_commits(id) ON DELETE CASCADE,
                snapshot_json   JSONB NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                access_count    BIGINT NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_version_commit_checkpoints_heat
                ON version_commit_checkpoints (last_accessed_at DESC, access_count DESC, created_at DESC);
            "#,
            None::<&str>,
        ),
        SqlMigration::new(
            8,
            "add_checkpoint_heat_columns",
            r#"
            ALTER TABLE version_commit_checkpoints
                ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                ADD COLUMN IF NOT EXISTS access_count BIGINT NOT NULL DEFAULT 0;
            CREATE INDEX IF NOT EXISTS idx_version_commit_checkpoints_heat
                ON version_commit_checkpoints (last_accessed_at DESC, access_count DESC, created_at DESC);
            "#,
            None::<&str>,
        ),
        SqlMigration::new(
            9,
            "create_version_conflict_sessions",
            r#"
            CREATE TABLE IF NOT EXISTS version_conflict_sessions (
                id             UUID PRIMARY KEY,
                scope          TEXT NOT NULL,
                operation_json JSONB NOT NULL,
                session_json   JSONB NOT NULL,
                status         TEXT NOT NULL DEFAULT 'open',
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                finished_at    TIMESTAMPTZ
            );
            CREATE INDEX IF NOT EXISTS idx_version_conflict_sessions_scope_status
                ON version_conflict_sessions (scope, status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_version_conflict_sessions_open
                ON version_conflict_sessions (updated_at DESC)
                WHERE status = 'open';
            "#,
            None::<&str>,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_have_sequential_versions() {
        let ms = context_db_migrations();
        for i in 1..ms.len() {
            assert!(
                ms[i].version > ms[i - 1].version,
                "migration versions must be ascending"
            );
        }
    }

    #[test]
    fn each_migration_has_name_and_sql() {
        let ms = context_db_migrations();
        for m in &ms {
            assert!(!m.name.is_empty());
            assert!(!m.up_sql.is_empty());
        }
    }
}
