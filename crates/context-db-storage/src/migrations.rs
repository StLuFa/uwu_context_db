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
                content_type    TEXT NOT NULL DEFAULT 'text',
                memory_class    TEXT,
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
                content_type    TEXT NOT NULL DEFAULT 'text',
                memory_class    TEXT,
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