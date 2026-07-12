//! Durable transactional outbox and index worker.
//!
//! Content mutations append an event in the same SQL transaction.  The vector index is a
//! projection: delivery is at-least-once, operations are idempotent, and expired leases are
//! reclaimed after a process crash.

use agent_context_db_core::{
    ContextEntry, ContextError, ContextUri, IndexPoint, Result, VectorIndex,
};
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, Sqlite, Transaction};
use std::{sync::Arc, time::Duration};
use tokio::{sync::watch, task::JoinHandle};
use uuid::Uuid;
use uwu_database::DbPool;

pub const DEFAULT_COLLECTION: &str = "context";
const VECTOR_KEY: &str = "embedding_vector";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum IndexMutation {
    Upsert {
        collection: String,
        point: IndexPoint,
    },
    Delete {
        collection: String,
        uri: ContextUri,
    },
    Rename {
        collection: String,
        from: ContextUri,
        point: Option<IndexPoint>,
    },
}

/// Converts the persisted entry metadata into an index point. Entries without an embedding are
/// deliberately represented by `None`: they remain content-only until an embedding write occurs.
pub fn collection_from_entry(entry: &ContextEntry) -> String {
    entry
        .metadata
        .custom
        .get("vector_collection")
        .and_then(|value| value.as_str())
        .unwrap_or(DEFAULT_COLLECTION)
        .to_owned()
}

pub fn upsert_mutation(
    entry: &ContextEntry,
    version: agent_context_db_core::MvccVersion,
) -> Result<Option<IndexMutation>> {
    let mut entry = entry.clone();
    entry.mvcc_version = version;
    Ok(
        point_from_entry(&entry)?.map(|point| IndexMutation::Upsert {
            collection: collection_from_entry(&entry),
            point,
        }),
    )
}

pub fn point_from_entry(entry: &ContextEntry) -> Result<Option<IndexPoint>> {
    let Some(value) = entry.metadata.custom.get(VECTOR_KEY) else {
        return Ok(None);
    };
    let vector: Vec<f32> = serde_json::from_value(value.clone())?;
    if vector.is_empty() || vector.iter().any(|v| !v.is_finite()) {
        return Err(ContextError::Storage(
            "embedding_vector must contain finite values".into(),
        ));
    }
    let collection = entry
        .metadata
        .custom
        .get("vector_collection")
        .and_then(|v| v.as_str())
        .unwrap_or(DEFAULT_COLLECTION);
    let payload = serde_json::json!({
        "collection": collection,
        "mvcc_version": entry.mvcc_version.0,
        "content_type": entry.metadata.content_type.map(|v| v.as_path_segment()),
    });
    Ok(Some(IndexPoint {
        uri: entry.uri.clone(),
        vector: vector.clone(),
        embedding_model_id: entry
            .metadata
            .custom
            .get("embedding_model_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        embedding_dim: Some(vector.len()),
        embedding_version: entry
            .metadata
            .custom
            .get("embedding_version")
            .and_then(|v| v.as_u64()),
        payload,
    }))
}

fn mutation_order(mutation: &IndexMutation) -> (String, Option<i64>) {
    match mutation {
        IndexMutation::Upsert { point, .. } => (
            point.uri.to_string(),
            point
                .payload
                .get("mvcc_version")
                .and_then(|v| v.as_u64())
                .and_then(|v| i64::try_from(v).ok()),
        ),
        IndexMutation::Delete { uri, .. } => (uri.to_string(), None),
        IndexMutation::Rename { point, from, .. } => (
            point
                .as_ref()
                .map_or_else(|| from.to_string(), |p| p.uri.to_string()),
            point
                .as_ref()
                .and_then(|p| p.payload.get("mvcc_version"))
                .and_then(|v| v.as_u64())
                .and_then(|v| i64::try_from(v).ok()),
        ),
    }
}

pub async fn enqueue_pg(
    tx: &mut Transaction<'_, Postgres>,
    mutation: &IndexMutation,
) -> Result<()> {
    let (uri, version) = mutation_order(mutation);
    sqlx::query("INSERT INTO context_index_outbox (id, mutation_json, uri, mvcc_version, status, attempts, available_at, created_at, updated_at) VALUES ($1,$2,$3,$4,'pending',0,now(),now(),now())")
        .bind(Uuid::now_v7()).bind(serde_json::to_value(mutation)?).bind(uri).bind(version)
        .execute(&mut **tx).await.map_err(|e| ContextError::Storage(format!("enqueue index outbox: {e}")))?;
    Ok(())
}

pub async fn enqueue_sqlite(
    tx: &mut Transaction<'_, Sqlite>,
    mutation: &IndexMutation,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let (uri, version) = mutation_order(mutation);
    sqlx::query("INSERT INTO context_index_outbox (id, mutation_json, uri, mvcc_version, status, attempts, available_at, created_at, updated_at) VALUES (?,?,?,?,'pending',0,?,?,?)")
        .bind(Uuid::now_v7().to_string()).bind(serde_json::to_string(mutation)?).bind(uri).bind(version).bind(&now).bind(&now).bind(&now)
        .execute(&mut **tx).await.map_err(|e| ContextError::Storage(format!("enqueue index outbox: {e}")))?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct OutboxConfig {
    pub poll_interval: Duration,
    pub reconciliation_interval: Duration,
    pub lease: Duration,
    pub max_attempts: u32,
    pub batch_size: usize,
}
impl Default for OutboxConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(200),
            reconciliation_interval: Duration::from_secs(300),
            lease: Duration::from_secs(30),
            max_attempts: 8,
            batch_size: 64,
        }
    }
}

pub struct OutboxRuntime {
    stop: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
}
impl OutboxRuntime {
    pub async fn shutdown(mut self) -> Result<()> {
        self.stop
            .send(true)
            .map_err(|error| ContextError::Storage(format!("stop outbox worker: {error}")))?;
        if let Some(task) = self.task.take() {
            task.await
                .map_err(|error| ContextError::Storage(format!("join outbox worker: {error}")))?;
        }
        Ok(())
    }
}
impl Drop for OutboxRuntime {
    fn drop(&mut self) {
        if let Err(error) = self.stop.send(true) {
            tracing::warn!(error = ?agent_context_db_core::ErrorReport::from_error(&error), "failed to signal outbox worker during drop");
        }
        if let Some(task) = self.task.take() {
            task.abort();
            tracing::debug!("aborted outbox worker during drop");
        }
    }
}

pub fn start_worker(
    pool: Arc<DbPool>,
    index: Arc<dyn VectorIndex>,
    config: OutboxConfig,
) -> OutboxRuntime {
    let (stop, mut stopped) = watch::channel(false);
    let task = tokio::spawn(async move {
        let mut reconcile = tokio::time::interval(config.reconciliation_interval);
        loop {
            tokio::select! {
                _ = stopped.changed() => if *stopped.borrow() { break },
                _ = reconcile.tick() => { if let Err(e) = reconcile_all(&pool).await { tracing::error!(error = ?agent_context_db_core::ErrorReport::from_error(&e), "index reconciliation failed"); } },
                _ = tokio::time::sleep(config.poll_interval) => {
                    if let Err(e) = process_batch(&pool, index.as_ref(), &config).await { tracing::error!(error = ?agent_context_db_core::ErrorReport::from_error(&e), "index outbox batch failed"); }
                }
            }
        }
    });
    OutboxRuntime {
        stop,
        task: Some(task),
    }
}

async fn process_batch(pool: &DbPool, index: &dyn VectorIndex, cfg: &OutboxConfig) -> Result<()> {
    let jobs = claim(pool, cfg).await?;
    for (id, mutation, attempts) in jobs {
        let result = apply(index, &mutation).await;
        finish(pool, &id, attempts, cfg.max_attempts, result.as_ref().err()).await?;
    }
    Ok(())
}

async fn apply(index: &dyn VectorIndex, mutation: &IndexMutation) -> Result<()> {
    match mutation {
        IndexMutation::Upsert { collection, point } => {
            index.upsert(collection, point.clone()).await
        }
        IndexMutation::Delete { collection, uri } => index.delete(collection, uri).await,
        IndexMutation::Rename {
            collection,
            from,
            point,
        } => {
            // Delete first; retries repeat both operations safely. A failed upsert therefore cannot
            // leave the stale URI visible indefinitely.
            index.delete(collection, from).await?;
            if let Some(point) = point {
                index.upsert(collection, point.clone()).await?;
            }
            Ok(())
        }
    }
}

async fn claim(pool: &DbPool, cfg: &OutboxConfig) -> Result<Vec<(String, IndexMutation, u32)>> {
    let lease_secs = i64::try_from(cfg.lease.as_secs()).unwrap_or(i64::MAX);
    let limit = i64::try_from(cfg.batch_size).unwrap_or(i64::MAX);
    match pool.backend() {
        uwu_database::SqlBackend::Postgres => {
            let pg = pool
                .as_postgres()
                .map_err(|e| ContextError::Storage(e.to_string()))?;
            let mut tx = pg.begin().await.map_err(err)?;
            let rows: Vec<(Uuid,serde_json::Value,i32)>=sqlx::query_as("WITH picked AS (SELECT id FROM context_index_outbox WHERE (status IN ('pending','failed') AND available_at <= now()) OR (status='processing' AND lease_until < now()) AND NOT EXISTS (SELECT 1 FROM context_index_outbox older WHERE older.uri=context_index_outbox.uri AND older.status NOT IN ('done','dead') AND (older.mvcc_version < context_index_outbox.mvcc_version OR (older.mvcc_version = context_index_outbox.mvcc_version AND older.created_at < context_index_outbox.created_at))) ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT $1) UPDATE context_index_outbox o SET status='processing', lease_until=now()+($2 * interval '1 second'), attempts=attempts+1, updated_at=now() FROM picked WHERE o.id=picked.id RETURNING o.id,o.mutation_json,o.attempts")
                .bind(limit).bind(lease_secs).fetch_all(&mut *tx).await.map_err(err)?;
            tx.commit().await.map_err(err)?;
            rows.into_iter()
                .map(|(id, v, a)| Ok((id.to_string(), serde_json::from_value(v)?, a as u32)))
                .collect()
        }
        uwu_database::SqlBackend::Sqlite => {
            let db = pool
                .as_sqlite()
                .map_err(|e| ContextError::Storage(e.to_string()))?;
            let mut tx = db.begin().await.map_err(err)?;
            let now = chrono::Utc::now();
            let lease = (now + chrono::Duration::seconds(lease_secs)).to_rfc3339();
            let rows: Vec<(String,String,i64)>=sqlx::query_as("UPDATE context_index_outbox SET status='processing', lease_until=?, attempts=attempts+1, updated_at=? WHERE id IN (SELECT id FROM context_index_outbox WHERE ((status IN ('pending','failed') AND available_at <= ?) OR (status='processing' AND lease_until < ?)) AND NOT EXISTS (SELECT 1 FROM context_index_outbox older WHERE older.uri=context_index_outbox.uri AND older.status NOT IN ('done','dead') AND (older.mvcc_version < context_index_outbox.mvcc_version OR (older.mvcc_version = context_index_outbox.mvcc_version AND older.created_at < context_index_outbox.created_at))) ORDER BY created_at LIMIT ?) RETURNING id,mutation_json,attempts")
                .bind(lease).bind(now.to_rfc3339()).bind(now.to_rfc3339()).bind(now.to_rfc3339()).bind(limit).fetch_all(&mut *tx).await.map_err(err)?;
            tx.commit().await.map_err(err)?;
            rows.into_iter()
                .map(|(id, v, a)| Ok((id, serde_json::from_str(&v)?, a as u32)))
                .collect()
        }
        b => Err(ContextError::Storage(format!(
            "unsupported outbox backend: {b:?}"
        ))),
    }
}

async fn finish(
    pool: &DbPool,
    id: &str,
    attempts: u32,
    max: u32,
    error: Option<&ContextError>,
) -> Result<()> {
    let dead = error.is_some() && attempts >= max;
    let status = if error.is_none() {
        "done"
    } else if dead {
        "dead"
    } else {
        "failed"
    };
    let delay = 2_i64.saturating_pow(attempts.min(16)).min(3600);
    let message = error.map(ToString::to_string);
    match pool.backend() {
        uwu_database::SqlBackend::Postgres => {
            sqlx::query("UPDATE context_index_outbox SET status=$2,last_error=$3,available_at=now()+($4*interval '1 second'),lease_until=NULL,updated_at=now(),finished_at=CASE WHEN $2 IN ('done','dead') THEN now() ELSE NULL END WHERE id=$1")
            .bind(Uuid::parse_str(id).map_err(|e|ContextError::Storage(e.to_string()))?).bind(status).bind(message).bind(delay).execute(pool.as_postgres().map_err(|e|ContextError::Storage(e.to_string()))?).await.map_err(err)?;
        }
        uwu_database::SqlBackend::Sqlite => {
            let now = chrono::Utc::now();
            sqlx::query("UPDATE context_index_outbox SET status=?,last_error=?,available_at=?,lease_until=NULL,updated_at=?,finished_at=CASE WHEN ? IN ('done','dead') THEN ? ELSE NULL END WHERE id=?")
            .bind(status).bind(message).bind((now+chrono::Duration::seconds(delay)).to_rfc3339()).bind(now.to_rfc3339()).bind(status).bind(now.to_rfc3339()).bind(id).execute(pool.as_sqlite().map_err(|e|ContextError::Storage(e.to_string()))?).await.map_err(err)?;
        }
        b => {
            return Err(ContextError::Storage(format!(
                "unsupported outbox backend: {b:?}"
            )));
        }
    }
    Ok(())
}

/// Reconciliation is durable: it emits idempotent upserts for every embedded fact and deletes are
/// already represented by transactional tombstones. This repairs missing/corrupt points after an
/// index reset without requiring vector-backend enumeration support.
async fn reconcile_all(pool: &DbPool) -> Result<()> {
    match pool.backend() {
        uwu_database::SqlBackend::Postgres => {
            let pg = pool
                .as_postgres()
                .map_err(|e| ContextError::Storage(e.to_string()))?;
            let mut tx = pg.begin().await.map_err(err)?;
            let rows: Vec<serde_json::Value> =
                sqlx::query_scalar("SELECT entry_json FROM context_entries ORDER BY uri")
                    .fetch_all(&mut *tx)
                    .await
                    .map_err(err)?;
            for value in rows {
                let entry: ContextEntry = serde_json::from_value(value)?;
                if let Some(point) = point_from_entry(&entry)? {
                    enqueue_pg(
                        &mut tx,
                        &IndexMutation::Upsert {
                            collection: collection_from_entry(&entry),
                            point,
                        },
                    )
                    .await?;
                }
            }
            tx.commit().await.map_err(err)?;
        }
        uwu_database::SqlBackend::Sqlite => {
            let db = pool
                .as_sqlite()
                .map_err(|e| ContextError::Storage(e.to_string()))?;
            let mut tx = db.begin().await.map_err(err)?;
            let rows: Vec<String> =
                sqlx::query_scalar("SELECT entry_json FROM context_entries ORDER BY uri")
                    .fetch_all(&mut *tx)
                    .await
                    .map_err(err)?;
            for value in rows {
                let entry: ContextEntry = serde_json::from_str(&value)?;
                if let Some(point) = point_from_entry(&entry)? {
                    enqueue_sqlite(
                        &mut tx,
                        &IndexMutation::Upsert {
                            collection: collection_from_entry(&entry),
                            point,
                        },
                    )
                    .await?;
                }
            }
            tx.commit().await.map_err(err)?;
        }
        b => {
            return Err(ContextError::Storage(format!(
                "unsupported reconciliation backend: {b:?}"
            )));
        }
    }
    Ok(())
}
fn err(e: impl std::fmt::Display) -> ContextError {
    ContextError::Storage(format!("index outbox: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SqliteContextStore, migrate_sqlite};
    use agent_context_db_core::{
        ContentRepo, ContentType, ContextMeta, IndexHit, MediaType, MvccVersion, TenantId,
    };
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uwu_database::config::{
        CacheBackend, CacheConfig, DbConfig, DeployConfig, RuntimeConfig, SqlBackend,
        VectorBackend, VectorConfig,
    };

    #[derive(Default)]
    struct TestIndex {
        points: Mutex<HashMap<(String, ContextUri), IndexPoint>>,
        deletes: Mutex<Vec<(String, ContextUri)>>,
        failures: AtomicUsize,
        calls: AtomicUsize,
    }

    impl TestIndex {
        fn failing(times: usize) -> Self {
            Self {
                failures: AtomicUsize::new(times),
                ..Self::default()
            }
        }

        fn contains(&self, collection: &str, uri: &ContextUri) -> bool {
            self.points
                .lock()
                .contains_key(&(collection.to_owned(), uri.clone()))
        }
    }

    #[async_trait]
    impl VectorIndex for TestIndex {
        async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self
                .failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                    left.checked_sub(1)
                })
                .is_ok()
            {
                return Err(ContextError::Storage("injected index failure".into()));
            }
            self.points
                .lock()
                .insert((collection.to_owned(), point.uri.clone()), point);
            Ok(())
        }

        async fn search(
            &self,
            _collection: &str,
            _query: Vec<f32>,
            _top_k: usize,
            _filter: Option<serde_json::Value>,
        ) -> Result<Vec<IndexHit>> {
            Ok(Vec::new())
        }

        async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.points
                .lock()
                .remove(&(collection.to_owned(), uri.clone()));
            self.deletes
                .lock()
                .push((collection.to_owned(), uri.clone()));
            Ok(())
        }
    }

    async fn database() -> (Arc<DbPool>, SqliteContextStore) {
        let config = RuntimeConfig {
            deploy: DeployConfig::default(),
            database: DbConfig {
                backend: SqlBackend::Sqlite,
                url: "sqlite::memory:".into(),
                max_connections: 1,
                min_connections: 0,
                acquire_timeout_secs: 5,
                idle_timeout_secs: 60,
                max_lifetime_secs: 300,
                test_before_acquire: false,
                statement_cache_capacity: 100,
                application_name: None,
            },
            cache: CacheConfig {
                backend: CacheBackend::None,
                url: None,
                capacity: 0,
            },
            vector: VectorConfig {
                backend: VectorBackend::Memory,
                url: None,
                api_key: None,
            },
        };
        let database = uwu_database::Database::connect(&config).await.unwrap();
        migrate_sqlite(&database.pool).await.unwrap();
        let pool = Arc::new(database.pool);
        let store =
            SqliteContextStore::try_new(pool.clone(), crate::GraphCentralityConfig::default())
                .unwrap();
        (pool, store)
    }

    fn entry(name: &str, collection: &str) -> ContextEntry {
        let mut entry = ContextEntry::new_text(
            ContextUri::parse(format!("uwu://tenant/agent/a/memory/fact/outbox/{name}")).unwrap(),
            TenantId(Uuid::new_v4()),
            name,
        );
        entry.metadata = ContextMeta {
            content_type: Some(ContentType::Fact),
            custom: serde_json::json!({
                "embedding_vector": [0.1, 0.2],
                "vector_collection": collection,
            }),
            ..ContextMeta::default()
        };
        entry.media_type = MediaType::Text;
        entry
    }

    fn config(max_attempts: u32) -> OutboxConfig {
        OutboxConfig {
            poll_interval: Duration::from_millis(5),
            reconciliation_interval: Duration::from_secs(3600),
            lease: Duration::from_secs(1),
            max_attempts,
            batch_size: 32,
        }
    }

    async fn force_ready(pool: &DbPool) {
        sqlx::query("UPDATE context_index_outbox SET available_at = ? WHERE status = 'failed'")
            .bind((chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339())
            .execute(pool.as_sqlite().unwrap())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn sqlite_content_and_outbox_roll_back_together() {
        let (pool, store) = database().await;
        sqlx::query("DROP TABLE context_index_outbox")
            .execute(pool.as_sqlite().unwrap())
            .await
            .unwrap();
        let value = entry("rollback", "atomic");
        assert!(ContentRepo::write(&store, value.clone()).await.is_err());
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM context_entries WHERE uri = ?")
            .bind(value.uri.to_string())
            .fetch_one(pool.as_sqlite().unwrap())
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn sqlite_write_batch_rename_and_delete_emit_correct_mutations() {
        let (pool, store) = database().await;
        let first = entry("first", "alpha");
        let second = entry("second", "beta");
        ContentRepo::batch_write(&store, &[first.clone(), second.clone()])
            .await
            .unwrap();
        let renamed = ContextUri::parse("uwu://tenant/agent/a/memory/fact/outbox/renamed").unwrap();
        ContentRepo::rename(&store, &first.uri, &renamed)
            .await
            .unwrap();
        ContentRepo::delete(&store, &second.uri).await.unwrap();
        let rows: Vec<String> = sqlx::query_scalar(
            "SELECT mutation_json FROM context_index_outbox ORDER BY created_at, id",
        )
        .fetch_all(pool.as_sqlite().unwrap())
        .await
        .unwrap();
        let mutations: Vec<IndexMutation> = rows
            .iter()
            .map(|row| serde_json::from_str(row).unwrap())
            .collect();
        assert_eq!(mutations.len(), 4);
        assert!(
            matches!(&mutations[2], IndexMutation::Rename { collection, from, point: Some(point) } if collection == "alpha" && from == &first.uri && point.uri == renamed)
        );
        assert!(
            matches!(&mutations[3], IndexMutation::Delete { collection, uri } if collection == "beta" && uri == &second.uri)
        );
    }

    #[tokio::test]
    async fn duplicate_delivery_is_idempotent() {
        let (pool, store) = database().await;
        let value = entry("duplicate", "dedup");
        ContentRepo::write(&store, value.clone()).await.unwrap();
        let mutation = upsert_mutation(&value, MvccVersion(1)).unwrap().unwrap();
        let mut tx = pool.as_sqlite().unwrap().begin().await.unwrap();
        enqueue_sqlite(&mut tx, &mutation).await.unwrap();
        tx.commit().await.unwrap();
        let index = TestIndex::default();
        process_batch(&pool, &index, &config(3)).await.unwrap();
        assert!(index.contains("dedup", &value.uri));
        assert_eq!(index.points.lock().len(), 1);
        assert_eq!(index.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failures_back_off_and_eventually_become_dead() {
        let (pool, store) = database().await;
        ContentRepo::write(&store, entry("dead", "failures"))
            .await
            .unwrap();
        let index = TestIndex::failing(2);
        let cfg = config(2);
        process_batch(&pool, &index, &cfg).await.unwrap();
        let (status, attempts, available_at): (String, i64, String) =
            sqlx::query_as("SELECT status, attempts, available_at FROM context_index_outbox")
                .fetch_one(pool.as_sqlite().unwrap())
                .await
                .unwrap();
        assert_eq!(status, "failed");
        assert_eq!(attempts, 1);
        assert!(available_at > chrono::Utc::now().to_rfc3339());
        force_ready(&pool).await;
        process_batch(&pool, &index, &cfg).await.unwrap();
        let (status, attempts, finished_at): (String, i64, Option<String>) =
            sqlx::query_as("SELECT status, attempts, finished_at FROM context_index_outbox")
                .fetch_one(pool.as_sqlite().unwrap())
                .await
                .unwrap();
        assert_eq!((status.as_str(), attempts), ("dead", 2));
        assert!(finished_at.is_some());
    }

    #[tokio::test]
    async fn expired_processing_lease_is_reclaimed() {
        let (pool, store) = database().await;
        let value = entry("lease", "recovery");
        ContentRepo::write(&store, value.clone()).await.unwrap();
        sqlx::query("UPDATE context_index_outbox SET status='processing', lease_until=?")
            .bind((chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339())
            .execute(pool.as_sqlite().unwrap())
            .await
            .unwrap();
        let index = TestIndex::default();
        process_batch(&pool, &index, &config(3)).await.unwrap();
        assert!(index.contains("recovery", &value.uri));
        let (status, attempts): (String, i64) =
            sqlx::query_as("SELECT status, attempts FROM context_index_outbox")
                .fetch_one(pool.as_sqlite().unwrap())
                .await
                .unwrap();
        assert_eq!((status.as_str(), attempts), ("done", 1));
    }

    #[tokio::test]
    async fn reconciliation_rebuilds_a_missing_point() {
        let (pool, store) = database().await;
        let value = entry("reconcile", "repair");
        ContentRepo::write(&store, value.clone()).await.unwrap();
        sqlx::query("DELETE FROM context_index_outbox")
            .execute(pool.as_sqlite().unwrap())
            .await
            .unwrap();
        reconcile_all(&pool).await.unwrap();
        let index = TestIndex::default();
        process_batch(&pool, &index, &config(3)).await.unwrap();
        assert!(index.contains("repair", &value.uri));
    }

    #[tokio::test]
    async fn runtime_stays_alive_until_last_owner_is_dropped() {
        let (pool, store) = database().await;
        let index = Arc::new(TestIndex::default());
        let runtime = Arc::new(start_worker(pool, index.clone(), config(3)));
        let clone = runtime.clone();
        drop(runtime);
        ContentRepo::write(&store, entry("runtime", "lifecycle"))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while index.calls.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(Arc::strong_count(&clone), 1);
        drop(clone);
    }
}
