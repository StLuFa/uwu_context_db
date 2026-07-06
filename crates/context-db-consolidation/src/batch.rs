//! MapReduceConsolidator — 分块并行巩固。

use crate::{ConsolidationEngine, ConsolidationProduct};
use agent_context_db_core::{ContextEntry, Result};
use std::sync::Arc;

pub struct MapReduceConsolidator {
    engine: Arc<ConsolidationEngine>,
    chunk_size: usize,
    max_concurrency: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BatchReport { pub total: usize, pub products: usize, pub chunks: usize, pub failures: usize }

impl MapReduceConsolidator {
    pub fn new(engine: ConsolidationEngine) -> Self { Self { engine: Arc::new(engine), chunk_size: 100, max_concurrency: 4 } }
    pub async fn batch_consolidate(&self, entries: &[ContextEntry]) -> Result<(Vec<ConsolidationProduct>, BatchReport)> {
        let mut report = BatchReport { total: entries.len(), ..Default::default() };
        let chunks: Vec<&[ContextEntry]> = entries.chunks(self.chunk_size).collect();
        report.chunks = chunks.len();
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(self.max_concurrency));
        let mut handles = Vec::new();
        for chunk in chunks {
            let permit = sem.clone().acquire_owned().await;
            let engine = self.engine.clone();
            let c: Vec<ContextEntry> = chunk.to_vec();
            handles.push(tokio::spawn(async move { let _p = permit; engine.consolidate_batch(&c).await }));
        }
        let mut all = Vec::new();
        for h in handles {
            match h.await { Ok(Ok(p)) => { report.products += p.len(); all.extend(p); } _ => { report.failures += 1; } }
        }
        Ok((all, report))
    }
}
