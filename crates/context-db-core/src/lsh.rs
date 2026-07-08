//! LSH 索引 — 增量更新的近似近邻索引，替代 O(N²) 暴力比较。

use crate::ContextUri;
use std::collections::{HashMap, HashSet};

/// LSH 哈希表。
pub struct LshIndex {
    tables: Vec<LshHashTable>,
    dimension: Option<usize>,
    /// 每个 hash bucket 中的所有 URI。
    buckets: HashMap<u64, Vec<ContextUri>>,
}

impl LshIndex {
    pub fn new(num_tables: usize, hash_size: usize) -> Self {
        Self {
            tables: (0..num_tables)
                .map(|table_id| LshHashTable::new(hash_size, table_id as u64))
                .collect(),
            dimension: None,
            buckets: HashMap::new(),
        }
    }

    /// 插入 embedding，只与同 bucket 的条目比较。
    pub fn insert(&mut self, uri: &ContextUri, embedding: &[f32]) {
        if embedding.is_empty() {
            return;
        }
        self.ensure_dimension(embedding.len());
        for (table_id, table) in self.tables.iter().enumerate() {
            let hash = table.hash(embedding);
            let key = bucket_key(table_id, hash);
            let bucket = self.buckets.entry(key).or_default();
            if !bucket.iter().any(|u| u == uri) {
                bucket.push(uri.clone());
            }
        }
    }

    /// 查询近邻 — O(1) per hash table。
    pub fn query(&mut self, embedding: &[f32]) -> Vec<ContextUri> {
        if embedding.is_empty() {
            return vec![];
        }
        self.ensure_dimension(embedding.len());
        let mut candidates = HashSet::new();
        for (table_id, table) in self.tables.iter().enumerate() {
            let hash = table.hash(embedding);
            let key = bucket_key(table_id, hash);
            if let Some(bucket) = self.buckets.get(&key) {
                for uri in bucket {
                    candidates.insert(uri.clone());
                }
            }
        }
        candidates.into_iter().collect()
    }

    /// 移除一个 URI。
    pub fn remove(&mut self, uri: &ContextUri) {
        for bucket in self.buckets.values_mut() {
            bucket.retain(|u| u != uri);
        }
    }

    pub fn len(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }

    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    fn ensure_dimension(&mut self, dimension: usize) {
        if self.dimension == Some(dimension) {
            return;
        }
        self.dimension = Some(dimension);
        self.buckets.clear();
        for table in &mut self.tables {
            table.initialize(dimension);
        }
    }
}

/// 单个 LSH 哈希表 — 使用随机投影。
pub struct LshHashTable {
    hash_size: usize,
    seed: u64,
    random_vectors: Vec<Vec<f32>>,
}

impl LshHashTable {
    pub fn new(hash_size: usize, seed: u64) -> Self {
        Self {
            hash_size: hash_size.min(64),
            seed,
            random_vectors: vec![],
        }
    }

    pub fn initialize(&mut self, dimension: usize) {
        if dimension == 0 {
            self.random_vectors.clear();
            return;
        }
        self.random_vectors = (0..self.hash_size)
            .map(|bit| {
                let mut norm = 0.0f32;
                let mut vector = (0..dimension)
                    .map(|dim| {
                        let v = projection_value(self.seed, bit as u64, dim as u64);
                        norm += v * v;
                        v
                    })
                    .collect::<Vec<_>>();
                let norm = norm.sqrt().max(f32::EPSILON);
                for value in &mut vector {
                    *value /= norm;
                }
                vector
            })
            .collect();
    }

    pub fn is_initialized(&self) -> bool {
        !self.random_vectors.is_empty()
    }

    /// 对 embedding 做 LSH 哈希：取每个随机投影的正负号。
    pub fn hash(&self, embedding: &[f32]) -> u64 {
        let mut hash: u64 = 0;
        for (i, rv) in self.random_vectors.iter().enumerate() {
            let dot: f32 = embedding.iter().zip(rv.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                hash |= 1 << i;
            }
        }
        hash
    }
}

fn bucket_key(table_id: usize, hash: u64) -> u64 {
    ((table_id as u64) << 56) ^ hash
}

fn projection_value(seed: u64, bit: u64, dim: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(bit.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(dim.wrapping_mul(0x94D0_49BB_1331_11EB));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = (x as f64 / u64::MAX as f64) as f32;
    unit * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsh_insert_and_query() {
        let mut index = LshIndex::new(3, 16);
        let uri1 = ContextUri::parse("uwu://t/a/x/fact/1").unwrap();
        let uri2 = ContextUri::parse("uwu://t/a/x/fact/2").unwrap();
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.9, 0.1, 0.0];
        index.insert(&uri1, &v1);
        index.insert(&uri2, &v2);
        assert_eq!(index.dimension(), Some(3));
        assert!(index.tables.iter().all(LshHashTable::is_initialized));
        let results = index.query(&v1);
        assert!(results.contains(&uri1));
    }

    #[test]
    fn lsh_uses_table_specific_buckets_without_duplicates() {
        let mut index = LshIndex::new(4, 12);
        let uri = ContextUri::parse("uwu://t/a/x/fact/1").unwrap();
        let v = vec![0.2, 0.4, -0.1, 0.7];
        index.insert(&uri, &v);
        index.insert(&uri, &v);
        assert_eq!(index.len(), 4);
        assert_eq!(index.query(&v), vec![uri]);
    }
}
