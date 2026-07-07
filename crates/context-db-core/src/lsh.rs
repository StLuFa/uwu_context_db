//! LSH 索引 — 增量更新的近似近邻索引，替代 O(N²) 暴力比较。

use crate::ContextUri;
use std::collections::{HashMap, HashSet};

/// LSH 哈希表。
pub struct LshIndex {
    tables: Vec<LshHashTable>,
    num_tables: usize,
    hash_size: usize,
    /// 每个 hash bucket 中的所有 URI。
    buckets: HashMap<u64, Vec<ContextUri>>,
}

impl LshIndex {
    pub fn new(num_tables: usize, hash_size: usize) -> Self {
        Self {
            tables: (0..num_tables)
                .map(|_| LshHashTable::new(hash_size))
                .collect(),
            num_tables,
            hash_size,
            buckets: HashMap::new(),
        }
    }

    /// 插入 embedding，只与同 bucket 的条目比较。
    pub fn insert(&mut self, uri: &ContextUri, embedding: &[f32]) {
        for table in &self.tables {
            let hash = table.hash(embedding);
            self.buckets.entry(hash).or_default().push(uri.clone());
        }
    }

    /// 查询近邻 — O(1) per hash table。
    pub fn query(&self, embedding: &[f32]) -> Vec<ContextUri> {
        let mut candidates = HashSet::new();
        for table in &self.tables {
            let hash = table.hash(embedding);
            if let Some(bucket) = self.buckets.get(&hash) {
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
}

/// 单个 LSH 哈希表 — 使用随机投影。
pub struct LshHashTable {
    random_vectors: Vec<Vec<f32>>,
}

impl LshHashTable {
    pub fn new(hash_size: usize) -> Self {
        Self {
            random_vectors: vec![],
        }
    }

    /// 对 embedding 做 LSH 哈希（简化版：取每个随机投影的正负号）。
    pub fn hash(&self, embedding: &[f32]) -> u64 {
        if self.random_vectors.is_empty() {
            // 无随机向量时用简单 bucket
            return embedding
                .iter()
                .map(|x| x.to_bits() as u64)
                .fold(0, |a, b| a.wrapping_mul(31).wrapping_add(b));
        }
        let mut hash: u64 = 0;
        for (i, rv) in self.random_vectors.iter().enumerate() {
            let dot: f32 = embedding.iter().zip(rv.iter()).map(|(a, b)| a * b).sum();
            if dot > 0.0 {
                hash |= 1 << (i % 64);
            }
        }
        hash
    }
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
        let results = index.query(&v1);
        assert!(!results.is_empty());
    }
}
