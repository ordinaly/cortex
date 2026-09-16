//! Sparse relation and dependency primitives for the Cortex v1 runtime.
//! The store allocates only represented relations; absent pairs cost no relation cell.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

pub type NodeId = u32;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RelationCell {
    pub value: f64,
    pub confidence: f64,
    pub last_updated: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SparseRelationStore {
    directed: HashMap<(NodeId, NodeId), RelationCell>,
    outgoing: HashMap<NodeId, HashSet<NodeId>>,
    incoming: HashMap<NodeId, HashSet<NodeId>>,
}

impl SparseRelationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.directed.len()
    }
    pub fn is_empty(&self) -> bool {
        self.directed.is_empty()
    }

    pub fn get(&self, a: NodeId, b: NodeId) -> Option<&RelationCell> {
        self.directed.get(&(a, b))
    }

    pub fn upsert(&mut self, a: NodeId, b: NodeId, cell: RelationCell) -> Option<RelationCell> {
        self.outgoing.entry(a).or_default().insert(b);
        self.incoming.entry(b).or_default().insert(a);
        self.directed.insert((a, b), cell)
    }

    pub fn remove(&mut self, a: NodeId, b: NodeId) -> Option<RelationCell> {
        let old = self.directed.remove(&(a, b));
        if old.is_some() {
            if let Some(s) = self.outgoing.get_mut(&a) {
                s.remove(&b);
                if s.is_empty() {
                    self.outgoing.remove(&a);
                }
            }
            if let Some(s) = self.incoming.get_mut(&b) {
                s.remove(&a);
                if s.is_empty() {
                    self.incoming.remove(&b);
                }
            }
        }
        old
    }

    pub fn outgoing(&self, a: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.outgoing
            .get(&a)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }

    pub fn incoming(&self, b: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.incoming
            .get(&b)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    dependents: HashMap<NodeId, Vec<NodeId>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_dependency(&mut self, source: NodeId, dependent: NodeId) {
        let v = self.dependents.entry(source).or_default();
        if !v.contains(&dependent) {
            v.push(dependent);
        }
    }

    pub fn closure<I>(&self, dirty: I) -> Vec<NodeId>
    where
        I: IntoIterator<Item = NodeId>,
    {
        let mut q = VecDeque::new();
        let mut seen = HashSet::new();
        for n in dirty {
            if seen.insert(n) {
                q.push_back(n);
            }
        }
        while let Some(n) = q.pop_front() {
            if let Some(ds) = self.dependents.get(&n) {
                for &d in ds {
                    if seen.insert(d) {
                        q.push_back(d);
                    }
                }
            }
        }
        let mut out: Vec<_> = seen.into_iter().collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_store_only_allocates_present_edges() {
        let mut s = SparseRelationStore::new();
        s.upsert(
            1,
            9,
            RelationCell {
                value: 0.7,
                confidence: 0.8,
                last_updated: 1,
            },
        );
        assert_eq!(s.len(), 1);
        assert_eq!(s.outgoing(1).collect::<Vec<_>>(), vec![9]);
        assert!(s.get(9, 1).is_none());
        s.remove(1, 9);
        assert!(s.is_empty());
    }

    #[test]
    fn dependency_closure_is_local_and_transitive() {
        let mut g = DependencyGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        g.add_dependency(9, 10);
        assert_eq!(g.closure([1]), vec![1, 2, 3]);
    }
}
