//! Sparse relation, causal-evidence, and dependency primitives for Cortex.
//!
//! Unlike the frozen Python v0.7 `PairIndex`, the native evidence carrier does
//! not preallocate all O(N^2) entity pairs. A relation or directed causal cell
//! is created only after a visible frame supplies evidence for that pair.

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EvidenceConfig {
    pub relation_min_exposure: u32,
    pub relation_active_threshold: f64,
    pub relation_absent_threshold: f64,
    pub causal_min_do: u32,
    pub causal_min_control: u32,
    pub causal_positive_diff: f64,
    pub causal_null_abs_diff: f64,
}

impl Default for EvidenceConfig {
    fn default() -> Self {
        Self {
            relation_min_exposure: 8,
            relation_active_threshold: 0.68,
            relation_absent_threshold: 0.32,
            causal_min_do: 8,
            causal_min_control: 8,
            causal_positive_diff: 0.24,
            causal_null_abs_diff: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct RelationEvidence {
    pub a: f64,
    pub b: f64,
    pub exposure: u32,
    /// -1 absent, 0 unresolved, +1 active.
    pub state: i8,
    pub last_updated: u64,
}

impl Default for RelationEvidence {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 1.0,
            exposure: 0,
            state: 0,
            last_updated: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CausalEvidence {
    pub do_a: f64,
    pub do_b: f64,
    pub ctrl_a: f64,
    pub ctrl_b: f64,
    pub do_n: u32,
    pub ctrl_n: u32,
    /// -1 null, 0 unresolved, +1 causal.
    pub state: i8,
    pub last_updated: u64,
}

impl Default for CausalEvidence {
    fn default() -> Self {
        Self {
            do_a: 1.0,
            do_b: 1.0,
            ctrl_a: 1.0,
            ctrl_b: 1.0,
            do_n: 0,
            ctrl_n: 0,
            state: 0,
            last_updated: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationEvidenceEntry {
    pub a: NodeId,
    pub b: NodeId,
    pub evidence: RelationEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalEvidenceEntry {
    pub source: NodeId,
    pub target: NodeId,
    pub evidence: CausalEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceSnapshot {
    pub time: u64,
    pub relations: Vec<RelationEvidenceEntry>,
    pub causal: Vec<CausalEvidenceEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceRead {
    pub relation_changed: Vec<(NodeId, NodeId)>,
    pub causal_changed: Vec<(NodeId, NodeId)>,
    pub represented_relations: usize,
    pub represented_causal: usize,
}

/// Lightweight graph-state statistics needed by the public articulation vector.
///
/// These counters are maintained incrementally so the hot path does not need to
/// materialize or sort a full graph snapshot on every frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceSummary {
    pub relation_active: usize,
    pub relation_resolved: usize,
    pub causal_active: usize,
}

/// Sparse online sufficient statistics for v0.7 relation and causal semantics.
///
/// Relation cells are undirected and normalized as `(min(a,b), max(a,b))`.
/// Causal cells are directed `(source, target)`. Cells are allocated lazily on
/// first exposure, so entity capacity by itself does not imply O(N^2) storage.
#[derive(Debug, Clone)]
pub struct SparseEvidenceGraph {
    pub cfg: EvidenceConfig,
    relations: HashMap<(NodeId, NodeId), RelationEvidence>,
    causal: HashMap<(NodeId, NodeId), CausalEvidence>,
    relation_active: usize,
    relation_resolved: usize,
    causal_active: usize,
    time: u64,
}

impl SparseEvidenceGraph {
    pub fn new(cfg: EvidenceConfig) -> Result<Self, String> {
        if cfg.relation_min_exposure == 0 || cfg.causal_min_do == 0 || cfg.causal_min_control == 0 {
            return Err("evidence count thresholds must be positive".into());
        }
        if !(0.0..=1.0).contains(&cfg.relation_absent_threshold)
            || !(0.0..=1.0).contains(&cfg.relation_active_threshold)
            || cfg.relation_absent_threshold > cfg.relation_active_threshold
        {
            return Err("invalid relation thresholds".into());
        }
        if !cfg.causal_positive_diff.is_finite()
            || !cfg.causal_null_abs_diff.is_finite()
            || cfg.causal_positive_diff < 0.0
            || cfg.causal_null_abs_diff < 0.0
        {
            return Err("invalid causal thresholds".into());
        }
        Ok(Self {
            cfg,
            relations: HashMap::new(),
            causal: HashMap::new(),
            relation_active: 0,
            relation_resolved: 0,
            causal_active: 0,
            time: 0,
        })
    }

    pub fn represented_relations(&self) -> usize {
        self.relations.len()
    }

    pub fn represented_causal(&self) -> usize {
        self.causal.len()
    }

    pub fn summary(&self) -> EvidenceSummary {
        EvidenceSummary {
            relation_active: self.relation_active,
            relation_resolved: self.relation_resolved,
            causal_active: self.causal_active,
        }
    }

    pub fn relation(&self, a: NodeId, b: NodeId) -> Option<&RelationEvidence> {
        self.relations.get(&undirected_key(a, b))
    }

    pub fn causal(&self, source: NodeId, target: NodeId) -> Option<&CausalEvidence> {
        self.causal.get(&(source, target))
    }

    /// Update graph evidence from one already-bound frame.
    ///
    /// `relation_obs` uses detection indices `(i,j,bit)`. As in frozen v0.7,
    /// every co-visible entity pair receives a relation exposure; omitted pairs
    /// are observed as bit 0. `outcomes` attaches a binary outcome to a target
    /// detection index. For a visible ordered source-target pair, evidence is
    /// counted as interventional only when the source detection equals
    /// `intervention_src_det`; otherwise it is a matched control observation.
    pub fn observe(
        &mut self,
        bindings: &[NodeId],
        relation_obs: &[(usize, usize, u8)],
        intervention_src_det: Option<usize>,
        outcomes: &[(usize, u8)],
    ) -> Result<EvidenceRead, String> {
        for &(_, _, y) in relation_obs {
            if y > 1 {
                return Err("relation observations must be binary".into());
            }
        }
        for &(_, y) in outcomes {
            if y > 1 {
                return Err("outcomes must be binary".into());
            }
        }
        self.time += 1;

        // Python v0.7 builds dictionaries, so later duplicate observations win.
        let mut rel_lookup: HashMap<(usize, usize), u8> = HashMap::new();
        for &(i, j, y) in relation_obs {
            rel_lookup.insert((i.min(j), i.max(j)), y);
        }
        let mut out_lookup: HashMap<usize, u8> = HashMap::new();
        for &(i, y) in outcomes {
            out_lookup.insert(i, y);
        }

        let mut relation_changed = Vec::new();
        for i in 0..bindings.len() {
            for j in (i + 1)..bindings.len() {
                let a = bindings[i];
                let b = bindings[j];
                if a == b {
                    continue;
                }
                let key = undirected_key(a, b);
                let y = *rel_lookup.get(&(i, j)).unwrap_or(&0);
                let cell = self.relations.entry(key).or_default();
                cell.a += f64::from(y);
                cell.b += f64::from(1 - y);
                cell.exposure += 1;
                cell.last_updated = self.time;
                let next = derive_relation_state(self.cfg, *cell);
                if next != cell.state {
                    let previous = cell.state;
                    cell.state = next;
                    if previous == 1 {
                        self.relation_active -= 1;
                    }
                    if next == 1 {
                        self.relation_active += 1;
                    }
                    if previous != 0 {
                        self.relation_resolved -= 1;
                    }
                    if next != 0 {
                        self.relation_resolved += 1;
                    }
                    relation_changed.push(key);
                }
            }
        }

        let mut causal_changed = Vec::new();
        for (i, &source) in bindings.iter().enumerate() {
            for (j, &target) in bindings.iter().enumerate() {
                if i == j || source == target {
                    continue;
                }
                let Some(&y) = out_lookup.get(&j) else {
                    continue;
                };
                let key = (source, target);
                let cell = self.causal.entry(key).or_default();
                if intervention_src_det == Some(i) {
                    cell.do_a += f64::from(y);
                    cell.do_b += f64::from(1 - y);
                    cell.do_n += 1;
                } else {
                    cell.ctrl_a += f64::from(y);
                    cell.ctrl_b += f64::from(1 - y);
                    cell.ctrl_n += 1;
                }
                cell.last_updated = self.time;
                let next = derive_causal_state(self.cfg, *cell);
                if next != cell.state {
                    let previous = cell.state;
                    cell.state = next;
                    if previous == 1 {
                        self.causal_active -= 1;
                    }
                    if next == 1 {
                        self.causal_active += 1;
                    }
                    causal_changed.push(key);
                }
            }
        }

        relation_changed.sort_unstable();
        causal_changed.sort_unstable();
        Ok(EvidenceRead {
            relation_changed,
            causal_changed,
            represented_relations: self.relations.len(),
            represented_causal: self.causal.len(),
        })
    }

    pub fn snapshot(&self) -> EvidenceSnapshot {
        let mut relations: Vec<_> = self
            .relations
            .iter()
            .map(|(&(a, b), &evidence)| RelationEvidenceEntry { a, b, evidence })
            .collect();
        relations.sort_by_key(|e| (e.a, e.b));
        let mut causal: Vec<_> = self
            .causal
            .iter()
            .map(|(&(source, target), &evidence)| CausalEvidenceEntry {
                source,
                target,
                evidence,
            })
            .collect();
        causal.sort_by_key(|e| (e.source, e.target));
        EvidenceSnapshot {
            time: self.time,
            relations,
            causal,
        }
    }
}

fn undirected_key(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a <= b { (a, b) } else { (b, a) }
}

fn derive_relation_state(cfg: EvidenceConfig, cell: RelationEvidence) -> i8 {
    if cell.exposure < cfg.relation_min_exposure {
        return 0;
    }
    let p = cell.a / (cell.a + cell.b);
    if p >= cfg.relation_active_threshold {
        1
    } else if p <= cfg.relation_absent_threshold {
        -1
    } else {
        0
    }
}

fn derive_causal_state(cfg: EvidenceConfig, cell: CausalEvidence) -> i8 {
    if cell.do_n < cfg.causal_min_do || cell.ctrl_n < cfg.causal_min_control {
        return 0;
    }
    let p_do = cell.do_a / (cell.do_a + cell.do_b);
    let p_ctrl = cell.ctrl_a / (cell.ctrl_a + cell.ctrl_b);
    let diff = p_do - p_ctrl;
    if diff >= cfg.causal_positive_diff {
        1
    } else if diff.abs() <= cfg.causal_null_abs_diff {
        -1
    } else {
        0
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
    fn evidence_cells_are_allocated_only_after_exposure() {
        let mut g = SparseEvidenceGraph::new(EvidenceConfig::default()).unwrap();
        assert_eq!(g.represented_relations(), 0);
        assert_eq!(g.represented_causal(), 0);
        g.observe(&[4, 9], &[(0, 1, 1)], Some(0), &[(1, 1)])
            .unwrap();
        assert_eq!(g.represented_relations(), 1);
        assert_eq!(g.represented_causal(), 1);
        assert!(g.relation(4, 9).is_some());
        assert!(g.causal(4, 9).is_some());
    }

    #[test]
    fn relation_and_causal_thresholds_match_v07_contract() {
        let mut g = SparseEvidenceGraph::new(EvidenceConfig::default()).unwrap();
        for _ in 0..8 {
            g.observe(&[0, 1], &[(0, 1, 1)], Some(0), &[(1, 1)])
                .unwrap();
        }
        assert_eq!(g.relation(0, 1).unwrap().state, 1);
        assert_eq!(g.causal(0, 1).unwrap().state, 0); // no controls yet
        assert_eq!(
            g.summary(),
            EvidenceSummary {
                relation_active: 1,
                relation_resolved: 1,
                causal_active: 0,
            }
        );
        for _ in 0..8 {
            g.observe(&[0, 1], &[(0, 1, 1)], None, &[(1, 0)])
                .unwrap();
        }
        assert_eq!(g.causal(0, 1).unwrap().state, 1);
        assert_eq!(g.summary().causal_active, 1);
    }

    #[test]
    fn summary_counts_follow_relation_state_transitions() {
        let mut cfg = EvidenceConfig::default();
        cfg.relation_min_exposure = 1;
        cfg.relation_active_threshold = 0.75;
        cfg.relation_absent_threshold = 0.25;
        let mut g = SparseEvidenceGraph::new(cfg).unwrap();

        for _ in 0..2 {
            g.observe(&[0, 1], &[(0, 1, 1)], None, &[]).unwrap();
        }
        assert_eq!(g.relation(0, 1).unwrap().state, 1);
        assert_eq!(g.summary().relation_active, 1);
        assert_eq!(g.summary().relation_resolved, 1);

        g.observe(&[0, 1], &[(0, 1, 0)], None, &[]).unwrap();
        assert_eq!(g.relation(0, 1).unwrap().state, 0);
        assert_eq!(g.summary().relation_active, 0);
        assert_eq!(g.summary().relation_resolved, 0);

        for _ in 0..7 {
            g.observe(&[0, 1], &[(0, 1, 0)], None, &[]).unwrap();
        }
        assert_eq!(g.relation(0, 1).unwrap().state, -1);
        assert_eq!(g.summary().relation_active, 0);
        assert_eq!(g.summary().relation_resolved, 1);
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
