#!/usr/bin/env python3
"""One-shot migration repair used by the CI-fix branch.

Align Rust tie-breaking with the frozen NumPy reference before differential tests:
NumPy argmax returns the first maximum, while Rust Iterator::max_by returns the
last equal maximum.
"""
from pathlib import Path

reasoner = Path("crates/cortex-core/src/reasoner.rs")
s = reasoner.read_text()

old = """                    cand.selected = cand.directions.iter().enumerate().max_by(|(_,a),(_,b)| a.mean_gain().partial_cmp(&b.mean_gain()).unwrap_or(Ordering::Equal)).map(|x| x.0);"""
new = """                    // Match NumPy's np.argmax semantics in the frozen Python reference:
                    // on an exact tie, keep the first maximum. Iterator::max_by
                    // returns the last equal maximum, which changes split geometry.
                    let mut best_idx = 0usize;
                    let mut best_gain = f64::NEG_INFINITY;
                    for (idx, sh) in cand.directions.iter().enumerate() {
                        let gain = sh.mean_gain();
                        if gain > best_gain {
                            best_gain = gain;
                            best_idx = idx;
                        }
                    }
                    cand.selected = Some(best_idx);"""
if old not in s:
    raise SystemExit("split-selection pattern not found")
s = s.replace(old, new)

old = """            active.push(m.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|x| x.0).unwrap_or(0));"""
new = """            // Match np.argmax: preserve the first index on exact ties.
            let mut best_idx = 0usize;
            let mut best_value = f64::NEG_INFINITY;
            for (idx, &value) in m.iter().enumerate() {
                if value > best_value {
                    best_value = value;
                    best_idx = idx;
                }
            }
            active.push(best_idx);"""
if old not in s:
    raise SystemExit("membership argmax pattern not found")
reasoner.write_text(s.replace(old, new))

math = Path("crates/cortex-core/src/math.rs")
s = math.read_text().replace("use std::cmp::Ordering;\n\n", "")
old = """    if let Some((j, _)) = u.iter().enumerate().max_by(|(_, a), (_, b)| {
        a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal)
    }) {
        if u[j] < 0.0 {
            for x in &mut u { *x = -*x; }
        }
    }"""
new = """    // Match np.argmax(np.abs(u)): preserve the first index on exact ties.
    let mut j = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (idx, value) in u.iter().enumerate() {
        let score = value.abs();
        if score > best {
            best = score;
            j = idx;
        }
    }
    if u[j] < 0.0 {
        for x in &mut u { *x = -*x; }
    }"""
if old not in s:
    raise SystemExit("canonicalization argmax pattern not found")
math.write_text(s.replace(old, new))
