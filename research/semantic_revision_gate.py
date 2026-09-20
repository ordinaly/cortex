"""Frozen gates for semantic-revision-v1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import mean
from semantic_revision_campaign import DEVELOPMENT_SEEDS, OFFICIAL_SEEDS, FAMILY_EDGES, PROTOCOL

def summarize(rows,official):
    seeds=OFFICIAL_SEEDS if official else DEVELOPMENT_SEEDS
    metrics={
      "initial_correct_rate":mean(int(r["initial_correct"]) for r in rows),
      "shock_preservation_rate":mean(int(r["shock_preserved"]) for r in rows),
      "shift_success_rate":mean(int(r["shifted"]) for r in rows),
      "recovery_success_rate":mean(int(r["recovered"]) for r in rows),
      "revision_path_valid_rate":mean(int(r["revision_path_valid"]) for r in rows),
      "unaffected_preservation_rate":mean(int(r["unaffected_preserved"]) for r in rows),
      "max_shift_latency":max(r["shift_latency"] for r in rows),
      "max_recovery_latency":max(r["recovery_latency"] for r in rows),
      "bounded_neural_history_rate":mean(int(r["neural_history_bounded"]) for r in rows),
      "bounded_structural_history_rate":mean(int(r["structural_history_bounded"]) for r in rows),
    }
    checks={
      "protocol_exact":all(r["protocol"]==PROTOCOL for r in rows),
      "row_count":len(rows)==len(seeds)*len(FAMILY_EDGES),
      "seeds_exact":sorted({r["seed"] for r in rows})==list(seeds),
      "initial_correct":metrics["initial_correct_rate"]==1.0,
      "single_shock_resistance":metrics["shock_preservation_rate"]==1.0,
      "shift_success":metrics["shift_success_rate"]==1.0,
      "recovery_success":metrics["recovery_success_rate"]==1.0,
      "explicit_revision_path":metrics["revision_path_valid_rate"]==1.0,
      "locality":metrics["unaffected_preservation_rate"]==1.0,
      "shift_latency":metrics["max_shift_latency"]<=8,
      "recovery_latency":metrics["max_recovery_latency"]<=8,
      "bounded_neural_history":metrics["bounded_neural_history_rate"]==1.0,
      "bounded_structural_history":metrics["bounded_structural_history_rate"]==1.0,
    }
    return {"protocol":PROTOCOL,"mode":"official" if official else "development","rows":len(rows),"metrics":metrics,"checks":checks,"passed":all(checks.values())}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--official",action="store_true"); a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    out=summarize(rows,a.official); print(json.dumps(out,sort_keys=True))
    if not out["passed"]: raise SystemExit("failed gates: "+", ".join(k for k,v in out["checks"].items() if not v))
if __name__=="__main__": main()
