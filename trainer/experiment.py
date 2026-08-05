"""Side-experiment persistence — a SECOND evidence SOURCE for the ONE thesis/hypothesis concept.

There is a SINGLE thesis concept (the app's hypothesis feature); it is fed from MULTIPLE sources and a thesis
may require BOTH: (1) `blackswan-run` RL MODEL runs, and (2) side-experiment runs (diagnostics that produce NO
model — deterministic-baseline scans, the diversified-trend breadth analysis, ablations, correctness probes).
The two STORES stay separate so the RL run store remains apples-to-apples pure; the HYPOTHESIS is the unifying
layer that aggregates evidence across both. Side-experiments persist here under `experiments/<id>/`, inspectable
+ re-runnable, and LINK to the thesis they feed via `hypothesis_id`. Because a side-experiment `cell` config
shares the run config schema, the same spec-matching that finds RL evidence also finds side-experiment evidence.

Pure (record construction + save/load round-trip + the markdown report). The RUNNER that spawns `trainer.run`
per cell lives in `scripts/` — it writes each cell's RunSummary under the experiment dir, never into the run store.
"""

import json
import os
from typing import Any, Dict, List, Optional

# Stamped on every record so a side-experiment can never be mistaken for an RL model run.
KIND = "side-experiment (diagnostic; NOT an RL model run)"


def build_record(exp_id: str, name: str, thesis: str, purpose: str, matrix: Dict[str, Any],
                 cells: List[Dict[str, Any]], aggregate: Dict[str, Any], verdict: str, status: str,
                 created_at: str, hypothesis_id: Optional[str] = None,
                 provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A self-describing side-experiment record — one EVIDENCE SOURCE for a thesis, not a rival hypothesis.
    `hypothesis_id` links it to the ONE thesis it feeds (None = standalone/exploratory, not yet linked). Holds
    what was swept (`matrix`), every cell's config+metrics (`cells`, spec-matchable evidence for the thesis),
    the aggregated analysis (`aggregate`), the plain-language `verdict`, and provenance. `status`
    (supports | refutes | inconclusive) is THIS experiment's LOCAL contribution to `thesis`; the AUTHORITATIVE
    verdict is the hypothesis's, aggregated across all its sources (RL runs + side-experiments)."""
    return {
        "id": exp_id,
        "name": name,
        "kind": KIND,
        "hypothesisId": hypothesis_id,
        "thesis": thesis,
        "status": status,
        "purpose": purpose,
        "created_at": created_at,
        "provenance": provenance or {},
        "matrix": matrix,
        "verdict": verdict,
        "aggregate": aggregate,
        "cells": cells,
    }


def experiment_dir(exp_id: str, root: str = "experiments") -> str:
    return os.path.join(root, exp_id)


def save_record(record: Dict[str, Any], root: str = "experiments") -> str:
    """Persist `record.json` + a human-readable `report.md` under experiments/<id>/. Returns the dir."""
    d = experiment_dir(record["id"], root)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "record.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write(format_report(record))
    return d


def load_record(exp_id: str, root: str = "experiments") -> Dict[str, Any]:
    with open(os.path.join(experiment_dir(exp_id, root), "record.json")) as f:
        return json.load(f)


def format_report(record: Dict[str, Any]) -> str:
    prov = record.get("provenance", {})
    lines = [
        f"# Side-experiment: {record['name']}",
        "",
        f"**{record['kind']}** — persisted separately from the `blackswan-run` RL model store so RL runs "
        f"stay apples-to-apples.",
        "",
        f"- **id:** `{record['id']}`",
        f"- **feeds hypothesis:** {('`' + record['hypothesisId'] + '`') if record.get('hypothesisId') else '(standalone / not yet linked)'}",
        f"- **thesis:** {record.get('thesis', '')}",
        f"- **status (this source's contribution):** **{record.get('status', 'inconclusive').upper()}**",
        f"- **purpose:** {record['purpose']}",
        f"- **created:** {record.get('created_at')}",
        f"- **cells:** {len(record.get('cells', []))}",
        f"- **git:** {prov.get('gitCommit', 'n/a')} (dirty={prov.get('gitDirty', 'n/a')})",
        "",
        "## Verdict",
        "",
        str(record.get("verdict", "")).strip(),
        "",
        "## Aggregate",
        "",
        "```json",
        json.dumps(record.get("aggregate", {}), indent=2),
        "```",
        "",
        "## Matrix (the swept run space)",
        "",
        "```json",
        json.dumps(record.get("matrix", {}), indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"
