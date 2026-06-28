"""
H1 × H3 Cross-Sweep — Proposition 1: sharing scope gates, prevention amplifies
================================================================================
HYPOTHESIS (H1 × H3 → Proposition 1)
    Sharing scope is a precondition; prevention strength only amplifies what
    sharing has already exposed.
      - Under NONE sharing, no value of β (prevention_effect) reduces incidents
        below baseline — the NONE row is flat.
      - Under wider scopes (LOCAL → NEIGHBOR → GLOBAL), β reduces incidents in
        proportion to how much knowledge each scope lets through; under GLOBAL
        the amplification is strongest.
    If the NONE row stays flat while the GLOBAL row responds strongly to β,
    then sharing scope and prevention strength are not interchangeable
    investments — scope opens the gate, β amplifies what comes through.

WHAT THIS SCRIPT VARIES  (a full 4 × 3 factorial)
    learning_scenario  ∈ {NONE, LOCAL, NEIGHBOR, GLOBAL}     (H1 dimension)
    prevention_effect  ∈ {0.0, 0.1, 0.5}                     (H3 dimension)
    50 seeds per cell → 4 × 3 × 50 = 600 simulations.
    Everything else in SimulationParams is held at the BASE_PARAMS below
    (defaults consistent with run_experiments.py).

WHAT IT MEASURES
    total_incidents per organization-year, aggregated over the seeds for each
    cell (mean / std / 95% CI / min / max). Also overall_availability and final
    prevention-knowledge means.

OUTPUT  (prefix: h1xh3_crosssweep)
    results/h1xh3_crosssweep_<timestamp>.json
    Top-level key `cells`; cell keys take the form SCOPE__prevX.Y
    (e.g. NONE__prev0.0, GLOBAL__prev0.5).

HOW TO RUN  (from the repo root — paths are resolved from this file's location)
    python experiments/h1xh3_crosssweep/run.py              # full: 50 seeds
    python experiments/h1xh3_crosssweep/run.py --quick      # smoke: 2 seeds
    python experiments/h1xh3_crosssweep/run.py --seeds 50   # custom seeds
    python experiments/h1xh3_crosssweep/run.py --outdir /tmp/x  # temp outdir

This script imports the shared simulation engine from the repo-root model.py
(located by walking up the directory tree) plus numpy. The experiment body is a
faithful copy of the standalone src/h1xh3_crosssweep.py; only the I/O paths and
CLI were adapted for the flat layout.
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Locate the repo-root model.py by walking up the tree, then make it importable.
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))
from model import (
    LearningScenario,
    SimulationParams,
    run_simulation,
)

warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Results land in this experiment folder's own results/ directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# NUM_SEEDS: seeds used for the committed paper results. QUICK_SEEDS: fast smoke-test.
NUM_SEEDS = 50
QUICK_SEEDS = 2

SCENARIOS = [
    LearningScenario.NONE,
    LearningScenario.LOCAL,
    LearningScenario.NEIGHBOR,
    LearningScenario.GLOBAL,
]

PREVENTION_EFFECTS = [0.0, 0.1, 0.5]

# Base parameters: defaults consistent with experiment_3_exploitation_effectiveness
# (20 teams, 365 days, watts_strogatz topology with ws_k=4 default)
BASE_PARAMS: Dict[str, Any] = {
    "num_teams": 20,
    "steps": 365,
    "network_topology": "watts_strogatz",
    "ws_k": 4,
    "base_incident_rate": 0.05,
    "deployment_rate": 0.1,
    "transformation_probability": 0.6,
}


# ==============================================================================
# HELPERS
# ==============================================================================

def convert(obj):
    """Recursively convert numpy scalars/arrays inside obj to plain Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj


def aggregate_total_incidents(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, 95% CI, min, max for a list of total_incidents values."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    se = std / np.sqrt(n) if n > 0 else 0.0
    ci_margin = 1.96 * se
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - ci_margin,
        "ci_upper": mean + ci_margin,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(n),
        "values": [float(v) for v in arr],
    }


# ==============================================================================
# MAIN SWEEP
# ==============================================================================

def run_crosssweep(seeds: List[int]) -> Dict[str, Any]:
    """Run the full 4x3 scenario x prevention_effect cross-sweep.

    For every (learning_scenario, prevention_effect) cell, runs one simulation
    per seed, aggregates total_incidents/availability/prevention-knowledge, and
    prints a summary grid. Returns the results dict (experiment, metadata, cells).
    """
    print("=" * 70)
    print("H1 x H3 CROSS-SWEEP: sharing_scenario x prevention_effect")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scenarios:         {[s.name for s in SCENARIOS]}")
    print(f"Prevention effects:{PREVENTION_EFFECTS}")
    print(f"Seeds per cell:    {len(seeds)}")
    print(f"Total runs:        {len(SCENARIOS) * len(PREVENTION_EFFECTS) * len(seeds)}")
    print(f"Base params:       {BASE_PARAMS}")
    print("=" * 70)

    results: Dict[str, Any] = {
        "experiment": "H1 x H3 Cross-Sweep",
        "metadata": {
            "scenarios": [s.name for s in SCENARIOS],
            "prevention_effects": PREVENTION_EFFECTS,
            "num_seeds": len(seeds),
            "base_params": BASE_PARAMS,
            "started_at": datetime.now().isoformat(),
        },
        "cells": {},
    }

    overall_t0 = time.time()
    cell_idx = 0
    n_cells = len(SCENARIOS) * len(PREVENTION_EFFECTS)

    for scenario in SCENARIOS:
        for prev_effect in PREVENTION_EFFECTS:
            cell_idx += 1
            cell_key = f"{scenario.name}__prev{prev_effect}"
            print(
                f"\n[{cell_idx}/{n_cells}] Cell: scenario={scenario.name}, "
                f"prevention_effect={prev_effect}"
            )

            cell_t0 = time.time()
            total_incidents_list: List[float] = []
            avail_list: List[float] = []
            prev_k_list: List[float] = []

            for seed in seeds:
                params = SimulationParams(
                    seed=seed,
                    learning_scenario=scenario,
                    prevention_effect=prev_effect,
                    **BASE_PARAMS,
                )
                r = run_simulation(params)
                total_incidents_list.append(r["summary"]["total_incidents"])
                avail_list.append(r["summary"]["overall_availability"])
                if r["time_series"]["avg_prevention_knowledge"]:
                    prev_k_list.append(
                        r["time_series"]["avg_prevention_knowledge"][-1]
                    )

            cell_elapsed = time.time() - cell_t0

            cell_summary = {
                "scenario": scenario.name,
                "prevention_effect": prev_effect,
                "total_incidents": aggregate_total_incidents(total_incidents_list),
                "overall_availability_mean": float(np.mean(avail_list)),
                "final_prevention_knowledge_mean": (
                    float(np.mean(prev_k_list)) if prev_k_list else None
                ),
                "elapsed_seconds": cell_elapsed,
            }
            results["cells"][cell_key] = cell_summary

            print(
                f"  Done in {cell_elapsed:.1f}s. "
                f"Mean total_incidents = {cell_summary['total_incidents']['mean']:.2f} "
                f"(±{cell_summary['total_incidents']['std']:.2f})"
            )

    overall_elapsed = time.time() - overall_t0
    results["metadata"]["finished_at"] = datetime.now().isoformat()
    results["metadata"]["total_elapsed_seconds"] = overall_elapsed
    results["metadata"]["total_elapsed_minutes"] = overall_elapsed / 60.0

    print("\n" + "=" * 70)
    print(f"CROSS-SWEEP SUMMARY (mean total_incidents, {len(seeds)} seeds per cell)")
    print("=" * 70)
    header = f"  {'scenario':<10} | " + " | ".join(
        f"prev={pe:<5}" for pe in PREVENTION_EFFECTS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for scenario in SCENARIOS:
        row = f"  {scenario.name:<10} | "
        cells_text = []
        for pe in PREVENTION_EFFECTS:
            key = f"{scenario.name}__prev{pe}"
            cell = results["cells"][key]
            mean = cell["total_incidents"]["mean"]
            std = cell["total_incidents"]["std"]
            cells_text.append(f"{mean:>6.1f} (±{std:>4.1f})")
        row += " | ".join(cells_text)
        print(row)

    print("\n" + "=" * 70)
    print(f"Total runtime: {overall_elapsed/60.0:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


def save_results(results: Dict[str, Any], outdir: Path) -> Path:
    """Write results to a timestamped h1xh3_crosssweep_*.json in outdir; return the path."""
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = outdir / f"h1xh3_crosssweep_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved results to: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H1 x H3 cross-sweep (sharing scope x prevention effect)"
    )
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    results = run_crosssweep(seed_list)
    save_results(results, outdir)
