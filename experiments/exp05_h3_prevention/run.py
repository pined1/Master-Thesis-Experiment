"""
Experiment 05 — H3: Prevention Effectiveness (β gradient, high resolution)
================================================================================
HYPOTHESIS (H3)
    Increasing the prevention coefficient β (the maximum fraction by which
    accumulated prevention knowledge can suppress incident probability) reduces
    total annual incident counts. The open question is the SHAPE of that
    response: is it linear, or does it bend (diminishing returns / saturation)?
    This high-resolution sweep exists to settle that — 100 seeds left the curve
    too noisy to call, so we re-run at 500 seeds.

WHAT THIS SCRIPT VARIES
    prevention_effect (β) ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
        - 0.0   → knowledge has no effect on incident probability (baseline)
        - 0.5   → at full knowledge, incident probability is halved
        The range is deliberately extended past the paper default (0.5) and
        densely sampled at the low end (0.01, 0.02) to expose any sublinear
        signal and to locate a possible saturation zone.
    Everything else in SimulationParams is held at the BASE_PARAMS values below
    (sharing scope is held at NEIGHBOR).

WHAT IT MEASURES
    total_incidents per organization-year, aggregated over `--seeds` runs
    (mean / std / 95% CI / raw / n). A slope analysis on the per-unit-β deltas
    decides whether the relationship is linear/accelerating (H3 rejected) or
    shows diminishing returns (H3 supported).

OUTPUT
    ../results/publication_h3_500seeds_<timestamp>.json
    Output prefix: publication_h3   (this folder owns that prefix exclusively).
    Top-level JSON keys are the β values as strings: "0.0", "0.01", "0.02",
    "0.05", "0.1", "0.2", "0.5".

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp05_h3_prevention/run.py             # full: 500 seeds
    python experiments/exp05_h3_prevention/run.py --quick     # smoke test: 5 seeds
    python experiments/exp05_h3_prevention/run.py --seeds 50  # custom seed count

    NOTE: the full run is 7 β values × 500 seeds = 3,500 simulations and is
    SLOW (~20 minutes). Use --quick or --seeds for fast checks.

This script imports the shared simulation engine from the repo-root model.py
(located by walking up the parent directories) plus numpy. The experiment body
is a faithful copy of sweep_h3_500seeds() from the original publication_tests.py;
only the I/O paths and the model import were adapted.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# --- make the repo-root model.py importable no matter the current directory --
# noqa: E402 — import follows the sys.path bootstrap above.
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))
from model import SimulationParams, LearningScenario, run_simulation

warnings.filterwarnings("ignore")

# Results land in this experiment folder's own results/ directory.
# NUM_SEEDS is the committed (high-resolution) paper count; QUICK_SEEDS is the fast smoke-test count.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
NUM_SEEDS = 500
QUICK_SEEDS = 5

# Fixed configuration for this experiment (identical to the paper run).
BASE_PARAMS = {
    "num_teams": 20,
    "steps": 365,
    "network_topology": "watts_strogatz",
    "base_incident_rate": 0.05,
    "deployment_rate": 0.1,
    "transformation_probability": 0.6,
    "learning_scenario": LearningScenario.NEIGHBOR,
}


# ==============================================================================
# SHARED HELPERS  (copied verbatim from publication_tests.py)
# ==============================================================================

def run_with_seeds(base_params: dict, seeds: List[int]) -> List[Dict]:
    """Run one configuration once per seed and collect the raw result dicts."""
    results = []
    for seed in seeds:
        params = SimulationParams(**{**base_params, "seed": seed})
        results.append(run_simulation(params))
    return results


def aggregate(results: List[Dict]) -> Dict[str, Any]:
    """Reduce a list of per-seed runs to mean / std / 95% CI / raw / n."""
    raw = {
        "total_incidents": [],
        "overall_availability": [],
        "final_prevention_knowledge": [],
        "transformation_rate": [],
    }
    for r in results:
        raw["total_incidents"].append(r["summary"]["total_incidents"])
        raw["overall_availability"].append(r["summary"]["overall_availability"])
        ts = r["time_series"]
        if ts.get("avg_prevention_knowledge"):
            raw["final_prevention_knowledge"].append(ts["avg_prevention_knowledge"][-1])
        if ts.get("transformation_rate"):
            raw["transformation_rate"].append(ts["transformation_rate"][-1])
    stats = {}
    for key, values in raw.items():
        if values:
            n = len(values)
            arr = np.array(values)
            std = float(np.std(arr))
            se = std / np.sqrt(n)
            ci = 1.96 * se
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": std,
                "ci_lower": float(np.mean(arr)) - ci,
                "ci_upper": float(np.mean(arr)) + ci,
                "raw": values,
                "n": n,
            }
    return stats


def save_results(name: str, results: Dict[str, Any], outdir: Path) -> Path:
    """Write a timestamped JSON, converting any numpy scalars to plain Python."""
    outdir.mkdir(parents=True, exist_ok=True)
    filepath = outdir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def convert(obj):
        """Recursively coerce numpy scalars/arrays in `obj` to JSON-native types."""
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

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved: {filepath}")
    return filepath


# ==============================================================================
# THE EXPERIMENT  (faithful copy of sweep_h3_500seeds)
# ==============================================================================

def sweep_h3_500seeds(seeds_500: List[int], outdir: Path) -> Dict[str, Any]:
    """
    H3: Prevention effectiveness — high-resolution β gradient at 500 seeds.

    Sweeps prevention_effect (β) over an extended, densely-sampled range and
    checks whether the incident-vs-β relationship is linear/accelerating or
    shows diminishing returns. Top-level result keys are the β values.
    """
    print("\n" + "=" * 70)
    print("PUBLICATION TEST: H3 Prevention Effectiveness — 500 Seeds")
    print("Sweep: prevention_effect = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]")
    print("=" * 70)
    print("  Extended range (adds 0.2 and 0.5) to find saturation zone.")
    print("  Question: Is the linear finding real, or did 100 seeds miss sublinear signal?")

    effect_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results: Dict[str, Any] = {}

    for effect in effect_levels:
        print(f"\n  Running prevention_effect={effect} ({len(seeds_500)} seeds)...", end=" ", flush=True)
        params = {**BASE_PARAMS, "prevention_effect": effect}
        r = aggregate(run_with_seeds(params, seeds_500))
        results[str(effect)] = r
        print(f"Done. Incidents: {r['total_incidents']['mean']:.1f}")

    print("\n  RESULTS SUMMARY:")
    print("  " + "-" * 55)
    prev_inc = None
    for effect in effect_levels:
        inc = results[str(effect)]["total_incidents"]["mean"]
        delta = f"  Δ={inc - prev_inc:+.1f}" if prev_inc is not None else ""
        print(f"  effect={effect:<5}  incidents={inc:>7.1f}{delta}")
        prev_inc = inc

    # Per-unit-β slope deltas: rising => linear/accelerating (H3 rejected),
    # falling => diminishing returns / saturation (H3 supported).
    deltas = []
    incs = [results[str(e)]["total_incidents"]["mean"] for e in effect_levels]
    for i in range(1, len(incs)):
        step = effect_levels[i] - effect_levels[i - 1]
        deltas.append((incs[i] - incs[i - 1]) / step if step > 0 else 0)

    print("\n  Slope analysis (incidents per unit effect):")
    for effect, delta in zip(effect_levels[1:], deltas):
        print(f"  At effect={effect}: slope={delta:.1f}")

    if all(deltas[i] >= deltas[i - 1] for i in range(1, len(deltas))):
        print("  → Relationship is LINEAR or accelerating (H3 rejected)")
    else:
        print("  → Diminishing returns detected (H3 supported)")

    save_results("publication_h3_500seeds", results, outdir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3 prevention-effect β gradient sweep")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    print("(full run is 7 β values × 500 seeds = 3,500 sims — this is slow, ~20 min)")
    sweep_h3_500seeds(seed_list, outdir)
