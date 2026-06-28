"""
Experiment 01 — H1: Sharing Scope (the foundational learning-scenario sweep)
================================================================================
HYPOTHESIS (H1)
    As the postmortem sharing scope expands from NONE → LOCAL → NEIGHBOR →
    GLOBAL, total annual incident counts decrease monotonically. Wider scope
    lets more teams traverse the four-stage absorptive-capacity pipeline per
    incident, so more postmortems become knowledge-matrix updates.

WHAT THIS SCRIPT VARIES
    learning_scenario ∈ {NONE, LOCAL, NEIGHBOR, GLOBAL}   (the ONLY variable)
    Everything else in SimulationParams is held at the baseline values below
    (identical to the paper run).

WHAT IT MEASURES
    Per organization-year, aggregated over `--seeds` runs (mean / std / 95% CI /
    min / max / n):
        total_incidents                  — primary H1 outcome (should fall NONE→GLOBAL)
        overall_availability             — reliability proxy (should rise NONE→GLOBAL)
        final_prevention/detection/mitigation_knowledge
        transformation_rate             — cross-team transfer (0% for NONE/LOCAL, high for GLOBAL)
    Expected ordering: NONE > LOCAL > NEIGHBOR > GLOBAL for incidents.

OUTPUT
    ../results/exp1_learning_scenarios_<timestamp>.json
    (top-level keys: "NONE", "LOCAL", "NEIGHBOR", "GLOBAL")

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp01_h1_sharing_scope/run.py            # full: 100 seeds
    python experiments/exp01_h1_sharing_scope/run.py --quick    # smoke test: 5 seeds
    python experiments/exp01_h1_sharing_scope/run.py --seeds 50 # custom seed count
    python experiments/exp01_h1_sharing_scope/run.py --outdir /tmp/x  # custom output dir

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy.
The experiment body is a faithful copy of experiment_1_learning_scenarios()
from the original run_experiments.py; only the I/O paths were adapted so the
folder stands on its own.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make the local model.py importable no matter the current directory.
# noqa: E402 — import follows the sys.path bootstrap above.
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))
from model import SimulationParams, LearningScenario, run_simulation

warnings.filterwarnings("ignore")

# Results land in this experiment folder's own results/ directory.
# NUM_SEEDS is the committed paper count; QUICK_SEEDS is the fast smoke-test count.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
NUM_SEEDS = 100
QUICK_SEEDS = 5


# ==============================================================================
# SHARED HELPERS  (copied verbatim from run_experiments.py)
# ==============================================================================

def run_with_seeds(base_params: dict, seeds: List[int]) -> List[Dict]:
    """Run one configuration once per seed and collect the raw result dicts."""
    results = []
    for seed in seeds:
        params = SimulationParams(**{**base_params, "seed": seed})
        results.append(run_simulation(params))
    return results


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Reduce a list of per-seed runs to mean / std / 95% CI / min / max / n."""
    aggregated = {
        "total_incidents": [],
        "total_engineering_cost": [],
        "total_learning_cost": [],
        "overall_availability": [],
        "final_prevention_knowledge": [],
        "final_detection_knowledge": [],
        "final_mitigation_knowledge": [],
        "transformation_rate": [],
    }

    for r in results:
        aggregated["total_incidents"].append(r["summary"]["total_incidents"])
        aggregated["total_engineering_cost"].append(r["summary"]["total_engineering_cost"])
        aggregated["total_learning_cost"].append(r["summary"]["total_learning_cost"])
        aggregated["overall_availability"].append(r["summary"]["overall_availability"])

        if r["time_series"]["avg_prevention_knowledge"]:
            aggregated["final_prevention_knowledge"].append(r["time_series"]["avg_prevention_knowledge"][-1])
            aggregated["final_detection_knowledge"].append(r["time_series"]["avg_detection_knowledge"][-1])
            aggregated["final_mitigation_knowledge"].append(r["time_series"]["avg_mitigation_knowledge"][-1])

        if r["time_series"].get("transformation_rate"):
            aggregated["transformation_rate"].append(r["time_series"]["transformation_rate"][-1])

    stats = {}
    for key, values in aggregated.items():
        if values:
            n = len(values)
            std = float(np.std(values))
            se = std / np.sqrt(n)
            ci_margin = 1.96 * se
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": std,
                "ci_lower": float(np.mean(values)) - ci_margin,
                "ci_upper": float(np.mean(values)) + ci_margin,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
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
# THE EXPERIMENT  (faithful copy of experiment_1_learning_scenarios)
# ==============================================================================

def experiment_1_learning_scenarios(seeds: List[int], outdir: Path) -> Dict[str, Any]:
    """
    Compare four learning scenarios: NONE, LOCAL, NEIGHBOR, GLOBAL.

    This is the core experiment demonstrating the value of knowledge sharing.
    Expected ordering: GLOBAL > NEIGHBOR > LOCAL > NONE for reliability
    (equivalently NONE > LOCAL > NEIGHBOR > GLOBAL for incident counts).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Learning Scenario Comparison")
    print("=" * 70)

    # Fixed configuration for this experiment (identical to the paper run);
    # steps=365 is one simulated year.
    base_params = {
        "num_teams": 20,
        "steps": 365,
        "network_topology": "watts_strogatz",
        "base_incident_rate": 0.05,
        "deployment_rate": 0.1,
        "transformation_probability": 0.6,
    }

    results = {}

    for scenario in LearningScenario:
        print(f"\n  Running {scenario.name}...", end=" ", flush=True)
        scenario_params = {**base_params, "learning_scenario": scenario}
        scenario_results = run_with_seeds(scenario_params, seeds)
        results[scenario.name] = aggregate_results(scenario_results)
        print(f"Done. Incidents: {results[scenario.name]['total_incidents']['mean']:.1f}")

    print("\n  RESULTS SUMMARY:")
    print("  " + "-" * 70)
    print(f"  {'Scenario':<15} {'Incidents':>12} {'Availability':>15} {'Prevention K':>14} {'Transform %':>12}")
    print("  " + "-" * 70)

    for scenario in ["NONE", "LOCAL", "NEIGHBOR", "GLOBAL"]:
        r = results[scenario]
        transform_rate = r.get('transformation_rate', {}).get('mean', None)
        transform_str = f"{transform_rate:.1%}" if transform_rate is not None else "N/A"
        print(
            f"  {scenario:<15} {r['total_incidents']['mean']:>12.1f} "
            f"{r['overall_availability']['mean']:>15.4f} "
            f"{r['final_prevention_knowledge']['mean']:>14.3f} "
            f"{transform_str:>12}"
        )

    # Monotonicity check (the H1 criterion): incidents should be non-increasing
    # in the order NONE > LOCAL > NEIGHBOR > GLOBAL.
    order = ["NONE", "LOCAL", "NEIGHBOR", "GLOBAL"]
    inc = [results[s]["total_incidents"]["mean"] for s in order]
    mono = all(inc[i] >= inc[i + 1] for i in range(len(inc) - 1))
    print(f"\n  Monotonicity (NONE>=LOCAL>=NEIGHBOR>=GLOBAL): "
          f"{'PASS' if mono else 'FAIL (likely noise at low seeds)'}")

    save_results("exp1_learning_scenarios", results, outdir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H1 sharing-scope / learning-scenario sweep")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    experiment_1_learning_scenarios(seed_list, outdir)
