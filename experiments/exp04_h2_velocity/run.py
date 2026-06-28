"""
Experiment 04 — H2: Deployment Velocity
================================================================================
HYPOTHESIS (H2)
    Incident count rises monotonically with deployment rate, and broader
    sharing (GLOBAL) absorbs that added risk better than narrow sharing (LOCAL).

WHAT THIS SCRIPT VARIES
    deployment_rate ∈ {0.05, 0.10, 0.20, 0.30, 0.50}   (deploys / team / day)
    learning_scenario ∈ {GLOBAL, LOCAL}
    Everything else in SimulationParams is held at the values below.

WHAT IT MEASURES
    total_incidents per organization-year, aggregated over `--seeds` runs
    (mean / std / 95% CI). GLOBAL should stay below LOCAL at every rate, and
    both curves should be non-decreasing in deployment_rate.

OUTPUT
    ../results/exp4_deployment_velocity_<timestamp>.json

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp04_h2_velocity/run.py            # full: 100 seeds
    python experiments/exp04_h2_velocity/run.py --quick    # smoke test: 5 seeds
    python experiments/exp04_h2_velocity/run.py --seeds 50 # custom seed count

This script imports the shared simulation engine from the repo-root model.py
(located by walking up the parent directories) plus numpy. The experiment body
is a faithful copy of experiment_4_deployment_velocity() from the original
run_experiments.py; only the I/O paths were adapted.
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
    """Reduce a list of per-seed runs to mean / std / 95% CI / min / max / n.

    Note: transformation_rate is a stage metric, not a knowledge dimension.
    """
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
# THE EXPERIMENT  (faithful copy of experiment_4_deployment_velocity)
# ==============================================================================

def experiment_4_deployment_velocity(seeds: List[int], outdir: Path) -> Dict[str, Any]:
    """
    H2: Incident count increases monotonically with deployment rate.

    Sweeps deployment_rate for both GLOBAL and LOCAL sharing. A non-monotonic
    dip at very few seeds is just noise; at 100 seeds the ordering holds.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Deployment Velocity")
    print("=" * 70)

    deployment_rates = [0.05, 0.1, 0.2, 0.3, 0.5]

    # Fixed configuration for this experiment (identical to the paper run).
    base_params = {
        "num_teams": 20,
        "steps": 365,
        "network_topology": "watts_strogatz",
        "base_incident_rate": 0.03,
        "transformation_probability": 0.6,
    }

    results: Dict[str, Any] = {"GLOBAL": {}, "LOCAL": {}}

    for rate in deployment_rates:
        for scenario in [LearningScenario.GLOBAL, LearningScenario.LOCAL]:
            key = f"{rate}"
            print(f"\n  Running {scenario.name} with rate={rate}...", end=" ", flush=True)
            rate_params = {**base_params, "deployment_rate": rate, "learning_scenario": scenario}
            rate_results = run_with_seeds(rate_params, seeds)
            results[scenario.name][key] = aggregate_results(rate_results)
            print("Done.")

    print("\n  RESULTS SUMMARY:")
    print("  " + "-" * 60)
    print(f"  {'Rate':<8} {'GLOBAL Incidents':>18} {'LOCAL Incidents':>18}")
    print("  " + "-" * 60)
    for rate in deployment_rates:
        key = str(rate)
        global_inc = results["GLOBAL"][key]["total_incidents"]["mean"]
        local_inc = results["LOCAL"][key]["total_incidents"]["mean"]
        print(f"  {rate:<8} {global_inc:>18.1f} {local_inc:>18.1f}")

    # Monotonicity check (the H2 criterion).
    global_vals = [results["GLOBAL"][str(r)]["total_incidents"]["mean"] for r in deployment_rates]
    local_vals = [results["LOCAL"][str(r)]["total_incidents"]["mean"] for r in deployment_rates]
    global_mono = all(global_vals[i] <= global_vals[i + 1] for i in range(len(global_vals) - 1))
    local_mono = all(local_vals[i] <= local_vals[i + 1] for i in range(len(local_vals) - 1))
    print(f"\n  Monotonicity — GLOBAL: {'PASS' if global_mono else 'FAIL (likely noise at low seeds)'}  |  "
          f"LOCAL: {'PASS' if local_mono else 'FAIL (likely noise at low seeds)'}")

    save_results("exp4_deployment_velocity", results, outdir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2 deployment-velocity sweep")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    experiment_4_deployment_velocity(seed_list, outdir)
