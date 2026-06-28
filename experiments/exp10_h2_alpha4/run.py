"""
Experiment 10 — H2 (secondary): Deployment Velocity × Stage-4 Exploitation (α₄)
================================================================================
HYPOTHESIS (H2, secondary / robustness)
    Deployment velocity and the Stage-4 exploitation parameter α₄ act on
    INDEPENDENT dimensions of the model. Varying one should not change the
    effect of varying the other. Concretely: incident count should rise with
    deployment rate but stay flat across α₄ at any fixed deployment rate.

WHAT THIS SCRIPT VARIES
    deployment_rate          ∈ {0.05, 0.1, 0.3}   (deploys / team / day)
    exploitation_probability ∈ {0.2, 0.6, 0.9}    (Stage-4 α₄ baseline)
    A full 3×3 cross-sweep. learning_scenario is held at NEIGHBOR and
    everything else in SimulationParams is held at the values below.

WHAT IT MEASURES
    total_incidents (and overall_availability, etc.) per organization-year,
    aggregated over `--seeds` runs (mean / std / 95% CI). Reading ACROSS a row
    (changing α₄) should barely move incidents; reading DOWN a column (changing
    deployment rate) should add a consistent number of incidents regardless of
    α₄ — the signature of two independent axes.

OUTPUT
    ../results/exp10_robustness_deployment_learning_<timestamp>.json
    Top-level keys: "experiment", "configurations".
    The "configurations" sub-dict is keyed by cell, e.g. "dep0.05_exp0.2",
    "dep0.05_exp0.6", ..., "dep0.3_exp0.9" (preserved exactly).

HOW TO RUN  (from the repo root — paths are resolved from this file's location)
    python experiments/exp10_h2_alpha4/run.py            # full: 100 seeds
    python experiments/exp10_h2_alpha4/run.py --quick    # smoke test: 5 seeds
    python experiments/exp10_h2_alpha4/run.py --seeds 50 # custom seed count

This script imports the shared simulation engine from the repo-root model.py
(walking up parent dirs until it finds it) plus numpy. The experiment body is a
faithful copy of experiment_robustness_deployment_learning() from the original
run_experiments.py; only the I/O paths were adapted for the flat layout.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make the repo-root model.py importable regardless of the current directory.
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))
from model import SimulationParams, LearningScenario, run_simulation

warnings.filterwarnings("ignore")

# Results land in this experiment folder's own results/ directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# NUM_SEEDS: seeds used for the committed paper results. QUICK_SEEDS: fast smoke-test.
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

    transformation_rate is a stage metric, not a knowledge dimension.
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
        """Recursively turn numpy arrays/ints/floats into JSON-serializable types."""
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
# THE EXPERIMENT  (faithful copy of experiment_robustness_deployment_learning)
# ==============================================================================

def experiment_robustness_deployment_learning(seeds: List[int], outdir: Path) -> Dict[str, Any]:
    """Robustness Part 3: 3x3 cross-sweep of deployment rate x learning effectiveness."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Robustness — Deployment x Learning Effectiveness")
    print("=" * 70)

    deployment_rates = [0.05, 0.1, 0.3]
    exploitation_probabilities = [0.2, 0.6, 0.9]

    base_params = {
        "num_teams": 20,
        "steps": 365,
        "network_topology": "watts_strogatz",
        "base_incident_rate": 0.05,
        "transformation_probability": 0.6,
        "learning_scenario": LearningScenario.NEIGHBOR,
    }

    configurations = {}

    for dep_rate in deployment_rates:
        for exp_prob in exploitation_probabilities:
            combo_key = f"dep{dep_rate}_exp{exp_prob}"
            print(f"\n  Running deployment_rate={dep_rate}, exploitation_probability={exp_prob}...", end=" ", flush=True)
            params = {
                **base_params,
                "deployment_rate": dep_rate,
                "exploitation_probability": exp_prob,
            }
            combo_results = run_with_seeds(params, seeds)
            configurations[combo_key] = aggregate_results(combo_results)
            incidents = configurations[combo_key]["total_incidents"]["mean"]
            avail = configurations[combo_key]["overall_availability"]["mean"]
            print(f"Done. Incidents: {incidents:.1f}, Availability: {avail:.4f}")

    print("\n  RESULTS SUMMARY (Incidents / Availability):")
    print("  " + "-" * 70)
    header = f"  {'dep \\ exp':<12}" + "".join(f"  exp={e:<6}" for e in exploitation_probabilities)
    print(header)
    print("  " + "-" * 70)
    for dep_rate in deployment_rates:
        row = f"  dep={dep_rate:<8}"
        for exp_prob in exploitation_probabilities:
            key = f"dep{dep_rate}_exp{exp_prob}"
            inc = configurations[key]["total_incidents"]["mean"]
            row += f"  {inc:>8.1f}"
        print(row)

    result = {"experiment": "Robustness: Deployment x Learning", "configurations": configurations}
    save_results("exp10_robustness_deployment_learning", result, outdir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2 (secondary) deployment × α₄ cross-sweep")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    experiment_robustness_deployment_learning(seed_list, outdir)
