"""
Experiment 09 — Robustness: Team Count Variation
================================================================================
HYPOTHESIS (H1 robustness)
    The H1 ordering of learning scenarios by incident count
    (NONE > LOCAL > NEIGHBOR > GLOBAL — broader sharing means fewer incidents)
    is a property of the learning mechanism, not an artifact of one particular
    organization size. It should therefore hold across small, medium, and large
    teams.

WHAT THIS SCRIPT VARIES
    num_teams ∈ {6, 20, 50}                              (organization size)
    learning_scenario ∈ {NONE, LOCAL, NEIGHBOR, GLOBAL}  (all four scenarios)
    Everything else in SimulationParams is held at the values below.

WHAT IT MEASURES
    total_incidents per organization-year for every (team-count, scenario)
    cell, aggregated over `--seeds` runs (mean / std / 95% CI). At each team
    count the four scenarios should preserve the H1 ordering, demonstrating
    that the result is robust to org size.

OUTPUT
    ../results/exp9_robustness_team_count_<timestamp>.json
    Top-level keys: "experiment" and "configurations". The "configurations"
    sub-dict is keyed by team count ("6"/"20"/"50"), then by scenario name.

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp09_team_count/run.py            # full: 100 seeds
    python experiments/exp09_team_count/run.py --quick    # smoke test: 5 seeds
    python experiments/exp09_team_count/run.py --seeds 50 # custom seed count

This script imports the shared simulation engine from the repo-root model.py
(located by walking up from this file until a model.py is found) plus numpy.
The experiment body is a faithful copy of experiment_robustness_team_count()
from the original run_experiments.py.
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
# THE EXPERIMENT  (faithful copy of experiment_robustness_team_count)
# ==============================================================================

def experiment_robustness_team_count(seeds: List[int], outdir: Path) -> Dict[str, Any]:
    """Robustness Part 1: Verify H1 ordering holds across team counts 6, 20, 50."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Robustness — Team Count Variation")
    print("=" * 70)

    team_counts = [6, 20, 50]

    base_params = {
        "steps": 365,
        "network_topology": "watts_strogatz",
        "base_incident_rate": 0.05,
        "deployment_rate": 0.1,
        "transformation_probability": 0.6,
    }

    configurations = {}

    for num_teams in team_counts:
        key = str(num_teams)
        configurations[key] = {}
        for scenario in LearningScenario:
            print(f"\n  Running num_teams={num_teams}, {scenario.name}...", end=" ", flush=True)
            params = {**base_params, "num_teams": num_teams, "learning_scenario": scenario}
            scenario_results = run_with_seeds(params, seeds)
            configurations[key][scenario.name] = aggregate_results(scenario_results)
            incidents = configurations[key][scenario.name]["total_incidents"]["mean"]
            print(f"Done. Incidents: {incidents:.1f}")

    print("\n  RESULTS SUMMARY:")
    print("  " + "-" * 70)
    print(f"  {'Teams':<8} {'NONE':>10} {'LOCAL':>10} {'NEIGHBOR':>10} {'GLOBAL':>10}")
    print("  " + "-" * 70)
    for num_teams in team_counts:
        key = str(num_teams)
        c = configurations[key]
        print(
            f"  {num_teams:<8} "
            f"{c['NONE']['total_incidents']['mean']:>10.1f} "
            f"{c['LOCAL']['total_incidents']['mean']:>10.1f} "
            f"{c['NEIGHBOR']['total_incidents']['mean']:>10.1f} "
            f"{c['GLOBAL']['total_incidents']['mean']:>10.1f}"
        )

    result = {"experiment": "Robustness: Team Count", "configurations": configurations}
    save_results("exp9_robustness_team_count", result, outdir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H1 robustness across team counts")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    experiment_robustness_team_count(seed_list, outdir)
