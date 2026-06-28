"""
Experiment 11 — Ablation: Knowledge Decay Disabled
================================================================================
PURPOSE (Ablation of H1)
    H1 establishes the sharing-scope ordering on total incidents:
    NONE > LOCAL > NEIGHBOR > GLOBAL. This ablation removes knowledge decay
    (no daily forgetting) and reruns the four sharing scenarios to confirm that
    the H1 ordering survives. If the ordering holds, it is structural and does
    not depend on the specific decay constant; if it collapses, decay was a
    load-bearing assumption — a model limitation a committee should know about.

WHAT THIS SCRIPT VARIES
    configuration ∈ {default_decay, no_decay}
        default_decay : knowledge_decay = 0.001        (the paper baseline)
        no_decay      : disable_knowledge_decay = True (the ablation knob)
    learning_scenario ∈ {NONE, LOCAL, NEIGHBOR, GLOBAL}  (all four scopes)
    Everything else in SimulationParams is held at the values below.

WHAT IT MEASURES
    total_incidents per organization-year, aggregated over `--seeds` runs
    (mean / std / 95% CI / min / max / n). The H1 ordering
    NONE > LOCAL > NEIGHBOR > GLOBAL should hold under BOTH configurations.

OUTPUT
    ../results/exp11_ablation_no_decay_<timestamp>.json
    Top-level keys: "experiment", "configurations".
    "configurations" -> {"default_decay", "no_decay"} -> per-scenario stats.

HOW TO RUN  (from the repo root — paths are resolved from this file's location)
    python experiments/exp11_no_decay/run.py            # full: 100 seeds
    python experiments/exp11_no_decay/run.py --quick    # smoke test: 5 seeds
    python experiments/exp11_no_decay/run.py --seeds 50 # custom seed count

This script imports the shared simulation engine from the repo-root model.py
(walking up parent dirs until it finds it) plus numpy. The experiment body is a
faithful copy of experiment_ablation_no_decay() from the original
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
# THE EXPERIMENT  (faithful copy of experiment_ablation_no_decay)
# ==============================================================================

def experiment_ablation_no_decay(seeds: List[int], outdir: Path) -> Dict[str, Any]:
    """Ablation 1: Compare H1 with decay disabled vs default decay."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: Ablation — No Decay")
    print("=" * 70)

    base_params = {
        "num_teams": 20,
        "steps": 365,
        "network_topology": "watts_strogatz",
        "base_incident_rate": 0.05,
        "deployment_rate": 0.1,
        "transformation_probability": 0.6,
    }

    configurations = {"default_decay": {}, "no_decay": {}}

    # For each sharing scope, run the paper baseline against the ablation.
    # default_decay keeps knowledge_decay=0.001; no_decay sets the
    # disable_knowledge_decay ablation knob.
    for scenario in LearningScenario:
        print(f"\n  Running default_decay, {scenario.name}...", end=" ", flush=True)
        params = {**base_params, "knowledge_decay": 0.001, "learning_scenario": scenario}
        scenario_results = run_with_seeds(params, seeds)
        configurations["default_decay"][scenario.name] = aggregate_results(scenario_results)
        print(f"Done. Incidents: {configurations['default_decay'][scenario.name]['total_incidents']['mean']:.1f}")

        print(f"  Running no_decay, {scenario.name}...", end=" ", flush=True)
        params_no_decay = {**base_params, "disable_knowledge_decay": True, "learning_scenario": scenario}
        no_decay_results = run_with_seeds(params_no_decay, seeds)
        configurations["no_decay"][scenario.name] = aggregate_results(no_decay_results)
        print(f"Done. Incidents: {configurations['no_decay'][scenario.name]['total_incidents']['mean']:.1f}")

    print("\n  RESULTS SUMMARY (mean total_incidents):")
    print("  " + "-" * 60)
    print(f"  {'Scenario':<12} {'Default Decay':>15} {'No Decay':>12}")
    print("  " + "-" * 60)
    for scenario in ["NONE", "LOCAL", "NEIGHBOR", "GLOBAL"]:
        default_inc = configurations["default_decay"][scenario]["total_incidents"]["mean"]
        no_dec_inc = configurations["no_decay"][scenario]["total_incidents"]["mean"]
        print(f"  {scenario:<12} {default_inc:>15.1f} {no_dec_inc:>12.1f}")

    result = {"experiment": "Ablation: No Decay", "configurations": configurations}
    save_results("exp11_ablation_no_decay", result, outdir)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation: knowledge decay disabled")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    experiment_ablation_no_decay(seed_list, outdir)
