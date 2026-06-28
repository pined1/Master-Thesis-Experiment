"""
Experiment 02 — H4: Network Topology
================================================================================
HYPOTHESIS (H4)
    Under fixed NEIGHBOR sharing, the network topology archetype connecting the
    teams changes total incident counts substantially. Centralized (Star)
    topologies underperform (the hub bottlenecks redistribution); dense /
    small-world topologies spread knowledge faster and suffer fewer incidents.
    The deeper claim (see publication_sweeps.py) is that what really matters is
    connectivity DENSITY, not topology family.

WHAT THIS SCRIPT VARIES
    network_topology ∈ {erdos_renyi, complete, watts_strogatz,
                        barabasi_albert, star}
    learning_scenario is held at NEIGHBOR — the only scope where topology bites
    (NONE/LOCAL bypass the network; GLOBAL ignores its structure). Density knobs
    (ws_k, ba_m, edge probability) stay at the model defaults here; sweeping
    them is the job of the companion publication_sweeps.py.

WHAT IT MEASURES
    Per topology, aggregated over `--seeds` runs (mean / std / 95% CI / min /
    max / n):
        total_incidents, costs, overall_availability, final knowledge dims,
        transformation_rate, and a bespoke H4 metric:
        midpoint_prevention_knowledge — the average prevention knowledge at
        step ~182 (the mid-year snapshot), which captures diffusion SPEED, not
        just the endpoint.

OUTPUT
    ../results/exp2_network_topology_<timestamp>.json
    Top-level keys are the five topology names; each maps to the aggregated
    metric dict described above.

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp02_h4_topology/run.py            # full: 100 seeds
    python experiments/exp02_h4_topology/run.py --quick    # smoke: 5 seeds
    python experiments/exp02_h4_topology/run.py --seeds 50 # custom count
    python experiments/exp02_h4_topology/run.py --outdir /tmp/x  # elsewhere

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy. The experiment
body is a faithful copy of experiment_2_network_topology() from the original
run_experiments.py — including the midpoint_prevention_knowledge logic — with
only the I/O paths adapted so the folder stands on its own.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# --- make the local model.py importable no matter the current directory ------
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
# THE EXPERIMENT  (faithful copy of experiment_2_network_topology)
# ==============================================================================

def experiment_2_network_topology(seeds: List[int], outdir: Path) -> Dict[str, Any]:
    """
    Compare different network topologies with NEIGHBOR learning.

    Tests: erdos_renyi, complete, watts_strogatz, barabasi_albert, star
    Expected: Complete and dense networks spread knowledge faster.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Network Topology Effect")
    print("=" * 70)

    topologies = ["erdos_renyi", "complete", "watts_strogatz", "barabasi_albert", "star"]

    base_params = {
        "num_teams": 20,
        "steps": 365,
        "learning_scenario": LearningScenario.NEIGHBOR,
        "base_incident_rate": 0.05,
        "transformation_probability": 0.6,
    }

    results = {}

    for topology in topologies:
        print(f"\n  Running {topology}...", end=" ", flush=True)
        topo_params = {**base_params, "network_topology": topology}
        topo_results = run_with_seeds(topo_params, seeds)
        results[topology] = aggregate_results(topo_results)

        # Extract midpoint prevention knowledge at step ~182 (H4 midpoint metric)
        midpoint_values = []
        for r in topo_results:
            ts = r["time_series"]["avg_prevention_knowledge"]
            idx = min(182, len(ts) - 1)
            midpoint_values.append(ts[idx])
        if midpoint_values:
            n = len(midpoint_values)
            std = float(np.std(midpoint_values))
            se = std / np.sqrt(n)
            ci_margin = 1.96 * se
            results[topology]["midpoint_prevention_knowledge"] = {
                "mean": float(np.mean(midpoint_values)),
                "std": std,
                "ci_lower": float(np.mean(midpoint_values)) - ci_margin,
                "ci_upper": float(np.mean(midpoint_values)) + ci_margin,
                "min": float(np.min(midpoint_values)),
                "max": float(np.max(midpoint_values)),
                "n": n,
            }

        print(f"Done. Knowledge: {results[topology]['final_prevention_knowledge']['mean']:.3f}")

    print("\n  RESULTS SUMMARY:")
    print("  " + "-" * 70)
    for topology in topologies:
        r = results[topology]
        midpoint_mean = r.get("midpoint_prevention_knowledge", {}).get("mean", float("nan"))
        print(
            f"  {topology:<20} "
            f"Incidents: {r['total_incidents']['mean']:>6.1f}  "
            f"Knowledge: {r['final_prevention_knowledge']['mean']:.3f}  "
            f"Midpoint K: {midpoint_mean:.3f}"
        )

    save_results("exp2_network_topology", results, outdir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H4 network-topology comparison")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir}")
    experiment_2_network_topology(seed_list, outdir)
