"""
Experiment 02 — H4: Network Topology — Publication Density Sweeps
================================================================================
HYPOTHESIS (H4, density refinement)
    The companion run.py shows topology FAMILY matters under NEIGHBOR sharing.
    These two publication-grade sweeps push the deeper claim that what really
    drives the effect is connectivity DENSITY, not family identity:

      * Barabási–Albert m sweep — increasing m (edges each new node attaches)
        densifies the scale-free graph. The committee question: "at what m does
        BA stop underperforming Watts–Strogatz?" A watts_strogatz run is
        included as a fixed baseline for the comparison.
      * Watts–Strogatz k sweep — increasing k (direct neighbors per team)
        densifies the small-world graph. Shows how sensitive NEIGHBOR learning
        is to network degree.

WHAT THESE SCRIPTS VARY
    ba_m ∈ {1, 2, 3, 4, 6}   with network_topology = barabasi_albert
    ws_k ∈ {2, 4, 6, 8, 10}  with network_topology = watts_strogatz
    Everything else is held at BASE_PARAMS (NEIGHBOR sharing, 20 teams, 365
    steps, base_incident_rate 0.05, deployment_rate 0.1, transform_prob 0.6) —
    identical to the original publication_tests.py.

WHAT THEY MEASURE
    Per density value, aggregated over `--seeds` runs (mean / std / 95% CI / n,
    plus the raw per-seed list):
        total_incidents, overall_availability, final_prevention_knowledge,
        transformation_rate.

OUTPUT  (one JSON per sweep, written to ../results)
    ../results/publication_ba_m_sweep_<timestamp>.json
        top-level keys: "1","2","3","4","6","watts_strogatz_baseline"
    ../results/publication_ws_k_sweep_<timestamp>.json
        top-level keys: "2","4","6","8","10"

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp02_h4_topology/publication_sweeps.py --which all
    python experiments/exp02_h4_topology/publication_sweeps.py --which ba_m
    python experiments/exp02_h4_topology/publication_sweeps.py --which ws_k
    ... --quick            # smoke test: 5 seeds
    ... --seeds 50         # custom seed count
    ... --outdir /tmp/x    # write elsewhere

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy. The two sweep
bodies, BASE_PARAMS, and the aggregate/print helpers are a faithful copy of the
sweep_ba_m() and sweep_ws_k() functions from the original publication_tests.py
(02-Framework-Code) — seed counts and numeric config preserved exactly — with
only the I/O paths adapted so the folder stands on its own.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

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

# Held-fixed configuration — identical to publication_tests.py BASE_PARAMS.
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

def save_results(name: str, results: dict, outdir: Path) -> Path:
    """Write a timestamped JSON to `outdir`, converting numpy scalars to plain Python."""
    outdir.mkdir(parents=True, exist_ok=True)
    filepath = outdir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def convert(obj):
        """Recursively coerce numpy scalars/arrays in `obj` to JSON-native types."""
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Saved: {filepath}")
    return filepath


def run_with_seeds(base_params: dict, seeds: list) -> list:
    """Run one configuration once per seed and collect the raw result dicts."""
    results = []
    for seed in seeds:
        params = SimulationParams(**{**base_params, "seed": seed})
        results.append(run_simulation(params))
    return results


def aggregate(results: list) -> dict:
    """Reduce a list of per-seed runs to mean / std / 95% CI / raw / n per metric."""
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


def print_row(label: str, r: dict):
    """Print one labeled summary row (incidents, availability, prevK, transform)."""
    inc = r.get("total_incidents", {}).get("mean", float("nan"))
    avail = r.get("overall_availability", {}).get("mean", float("nan"))
    prevK = r.get("final_prevention_knowledge", {}).get("mean", float("nan"))
    tr = r.get("transformation_rate", {}).get("mean", None)
    tr_str = f"{tr:.1%}" if tr is not None else "N/A"
    print(f"  {label:<32} inc={inc:>7.1f}  avail={avail:.4f}  prevK={prevK:.3f}  transform={tr_str}")


# ==============================================================================
# ba_m Sweep (Barabási-Albert edge density)
#   faithful copy of sweep_ba_m() from publication_tests.py
# ==============================================================================

def sweep_ba_m(seeds, outdir: Path):
    """Sweep barabasi_albert ba_m over [1,2,3,4,6] under NEIGHBOR sharing, compare
    against a watts_strogatz baseline, write the JSON to `outdir`, and return
    the per-value aggregated results dict."""
    print("\n" + "=" * 70)
    print("PUBLICATION TEST: Barabási-Albert ba_m Sweep")
    print("Sweep: ba_m = [1, 2, 3, 4, 6] | NEIGHBOR scenario")
    print("=" * 70)
    print("  Question: At what ba_m does scale-free topology stop underperforming?")
    print("  Answers committee question on minimum network size for scale-free.")

    ba_m_values = [1, 2, 3, 4, 6]
    results = {}

    for m in ba_m_values:
        print(f"\n  Running ba_m={m}...", end=" ", flush=True)
        params = {**BASE_PARAMS, "network_topology": "barabasi_albert", "ba_m": m}
        r = aggregate(run_with_seeds(params, seeds))
        results[str(m)] = r
        print("Done.")
        print_row(f"ba_m={m}", r)

    print(f"\n  Running watts_strogatz baseline...", end=" ", flush=True)
    ws_r = aggregate(run_with_seeds(BASE_PARAMS, seeds))
    results["watts_strogatz_baseline"] = ws_r
    print("Done.")
    print_row("watts_strogatz (baseline)", ws_r)

    print("\n  INTERPRETATION: Does higher ba_m rescue BA performance?")
    ws_inc = ws_r["total_incidents"]["mean"]
    for m in ba_m_values:
        ba_inc = results[str(m)]["total_incidents"]["mean"]
        gap = ba_inc - ws_inc
        better = "BA better" if gap < 0 else f"BA worse by {gap:.1f}"
        print(f"  ba_m={m}: {ba_inc:.1f} vs WS {ws_inc:.1f} → {better}")

    save_results("publication_ba_m_sweep", results, outdir)
    return results


# ==============================================================================
# ws_k Sweep (Watts-Strogatz neighbor count)
#   faithful copy of sweep_ws_k() from publication_tests.py
# ==============================================================================

def sweep_ws_k(seeds, outdir: Path):
    """Sweep watts_strogatz ws_k over [2,4,6,8,10] under NEIGHBOR sharing to
    gauge NEIGHBOR learning's sensitivity to network degree, write the JSON to
    `outdir`, and return the per-value aggregated results dict."""
    print("\n" + "=" * 70)
    print("PUBLICATION TEST: Watts-Strogatz ws_k Sweep")
    print("Sweep: ws_k = [2, 4, 6, 8, 10] | NEIGHBOR scenario")
    print("=" * 70)
    print("  Question: How sensitive is NEIGHBOR learning to network degree?")
    print("  ws_k controls how many direct neighbors each team has.")

    ws_k_values = [2, 4, 6, 8, 10]
    results = {}

    for k in ws_k_values:
        print(f"\n  Running ws_k={k}...", end=" ", flush=True)
        params = {**BASE_PARAMS, "ws_k": k}
        r = aggregate(run_with_seeds(params, seeds))
        results[str(k)] = r
        print("Done.")
        print_row(f"ws_k={k}", r)

    print("\n  INTERPRETATION:")
    for k in ws_k_values:
        inc = results[str(k)]["total_incidents"]["mean"]
        prevK = results[str(k)]["final_prevention_knowledge"]["mean"]
        print(f"  ws_k={k}: {inc:.1f} incidents, prevK={prevK:.3f}")

    save_results("publication_ws_k_sweep", results, outdir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H4 density sweeps (BA m / WS k)")
    parser.add_argument("--which", choices=["ba_m", "ws_k", "all"], default="all",
                        help="which sweep(s) to run")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_list = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    print(f"Running with {n_seeds} seeds → {outdir} | which={args.which}")

    if args.which in ("ba_m", "all"):
        sweep_ba_m(seed_list, outdir)
    if args.which in ("ws_k", "all"):
        sweep_ws_k(seed_list, outdir)
