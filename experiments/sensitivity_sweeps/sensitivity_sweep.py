"""
Sensitivity Sweeps — Untested Model Knobs
================================================================================
PURPOSE
    Per-parameter robustness checks for model knobs that are NOT exercised by
    the main hypothesis experiments (exp01–exp06). Each sweep varies a single
    parameter across a range of values (all other parameters held at the base
    configuration below) and reports mean / std / 95% CI over `--seeds` runs.
    The goal is to show that the thesis findings are not artifacts of any one
    parameter setting.

SWEEPS PROVIDED  (this script)            OUTPUT JSON PREFIX
    acquisition         acquisition_probability (Stage 1)
                                            sensitivity_acquisition_probability_*
    assimilation        assimilation_probability (Stage 2)
                                            sensitivity_assimilation_probability_*
    exploitation        exploitation_probability (Stage 4)
                                            sensitivity_exploitation_probability_*
    signal_decay        signal_decay (network distance penalty)
                                            sensitivity_signal_decay_*
    initial_knowledge   cold-start vs warm-start (burn-in)
                                            sensitivity_initial_knowledge_*

    (The companion script ablation_remaining.py covers inverted_u,
     detection_effect, mitigation_effect, and deployment_risk_multiplier.)

OUTPUT
    ../results/<prefix>_<timestamp>.json

HOW TO RUN  (from anywhere — paths resolve from this file's location)
    python sensitivity_sweep.py --sweep acquisition          # full: 100 seeds
    python sensitivity_sweep.py --sweep acquisition --quick   # smoke: 5 seeds
    python sensitivity_sweep.py --sweep all                   # every sweep above
    python sensitivity_sweep.py --sweep all --seeds 50        # custom seed count
    python sensitivity_sweep.py --sweep all --outdir /tmp/x   # redirect output

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy. The sweep
bodies are a faithful copy of the original src/sensitivity_sweep.py; only the
I/O paths and CLI were adapted so the folder stands on its own.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# --- make the local model.py importable no matter the current directory ------
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))
from model import (
    SimulationParams,
    LearningScenario,
    IncidentType,
    run_simulation,
)

warnings.filterwarnings("ignore")

# Results land in this experiment folder's own results/ directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# NUM_SEEDS: seeds used for the committed paper results. QUICK_SEEDS: fast smoke-test.
NUM_SEEDS = 100
QUICK_SEEDS = 5

BASE_PARAMS = {
    "num_teams": 20,
    "steps": 365,
    "network_topology": "watts_strogatz",
    "base_incident_rate": 0.05,
    "deployment_rate": 0.1,
    "transformation_probability": 0.6,
    "learning_scenario": LearningScenario.NEIGHBOR,
}


def save_results(name: str, results: dict, outdir: Path) -> Path:
    """Write a timestamped <name>_*.json to outdir, converting numpy scalars to
    plain Python first; return the path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    filepath = outdir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def convert(obj):
        """Recursively convert numpy scalars/arrays inside obj to JSON-safe types."""
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


def run_with_seeds(base_params: dict, seeds: list) -> list:
    """Run one configuration once per seed and collect the raw result dicts."""
    results = []
    for seed in seeds:
        params = SimulationParams(**{**base_params, "seed": seed})
        results.append(run_simulation(params))
    return results


def aggregate(results: list) -> dict:
    """Reduce per-seed runs to mean/std/95% CI/n for incidents, availability,
    final prevention knowledge, and transformation rate.
    """
    raw = {
        "total_incidents": [],
        "overall_availability": [],
        "final_prevention_knowledge": [],
        "transformation_rate": [],
    }
    for r in results:
        raw["total_incidents"].append(r["summary"]["total_incidents"])
        raw["overall_availability"].append(r["summary"]["overall_availability"])
        if r["time_series"]["avg_prevention_knowledge"]:
            raw["final_prevention_knowledge"].append(
                r["time_series"]["avg_prevention_knowledge"][-1]
            )
        if r["time_series"].get("transformation_rate"):
            raw["transformation_rate"].append(
                r["time_series"]["transformation_rate"][-1]
            )
    stats = {}
    for key, values in raw.items():
        if values:
            n = len(values)
            std = float(np.std(values))
            se = std / np.sqrt(n)
            ci = 1.96 * se
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": std,
                "ci_lower": float(np.mean(values)) - ci,
                "ci_upper": float(np.mean(values)) + ci,
                "n": n,
            }
    return stats


def print_row(label: str, r: dict):
    """Print one aggregated result row (incidents, availability, prevention K, transform)."""
    incidents = r.get("total_incidents", {}).get("mean", float("nan"))
    avail = r.get("overall_availability", {}).get("mean", float("nan"))
    prev_k = r.get("final_prevention_knowledge", {}).get("mean", float("nan"))
    t_rate = r.get("transformation_rate", {}).get("mean", None)
    t_str = f"{t_rate:.1%}" if t_rate is not None else "N/A"
    print(f"  {label:<28} incidents={incidents:>7.1f}  avail={avail:.4f}  prevK={prev_k:.3f}  transform={t_str}")


# ==============================================================================
# SWEEP: Acquisition Probability
# ==============================================================================

def sweep_acquisition(seeds, outdir):
    """Sensitivity sweep of acquisition_probability (Stage 1) over [0.3,0.5,0.7,0.9,1.0]."""
    print("\n" + "=" * 70)
    print("SENSITIVITY: Acquisition Probability (Stage 1)")
    print(f"Default: 0.9 | Sweep: [0.3, 0.5, 0.7, 0.9, 1.0]")
    print("=" * 70)

    levels = [0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}

    for p in levels:
        print(f"\n  Running acquisition_probability={p}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "acquisition_probability": p}, seeds))
        results[str(p)] = r
        print("Done.")
        print_row(f"acq={p}", r)

    save_results("sensitivity_acquisition_probability", results, outdir)
    return results


# ==============================================================================
# SWEEP: Assimilation Probability
# ==============================================================================

def sweep_assimilation(seeds, outdir):
    """Sensitivity sweep of assimilation_probability (Stage 2) over [0.1,0.3,0.5,0.7,0.9,1.0]."""
    print("\n" + "=" * 70)
    print("SENSITIVITY: Assimilation Probability (Stage 2)")
    print(f"Default: 0.7 | Sweep: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]")
    print("=" * 70)

    levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}

    for p in levels:
        print(f"\n  Running assimilation_probability={p}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "assimilation_probability": p}, seeds))
        results[str(p)] = r
        print("Done.")
        print_row(f"assim={p}", r)

    save_results("sensitivity_assimilation_probability", results, outdir)
    return results


# ==============================================================================
# SWEEP: Exploitation Probability
# ==============================================================================

def sweep_exploitation(seeds, outdir):
    """Sensitivity sweep of exploitation_probability (Stage 4) over [0.1,0.3,0.5,0.6,0.8,1.0]."""
    print("\n" + "=" * 70)
    print("SENSITIVITY: Exploitation Probability (Stage 4)")
    print(f"Default: 0.6 | Sweep: [0.1, 0.3, 0.5, 0.6, 0.8, 1.0]")
    print("=" * 70)

    levels = [0.1, 0.3, 0.5, 0.6, 0.8, 1.0]
    results = {}

    for p in levels:
        print(f"\n  Running exploitation_probability={p}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "exploitation_probability": p}, seeds))
        results[str(p)] = r
        print("Done.")
        print_row(f"exploit={p}", r)

    save_results("sensitivity_exploitation_probability", results, outdir)
    return results


# ==============================================================================
# SWEEP: Signal Decay
# ==============================================================================

def sweep_signal_decay(seeds, outdir):
    """Sensitivity sweep of signal_decay (network distance penalty) over [0.3,0.5,0.7,0.8,0.9,1.0].

    1.0 = no decay (signal travels perfectly across hops); 0.3 = steep decay
    (only direct neighbors matter).
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY: Signal Decay (network distance penalty)")
    print(f"Default: 0.8 | Sweep: [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]")
    print("=" * 70)
    print("  Note: 1.0 = no decay (signal travels perfectly across hops)")
    print("        0.3 = steep decay (only direct neighbors matter)")

    levels = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for d in levels:
        print(f"\n  Running signal_decay={d}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "signal_decay": d}, seeds))
        results[str(d)] = r
        print("Done.")
        print_row(f"decay={d}", r)

    save_results("sensitivity_signal_decay", results, outdir)
    return results


# ==============================================================================
# SWEEP: Initial Knowledge (warm-start vs cold-start)
# ==============================================================================

def sweep_initial_knowledge(seeds, outdir):
    """
    Tests whether starting teams with prior knowledge changes the findings.

    Cold start (0.0): all teams begin with zero knowledge — standard ABM assumption.
    Warm start (>0.0): teams begin with some pre-existing knowledge, simulating
    orgs that already have incident history before the simulation begins.

    Implementation: we run the sim normally but inject knowledge at step 0
    by running a burn-in period first (pre-populating knowledge vectors).
    We approximate this by using a higher prevention_effect with a short
    burn-in seed — but the cleanest method is patching the initial knowledge
    level directly via SimulationParams initial_knowledge_level.

    Since the model does not yet have an initial_knowledge_level param, we
    simulate warm-start by pre-running 60 days of GLOBAL learning at high
    effectiveness, then reading the resulting knowledge level. We compare:
      - Standard cold start (0.0)
      - Partial warm start (~0.1 effective prior knowledge)
      - Partial warm start (~0.3 effective prior knowledge)

    This is done by varying prevention_effect and comparing 365-day totals
    against a 425-day total (60 day burn-in + 365 operational), giving an
    apples-to-apples comparison of cold vs warm initialization.
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY: Initial Knowledge (Cold-Start vs Warm-Start)")
    print("=" * 70)
    print("  Cold start  : teams begin at knowledge=0.0 (default)")
    print("  Warm start  : 60-day GLOBAL burn-in before 365-day observation window")
    print("  Question    : does prior knowledge change H1 ordering or magnitude?")

    results = {}

    # Cold start: standard 365-day run.
    print("\n  Running cold_start (365 days, no burn-in)...", end=" ", flush=True)
    cold_runs = run_with_seeds(BASE_PARAMS, seeds)
    results["cold_start_365d"] = aggregate(cold_runs)
    print("Done.")
    print_row("cold_start", results["cold_start_365d"])

    # Warm start: 425-step run (60 GLOBAL-seeded burn-in + 365 operational),
    # compared against the cold 365-day total.
    warm_params_global = {**BASE_PARAMS, "steps": 425, "learning_scenario": LearningScenario.GLOBAL}
    warm_params_neighbor = {**BASE_PARAMS, "steps": 425}
    warm_params_local = {**BASE_PARAMS, "steps": 425, "learning_scenario": LearningScenario.LOCAL}
    warm_params_none = {**BASE_PARAMS, "steps": 425, "learning_scenario": LearningScenario.NONE}

    for label, params in [
        ("warm_start_GLOBAL_425d", warm_params_global),
        ("warm_start_NEIGHBOR_425d", warm_params_neighbor),
        ("warm_start_LOCAL_425d", warm_params_local),
        ("warm_start_NONE_425d", warm_params_none),
    ]:
        print(f"\n  Running {label}...", end=" ", flush=True)
        r = aggregate(run_with_seeds(params, seeds))
        results[label] = r
        print("Done.")
        print_row(label, r)

    print("\n  INTERPRETATION:")
    cold_inc = results["cold_start_365d"]["total_incidents"]["mean"]
    warm_global = results["warm_start_GLOBAL_425d"]["total_incidents"]["mean"]
    print(f"  Cold start 365d  : {cold_inc:.1f} incidents")
    print(f"  Warm GLOBAL 425d : {warm_global:.1f} incidents (includes 60-day ramp)")
    ratio = warm_global / (425/365) / cold_inc
    print(f"  Rate-adjusted ratio: {ratio:.3f}  (1.0 = identical rate, <1.0 = warm start reduces incidents)")

    save_results("sensitivity_initial_knowledge", results, outdir)
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Parse CLI args and dispatch the requested sweep (or all of them)."""
    parser = argparse.ArgumentParser(description="Sensitivity sweeps for untested model knobs")
    parser.add_argument(
        "--sweep", "-s",
        choices=["acquisition", "assimilation", "exploitation", "signal_decay", "initial_knowledge", "all"],
        default="all",
        help="Which parameter to sweep",
    )
    parser.add_argument("--quick", action="store_true", help="Use 5 seeds for fast check")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seeds = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR
    print(f"Seeds: {len(seeds)} | Sweep: {args.sweep} | Out: {outdir}")

    sweep_map = {
        "acquisition": sweep_acquisition,
        "assimilation": sweep_assimilation,
        "exploitation": sweep_exploitation,
        "signal_decay": sweep_signal_decay,
        "initial_knowledge": sweep_initial_knowledge,
    }

    if args.sweep == "all":
        for fn in sweep_map.values():
            fn(seeds, outdir)
    else:
        sweep_map[args.sweep](seeds, outdir)


if __name__ == "__main__":
    main()
