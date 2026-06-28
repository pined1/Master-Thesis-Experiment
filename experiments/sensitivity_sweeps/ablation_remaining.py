"""
Remaining Ablations & Sensitivity Sweeps
================================================================================
PURPOSE
    Per-parameter robustness checks and one design-decision ablation for knobs
    not exercised by the main hypothesis experiments (exp01–exp06) or by the
    companion sensitivity_sweep.py. Each sweep varies a single parameter across
    a range (all other parameters held at the base configuration below) and
    reports mean / std / 95% CI over `--seeds` runs. The inverted_u ablation
    additionally re-runs all four learning scenarios under both cognitive
    models to confirm the H1 ordering is unaffected by that design choice.

SWEEPS PROVIDED  (this script)            OUTPUT JSON PREFIX
    inverted_u          use_inverted_u (Nooteboom inverted-U vs
                        Cohen & Levinthal linear), all 4 scenarios
                                            ablation_inverted_u_vs_linear_*
    detection_effect    detection_effect (knowledge → MTTD reduction)
                                            sensitivity_detection_effect_*
    mitigation_effect   mitigation_effect (knowledge → severity reduction)
                                            sensitivity_mitigation_effect_*
    deployment_risk     deployment_risk_multiplier
                                            sensitivity_deployment_risk_multiplier_*

    (The companion script sensitivity_sweep.py covers acquisition,
     assimilation, exploitation, signal_decay, and initial_knowledge.)

OUTPUT
    ../results/<prefix>_<timestamp>.json

HOW TO RUN  (from anywhere — paths resolve from this file's location)
    python ablation_remaining.py --sweep inverted_u           # full: 100 seeds
    python ablation_remaining.py --sweep inverted_u --quick    # smoke: 5 seeds
    python ablation_remaining.py --sweep all                   # every sweep above
    python ablation_remaining.py --sweep all --seeds 50        # custom seed count
    python ablation_remaining.py --sweep all --outdir /tmp/x   # redirect output

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy. The sweep
bodies are a faithful copy of the original src/ablation_remaining.py; only the
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
from model import SimulationParams, LearningScenario, run_simulation

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
    """Reduce per-seed runs to mean/std/95% CI/n for incidents, availability, the
    three final knowledge dimensions, and transformation rate.
    """
    raw = {
        "total_incidents": [],
        "overall_availability": [],
        "final_prevention_knowledge": [],
        "final_detection_knowledge": [],
        "final_mitigation_knowledge": [],
        "transformation_rate": [],
    }
    for r in results:
        raw["total_incidents"].append(r["summary"]["total_incidents"])
        raw["overall_availability"].append(r["summary"]["overall_availability"])
        ts = r["time_series"]
        if ts.get("avg_prevention_knowledge"):
            raw["final_prevention_knowledge"].append(ts["avg_prevention_knowledge"][-1])
            raw["final_detection_knowledge"].append(ts["avg_detection_knowledge"][-1])
            raw["final_mitigation_knowledge"].append(ts["avg_mitigation_knowledge"][-1])
        if ts.get("transformation_rate"):
            raw["transformation_rate"].append(ts["transformation_rate"][-1])
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
    """Print one aggregated result row (incidents, availability, the three knowledge
    dimensions, and transformation rate).
    """
    inc = r.get("total_incidents", {}).get("mean", float("nan"))
    avail = r.get("overall_availability", {}).get("mean", float("nan"))
    prevK = r.get("final_prevention_knowledge", {}).get("mean", float("nan"))
    detK = r.get("final_detection_knowledge", {}).get("mean", float("nan"))
    mitK = r.get("final_mitigation_knowledge", {}).get("mean", float("nan"))
    tr = r.get("transformation_rate", {}).get("mean", None)
    tr_str = f"{tr:.1%}" if tr is not None else "N/A"
    print(f"  {label:<30} inc={inc:>7.1f}  avail={avail:.4f}  prevK={prevK:.3f}  detK={detK:.3f}  mitK={mitK:.3f}  transform={tr_str}")


# ==============================================================================
# ABLATION: Inverted-U vs Linear Cognitive Model
# ==============================================================================

def sweep_inverted_u(seeds, outdir):
    """Ablation: inverted-U (Nooteboom) vs linear (Cohen & Levinthal) cognitive model.

    use_inverted_u was changed True->False on 2026-03-27; this quantifies the
    impact of that design decision. Runs all four scenarios under both models to
    check whether the change affects the H1 ordering.
    """
    print("\n" + "=" * 70)
    print("ABLATION: Inverted-U (Nooteboom) vs Linear (Cohen & Levinthal)")
    print("Default: use_inverted_u=False (linear) | Compare: True vs False")
    print("=" * 70)
    print("  Context: use_inverted_u was changed from True→False on 2026-03-27.")
    print("  This ablation quantifies the impact of that design decision.")
    print("  All 4 scenarios tested to see if the change affects H1 ordering.")

    results = {"linear_false": {}, "inverted_u_true": {}}

    for scenario in LearningScenario:
        label = scenario.name
        print(f"\n  Running linear (use_inverted_u=False), {label}...", end=" ", flush=True)
        p_linear = {**BASE_PARAMS, "learning_scenario": scenario, "use_inverted_u": False}
        results["linear_false"][label] = aggregate(run_with_seeds(p_linear, seeds))
        print("Done.")
        print_row(f"linear / {label}", results["linear_false"][label])

        print(f"  Running inverted_u (use_inverted_u=True), {label}...", end=" ", flush=True)
        p_inv = {**BASE_PARAMS, "learning_scenario": scenario, "use_inverted_u": True}
        results["inverted_u_true"][label] = aggregate(run_with_seeds(p_inv, seeds))
        print("Done.")
        print_row(f"inverted_u / {label}", results["inverted_u_true"][label])

    print("\n  RESULTS SUMMARY (mean total_incidents):")
    print("  " + "-" * 65)
    print(f"  {'Scenario':<12} {'Linear (current)':>18} {'Inverted-U (old)':>18} {'Delta':>10}")
    print("  " + "-" * 65)
    for scenario in LearningScenario:
        label = scenario.name
        lin = results["linear_false"][label]["total_incidents"]["mean"]
        inv = results["inverted_u_true"][label]["total_incidents"]["mean"]
        delta = inv - lin
        print(f"  {label:<12} {lin:>18.1f} {inv:>18.1f} {delta:>+10.1f}")

    save_results("ablation_inverted_u_vs_linear", results, outdir)
    return results


# ==============================================================================
# SWEEP: Detection Effect
# ==============================================================================

def sweep_detection_effect(seeds, outdir):
    """Sensitivity sweep of detection_effect (knowledge -> MTTD reduction) over [0.0,0.1,0.2,0.3,0.5,0.8]."""
    print("\n" + "=" * 70)
    print("SENSITIVITY: Detection Effect (knowledge → MTTD reduction)")
    print("Default: 0.3 | Sweep: [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]")
    print("=" * 70)

    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    results = {}

    for d in levels:
        print(f"\n  Running detection_effect={d}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "detection_effect": d}, seeds))
        results[str(d)] = r
        print("Done.")
        print_row(f"det_effect={d}", r)

    save_results("sensitivity_detection_effect", results, outdir)
    return results


# ==============================================================================
# SWEEP: Mitigation Effect
# ==============================================================================

def sweep_mitigation_effect(seeds, outdir):
    """Sensitivity sweep of mitigation_effect (knowledge -> severity/duration reduction) over [0.0,0.1,0.2,0.3,0.5,0.8]."""
    print("\n" + "=" * 70)
    print("SENSITIVITY: Mitigation Effect (knowledge → severity/duration reduction)")
    print("Default: 0.3 | Sweep: [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]")
    print("=" * 70)

    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    results = {}

    for m in levels:
        print(f"\n  Running mitigation_effect={m}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "mitigation_effect": m}, seeds))
        results[str(m)] = r
        print("Done.")
        print_row(f"mit_effect={m}", r)

    save_results("sensitivity_mitigation_effect", results, outdir)
    return results


# ==============================================================================
# SWEEP: Deployment Risk Multiplier
# ==============================================================================

def sweep_deployment_risk(seeds, outdir):
    """Sensitivity sweep of deployment_risk_multiplier over [1.0,1.2,1.5,2.0,3.0,5.0].

    1.0 = deployments add zero extra risk; 5.0 = deployments quintuple the
    incident rate for that timestep.
    """
    print("\n" + "=" * 70)
    print("SENSITIVITY: Deployment Risk Multiplier")
    print("Default: 1.5 | Sweep: [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]")
    print("=" * 70)
    print("  1.0 = deployments add zero extra risk")
    print("  5.0 = deployments quintuple incident rate for that timestep")

    levels = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    results = {}

    for m in levels:
        print(f"\n  Running deployment_risk_multiplier={m}...", end=" ", flush=True)
        r = aggregate(run_with_seeds({**BASE_PARAMS, "deployment_risk_multiplier": m}, seeds))
        results[str(m)] = r
        print("Done.")
        print_row(f"risk_mult={m}", r)

    save_results("sensitivity_deployment_risk_multiplier", results, outdir)
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Parse CLI args and dispatch the requested sweep/ablation (or all of them)."""
    parser = argparse.ArgumentParser(description="Remaining ablations and sensitivity sweeps")
    parser.add_argument(
        "--sweep", "-s",
        choices=["inverted_u", "detection_effect", "mitigation_effect", "deployment_risk", "all"],
        default="all",
    )
    parser.add_argument("--quick", action="store_true", help="5 seeds for fast check")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seeds = list(range(n_seeds))
    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR
    print(f"Seeds: {len(seeds)} | Sweep: {args.sweep} | Out: {outdir}")

    sweep_map = {
        "inverted_u": sweep_inverted_u,
        "detection_effect": sweep_detection_effect,
        "mitigation_effect": sweep_mitigation_effect,
        "deployment_risk": sweep_deployment_risk,
    }

    if args.sweep == "all":
        for fn in sweep_map.values():
            fn(seeds, outdir)
    else:
        sweep_map[args.sweep](seeds, outdir)


if __name__ == "__main__":
    main()
