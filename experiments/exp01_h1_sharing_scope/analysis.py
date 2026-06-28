"""
Experiment 01 — H1: Sharing Scope — POST-HOC ANALYSES
================================================================================
HYPOTHESIS (H1)
    As the postmortem sharing scope expands from NONE → LOCAL → NEIGHBOR →
    GLOBAL, total annual incident counts decrease monotonically. This file does
    NOT re-test H1 with the headline sweep (that is run.py); instead it produces
    the two supporting analyses the paper reports alongside H1.

WHAT THIS SCRIPT VARIES
    learning_scenario ∈ {NONE, LOCAL, NEIGHBOR, GLOBAL}   (the ONLY variable)
    Everything else in SimulationParams is held at the same baseline as run.py.

WHAT IT MEASURES  (two analyses, two output prefixes)
    1. time_dynamics  →  analysis_time_dynamics_<timestamp>.json
       Re-runs H1 collecting per-day time series, then bins incidents into
       30-day windows to answer "WHEN does the H1 ordering emerge?". Also tracks
       prevention-knowledge accumulation and cross-team transformation over time.
       Top-level keys: "NONE", "LOCAL", "NEIGHBOR", "GLOBAL"; each holds
       {avg_incidents_per_window, avg_knowledge, avg_transform}.
       Committed full run uses 50 seeds.

    2. cohens_d  →  analysis_cohens_d_<timestamp>.json
       Re-runs H1 keeping raw per-seed incident counts, then computes Cohen's d
       effect sizes and Welch t-test p-values for every key pairwise comparison.
       Top-level keys are the comparison labels, e.g.
       "NONE vs GLOBAL (main finding)"; each holds {cohens_d, p_value, magnitude}.
       Committed full run uses 100 seeds.

HOW TO RUN  (from anywhere — paths are resolved from this file's location)
    python experiments/exp01_h1_sharing_scope/analysis.py             # both, committed seed counts (50 / 100)
    python experiments/exp01_h1_sharing_scope/analysis.py --analysis cohens_d
    python experiments/exp01_h1_sharing_scope/analysis.py --quick     # smoke test: 2 seeds each
    python experiments/exp01_h1_sharing_scope/analysis.py --seeds 30  # override BOTH analyses
    python experiments/exp01_h1_sharing_scope/analysis.py --outdir /tmp/x

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy/scipy.
The two analysis bodies are faithful copies of analyze_time_dynamics() and
analyze_cohens_d() from 02-Framework-Code/analysis_extra.py; only the I/O paths
and seed-count plumbing were adapted so the folder stands on its own. The
committed seed counts (50 for time_dynamics, 100 for cohens_d) and the numeric
base_params are preserved exactly.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Make the local model.py importable no matter the current directory.
# noqa: E402 — import follows the sys.path bootstrap above.
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))
from model import SimulationParams, LearningScenario, run_simulation

warnings.filterwarnings("ignore")

# Results land in this experiment folder's own results/ directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Committed seed counts for the two analyses (preserved exactly from
# analysis_extra.py). A --seeds override replaces both; --quick uses 2.
TIME_DYNAMICS_SEEDS = 50
COHENS_D_SEEDS = 100
QUICK_SEEDS = 2

# Baseline configuration shared by both analyses (identical to run.py / paper).
BASE_PARAMS = {
    "num_teams": 20,
    "steps": 365,
    "network_topology": "watts_strogatz",
    "base_incident_rate": 0.05,
    "deployment_rate": 0.1,
    "transformation_probability": 0.6,
}


# ==============================================================================
# SHARED HELPERS  (copied verbatim from analysis_extra.py)
# ==============================================================================

def cohens_d(group1: list, group2: list) -> float:
    """Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def interpret_d(d: float) -> str:
    """Map a Cohen's d to a magnitude label (negligible/small/medium/LARGE)."""
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "LARGE"


# ==============================================================================
# ANALYSIS 1: Time Dynamics — When Does H1 Ordering Emerge?
# (faithful copy of analyze_time_dynamics; re-runs H1 to get per-day series)
# ==============================================================================

def analyze_time_dynamics(n_seeds: int, outdir: Path) -> Dict[str, Any]:
    """Re-run H1 over `n_seeds` collecting per-day series, bin incidents into
    30-day windows to find when the NONE>LOCAL>NEIGHBOR>GLOBAL ordering emerges,
    write the result JSON to `outdir`, and return the per-scenario series dict."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Time Dynamics — When Does H1 Ordering Emerge?")
    print("=" * 70)

    print(f"\n  Running a focused {n_seeds}-seed time-series extraction...")

    base_params = dict(BASE_PARAMS)

    seeds = list(range(n_seeds))
    window = 30

    scenario_ts: Dict[str, Any] = {}

    for scenario in LearningScenario:
        label = scenario.name
        print(f"\n  Extracting time series for {label} ({n_seeds} seeds)...", end=" ", flush=True)
        incident_series = []
        knowledge_series = []
        transform_series = []

        for seed in seeds:
            params = SimulationParams(**{**base_params, "learning_scenario": scenario, "seed": seed})
            result = run_simulation(params)
            ts = result["time_series"]
            # incident_frequency is a dict keyed by subsystem — sum across subsystems
            freq_dict = ts["incident_frequency"]
            total_per_day = np.sum([freq_dict[k] for k in freq_dict], axis=0).tolist()
            incident_series.append(total_per_day)
            knowledge_series.append(ts["avg_prevention_knowledge"])
            if ts.get("transformation_rate"):
                transform_series.append(ts["transformation_rate"])

        max_len = min(len(s) for s in incident_series)
        avg_incidents = np.mean([s[:max_len] for s in incident_series], axis=0)
        avg_knowledge = np.mean([s[:max_len] for s in knowledge_series], axis=0)
        avg_transform = np.mean([s[:max_len] for s in transform_series], axis=0) if transform_series else np.zeros(max_len)

        window_incidents = []
        for start in range(0, max_len - window + 1, window):
            window_incidents.append(float(np.sum(avg_incidents[start:start+window])))

        scenario_ts[label] = {
            "avg_incidents_per_window": window_incidents,
            "avg_knowledge": avg_knowledge.tolist(),
            "avg_transform": avg_transform.tolist(),
        }
        print(f"Done.")

    print(f"\n  INCIDENT COUNT BY 30-DAY WINDOW (mean across {n_seeds} seeds):")
    print("  " + "-" * 75)
    windows = list(range(1, len(scenario_ts["NONE"]["avg_incidents_per_window"]) + 1))
    header = f"  {'Window':<10}" + "".join(f"{'Day '+str(w*30):>12}" for w in windows[:12])
    print(header)
    print("  " + "-" * 75)
    for label in ["NONE", "LOCAL", "NEIGHBOR", "GLOBAL"]:
        row = f"  {label:<10}" + "".join(f"{v:>12.1f}" for v in scenario_ts[label]["avg_incidents_per_window"][:12])
        print(row)

    print("\n  ORDERING CHECK BY WINDOW (NONE > LOCAL > NEIGHBOR > GLOBAL):")
    print("  " + "-" * 50)
    n_windows = len(scenario_ts["NONE"]["avg_incidents_per_window"])
    first_hold = None
    for w in range(n_windows):
        none_v = scenario_ts["NONE"]["avg_incidents_per_window"][w]
        local_v = scenario_ts["LOCAL"]["avg_incidents_per_window"][w]
        neighbor_v = scenario_ts["NEIGHBOR"]["avg_incidents_per_window"][w]
        global_v = scenario_ts["GLOBAL"]["avg_incidents_per_window"][w]
        holds = none_v > local_v > neighbor_v > global_v
        status = "HOLDS" if holds else "not yet"
        print(f"  Days {w*30+1:>4}-{(w+1)*30:>4}: {status}  (NONE={none_v:.1f}, LOCAL={local_v:.1f}, NEIGHBOR={neighbor_v:.1f}, GLOBAL={global_v:.1f})")
        if holds and first_hold is None:
            first_hold = (w + 1) * 30

    if first_hold:
        print(f"\n  -> H1 ordering first holds consistently at DAY {first_hold}")
    else:
        print(f"\n  -> H1 ordering does not fully stabilize within 365 days")

    print("\n  KNOWLEDGE ACCUMULATION — when does GLOBAL diverge from NONE?")
    none_k = scenario_ts["NONE"]["avg_knowledge"]
    global_k = scenario_ts["GLOBAL"]["avg_knowledge"]
    for day in [30, 60, 90, 120, 180, 270, 365]:
        idx = min(day - 1, len(none_k) - 1)
        print(f"  Day {day:>4}: NONE={none_k[idx]:.3f}  GLOBAL={global_k[idx]:.3f}  gap={global_k[idx]-none_k[idx]:.3f}")

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"analysis_time_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(scenario_ts, f, indent=2)
    print(f"\n  Saved: {output_path}")

    return scenario_ts


# ==============================================================================
# ANALYSIS 2: Cohen's d Effect Sizes
# (faithful copy of analyze_cohens_d; re-runs H1 keeping raw distributions)
# ==============================================================================

def analyze_cohens_d(n_seeds: int, outdir: Path) -> Dict[str, Any]:
    """Re-run H1 over `n_seeds` keeping raw per-seed incident counts, compute
    Cohen's d and Welch t-test p-values for each key pairwise scenario
    comparison, write the result JSON to `outdir`, and return the results dict."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Cohen's d Effect Sizes")
    print("=" * 70)

    print(f"  Re-running H1 with {n_seeds} seeds to extract raw distributions...")

    base_params = dict(BASE_PARAMS)

    seeds = list(range(n_seeds))
    raw: Dict[str, list] = {}

    for scenario in LearningScenario:
        label = scenario.name
        print(f"  Running {label}...", end=" ", flush=True)
        incidents = []
        for seed in seeds:
            params = SimulationParams(**{**base_params, "learning_scenario": scenario, "seed": seed})
            result = run_simulation(params)
            incidents.append(result["summary"]["total_incidents"])
        raw[label] = incidents
        print(f"Done. Mean: {np.mean(incidents):.1f}, SD: {np.std(incidents):.1f}")

    print("\n  COHEN'S D EFFECT SIZES (pairwise):")
    print("  " + "-" * 65)
    print(f"  {'Comparison':<30} {'d':>8} {'|d|':>8} {'Magnitude':>12} {'p<0.001?':>10}")
    print("  " + "-" * 65)

    comparisons = [
        ("NONE", "GLOBAL",    "NONE vs GLOBAL (main finding)"),
        ("NONE", "NEIGHBOR",  "NONE vs NEIGHBOR"),
        ("NONE", "LOCAL",     "NONE vs LOCAL"),
        ("LOCAL", "NEIGHBOR", "LOCAL vs NEIGHBOR"),
        ("LOCAL", "GLOBAL",   "LOCAL vs GLOBAL"),
        ("NEIGHBOR", "GLOBAL","NEIGHBOR vs GLOBAL"),
    ]

    from scipy import stats as scipy_stats
    results: Dict[str, Any] = {}

    for g1, g2, label in comparisons:
        d = cohens_d(raw[g1], raw[g2])
        mag = interpret_d(d)
        t_stat, p_val = scipy_stats.ttest_ind(raw[g1], raw[g2])
        sig = "YES" if p_val < 0.001 else f"p={p_val:.4f}"
        print(f"  {label:<30} {d:>8.3f} {abs(d):>8.3f} {mag:>12} {sig:>10}")
        results[label] = {"cohens_d": d, "p_value": float(p_val), "magnitude": mag}

    print("\n  EFFECT SIZE REFERENCE: <0.2=negligible, 0.2-0.5=small, 0.5-0.8=medium, >0.8=LARGE")

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / f"analysis_cohens_d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_path}")

    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H1 post-hoc analyses (time dynamics + Cohen's d)")
    parser.add_argument(
        "--analysis", "-a",
        choices=["time_dynamics", "cohens_d", "all"],
        default="all",
        help="which analysis to run (default: all)",
    )
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None,
                        help="override seed count for BOTH analyses (default: committed 50 / 100)")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else RESULTS_DIR

    # Resolve seed counts. --seeds overrides both; --quick uses QUICK_SEEDS for
    # both; otherwise each analysis keeps its committed paper seed count.
    if args.seeds is not None:
        td_seeds = cd_seeds = args.seeds
    elif args.quick:
        td_seeds = cd_seeds = QUICK_SEEDS
    else:
        td_seeds = TIME_DYNAMICS_SEEDS
        cd_seeds = COHENS_D_SEEDS

    if args.analysis in ("all", "time_dynamics"):
        print(f"time_dynamics: {td_seeds} seeds → {outdir}")
        analyze_time_dynamics(td_seeds, outdir)
    if args.analysis in ("all", "cohens_d"):
        print(f"cohens_d: {cd_seeds} seeds → {outdir}")
        analyze_cohens_d(cd_seeds, outdir)
