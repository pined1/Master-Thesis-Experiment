"""
Magic-Number Pilot Sweeps — calibration robustness
================================================================================
PURPOSE
    Three calibration choices in the model are NOT anchored in prior literature:
      (1) the 0.7/0.3 assimilation weighting (cognitive vs. documentation),
      (2) the 0.5 relevance floor in the (0.5 + 0.5*ρ) relevance multiplier, and
      (3) the 3-day deployment risk window.
    This script confirms that the H1 ordering NONE > LOCAL > NEIGHBOR > GLOBAL
    survives across a range of reasonable values for each of the three
    un-anchored constants. If the ordering holds everywhere, the headline
    result is structural, not an artifact of these hand-picked numbers.

WHAT THIS SCRIPT VARIES  (three independent sweeps)
    Sweep 1 — assimilation_weights:
        (cognitive_weight, doc_weight) ∈ {(0.3,0.7), (0.5,0.5), (0.7,0.3)*, (0.9,0.1)}
    Sweep 2 — relevance_floor:
        floor ∈ {0.0, 0.25, 0.5*, 0.75}   (multiplier = floor + (1-floor)*ρ)
    Sweep 3 — deployment_window_days:
        window ∈ {1, 3*, 5, 7}            (* = model default)
    Each cell runs all 4 sharing scenarios (NONE / LOCAL / NEIGHBOR / GLOBAL)
    at NUM_SEEDS seeds, 20 teams, 365 steps, Watts-Strogatz topology.

MECHANISM
    Sweep 1 monkey-patches the two module-level weight constants directly.
    Sweeps 2 & 3 need to alter inline arithmetic, so they do source-level
    patching: the LOCAL model.py source is read, the relevant expressions are
    regex/string-substituted, and the patched source is exec'd into a fresh
    module namespace. The patch reads the model.py sitting next to this file.

OUTPUT  (one JSON per sweep, written to ../results/)
    pilot_assimilation_weights_<timestamp>.json
    pilot_relevance_floor_<timestamp>.json
    pilot_deployment_window_<timestamp>.json
    Each JSON has top-level keys {parameter, values}; values maps each tested
    parameter value to per-scenario incident stats + the ordering_holds flag.

HOW TO RUN  (from anywhere — paths resolve from this file's location)
    python experiments/magic_number_pilots/run.py              # full: 30 seeds
    python experiments/magic_number_pilots/run.py --quick      # smoke: 3 seeds
    python experiments/magic_number_pilots/run.py --seeds 10   # custom seed count
    python experiments/magic_number_pilots/run.py --outdir /tmp/x  # custom outdir

This script is SELF-CONTAINED: it imports the repo-root model.py
(the one shared engine at the repo root) plus numpy. Only the I/O
paths and the local-source lookup were adapted so the folder stands on its own.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Locate the repo-root model.py by walking up the tree, then make it importable.
# `model` is imported as a module (not just its names) so Sweep 1 can rebind its
# weight constants in place.
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists())))

import model
from model import (
    SimulationParams,
    LearningScenario,
    run_simulation,
)

# Results land in this experiment folder's own results/ directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
# 30 seeds is enough to confirm ordering; we are not estimating curvature.
NUM_SEEDS = 30
QUICK_SEEDS = 3
NUM_TEAMS = 20
STEPS = 365

SCENARIOS = [
    LearningScenario.NONE,
    LearningScenario.LOCAL,
    LearningScenario.NEIGHBOR,
    LearningScenario.GLOBAL,
]


def run_cell(scenario: LearningScenario, seeds: range) -> dict:
    """Run one learning scenario across all seeds; return mean/std/n of total_incidents."""
    incidents = []
    for s in seeds:
        params = SimulationParams(
            seed=s,
            num_teams=NUM_TEAMS,
            steps=STEPS,
            network_topology="watts_strogatz",
            learning_scenario=scenario,
        )
        r = run_simulation(params)
        incidents.append(r["summary"]["total_incidents"])
    return {
        "mean": float(np.mean(incidents)),
        "std": float(np.std(incidents, ddof=1)),
        "n": len(incidents),
    }


def run_block(label: str, seeds: range) -> dict:
    """Run all four scenarios for one parameter setting; report and flag whether the
    H1 ordering NONE > LOCAL > NEIGHBOR > GLOBAL holds. Returns cells/means/ordering_holds.
    """
    cells = {}
    for sc in SCENARIOS:
        cells[sc.name] = run_cell(sc, seeds)
    means = [cells[sc.name]["mean"] for sc in SCENARIOS]
    ordering_holds = means[0] > means[1] > means[2] > means[3]
    print(
        f"  [{label}] NONE={means[0]:.1f}  LOCAL={means[1]:.1f}  "
        f"NEIGHBOR={means[2]:.1f}  GLOBAL={means[3]:.1f}  "
        f"ordering: {'HELD' if ordering_holds else 'BROKEN'}"
    )
    return {"cells": cells, "ordering_holds": ordering_holds, "means": means}


# ---------------- Sweep 1: Assimilation weighting 0.7/0.3 ----------------
def sweep_assimilation_weights(seeds: range):
    """Sweep 1: vary the (cognitive, doc) assimilation weights.

    Rebinds the two module-level weight constants in place (no source patch
    needed), running the four scenarios at each weight pair, then restores the
    0.7/0.3 default so later sweeps see an unpatched module. (0.7, 0.3) is the
    model default. Returns {parameter, values}.
    """
    print("\n=== Sweep 1: Assimilation weighting (cognitive_weight, doc_weight) ===")
    out = {"parameter": "assimilation_weights", "values": {}}
    weight_pairs = [
        (0.3, 0.7),
        (0.5, 0.5),
        (0.7, 0.3),
        (0.9, 0.1),
    ]
    for cog_w, doc_w in weight_pairs:
        model.ASSIMILATION_COGNITIVE_WEIGHT = cog_w
        model.ASSIMILATION_DOC_WEIGHT = doc_w
        label = f"cog={cog_w}, doc={doc_w}"
        out["values"][f"{cog_w}_{doc_w}"] = run_block(label, seeds)
    # Restore the model defaults so later sweeps see an unpatched module.
    model.ASSIMILATION_COGNITIVE_WEIGHT = 0.7
    model.ASSIMILATION_DOC_WEIGHT = 0.3
    return out


# ---- Source-patching machinery for Sweeps 2 & 3 ----
# Sweeps 2 and 3 alter inline arithmetic that has no module-level constant to
# rebind, so they patch the model's SOURCE: the repo-root model.py text (the
# same shared engine imported above, located by walking up from this script —
# not a CWD file and not a local copy) is read, the target expressions are
# string-substituted, and the result is exec'd into a fresh module namespace.
# Patch targets:
#   Sweep 2 floor — the three (0.5 + 0.5 * X) relevance multipliers.
#   Sweep 3 window — the `recent_deployments[team.subsystem] = 3` assignment.

import re
import types

MODEL_SRC_PATH = next(p for p in Path(__file__).resolve().parents if (p / "model.py").exists()) / "model.py"


def make_patched_model(floor: float, deploy_window: int = None):
    """Return a freshly loaded model module with patched constants."""
    src = MODEL_SRC_PATH.read_text()
    rest = 1.0 - floor
    src = src.replace("(0.5 + 0.5 * relevance)", f"({floor} + {rest} * relevance)")
    src = src.replace("(0.5 + 0.5 * cognitive_factor)", f"({floor} + {rest} * cognitive_factor)")
    src = src.replace("(0.5 + 0.5 * params.documentation_quality)", f"({floor} + {rest} * params.documentation_quality)")
    if deploy_window is not None:
        src = src.replace("recent_deployments[team.subsystem] = 3", f"recent_deployments[team.subsystem] = {deploy_window}")
    patched = types.ModuleType("model_patched")
    patched.__file__ = "model_patched.py"
    exec(compile(src, "model_patched.py", "exec"), patched.__dict__)
    return patched


def run_cell_patched(patched, scenario, seeds):
    """Like run_cell, but uses a source-patched model module; return mean/std/n of incidents."""
    incidents = []
    for s in seeds:
        params = patched.SimulationParams(
            seed=s,
            num_teams=NUM_TEAMS,
            steps=STEPS,
            network_topology="watts_strogatz",
            learning_scenario=patched.LearningScenario[scenario.name],
        )
        r = patched.run_simulation(params)
        incidents.append(r["summary"]["total_incidents"])
    return {
        "mean": float(np.mean(incidents)),
        "std": float(np.std(incidents, ddof=1)),
        "n": len(incidents),
    }


def run_block_patched(patched, label, seeds):
    """Like run_block, but against a source-patched model; flags whether H1 ordering holds."""
    cells = {}
    for sc in SCENARIOS:
        cells[sc.name] = run_cell_patched(patched, sc, seeds)
    means = [cells[sc.name]["mean"] for sc in SCENARIOS]
    ordering_holds = means[0] > means[1] > means[2] > means[3]
    print(
        f"  [{label}] NONE={means[0]:.1f}  LOCAL={means[1]:.1f}  "
        f"NEIGHBOR={means[2]:.1f}  GLOBAL={means[3]:.1f}  "
        f"ordering: {'HELD' if ordering_holds else 'BROKEN'}"
    )
    return {"cells": cells, "ordering_holds": ordering_holds, "means": means}


# ---- Sweep 2: relevance multiplier floor (default 0.5) ----
def sweep_relevance_floor(seeds: range):
    """Sweep 2: vary the relevance-multiplier floor (model default 0.5).

    For each floor it builds a source-patched model and runs the four scenarios.
    Returns {parameter, values}.
    """
    print("\n=== Sweep 2: Relevance multiplier floor (default 0.5) ===")
    out = {"parameter": "relevance_floor", "values": {}}
    for floor in [0.0, 0.25, 0.5, 0.75]:
        patched = make_patched_model(floor=floor)
        out["values"][f"{floor}"] = run_block_patched(patched, f"floor={floor}", seeds)
    return out


# ---- Sweep 3: deployment risk window (default 3 days) ----
def sweep_deployment_window(seeds: range):
    """Sweep 3: vary the deployment risk window in days (model default 3).

    For each window it builds a source-patched model (floor held at 0.5) and runs
    the four scenarios. Returns {parameter, values}.
    """
    print("\n=== Sweep 3: Deployment risk window (default 3 days) ===")
    out = {"parameter": "deployment_window_days", "values": {}}
    for window in [1, 3, 5, 7]:
        patched = make_patched_model(floor=0.5, deploy_window=window)
        out["values"][f"{window}"] = run_block_patched(patched, f"window={window}d", seeds)
    return out


def save(name, results, outdir: Path):
    """Write results to a timestamped <name>_*.json in outdir; return the path."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return path


def main(seeds: range, outdir: Path):
    """Run all three pilot sweeps, save one JSON each, and print whether the H1
    ordering held at every tested value of each constant.
    """
    n_seeds = len(seeds)
    print(f"Running magic-number pilot sweeps")
    print(f"  Seeds per cell: {n_seeds}")
    print(f"  Teams: {NUM_TEAMS}  Steps: {STEPS}  Topology: Watts-Strogatz")
    print(f"  Scenarios: NONE / LOCAL / NEIGHBOR / GLOBAL")
    print(f"  Output dir: {outdir}")

    r1 = sweep_assimilation_weights(seeds)
    save("pilot_assimilation_weights", r1, outdir)

    r2 = sweep_relevance_floor(seeds)
    save("pilot_relevance_floor", r2, outdir)

    r3 = sweep_deployment_window(seeds)
    save("pilot_deployment_window", r3, outdir)

    print("\n=== Summary ===")
    for name, r in [("Assimilation weights", r1), ("Relevance floor", r2), ("Deployment window", r3)]:
        all_held = all(v["ordering_holds"] for v in r["values"].values())
        print(f"  {name}: H1 ordering held at every tested value? {'YES' if all_held else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magic-number calibration pilot sweeps")
    parser.add_argument("--quick", action="store_true", help="smoke test with few seeds")
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--outdir", type=str, default=None, help="override output directory")
    args = parser.parse_args()

    n_seeds = args.seeds if args.seeds else (QUICK_SEEDS if args.quick else NUM_SEEDS)
    seed_range = range(n_seeds)
    out_dir = Path(args.outdir) if args.outdir else RESULTS_DIR

    main(seed_range, out_dir)
