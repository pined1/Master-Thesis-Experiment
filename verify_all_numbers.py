#!/usr/bin/env python3
"""
verify_all_numbers.py

Checks every headline number in the paper against the committed JSON results
under experiments/*/results/. Prints each paper claim next to the value found
in the data with a ✓ (match) or ✗ (mismatch).

No arguments needed:

    python verify_all_numbers.py

For each claim the script loads the *oldest* file matching a known prefix in
each experiment's results/ folder — i.e. the committed result — so re-running an
experiment (which writes a newer-timestamped file) never changes the verdict.

Pure standard library; no third-party imports.
"""

import json
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

_passed = 0
_failed = 0
_cache = {}


def load(folder, prefix):
    """Oldest (committed) JSON in experiments/<folder>/results matching prefix."""
    key = (folder, prefix)
    if key in _cache:
        return _cache[key]
    hits = sorted(glob(str(EXP / folder / "results" / f"{prefix}*.json")))
    data = json.load(open(hits[0])) if hits else None
    _cache[key] = data
    return data


def dig(d, *keys):
    """Nested lookup; returns None if any key is missing."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def check(label, actual, paper, tol=0.5):
    """Compare a data value against a paper claim and print a ✓/✗ line.

    Increments the global pass/fail counters. Treats actual=None as a failure
    (value not found). Matches when abs(actual - paper) <= tol.

    Args:
        label: human-readable name for the claim.
        actual: value pulled from the JSON data (or None if absent).
        paper: the headline number reported in the paper.
        tol: maximum allowed absolute difference to count as a match.
    """
    global _passed, _failed
    if actual is None:
        _failed += 1
        print(f"  ✗ {label}: paper {paper}, value NOT FOUND in data")
        return
    if abs(float(actual) - float(paper)) <= tol:
        _passed += 1
        print(f"  ✓ {label}: {paper} ≈ {float(actual):.2f}")
    else:
        _failed += 1
        print(f"  ✗ {label}: paper {paper} vs data {float(actual):.2f} "
              f"(diff {abs(float(actual)-float(paper)):.2f})")


print("=" * 70)
print("PAPER NUMBER VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------- H1 (Table II)
print("\n[H1] Sharing scope — exp01 (incidents, knowledge K, transformation %)")
h1 = load("exp01_h1_sharing_scope", "exp1_learning_scenarios")
# scenario: (incidents, K, transform%)
H1 = {
    "NONE":     (484.3, 0.000, 0.0),
    "LOCAL":    (406.4, 0.555, 0.0),
    "NEIGHBOR": (336.0, 0.890, 14.0),
    "GLOBAL":   (265.6, 0.992, 89.5),
}
for sc, (inc, k, tr) in H1.items():
    check(f"{sc:8} incidents", dig(h1, sc, "total_incidents", "mean"), inc)
    check(f"{sc:8} knowledge", dig(h1, sc, "final_prevention_knowledge", "mean"), k, tol=0.01)
    actual_tr = dig(h1, sc, "transformation_rate", "mean")
    check(f"{sc:8} transform%", None if actual_tr is None else actual_tr * 100, tr)

# ---------------------------------------------------------------- H2 (exp04)
print("\n[H2] Deployment velocity — exp04 (0.05 vs 0.50 deploys/day)")
h2 = load("exp04_h2_velocity", "exp4_deployment_velocity")
for sc, lo, hi in [("LOCAL", 238.1, 288.6), ("GLOBAL", 152.5, 188.4)]:
    check(f"{sc:7} @0.05/day", dig(h2, sc, "0.05", "total_incidents", "mean"), lo)
    check(f"{sc:7} @0.50/day", dig(h2, sc, "0.5", "total_incidents", "mean"), hi)

# ---------------------------------------------------------------- H3 (Table VI)
print("\n[H3] Prevention coefficient β gradient — exp05 (500 seeds)")
h3 = load("exp05_h3_prevention", "publication_h3_500seeds")
for beta, val in [("0.0", 484.1), ("0.05", 468.8), ("0.1", 451.2),
                  ("0.2", 420.5), ("0.5", 335.3)]:
    check(f"β={beta:4} incidents", dig(h3, beta, "total_incidents", "mean"), val)

# ---------------------------------------------------------------- H1×H3 (Table VII)
print("\n[H1×H3] Cross-sweep — h1xh3 (50 seeds; paper rounds to integers)")
xs = load("h1xh3_crosssweep", "h1xh3_crosssweep")
cells = dig(xs, "cells") or {}
H1xH3 = {
    "NONE":     {"0.0": 482, "0.1": 482, "0.5": 482},
    "LOCAL":    {"0.0": 482, "0.1": 465, "0.5": 405},
    "NEIGHBOR": {"0.0": 479, "0.1": 448, "0.5": 334},
    "GLOBAL":   {"0.0": 484, "0.1": 439, "0.5": 266},
}
for sc, row in H1xH3.items():
    for beta, val in row.items():
        cell = cells.get(f"{sc}__prev{beta}")
        check(f"{sc:8} β={beta}", dig(cell, "total_incidents", "mean"), val, tol=1.0)

# ---------------------------------------------------------------- H4 (Table VIII)
print("\n[H4] Network topology — exp02 (NEIGHBOR sharing)")
h4 = load("exp02_h4_topology", "exp2_network_topology")
for key, name, val in [("complete", "Complete", 273.1),
                       ("erdos_renyi", "Erdos-Renyi", 323.3),
                       ("watts_strogatz", "Watts-Strogatz", 336.0),
                       ("barabasi_albert", "Barabasi-Albert", 346.9),
                       ("star", "Star", 382.1)]:
    check(f"{name:16} incidents", dig(h4, key, "total_incidents", "mean"), val)

# ---------------------------------------------------------------- exp10 (Table V)
print("\n[H2×α4] Deployment × exploitation — exp10 (NEIGHBOR)")
e10 = load("exp10_h2_alpha4", "exp10_robustness")
cfg10 = dig(e10, "configurations") or {}
EXP10 = {
    ("0.05", "0.2"): 310.67, ("0.05", "0.6"): 309.28, ("0.05", "0.9"): 310.12,
    ("0.1", "0.2"): 335.15,  ("0.1", "0.6"): 335.96,  ("0.1", "0.9"): 334.05,
    ("0.3", "0.2"): 365.86,  ("0.3", "0.6"): 363.55,  ("0.3", "0.9"): 364.03,
}
for (dep, a4), val in EXP10.items():
    cell = cfg10.get(f"dep{dep}_exp{a4}")
    check(f"dep={dep:4} α4={a4}", dig(cell, "total_incidents", "mean"), val)

# ---------------------------------------------------------------- Ablations (Table IX)
print("\n[Ablations] exp11 (no decay) & exp12 (no asymmetry)")
e11 = load("exp11_no_decay", "exp11_ablation")
nd = dig(e11, "configurations", "no_decay") or {}
for sc, val in [("NONE", 484.3), ("LOCAL", 398.5), ("NEIGHBOR", 323.0), ("GLOBAL", 263.3)]:
    check(f"no-decay  {sc:8}", dig(nd, sc, "total_incidents", "mean"), val)

e12 = load("exp12_no_asymmetry", "exp12_ablation")
na = dig(e12, "configurations", "no_asymmetry") or {}
for sc, val in [("NONE", 484.3), ("LOCAL", 437.5), ("NEIGHBOR", 348.7), ("GLOBAL", 265.2)]:
    check(f"no-asym   {sc:8}", dig(na, sc, "total_incidents", "mean"), val)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
total = _passed + _failed
print(f"{_passed}/{total} checks ✓" + ("" if _failed == 0 else f"   ({_failed} ✗)"))
print("=" * 70)
sys.exit(0 if _failed == 0 else 1)
