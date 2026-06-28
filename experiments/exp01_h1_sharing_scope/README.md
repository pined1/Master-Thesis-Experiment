# exp01 — H1: Sharing Scope

Tests **H1**: widening the sharing-scope boundary lowers systemic incidents.

## Why (motivation)

This tests H1: that widening the sharing-scope boundary lowers systemic
incidents, which is how I operationalize the Stage-1 acquisition gate of
relative absorptive capacity (Cohen & Levinthal 1990; Zahra & George 2002).

## How (manipulation)

The independent variable is the categorical sharing scope, swept across NONE,
LOCAL, NEIGHBOR, and GLOBAL, with everything else pinned at baseline. Incidents
per organization-year should drop monotonically NONE > LOCAL > NEIGHBOR >
GLOBAL, and the cross-team transformation rate should rise along the same axis.

## What is changing

| Variable | Values |
|----------|--------|
| `learning_scenario` | NONE, LOCAL, NEIGHBOR, GLOBAL |

Held at baseline: Watts–Strogatz (k=4, p_rewire=0.1), 20 teams, 365 days,
deployment 0.10/day, β=0.5. 4 cells × 100 seeds = 400 runs.

## How to run

```bash
python experiments/exp01_h1_sharing_scope/run.py            # H1 sweep (paper Table 4)
python experiments/exp01_h1_sharing_scope/analysis.py       # Cohen's d, 6 pairwise comparisons
python experiments/exp01_h1_sharing_scope/run.py --quick    # fast smoke test
```

## Results

- `run.py` → `exp1_learning_scenarios_*.json` (the H1 performance profile, Table 4).
- `analysis.py` → `analysis_cohens_d_*.json` and `analysis_time_dynamics_*.json`
  (effect sizes for the six pairwise comparisons; knowledge accumulation over time).

Each result is per-scenario aggregate stats (mean / std / 95% CI / n).
