# exp09 — Robustness: Team Count

Robustness check: does the H1 ordering hold as organization size changes?

## Why (motivation)

We ran the simulation at three team counts to check whether organization size
has an effect on the result. If the NONE > LOCAL > NEIGHBOR > GLOBAL ordering
were an artifact of the baseline 20-team configuration, it would scramble at
other sizes.

## How (manipulation)

Re-runs the H1 baseline (four sharing scopes) at three organization sizes
N ∈ {6, 20, 50}. The ordering should hold at every size, and the reduction from
NONE to GLOBAL should stay within a 34%–48% band across all three.

## What is changing

| Variable | Values |
|----------|--------|
| `num_teams` | 6, 20, 50 |
| `learning_scenario` | NONE, LOCAL, NEIGHBOR, GLOBAL |

Held at baseline: Watts–Strogatz (k=4), 365 days, deployment 0.10/day, β=0.5.
3 sizes × 4 scopes × 100 seeds = 1,200 runs.

## How to run

```bash
python experiments/exp09_team_count/run.py
python experiments/exp09_team_count/run.py --quick
```

## Results

`exp9_robustness_team_count_*.json` (§5.5). `configurations` keyed by team count
(`6` / `20` / `50`), each by scenario, with aggregate stats (mean / std / 95% CI / n).
