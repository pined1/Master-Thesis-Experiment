# exp04 — H2: Deployment Velocity (primary sweep)

Tests **H2**: a wider sharing scope flattens the velocity–incident curve.

## Why (motivation)

This tests H2: that a wider sharing scope flattens the velocity–incident curve,
reproducing the DevOps result that high deployment rate and stability coexist
when feedback is fast (Forsgren et al. 2018; Kim et al. 2016).

## How (manipulation)

A mixed factorial: two sharing scopes (LOCAL, GLOBAL) crossed with a 10×
deployment-rate sweep from 0.05 to 0.50 deployments per team per day. GLOBAL
should absorb the velocity penalty and bend the projected exponential incident
spike into sub-linear growth. (The secondary deployment × α₄ matrix is
`exp10_h2_alpha4`.)

## What is changing

| Variable | Values |
|----------|--------|
| `learning_scenario` | LOCAL, GLOBAL |
| `deployment_rate` | 0.05, 0.10, 0.20, 0.30, 0.50 /team/day |

Held at baseline: Watts–Strogatz (k=4), 20 teams, 365 days, β=0.5.
10 cells × 100 seeds = 1,000 runs.

## How to run

```bash
python experiments/exp04_h2_velocity/run.py
python experiments/exp04_h2_velocity/run.py --quick
```

## Results

`exp4_deployment_velocity_*.json`. Top-level keys `GLOBAL`, `LOCAL`, each with
per-rate aggregate stats (mean / std / 95% CI / n).
