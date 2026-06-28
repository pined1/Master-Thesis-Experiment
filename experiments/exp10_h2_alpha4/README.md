# exp10 — H2: Deployment Velocity × α₄ (secondary cross-sweep)

Secondary H2 matrix: does Stage-4 exploitation effectiveness interact with
deployment frequency?

## Why (motivation)

A secondary cross-sweep for H2. The H1×H3 cross-sweep already pins β as an
amplifier whose effect is conditional on sharing scope, so this matrix instead
asks whether downstream Stage-4 exploitation effectiveness (α₄) interacts with
deployment frequency independently.

## How (manipulation)

Crosses the daily deployment rate against the Stage-4 exploitation baseline gate
α₄ ∈ {0.2, 0.6, 0.9} under uniform NEIGHBOR sharing. The two axes are expected to
be independent: deployment rate adds incidents down each column while α₄ barely
moves the count across each row.

## What is changing

| Variable | Values |
|----------|--------|
| `deployment_rate` | 0.05, 0.10, 0.30 /team/day |
| Stage-4 exploitation (α₄) | 0.2, 0.6, 0.9 |

Held at baseline: NEIGHBOR sharing, Watts–Strogatz (k=4), 20 teams, 365 days,
β=0.5. 9 cells × 100 seeds = 900 runs.

## How to run

```bash
python experiments/exp10_h2_alpha4/run.py
python experiments/exp10_h2_alpha4/run.py --quick
```

## Results

`exp10_robustness_deployment_learning_*.json` (Table 5). `configurations` keyed
`dep<rate>_exp<α₄>`, each aggregate stats (mean / std / 95% CI / n).
