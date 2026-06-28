# exp11 — Ablation: Knowledge Depreciation Removal

Ablation: does the H1 ordering survive without knowledge decay?

## Why (motivation)

To see how sensitive the model really is to its core assumptions, I disable one
assumption at a time and re-run the full H1 baseline matrix. exp11 sets the daily
decay variable to δ = 0, disabling the calibrated depreciation routine (Darr et
al. 1995). This checks whether the H1 ordering survives without continuous
knowledge degradation.

## How (manipulation)

Re-runs the H1 baseline (four sharing scopes × 100 seeds × 365 days) with
knowledge decay turned off. Expected: removing decay changes the LOCAL–NEIGHBOR
spread amplitude but does not invert the ordering.

## What is changing

| Variable | Values |
|----------|--------|
| `disable_knowledge_decay` | off (δ = 0.001, baseline) → on (δ = 0) |

Held at baseline: the full H1 configuration — four scopes, Watts–Strogatz (k=4),
20 teams, 365 days. The run sweeps all four scopes under both the decay-on
baseline (`default_decay`) and the decay-off ablation (`no_decay`):
8 cells × 100 seeds = 800 runs.

## How to run

```bash
python experiments/exp11_no_decay/run.py
python experiments/exp11_no_decay/run.py --quick
```

## Results

`exp11_ablation_no_decay_*.json` (Table 9). `configurations` keyed
`default_decay` / `no_decay`, each by scenario (mean / std / 95% CI / n).
