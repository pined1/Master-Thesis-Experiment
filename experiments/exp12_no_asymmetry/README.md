# exp12 — Ablation: Source Asymmetry Suppression

Ablation: how much does experiential learning contribute?

## Why (motivation)

Removes the experiential-learning shortcut: the source team now has to traverse
all four stage gates right alongside the non-source agents. This isolates how
much experiential learning is actually contributing to the reliability outcome.

## How (manipulation)

Re-runs the H1 baseline with source asymmetry suppressed. Expected: it hits the
LOCAL cell hardest while leaving NONE > LOCAL > NEIGHBOR > GLOBAL intact.

## What is changing

| Variable | Values |
|----------|--------|
| `disable_source_asymmetry` | off (baseline) → on (source team runs the full pipeline) |

Held at baseline: the full H1 configuration — four scopes, Watts–Strogatz (k=4),
20 teams, 365 days. The run sweeps all four scopes under both the
asymmetry-on baseline (`default_asymmetry`) and the suppressed-asymmetry
ablation (`no_asymmetry`): 8 cells × 100 seeds = 800 runs.

## How to run

```bash
python experiments/exp12_no_asymmetry/run.py
python experiments/exp12_no_asymmetry/run.py --quick
```

## Results

`exp12_ablation_no_asymmetry_*.json` (Table 9). `configurations` keyed
`default_asymmetry` / `no_asymmetry`, each by scenario (mean / std / 95% CI / n).
