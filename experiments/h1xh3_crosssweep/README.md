# h1xh3_crosssweep — H1×H3 Factorial Cross-Sweep

The full-factorial scope × β matrix that isolates the H1×H3 interaction.

## Why (motivation)

The interdependence of sharing scope and prevention strength is definitively
isolated via the full factorial H1×H3 cross-sweep. The NONE-scope row should
stay flat across every β value: with no cross-team ingestion, the translation and
exploitation loops never fire in the first place — so prevention strength has
nothing to amplify.

## How (manipulation)

A full factorial: four sharing scopes (NONE → GLOBAL) crossed with three
prevention levels β ∈ {0.0, 0.1, 0.5}.

## What is changing

| Variable | Values |
|----------|--------|
| `learning_scenario` | NONE, LOCAL, NEIGHBOR, GLOBAL |
| `prevention_effect` (β) | 0.0, 0.1, 0.5 |

Held at baseline: Watts–Strogatz (k=4), 20 teams, 365 days, deployment 0.10/day.
12 cells × 50 seeds = 600 runs.

## How to run

```bash
python experiments/h1xh3_crosssweep/run.py
python experiments/h1xh3_crosssweep/run.py --quick
```

## Results

`h1xh3_crosssweep_*.json` (Table 7). `cells` keyed `SCOPE__prevβ`, each with
incident stats (mean / std / 95% CI / n). Cell values are rounded from a 50-seed
sample, so small offsets vs the higher-resolution tables are expected.
