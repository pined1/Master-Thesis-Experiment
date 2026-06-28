# exp05 — H3: Prevention Strength (β gradient)

Tests **H3**: the prevention coefficient β shows diminishing returns.

## Why (motivation)

This tests H3: that the prevention coefficient β (the Stage-4 exploitation scale
factor) shows diminishing returns, which is what isolates the "fix-it treadmill"
curve practitioners keep describing (Cohen & Levinthal 1990; Zahra & George
2002; Reed 2019).

## How (manipulation)

A high-resolution single-axis β gradient under fixed NEIGHBOR sharing. Incidents
should fall monotonically with β. (The companion full-factorial scope × β
cross-sweep is `h1xh3_crosssweep`.)

## What is changing

| Variable | Values |
|----------|--------|
| `prevention_effect` (β) | 0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 |

Held at baseline: NEIGHBOR sharing, Watts–Strogatz (k=4), 20 teams, 365 days.
7 β values × 500 seeds = 3,500 runs.

## How to run

```bash
python experiments/exp05_h3_prevention/run.py            # 500 seeds — slow
python experiments/exp05_h3_prevention/run.py --quick    # fast smoke test
```

## Results

`publication_h3_500seeds_*.json` (Table 6). Per-β aggregate stats
(mean / std / 95% CI / n).
