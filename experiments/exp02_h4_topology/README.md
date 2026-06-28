# exp02 — H4: Network Topology

Tests **H4**: network topology drives organizational learning rates.

## Why (motivation)

This tests H4: that network topology drives organizational learning rates by
shaping routing speed and postmortem visibility boundaries (Reagans & McEvily
2003).

## How (manipulation)

Sharing scope is fixed at NEIGHBOR — the only scope where graph edges actually
mediate propagation. The independent variable is topology, swept across the five
archetypes: Complete, Erdős–Rényi, Watts–Strogatz, Barabási–Albert, and Star.
Expected: a ~28% incident delta between the centralized Star bottleneck and
Complete connectivity, with small-world structures outperforming scale-free
(m=2) variants whose preferential-attachment hubs reproduce star-like
bottlenecks.

## What is changing

| Variable | Values |
|----------|--------|
| `network_topology` | Complete, Erdős–Rényi, Watts–Strogatz, Barabási–Albert, Star |

Held at baseline: NEIGHBOR sharing, 20 teams, 365 days, β=0.5, α at defaults.
5 cells × 100 seeds = 500 runs. Sub-sweeps on Watts–Strogatz k and
Barabási–Albert m run separately for sensitivity tracking.

## How to run

```bash
python experiments/exp02_h4_topology/run.py                 # 5-topology sweep (Table 8)
python experiments/exp02_h4_topology/publication_sweeps.py  # WS-k and BA-m sub-sweeps
python experiments/exp02_h4_topology/run.py --quick
```

## Results

- `run.py` → `exp2_network_topology_*.json` (Table 8).
- `publication_sweeps.py` → `publication_ws_k_sweep_*.json`, `publication_ba_m_sweep_*.json`.

Per-topology aggregate stats (mean / std / 95% CI / n).
