# Experiments

The paper reports four hypotheses (H1–H4) plus the H1×H3 cross-sweep, two
ablations, and robustness/calibration sweeps — about 12,000 simulation runs.
Each folder below is one of those, with its own README, runnable script(s), and
committed `results/`.

**Naming.** Numbered folders (`exp01`, `exp02`, …) match the paper's own
experiment labels. The cross-sweep, pilots, and sensitivity sweeps keep
descriptive names because the paper refers to them by name, not by number.

## The four hypotheses

| Hypothesis | What is manipulated | What we expect |
|-----------|---------------------|----------------|
| **H1** Sharing scope | NONE / LOCAL / NEIGHBOR / GLOBAL | Monotonic drop NONE > LOCAL > NEIGHBOR > GLOBAL |
| **H2** Deployment velocity | Deployment rate × sharing scope | GLOBAL absorbs the velocity penalty |
| **H3** Prevention strength | Prevention coefficient β | Diminishing returns; NONE row stays flat |
| **H4** Network topology | 5 graph archetypes | Star worst; ~28% spread to Complete |

## The 11 experiments

| Folder | Paper | Hypothesis | Runs |
|--------|-------|------------|------|
| `exp01_h1_sharing_scope` | Table 4 | H1 | 400 |
| `exp04_h2_velocity` | §H2 | H2 (primary sweep) | 1,000 |
| `exp10_h2_alpha4` | Table 5 | H2 (secondary, ×α₄) | 900 |
| `exp05_h3_prevention` | Table 6 | H3 (β gradient) | 3,500 |
| `h1xh3_crosssweep` | Table 7 | H1×H3 (Proposition 1) | 600 |
| `exp02_h4_topology` | Table 8 | H4 | 500 |
| `exp11_no_decay` | Table 9 | Ablation | 800 |
| `exp12_no_asymmetry` | Table 9 | Ablation | 800 |
| `exp09_team_count` | §5.5 | H1 (robustness) | 1,200 |
| `magic_number_pilots` | §Sensitivity | Calibration pilots | 1,440 |
| `sensitivity_sweeps` | §Sensitivity | Per-parameter robustness | ~4,300 |

## How to run

Each folder has a `run.py` (a couple have a second script). From the repo root:

```bash
python experiments/exp01_h1_sharing_scope/run.py
python experiments/exp01_h1_sharing_scope/run.py --quick   # fast smoke test
```

Every script imports the shared engine (`model.py` at the repo root) and writes
timestamped JSON into its own `results/`. See each folder's README for the exact
hypothesis, manipulation, and command.

## Reading order

Start with `exp01_h1_sharing_scope` (the H1 baseline), then `h1xh3_crosssweep`
(the H1×H3 cross-sweep that becomes Proposition 1 — the paper's central result).
