# Paper-to-Code Mapping

Every figure, table, and headline number in the paper is traceable to a specific experimental run and JSON file. This document is the cross-reference.

## Tables in the paper

Each JSON file lives in its experiment's own `experiments/<folder>/results/` directory.

| Paper table | Caption | Code that produces it | JSON file (in `experiments/<folder>/results/`) |
|-------------|---------|----------------------|-----------------------------------------|
| Table 1 (`tab:team-a-snapshot`) | Team A knowledge matrix at day 90 under NEIGHBOR, seed 42 | illustrative snapshot, hand-extracted from an instrumented seed-42 NEIGHBOR run of `model.py` | (no committed script; `run_simulation` returns only the end-of-run `final_knowledge`, not per-day per-team matrices) |
| Table 2 (`tab:sim-knobs`) | Configurable simulator parameters | Documentation only | (no JSON) |
| Table 3 (`tab:methodology-summary`) | Cross-experiment configuration matrix | Documentation only | (no JSON) |
| Table 4 (`tab:h1-core`) | H1 sharing scope performance profile | `experiments/exp01_h1_sharing_scope/run.py` | `exp01_h1_sharing_scope/results/exp1_learning_scenarios_*.json` |
| Table 5 (`tab:h2-crosssweep`) | exp10 H2×α₄ cross-sweep matrix | `experiments/exp10_h2_alpha4/run.py` | `exp10_h2_alpha4/results/exp10_robustness_*.json` |
| Table 6 (`tab:h3-sweep`) | β gradient sweep (7 values × 500 seeds) | `experiments/exp05_h3_prevention/run.py` | `exp05_h3_prevention/results/publication_h3_*.json` |
| Table 7 (`tab:h1xh3`) | H1×H3 cross-sweep (4 scopes × 3 β values) | `experiments/h1xh3_crosssweep/run.py` | `h1xh3_crosssweep/results/h1xh3_crosssweep_*.json` |
| Table 8 (`tab:h4-ranking`) | H4 network topology results | `experiments/exp02_h4_topology/run.py` | `exp02_h4_topology/results/exp2_network_topology_*.json` |
| Table 9 (`tab:ablations`) | Ablation verification (exp11 + exp12) | `experiments/exp11_no_decay/run.py`, `experiments/exp12_no_asymmetry/run.py` | `exp11_no_decay/results/exp11_ablation_no_decay_*.json`, `exp12_no_asymmetry/results/exp12_ablation_no_asymmetry_*.json` |

## Headline numbers in the paper

| Claim | Value | Source | Where it appears in the paper |
|-------|-------|--------|------------------------------|
| NONE baseline incidents | 484.3 | `exp1_learning_scenarios_*` | abstract, §5.1, conclusion |
| GLOBAL baseline incidents | 265.6 | `exp1_learning_scenarios_*` | abstract, §5.1, conclusion |
| Headline reduction | 45.1% | derived from 484.3 → 265.6 | abstract, §5.1 |
| Cohen's d, NONE vs. GLOBAL | 11.51 | `analysis_cohens_d_*` | abstract, §5.1 |
| Cohen's d, smallest pairwise | 3.91 | `analysis_cohens_d_*` | abstract, §5.1, conclusion |
| Total simulation runs | ~12,000 | sum across exp01–exp12 + sensitivity | abstract, intro |
| Magic-number pilot runs | 1,440 | `pilot_*` (3 sweeps × 480 runs) | abstract, conclusion |
| H1×H3 NONE row | 482 across all β | `h1xh3_crosssweep_*` | §5.3, §6 Proposition 1 |
| H1×H3 GLOBAL β=0.5 reduction | 216 (482 → 266) | `h1xh3_crosssweep_*` | §5.3, §6 |
| H4 Star (worst) | 382.1 | `exp2_network_topology_*` | §5.4 |
| H4 Complete (best) | 273.1 | `exp2_network_topology_*` | §5.4 |
| H4 topology spread | 28.5% (Star → Complete) | derived | §5.4, §6 |
| H4 BA m=3 inversion | 331 (beats WS) | (separate sub-sweep, see code) | §5.4 |
| exp11 (no decay) LOCAL shift | −7.9 incidents | `exp11_ablation_no_decay_*` | §5.5 |
| exp12 (no asymmetry) LOCAL shift | +31.1 incidents (7.7%) | `exp12_ablation_no_asymmetry_*` | §5.5 |
| Team-count band | 34%–48% reduction across n = 6, 20, 50 | `exp9_robustness_team_count_*` | §5.5 |
| Sensitivity spread (acquisition) | 11.1% across 0.3–1.0 sweep | `sensitivity_acquisition_*` | §5.5 |
| Sensitivity spread (assimilation) | 1.7% | `sensitivity_assimilation_*` | §5.5 |
| Sensitivity spread (exploitation) | 0.7% | `sensitivity_exploitation_*` | §5.5 |
| Knowledge half-life | 1.9 years (693 days) | derived from δ=0.001 | §3.2 |

## Figure 1 — the four-stage pipeline diagram

The pipeline diagram in §3.3 of the paper is a TikZ illustration (`methods-pipeline-diagram.tex` in the paper repo, not this code repo). It is hand-drawn from the model specification, not auto-generated from code. The visual mirrors the four stages as they run inline in `run_simulation`'s Phase 2 (acquisition, then assimilation, transformation, and exploitation) in the root `model.py`.

## The Cohen's d analysis

The paper reports Cohen's d for all six pairwise H1 comparisons. These are computed in a post-processing pass over the exp01 output:

```bash
# Run exp01, then its analysis.py produces analysis_cohens_d_*.json
# with all six pairwise effect sizes (written to the same results/ folder)
python experiments/exp01_h1_sharing_scope/run.py
python experiments/exp01_h1_sharing_scope/analysis.py
```

The six pairs are: NONE–LOCAL, NONE–NEIGHBOR, NONE–GLOBAL, LOCAL–NEIGHBOR, LOCAL–GLOBAL, NEIGHBOR–GLOBAL. The smallest (NONE vs. LOCAL, d=3.91) is the one reported as the conservative bound in the abstract.

## Time-series trajectory of K̄ (paper §5.1)

The paper claims that under GLOBAL sharing, K̄ reaches ~0.96 by day 90 and plateaus at ~0.99 by day 120, while under NEIGHBOR it drifts to ~0.44 by day 90, ~0.70 by day 180, and ~0.89 at year-end. These come from per-day K̄ tracking, which `run_simulation` records by default:

```bash
# Every run records the daily K̄ trajectory in its output under
#   result["time_series"]["avg_prevention_knowledge"]   # one value per day
# exp01's analysis.py aggregates this across seeds into the per-scenario
# "avg_knowledge" series written to analysis_time_dynamics_*.json:
python experiments/exp01_h1_sharing_scope/analysis.py --analysis time_dynamics
```

See `analysis_time_dynamics_*.json` (per scenario: `avg_knowledge`, `avg_incidents_per_window`, `avg_transform`) for the full trajectories.

## What does NOT appear in this code

- The TikZ figure (paper repo only)
- The LaTeX paper itself (paper repo only)
- Citation handling (paper repo only)

## What appears here but not in the paper

- `verify_all_numbers.py` — a one-command checker (repo root) that confirms every committed result against the paper numbers; a reproducibility aid, not part of the paper text
- The `docs/` folder itself — the model walkthrough, this mapping, and the parameter reference are written for code readers, not reproduced in the paper
- The `--quick` smoke-test paths in each `run.py`, and the calibration/sensitivity sweeps beyond the specific cells the paper tabulates
