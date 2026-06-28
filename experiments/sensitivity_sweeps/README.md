# sensitivity_sweeps — Per-Stage Parameter Sensitivity

Tests whether the reliability ordering depends on specific pipeline-stage
parameter values.

## Why (motivation)

Multi-parameter sensitivity runs that check whether the ordering depends on
specific parameter values. Changing the Stage-2 (assimilation) or Stage-4
(exploitation) parameters barely moves the ordering — the spread stays at 1.7%
and 0.7% respectively. Only the Stage-1 (acquisition) parameter produces a
noticeable shift: sweeping it from 0.3 to 1.0 changes incidents by 11.1%. Whether
a team hears about an incident at all matters more than how it processes it once
it has it.

## How (manipulation)

Sweeps each pipeline-stage probability (and related parameters) one at a time
under NEIGHBOR sharing, holding everything else at baseline.

## What is changing

| Script | Sweeps (`--sweep`) |
|--------|--------------------|
| `sensitivity_sweep.py` | acquisition, assimilation, exploitation, signal_decay, initial_knowledge |
| `ablation_remaining.py` | detection_effect, mitigation_effect, deployment_risk, inverted_u |

Held at baseline: NEIGHBOR sharing, Watts–Strogatz (k=4), 20 teams, 365 days.

## How to run

```bash
python experiments/sensitivity_sweeps/sensitivity_sweep.py  --sweep acquisition
python experiments/sensitivity_sweeps/ablation_remaining.py --sweep detection_effect
# --sweep all runs every sweep in that script
```

## Results

One `sensitivity_<parameter>_*.json` per sweep, plus
`ablation_inverted_u_vs_linear_*.json`. Each records incident stats across the
swept range (mean / std / 95% CI / n).
