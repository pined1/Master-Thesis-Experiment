# magic_number_pilots — Calibration Pilot Sweeps

Confirms the result does not hinge on three un-anchored constants.

## Why (motivation)

Three calibration choices in the model were not anchored in prior literature: the
Stage-2 weighting ratio, the relevance multiplier floor, and the length of the
deployment risk window. I ran a dedicated pilot sweep over each one to confirm the
core finding comes from the structure of the network pipeline, not from a lucky
pick of parameter values.

## How (manipulation)

Sweeps each of the three constants in turn and checks that the
NONE > LOCAL > NEIGHBOR > GLOBAL ordering holds across every combination tested.

## What is changing

| Variable | Values |
|----------|--------|
| Stage-2 weighting ratio (cognitive vs documentation) | swept |
| Relevance multiplier floor | swept |
| Deployment risk window length | swept |

Held at baseline: NEIGHBOR sharing, Watts–Strogatz (k=4), 20 teams, 365 days.

## How to run

```bash
python experiments/magic_number_pilots/run.py
python experiments/magic_number_pilots/run.py --quick
```

## Results

`pilot_assimilation_weights_*.json`, `pilot_relevance_floor_*.json`,
`pilot_deployment_window_*.json`. Each records the H1 ordering at every swept
value.
