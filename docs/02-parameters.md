# Simulator Parameters

Every parameter in the model, what it means in real-world terms, its default value, and the range we tested.

For the mathematical role of each parameter, see the paper's §3.7 (Configurable Platform Parameters). For its human/cooperative interpretation, see §3.8 (Human Interpretation of Parameters).

## Group A — Structural variables (manipulated in experiments)

| Parameter | Symbol | Real-world meaning | Default | Tested range |
|-----------|--------|--------------------|---------|--------------|
| Sharing scenario | — | Which teams receive postmortems | vary | NONE, LOCAL, NEIGHBOR, GLOBAL |
| Network topology | — | Shape of the communication graph between teams | vary | Complete, Erdős–Rényi, Watts–Strogatz, Barabási–Albert, Star |

## Group B — Incident dynamics

| Parameter | Symbol | Real-world meaning | Default | Tested range |
|-----------|--------|--------------------|---------|--------------|
| Base incident rate | `r_base` | Baseline daily incident probability per subsystem | 0.05 | 0.01–0.20 |
| Prevention coefficient | `β` | How much accumulated knowledge actually reduces failures | 0.5 | 0.00–0.50 |
| Deployment multiplier | `m_deploy` | Failure-probability bump during a deploy window | 1.0 / 1.5 | binary |
| Deployment velocity | — | Deployments per team per day | 0.10/day | 0.05–0.50/day |

## Group C — Pipeline gate baselines

| Parameter | Symbol | Real-world meaning | Default | Tested range |
|-----------|--------|--------------------|---------|--------------|
| Acquisition baseline | `α₁` | How readily a postmortem propagates via communication channels | 0.9 | 0.3–1.0 |
| Assimilation baseline | `α₂` | Whether another team has the technical vocabulary to interpret the writeup | 0.7 | 0.1–1.0 |
| Transformation baseline | `α₃` | Architectural and practice affinity between teams | 0.6 | fixed |
| Exploitation baseline | `α₄` | Organizational follow-through discipline | 0.8 | 0.1–1.0 |
| Signal-decay constant | `σ` | Channel fidelity across organizational distance | 0.8 | 0.3–1.0 |
| Documentation quality | `q_doc` | Honesty and detail of the postmortem writeup (proxy for psychological safety) | 0.5 | 0.1–0.9 |
| Per-success increment | `λ` | Knowledge gained per successful pipeline traversal | 0.1 | fixed |

## Group D — Knowledge decay

| Parameter | Symbol | Real-world meaning | Default | Tested range |
|-----------|--------|--------------------|---------|--------------|
| Daily decay coefficient | `δ` | How fast organizations forget lessons (team turnover, atrophy) | 0.001 | half-life: 2 weeks — 19 years |

## Group E — Engine execution

| Parameter | Symbol | Real-world meaning | Default | Tested range |
|-----------|--------|--------------------|---------|--------------|
| Total team count | `N` | Number of teams in the simulated organization | 20 | 6, 20, 50 |
| Simulation horizon | — | Length of the simulated year | 365 days | 180, 730, 1095 days |
| Seeds per cell | — | Number of independent stochastic runs averaged per configuration | 100 | 50–500 |

## How parameters appear in the code

In `model.py` (at the repository root), parameters live in the `SimulationParams` dataclass at the top of the file. To run with non-default values, construct a `SimulationParams` instance and pass it to `run_simulation()`. Every knob — including the scenario, seed, and run length — is a field on `SimulationParams`:

```python
from model import SimulationParams, LearningScenario, run_simulation

params = SimulationParams(
    learning_scenario=LearningScenario.NEIGHBOR,
    seed=42,
    steps=365,                    # run length in days
    prevention_effect=0.3,        # β = 0.3 instead of 0.5
    signal_decay=0.6,             # σ = 0.6 instead of 0.8
    disable_knowledge_decay=True, # exp11 ablation: turn off knowledge decay
)
result = run_simulation(params)
```

Note the code field names differ from the paper symbols: β is `prevention_effect`, σ is `signal_decay`, and δ is `knowledge_decay` (with `disable_knowledge_decay` as the on/off ablation flag).

For experiment-scale runs (multiple seeds, multiple parameter cells), use the `run.py` in the relevant `experiments/<folder>/` directory, which handles the seed loop and JSON output.

## Why each parameter has the default it does

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `r_base = 0.05` | 5% daily | Calibrated to produce ~365 × 0.05 × 20 = ~365 incidents/year baseline before learning |
| `β = 0.5` | 50% reduction at K̄=1.0 | Conservative; a fully-knowledgeable team still has 50% of baseline incidents |
| `α₁ = 0.9` | 90% | Direct neighbors miss about 10% of incidents on the day they happen |
| `α₂ = 0.7` | 70% | Cognitive gate; teams with high cognitive similarity assimilate most reports |
| `α₃ = 0.6` | 60% | Lower than α₂ because translation is harder than comprehension |
| `α₄ = 0.8` | 80% | Organizational discipline; most understood lessons get acted on |
| `σ = 0.8` | 80%/hop | Signal degrades 20% per network hop |
| `q_doc = 0.5` | medium-quality | Average postmortem quality across surveyed orgs |
| `δ = 0.001` | 1.9-year half-life | Calibrated against Darr, Argote & Epple (1995) franchise depreciation data |

The three calibration choices that are **not** anchored in prior literature are:

- The 0.7/0.3 weighting split between cognitive similarity and documentation quality in Stage 2
- The 0.5 floor for the relevance multiplier `ρ` (where the +0.5 mathematical modifier prevents zeroing out)
- The 3-day deployment risk window

These three are the targets of the magic-number pilot sweeps (`experiments/magic_number_pilots/run.py`) which confirm the H1 ordering holds across reasonable alternative values for each.
