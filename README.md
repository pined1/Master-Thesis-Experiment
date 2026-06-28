# Organizational Learning from Software Incidents: An Agent-Based Simulation

Code, data, and verification scripts for *Organizational Learning from
Software Incidents: An Agent-Based Simulation Study* (master's thesis, Brigham
Young University).

The simulator models 20 engineering teams learning from incidents over one year
(365 days) and tests four factors that could reduce incidents: **sharing scope
(H1), deployment velocity (H2), prevention strength (H3), network topology (H4)**.

## Quick start

```bash
pip install -r requirements.txt

python verify_all_numbers.py                          # check every paper number (~10s)
python experiments/exp01_h1_sharing_scope/run.py      # run the H1 experiment (~2 min)
python experiments/exp01_h1_sharing_scope/run.py --quick   # fast smoke run
```

`verify_all_numbers.py` prints each paper claim next to the value in the committed
results with a `✓`. To regenerate any experiment, run that folder's `run.py`.

## Layout

One engine (`model.py`) at the root; each experiment is a folder that imports it.

```
model.py                 the simulation engine — start here
verify_all_numbers.py    checks paper numbers against the committed results
experiments/
  expNN_name/
    README.md            what this experiment tests
    run.py               runs it, writes to results/
    results/*.json       its outputs
docs/                    optional deeper reading (model walkthrough, parameters, paper map)
figures/                 plot script + the paper's result figures
```

Every experiment folder is the same three things: a README, a `run.py`, and a
`results/`. The 11 experiments (the ones reported in the paper) are listed in
`experiments/README.md` and mapped to paper tables in `docs/03-paper-mapping.md`.

## The four hypotheses

| | Varies | Folder |
|---|---|---|
| **H1** | Sharing scope: NONE / LOCAL / NEIGHBOR / GLOBAL | `exp01_h1_sharing_scope` |
| **H2** | Deployment velocity | `exp04_h2_velocity` |
| **H3** | Prevention coefficient β | `exp05_h3_prevention` |
| **H4** | Network topology | `exp02_h4_topology` |

**Headline result:** across ~12,000 runs, incidents fall monotonically
`NONE > LOCAL > NEIGHBOR > GLOBAL`. In the H1×H3 cross-sweep the NONE row stays
flat at every β — sharing scope is the precondition; prevention strength only
amplifies what sharing has already exposed.

## Where to read more

- `model.py` — the engine (start here; the four learning stages run inline in `run_simulation`'s Phase 2)
- `docs/01-model-walkthrough.md` — the four-stage learning pipeline in plain English
- `docs/03-paper-mapping.md` — which paper table comes from which experiment
- `verify_all_numbers.py` — one command that checks every paper number against the committed results
- `figures/make_paper_figures.py` — regenerates the paper's three result figures from the committed results

## Dependencies

Python 3.10+ (tested on 3.13), NumPy, NetworkX, and SciPy (used by one analysis script). Pinned in `requirements.txt`.

## License & citation

MIT (see `LICENSE`).

```bibtex
@mastersthesis{pineda2026organizational,
  author = {Pineda, David},
  title  = {Organizational Learning from Software Incidents: An Agent-Based Simulation Study},
  school = {Brigham Young University},
  year   = {2026}
}
```
