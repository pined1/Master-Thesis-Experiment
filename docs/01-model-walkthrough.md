# Model Walkthrough

This document explains the simulator step by step in plain English. For the formal mathematical specification, see the paper's §3 (Simulation Framework). For the code, see `model.py` at the repository root.

## What the simulator does

We model 20 engineering teams over a 365-day year. Each team owns a subsystem (database, payment gateway, cache, etc.). Incidents happen randomly. When a team has an incident, it writes a postmortem. Other teams may or may not learn from that postmortem depending on the sharing strategy and the network structure of the organization.

The simulator answers the question: *which organizational factor matters most for reducing total incident counts?*

## The four sharing scenarios

The organization chooses one of four sharing strategies at the start of the year and sticks with it:

| Scenario | What happens with each postmortem |
|----------|------------------------------------|
| **NONE** | Nobody learns. Even the team that triggered the incident does not update its knowledge. |
| **LOCAL** | Only the team that triggered the incident learns from it. |
| **NEIGHBOR** | The source team plus its immediate network neighbors enter the learning pipeline. |
| **GLOBAL** | All 20 teams enter the learning pipeline simultaneously. |

These four scenarios are the manipulated variable for H1. They are also the y-axis of the H1×H3 cross-sweep that produces the paper's headline mechanistic result.

## The five network topologies

Under NEIGHBOR sharing, which teams count as "neighbors" depends on the communication graph. We test five archetypes:

| Topology | Description | Used in H4 |
|----------|-------------|------------|
| **Complete** | Every team is connected to every other team. Models a fully flat organization. | ✓ |
| **Erdős–Rényi** | Random graph with edge probability 0.3. Models unstructured communication. | ✓ |
| **Watts–Strogatz** | Small-world (k=4, rewire=0.1). Models silos bridged by occasional cross-functional ties. | ✓ (also baseline for H1/H2/H3) |
| **Barabási–Albert** | Scale-free (m=2). Models hub-and-spoke with platform/infrastructure teams as hubs. | ✓ |
| **Star** | One central hub, all other teams as spokes. Models routing everything through a single VP of reliability. | ✓ |

## The day loop

Each simulated day, each team does two things in sequence:

### Phase 1: incident generation

The team rolls a Bernoulli trial against its current incident probability:

```
P(incident) = r_base · (1 - K̄ · β) · m_deploy
```

- `r_base = 0.05` is the baseline daily incident probability.
- `K̄` is the team's mean prevention competence (averaged across all five incident types).
- `β = 0.5` is the prevention coefficient — how much accumulated knowledge actually reduces failures.
- `m_deploy` is a multiplier: 1.5 for three days after a deployment, 1.0 otherwise.

If the trial fires, the team has an incident. The incident type is sampled from the Microsoft Azure ARTS taxonomy (database timeout, configuration error, dependency failure, capacity issue, deployment problem).

### Phase 2: pipeline processing

If an incident occurred, every eligible team (depending on the sharing scenario) enters the four-stage absorptive-capacity pipeline. See the next section.

### End of day: knowledge decay

Every team's knowledge matrix decays by a fixed amount:

```
K ← K · (1 - δ)
```

with `δ = 0.001` — a 1.9-year half-life. This reflects team turnover, role rotations, and skill atrophy.

## The four-stage learning pipeline

When a postmortem reaches a team that didn't trigger the incident, that team must clear four sequential probabilistic gates to actually learn. Stage 1 is terminal — if a team doesn't hear about an incident on the day it happens, it never can. Stages 2-4 are retried on subsequent days until they succeed or the year ends.

### Stage 1: Acquisition — does the report reach you?

```
P(acquisition) = α₁ · d(ℓ)
```

where `ℓ` is the shortest path length from the source team to this receiver in the communication graph. For direct neighbors, `d(ℓ)` is the edge weight (default 1.0). For multi-hop paths, `d(ℓ) = σ^ℓ` where `σ = 0.8` is a signal-decay penalty.

In plain terms: if you're far from the source team on the org chart, news of the incident may not reach you. `α₁ = 0.9` is the baseline acquisition rate — even direct neighbors miss about 10% of incidents.

### Stage 2: Assimilation — do you understand what happened?

```
P(assimilation) = (0.7 φ_cog + 0.3 q_doc) · α₂ · (0.5 + 0.5 ρ)
```

- `φ_cog` is the cosine similarity between your knowledge matrix and the source team's knowledge matrix (a 15-dimensional vector). If your teams work on similar tech, this is high.
- `q_doc = 0.5` is the documentation quality of the postmortem writeup.
- `α₂ = 0.7` is the baseline assimilation rate.
- `ρ` is the incident relevance factor (Azure ARTS subsystem-by-type susceptibility), floored at 0.3.

The 0.7/0.3 weighting comes from absorptive-capacity theory (Cohen & Levinthal 1990): shared technical context matters more than documentation polish.

### Stage 3: Transformation — can you translate it to your codebase?

```
P(transformation) = α₃ · φ_cog · (0.5 + 0.5 ρ)
```

Same cognitive-similarity logic as Stage 2, but now the gate models the question "can you adapt this lesson to your specific architecture?" `α₃ = 0.6` is the baseline.

### Stage 4: Exploitation — do you actually ship a fix?

```
P(exploitation) = α₄ · (0.5 + 0.5 ρ)
```

No cognitive gate here — you understand the lesson and have translated it. The remaining question is execution discipline. `α₄ = 0.8` baseline.

### Knowledge matrix update

When all four stages clear, the team's knowledge matrix increments at the specific incident-type row. The increment is sampled uniformly from `Uniform(0.1, 0.25)` and scaled by relevance and documentation quality. All cells are capped at 1.0.

The update only touches the row for the specific incident type the team just learned about. This prevents one postmortem from boosting knowledge across unrelated failure modes.

## Source-team asymmetry

The team that triggered the incident does not go through the four-stage pipeline. It updates its knowledge matrix directly. This captures hands-on experiential learning: the engineers who handled the live outage already have the runtime data and architectural context.

This asymmetry is the target of the exp12 ablation: forcing the source team through the pipeline causes a 7.7% incident increase under LOCAL sharing, confirming the asymmetry is load-bearing.

## What is fixed vs. what is configurable

The simulator separates two classes of model parameters:

**Architectural constants** (changing these changes the model, not just a number):
- The sequential four-stage pipeline structure
- Source-team asymmetry (bypass)
- Row-isolated matrix updates (a postmortem only updates the row for its incident type)
- Exponential knowledge decay

**Configurable parameters** (these get swept in experiments):
- Sharing scope, network topology, deployment velocity, β prevention coefficient
- The four α gate baselines, σ signal decay, q_doc documentation quality, δ decay rate
- See `docs/02-parameters.md` for the complete list with defaults and tested ranges

## How runs become results

Each "run" is one simulated year (365 days) with one random seed. For most experiments we run 100 seeds per cell and report the mean incident count. For the H3 high-resolution sweep we use 500 seeds per cell. The simulator outputs a JSON file per experiment containing the mean, std, and per-seed values for every metric.

The `verify_all_numbers.py` script reads these JSONs and confirms each paper claim matches its corresponding JSON value.

## Where to look in the code

All of the simulation lives in `model.py`. The four learning stages are not
separate methods — they run inline inside `run_simulation`'s daily loop.

| Concept | Where in `model.py` |
|---------|---------------------|
| Team agent state (5×3 knowledge grid) | `class Team` |
| Subsystem × incident-type susceptibility | `DEFAULT_SUSCEPTIBILITY` |
| All tunable parameters | `class SimulationParams` |
| Network topology generation | `init_graph()` |
| Who can learn, per sharing scope | `get_learners_for_scenario()` |
| Cognitive-similarity gate | `cosine_similarity()` |
| Incident relevance | `calculate_relevance()` |
| Knowledge-matrix update | `Team.learn()` |
| Daily incident probability + generation | `run_simulation()` — Phase 1 |
| Knowledge decay (applied each day) | `run_simulation()` — top of the daily loop |
| Source-team asymmetry shortcut | `run_simulation()` — end of Phase 1 |
| Four-stage pipeline (acquisition → exploitation) | `run_simulation()` — Phase 2 |
| Metrics / output assembly | `run_simulation()` — Phase 3 and after |
| Experiment orchestration | each `experiments/<folder>/run.py` |
