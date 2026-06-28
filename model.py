"""
Organizational Learning from Software Incidents: Agent-Based Simulation

This model simulates how organizations learn from software incidents through
knowledge sharing across teams. Teams are agents that own subsystems and learn
from incidents via the four-stage absorptive capacity model (Zahra & George, 2002):

1. Acquisition: Hearing about an incident
2. Assimilation: Understanding the root cause and adapting to context
3. Transformation: Recombining new knowledge with existing mental models
4. Exploitation: Implementing preventive or mitigating changes

Key theoretical foundations:
- Absorptive capacity: Cohen & Levinthal (1990), Zahra & George (2002)
- Cognitive distance / inverted-U: Nooteboom et al. (2007)
- Exploration vs exploitation trade-off: March (1991)
- Learning from incidents: Lunney & Lueder (2016), Drupsteen & Guldenmund (2014)
- Incident taxonomy: Dogga et al. (2023) ARTS classification

The simulation supports four learning scenarios:
- NONE: No learning (baseline control)
- LOCAL: Teams learn only from their own incidents
- NEIGHBOR: Learning propagates to adjacent teams in communication network
- GLOBAL: All teams can learn from any incident

Transformation modes:
- MINIMAL (default): Single probability check with cognitive factors for transformation
- TIME-BASED (optional): Cumulative effort over multiple timesteps required for
  transformation success. Enabled via use_time_based_transformation parameter.
  Progress rate depends on cognitive alignment, relevance, and documentation quality.

Exploitation is operationalized as changes to team competence that reduce:
- Incident frequency (prevention knowledge)
- Time to detect (detection knowledge)
- Severity and duration (mitigation knowledge)
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from collections import defaultdict

import numpy as np
import networkx as nx


# ==============================================================================
# ENUMS AND TYPE DEFINITIONS
# ==============================================================================

class LearningScenario(Enum):
    """
    Four learning scenarios from the proposal.

    - NONE: no learning (baseline control)
    - LOCAL: teams learn only from their own incidents
    - NEIGHBOR: learning propagates to network neighbors
    - GLOBAL: all teams learn from all incidents
    """
    NONE = "none"
    LOCAL = "local"
    NEIGHBOR = "neighbor"
    GLOBAL = "global"


class IncidentType(Enum):
    """
    Incident types based on ARTS taxonomy (Dogga et al., 2023).
    Different subsystems are susceptible to different incident types.
    """
    DATABASE_TIMEOUT = "database_timeout"
    CONFIG_ERROR = "config_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    CAPACITY_ISSUE = "capacity_issue"
    DEPLOYMENT_PROBLEM = "deployment_problem"


class SubsystemType(Enum):
    """
    Subsystem types that teams own. Following Conway's Law,
    team structure mirrors system structure.
    """
    DATABASE = "database"
    PAYMENT = "payment"
    AUTH = "auth"
    FRONTEND = "frontend"
    API = "api"
    CACHE = "cache"


# ==============================================================================
# SUSCEPTIBILITY MATRIX
# ==============================================================================
# Probability (0-1) that each subsystem is susceptible to each incident type.
# This is what enables relevance-based learning transfer between teams.

DEFAULT_SUSCEPTIBILITY = {
    SubsystemType.DATABASE: {
        IncidentType.DATABASE_TIMEOUT: 0.9,
        IncidentType.CONFIG_ERROR: 0.6,
        IncidentType.DEPENDENCY_FAILURE: 0.3,
        IncidentType.CAPACITY_ISSUE: 0.8,
        IncidentType.DEPLOYMENT_PROBLEM: 0.4,
    },
    SubsystemType.PAYMENT: {
        IncidentType.DATABASE_TIMEOUT: 0.4,
        IncidentType.CONFIG_ERROR: 0.7,
        IncidentType.DEPENDENCY_FAILURE: 0.8,
        IncidentType.CAPACITY_ISSUE: 0.5,
        IncidentType.DEPLOYMENT_PROBLEM: 0.6,
    },
    SubsystemType.AUTH: {
        IncidentType.DATABASE_TIMEOUT: 0.3,
        IncidentType.CONFIG_ERROR: 0.8,
        IncidentType.DEPENDENCY_FAILURE: 0.5,
        IncidentType.CAPACITY_ISSUE: 0.4,
        IncidentType.DEPLOYMENT_PROBLEM: 0.6,
    },
    SubsystemType.FRONTEND: {
        IncidentType.DATABASE_TIMEOUT: 0.2,
        IncidentType.CONFIG_ERROR: 0.7,
        IncidentType.DEPENDENCY_FAILURE: 0.6,
        IncidentType.CAPACITY_ISSUE: 0.5,
        IncidentType.DEPLOYMENT_PROBLEM: 0.8,
    },
    SubsystemType.API: {
        IncidentType.DATABASE_TIMEOUT: 0.5,
        IncidentType.CONFIG_ERROR: 0.7,
        IncidentType.DEPENDENCY_FAILURE: 0.7,
        IncidentType.CAPACITY_ISSUE: 0.6,
        IncidentType.DEPLOYMENT_PROBLEM: 0.7,
    },
    SubsystemType.CACHE: {
        IncidentType.DATABASE_TIMEOUT: 0.3,
        IncidentType.CONFIG_ERROR: 0.6,
        IncidentType.DEPENDENCY_FAILURE: 0.4,
        IncidentType.CAPACITY_ISSUE: 0.9,
        IncidentType.DEPLOYMENT_PROBLEM: 0.5,
    },
}


# ==============================================================================
# MODEL CONSTANTS
# ==============================================================================
# Fixed model coefficients. Stage-2 assimilation probability is a weighted blend
# of cognitive alignment and postmortem quality:
#   p_assim = ASSIMILATION_COGNITIVE_WEIGHT * cognitive_factor
#           + ASSIMILATION_DOC_WEIGHT       * doc_quality
# BASE_RELEVANCE is the floor relevance dissimilar subsystems still get from an
# incident (general lessons about monitoring, runbooks, etc.).

ASSIMILATION_COGNITIVE_WEIGHT = 0.7
ASSIMILATION_DOC_WEIGHT = 0.3

EXPLOITATION_BASE_PROB = 0.6

EDGE_WEIGHT_MIN = 0.5
EDGE_WEIGHT_MAX = 1.0

KNOWLEDGE_DIMENSIONS = ["prevention", "detection", "mitigation"]

BASE_RELEVANCE = 0.3


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Team:
    """
    A team agent that owns a subsystem and learns from incidents.

    knowledge maps {IncidentType: {dimension: float}}, each dimension a
    competence level in [0, 1]. The four *_incidents sets track which incidents
    a team has carried through each absorptive-capacity stage; for the optional
    time-based mode, transformation_progress maps incident_id -> accumulated
    progress in [0, 1].
    """
    team_id: int
    subsystem: SubsystemType

    knowledge: Dict[IncidentType, Dict[str, float]] = field(default_factory=dict)

    acquired_incidents: Set[int] = field(default_factory=set)
    assimilated_incidents: Set[int] = field(default_factory=set)
    transformed_incidents: Set[int] = field(default_factory=set)
    exploited_incidents: Set[int] = field(default_factory=set)

    transformation_progress: Dict[int, float] = field(default_factory=dict)

    incidents_experienced: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize empty per-incident-type knowledge dimensions to 0.0 if none were supplied."""
        if not self.knowledge:
            self.knowledge = {
                it: {dim: 0.0 for dim in KNOWLEDGE_DIMENSIONS}
                for it in IncidentType
            }

    def get_susceptibility(self, incident_type: IncidentType) -> float:
        """Get this team's subsystem susceptibility to an incident type."""
        return DEFAULT_SUSCEPTIBILITY[self.subsystem][incident_type]

    def get_knowledge_vector(self) -> np.ndarray:
        """Flatten knowledge into a vector for similarity calculations."""
        values = []
        for it in IncidentType:
            for dim in KNOWLEDGE_DIMENSIONS:
                values.append(self.knowledge[it][dim])
        return np.array(values)

    def learn(self, incident_type: IncidentType, dimension: str, amount: float):
        """Add knowledge, capped at 1.0."""
        current = self.knowledge[incident_type][dimension]
        self.knowledge[incident_type][dimension] = min(1.0, current + amount)


@dataclass
class Incident:
    """
    Record of a single incident.

    Units: severity is a 1-5 scale; duration is total hours from detection to
    resolution; detection_time is hours to detect; engineering_cost is
    developer-hours.
    """
    incident_id: int
    timestep: int
    source_team_id: int
    subsystem: SubsystemType
    incident_type: IncidentType
    severity: float
    duration: float
    detection_time: float
    engineering_cost: float
    learnable_knowledge: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationParams:
    """
    Parameters for organizational learning simulation.

    Supports four learning scenarios:
    - NONE: No learning (baseline)
    - LOCAL: Teams learn only from own incidents
    - NEIGHBOR: Learning propagates to network neighbors
    - GLOBAL: All teams learn from all incidents
    """
    # Random seed for reproducibility
    seed: int = 42

    # Organization structure
    num_teams: int = 6

    # Simulation duration (timesteps = business days)
    steps: int = 365

    # Learning scenario
    learning_scenario: LearningScenario = LearningScenario.NEIGHBOR

    # Network topology: "erdos_renyi", "complete", "watts_strogatz", "barabasi_albert", "star"
    network_topology: str = "watts_strogatz"

    # Network parameters
    # er_p: Erdős-Rényi edge probability
    # ws_k: Watts-Strogatz neighbors
    # ws_p: Watts-Strogatz rewiring probability
    # ba_m: Barabási-Albert edges per new node
    er_p: float = 0.3
    ws_k: int = 4
    ws_p: float = 0.1
    ba_m: int = 2

    # Incident generation
    # base_incident_rate: per subsystem per timestep
    # deployment_rate: deployments per timestep
    # deployment_risk_multiplier: risk increase after deployment
    base_incident_rate: float = 0.05
    deployment_rate: float = 0.1
    deployment_risk_multiplier: float = 1.5

    # Incident characteristics
    # severity is a 1-5 scale; duration is in hours
    incident_severity_base: float = 3.0
    incident_severity_std: float = 1.0
    incident_duration_base: float = 2.0
    incident_duration_std: float = 1.0

    # Learning probabilities (per stage)
    acquisition_probability: float = 0.9
    assimilation_probability: float = 0.7
    transformation_probability: float = 0.7
    exploitation_probability: float = 0.6

    # Cognitive factors
    # documentation_quality: quality of postmortems [0, 1]
    # use_inverted_u: when False, use linear similarity (Cohen & Levinthal 1990)
    #   so more-similar teams assimilate more easily
    # signal_decay: signal decay over network distance
    documentation_quality: float = 0.5
    use_inverted_u: bool = False
    signal_decay: float = 0.8

    # Exploitation effectiveness (how much knowledge reduces incident impact)
    # prevention_effect: per unit knowledge reduction in incident rate
    # detection_effect: per unit knowledge reduction in detection time
    # mitigation_effect: per unit knowledge reduction in severity/duration
    prevention_effect: float = 0.5
    detection_effect: float = 0.3
    mitigation_effect: float = 0.3

    # Time-based transformation mode (optional)
    # transformation_effort_rate: progress per timestep (base rate)
    use_time_based_transformation: bool = False
    transformation_effort_rate: float = 0.2

    # Engineering costs
    # engineering_cost_base: hours per incident
    # learning_cost: hours per learning event
    engineering_cost_base: float = 4.0
    learning_cost: float = 2.0

    # Knowledge decay
    # knowledge_decay: daily decay rate δ (half-life ≈ 2 years per Darr et al.)
    # disable_knowledge_decay: set True for ablation with no decay
    # disable_source_asymmetry: set True for ablation with equal probs for
    #   source and other teams
    knowledge_decay: float = 0.001
    disable_knowledge_decay: bool = False
    disable_source_asymmetry: bool = False

    # Output options
    log_per_team: bool = False


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between knowledge vectors.
    Uses dot product divided by the product of vector norms.

    When one or both vectors are zero (no prior knowledge), similarity is
    undefined.  We return 0.5 in that case so that the inverted-U absorptive
    capacity function evaluates to its peak (1.0), reflecting the empirical
    finding that blank-slate learners have maximum capacity to absorb new
    knowledge (Cohen & Levinthal 1990).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.5
    return float(np.dot(a, b) / (norm_a * norm_b))


def inverted_u_absorptive_capacity(
    similarity: float,
    peak_location: float = 0.5,
    steepness: float = 4.0
) -> float:
    """
    Inverted-U relationship (Nooteboom et al., 2007).

    Learning is maximized at intermediate cognitive distance:
    - Too similar: nothing new to learn
    - Too different: can't understand/integrate

    Returns a value in [0, 1] where the peak is at peak_location.
    """
    normalization = steepness * peak_location * (1 - peak_location)
    if normalization == 0:
        return 0.0
    return float(steepness * similarity * (1 - similarity) / normalization)


def calculate_relevance(
    source_subsystem: SubsystemType,
    target_subsystem: SubsystemType,
    incident_type: IncidentType
) -> float:
    """
    Calculate how relevant an incident is to a target team.

    Relevance is driven by the target team's susceptibility to the incident
    type: a team susceptible to this kind of failure has more to learn from it.
    Teams with dissimilar profiles still pick up a general lesson (BASE_RELEVANCE).
    """
    target_susceptibility = DEFAULT_SUSCEPTIBILITY[target_subsystem][incident_type]

    # If target is also susceptible to this incident type, high relevance
    if target_susceptibility > 0.5:
        return target_susceptibility

    # Otherwise, base relevance for general lessons
    return BASE_RELEVANCE


def init_graph(params: SimulationParams, rng: np.random.Generator) -> nx.Graph:
    """
    Initialize communication network based on specified topology.
    """
    n = params.num_teams
    seed = int(rng.integers(1e9))

    # Handle edge case of single team
    if n <= 1:
        g = nx.Graph()
        g.add_node(0)
        return g

    if params.network_topology == "complete":
        g = nx.complete_graph(n)

    elif params.network_topology == "watts_strogatz":
        k = min(params.ws_k, n - 1)
        if k < 2:
            k = min(2, n - 1)
        if k < 1 or n < 3:
            # Fall back to complete graph for very small n
            g = nx.complete_graph(n)
        else:
            g = nx.watts_strogatz_graph(n, k, params.ws_p, seed=seed)

    elif params.network_topology == "barabasi_albert":
        m = min(params.ba_m, n - 1)
        if m < 1:
            m = 1
        g = nx.barabasi_albert_graph(n, m, seed=seed)

    elif params.network_topology == "star":
        g = nx.star_graph(n - 1)

    # Default topology: erdos_renyi
    else:
        g = nx.erdos_renyi_graph(n, params.er_p, seed=seed)

    # Add edge weights (communication strength)
    for u, v in g.edges:
        g[u][v]["weight"] = float(rng.uniform(EDGE_WEIGHT_MIN, EDGE_WEIGHT_MAX))

    return g


def get_learners_for_scenario(
    scenario: LearningScenario,
    source_team_id: int,
    all_team_ids: List[int],
    graph: nx.Graph
) -> List[int]:
    """
    Determine which teams can learn from an incident based on learning scenario.
    """
    if scenario == LearningScenario.NONE:
        return []

    elif scenario == LearningScenario.LOCAL:
        return [source_team_id]

    elif scenario == LearningScenario.NEIGHBOR:
        # Source team plus direct neighbors
        neighbors = list(graph.neighbors(source_team_id))
        return [source_team_id] + neighbors

    elif scenario == LearningScenario.GLOBAL:
        return all_team_ids

    return []


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def run_simulation(params: SimulationParams) -> Dict[str, Any]:
    """
    Run the organizational learning simulation.

    Returns a dictionary with:
    - time_series: Incident frequency, duration, severity per timestep
    - summary: Total incidents, costs, availability metrics
    - final_knowledge: Knowledge state of each team
    """
    rng = np.random.default_rng(params.seed)

    # Set up the organization: communication network plus teams assigned to
    # subsystems round-robin through the available subsystem types.
    graph = init_graph(params, rng)

    subsystem_types = list(SubsystemType)
    teams = [
        Team(
            team_id=i,
            subsystem=subsystem_types[i % len(subsystem_types)]
        )
        for i in range(params.num_teams)
    ]

    # Trackers: full incident log, plus per-subsystem samples used to compute
    # MTBF (mean time between failures), MTTR (resolution), and MTTD (detection).
    all_incidents: List[Incident] = []
    incident_counter = 0

    time_since_incident = {st: 0 for st in SubsystemType}
    mtbf_samples: Dict[SubsystemType, List[float]] = {st: [] for st in SubsystemType}
    mttr_samples: List[float] = []
    mttd_samples: List[float] = []

    metrics = {
        "incident_frequency": {st.name: [] for st in SubsystemType},
        "incident_duration": [],
        "incident_severity": [],
        "engineering_cost": [],
        "learning_cost": [],
        "cumulative_engineering_cost": 0.0,
        "cumulative_learning_cost": 0.0,
        "total_incidents": 0,
        "total_incidents_by_type": {it.name: 0 for it in IncidentType},
        "total_incidents_by_subsystem": {st.name: 0 for st in SubsystemType},
        # Stage rates per timestep
        "acquisition_rate": [],
        "assimilation_rate": [],
        "transformation_rate": [],
        "exploitation_rate": [],
        # Knowledge levels
        "avg_prevention_knowledge": [],
        "avg_detection_knowledge": [],
        "avg_mitigation_knowledge": [],
        # Availability
        "mtbf": [],
        "mttr": [],
        "mttd": [],
    }

    # A recent deployment elevates a subsystem's incident risk for a few days.
    recent_deployments = {st: 0 for st in SubsystemType}

    # ----- Main simulation loop: one iteration per business day -----
    # Each day, in order: age out deployment risk, roll new deployments (risk
    # stays elevated for 3 days), then apply knowledge decay before any new
    # learning lands. Decay follows K_t = K_{t-1}(1 - δ) + ΔK_t (Darr 1995),
    # so decay is applied before this step's knowledge gains are added.
    for t in range(params.steps):
        timestep_incidents: List[Incident] = []
        timestep_durations: List[float] = []
        timestep_severities: List[float] = []
        timestep_costs: List[float] = []
        incidents_this_step = {st: 0 for st in SubsystemType}

        for st in SubsystemType:
            recent_deployments[st] = max(0, recent_deployments[st] - 1)

        for team in teams:
            if rng.random() < params.deployment_rate:
                recent_deployments[team.subsystem] = 3

        if not params.disable_knowledge_decay:
            for team in teams:
                for it in IncidentType:
                    team.knowledge[it]["prevention"] *= (1 - params.knowledge_decay)
                    team.knowledge[it]["detection"] *= (1 - params.knowledge_decay)
                    team.knowledge[it]["mitigation"] *= (1 - params.knowledge_decay)

        # ==============================================================
        # PHASE 1: Incident Generation
        # ==============================================================
        # Each team rolls for an incident with probability
        #   p_incident = base_incident_rate * prevention_modifier * deployment_modifier
        # where prevention knowledge lowers the rate and a recent deployment
        # raises it. When one fires, its type is drawn weighted by the team's
        # subsystem susceptibility, and its severity / detection / resolution
        # times are sampled and then attenuated by the team's mitigation and
        # detection knowledge. Duration splits ~40% detection / ~60% resolution.
        for team in teams:
            avg_prevention = np.mean([
                team.knowledge[it]["prevention"] for it in IncidentType
            ])
            prevention_modifier = 1.0 - (avg_prevention * params.prevention_effect)

            deployment_modifier = 1.0
            if recent_deployments[team.subsystem] > 0:
                deployment_modifier = params.deployment_risk_multiplier

            p_incident = params.base_incident_rate * prevention_modifier * deployment_modifier

            if rng.random() < p_incident:
                susceptibilities = [
                    DEFAULT_SUSCEPTIBILITY[team.subsystem][it]
                    for it in IncidentType
                ]
                total = sum(susceptibilities)
                probs = [s / total for s in susceptibilities]
                incident_type = rng.choice(list(IncidentType), p=probs)

                avg_mitigation = np.mean([
                    team.knowledge[it]["mitigation"] for it in IncidentType
                ])
                severity_modifier = 1.0 - (avg_mitigation * params.mitigation_effect)

                severity = np.clip(
                    rng.normal(params.incident_severity_base, params.incident_severity_std) * severity_modifier,
                    1.0, 5.0
                )

                avg_detection = np.mean([
                    team.knowledge[it]["detection"] for it in IncidentType
                ])
                detection_modifier = 1.0 - (avg_detection * params.detection_effect)

                base_detection_time = params.incident_duration_base * 0.4
                detection_time = max(0.1, rng.normal(
                    base_detection_time,
                    params.incident_duration_std * 0.4
                ) * detection_modifier)

                base_resolution_time = params.incident_duration_base * 0.6
                resolution_time = max(0.4, rng.normal(
                    base_resolution_time,
                    params.incident_duration_std * 0.6
                ) * severity_modifier)

                duration = detection_time + resolution_time

                cost = params.engineering_cost_base * (severity / 3.0) * (duration / 2.0)

                learnable_knowledge = {
                    "prevention": rng.uniform(0.1, 0.25),
                    "detection": rng.uniform(0.1, 0.25),
                    "mitigation": rng.uniform(0.1, 0.25),
                }

                incident = Incident(
                    incident_id=incident_counter,
                    timestep=t,
                    source_team_id=team.team_id,
                    subsystem=team.subsystem,
                    incident_type=incident_type,
                    severity=severity,
                    duration=duration,
                    detection_time=detection_time,
                    engineering_cost=cost,
                    learnable_knowledge=learnable_knowledge,
                )

                all_incidents.append(incident)
                timestep_incidents.append(incident)
                timestep_durations.append(duration)
                timestep_severities.append(severity)
                timestep_costs.append(cost)

                incident_counter += 1
                metrics["total_incidents"] += 1
                metrics["total_incidents_by_type"][incident_type.name] += 1
                metrics["total_incidents_by_subsystem"][team.subsystem.name] += 1
                incidents_this_step[team.subsystem] += 1

                # Record availability samples for this subsystem.
                if time_since_incident[team.subsystem] > 0:
                    mtbf_samples[team.subsystem].append(time_since_incident[team.subsystem])
                time_since_incident[team.subsystem] = 0
                mttr_samples.append(duration)
                mttd_samples.append(detection_time)

                # Source-team asymmetry: the team that experiences an incident
                # learns from it immediately, short-circuiting the 4-stage
                # pipeline. Under NONE no learning happens at all (true
                # baseline); under disable_source_asymmetry the source team is
                # run through the pipeline like every other team.
                team.incidents_experienced.append({
                    "incident_id": incident.incident_id,
                    "timestep": t,
                    "type": incident_type,
                })
                if (params.learning_scenario != LearningScenario.NONE
                        and not params.disable_source_asymmetry):
                    team.acquired_incidents.add(incident.incident_id)
                    team.assimilated_incidents.add(incident.incident_id)
                    team.transformed_incidents.add(incident.incident_id)
                    team.exploited_incidents.add(incident.incident_id)

                    for dim in KNOWLEDGE_DIMENSIONS:
                        team.learn(incident_type, dim, learnable_knowledge[dim])

        for st in SubsystemType:
            if incidents_this_step[st] == 0:
                time_since_incident[st] += 1

        # ==============================================================
        # PHASE 2: Four-Stage Learning (Zahra & George 2002)
        # 1. Acquisition - hearing about the incident
        # 2. Assimilation - understanding root cause and context
        # 3. Transformation - recombining new knowledge with existing
        # 4. Exploitation - implementing preventive/mitigating changes
        # ==============================================================
        # Stage 1: Acquisition — only runs for incidents that occurred this timestep.
        # Teams discover an incident when it is reported; they cannot retroactively
        # hear about an incident from a prior day.
        for incident in timestep_incidents:
            potential_learners = get_learners_for_scenario(
                params.learning_scenario,
                incident.source_team_id,
                list(range(params.num_teams)),
                graph
            )

            # Under GLOBAL every learner hears at the base probability; otherwise
            # the signal attenuates with network distance — a direct edge scales
            # by its weight, and farther teams decay by signal_decay ** path_length
            # (no path means no chance of hearing). The source team is skipped
            # unless source asymmetry is disabled.
            for team in teams:
                if team.team_id == incident.source_team_id and not params.disable_source_asymmetry:
                    continue

                if team.team_id not in potential_learners:
                    continue

                if incident.incident_id not in team.acquired_incidents:
                    if params.learning_scenario == LearningScenario.GLOBAL:
                        p_acquire = params.acquisition_probability
                    else:
                        if graph.has_edge(team.team_id, incident.source_team_id):
                            edge_weight = graph[team.team_id][incident.source_team_id]["weight"]
                            p_acquire = params.acquisition_probability * edge_weight
                        else:
                            try:
                                path_length = nx.shortest_path_length(
                                    graph, team.team_id, incident.source_team_id
                                )
                                p_acquire = params.acquisition_probability * (params.signal_decay ** path_length)
                            except nx.NetworkXNoPath:
                                p_acquire = 0.0

                    if rng.random() < p_acquire:
                        team.acquired_incidents.add(incident.incident_id)

        # Stages 2–4: run every timestep over ALL incidents a team has acquired.
        # Each stage retries daily until it succeeds, matching the theory that
        # assimilation, transformation, and exploitation unfold over time — not
        # in a single day.
        for incident in all_incidents:
            potential_learners = get_learners_for_scenario(
                params.learning_scenario,
                incident.source_team_id,
                list(range(params.num_teams)),
                graph
            )

            for team in teams:
                if team.team_id == incident.source_team_id and not params.disable_source_asymmetry:
                    continue

                if team.team_id not in potential_learners:
                    continue

                # Stage 2: Assimilation — understand the root cause. Likelihood
                # blends cognitive alignment with the source team (cosine
                # similarity, optionally passed through the inverted-U absorptive
                # capacity curve) and postmortem quality, scaled by relevance.
                if (incident.incident_id in team.acquired_incidents and
                    incident.incident_id not in team.assimilated_incidents):

                    source_team = teams[incident.source_team_id]
                    similarity = cosine_similarity(
                        team.get_knowledge_vector(),
                        source_team.get_knowledge_vector()
                    )

                    if params.use_inverted_u:
                        cognitive_factor = inverted_u_absorptive_capacity(similarity)
                    else:
                        cognitive_factor = similarity

                    # Relevance
                    relevance = calculate_relevance(
                        source_team.subsystem, team.subsystem, incident.incident_type
                    )

                    p_assimilate = (
                        ASSIMILATION_COGNITIVE_WEIGHT * cognitive_factor +
                        ASSIMILATION_DOC_WEIGHT * params.documentation_quality
                    ) * params.assimilation_probability * (0.5 + 0.5 * relevance)

                    if rng.random() < p_assimilate:
                        team.assimilated_incidents.add(incident.incident_id)

                # Stage 3: Transformation — recombine new knowledge with the
                # team's existing understanding. MINIMAL mode (default) is a
                # single probability check; the optional TIME-BASED mode instead
                # accumulates progress over multiple timesteps and succeeds once
                # it reaches 1.0. Both are driven by cognitive alignment,
                # relevance, and documentation quality.
                if (incident.incident_id in team.assimilated_incidents and
                    incident.incident_id not in team.transformed_incidents):

                    source_team = teams[incident.source_team_id]
                    similarity = cosine_similarity(
                        team.get_knowledge_vector(),
                        source_team.get_knowledge_vector()
                    )

                    if params.use_inverted_u:
                        cognitive_factor = inverted_u_absorptive_capacity(similarity)
                    else:
                        cognitive_factor = similarity

                    relevance = calculate_relevance(
                        source_team.subsystem, team.subsystem, incident.incident_type
                    )

                    if params.use_time_based_transformation:
                        if incident.incident_id not in team.transformation_progress:
                            team.transformation_progress[incident.incident_id] = 0.0

                        progress_rate = (
                            params.transformation_effort_rate *
                            (0.5 + 0.5 * cognitive_factor) *
                            (0.5 + 0.5 * relevance) *
                            (0.5 + 0.5 * params.documentation_quality)
                        )

                        team.transformation_progress[incident.incident_id] += progress_rate

                        if team.transformation_progress[incident.incident_id] >= 1.0:
                            team.transformed_incidents.add(incident.incident_id)
                    else:
                        p_transform = (
                            0.8 * cognitive_factor +
                            0.2 * params.documentation_quality
                        ) * params.transformation_probability * (0.5 + 0.5 * relevance)

                        if rng.random() < p_transform:
                            team.transformed_incidents.add(incident.incident_id)

                # Stage 4: Exploitation — implement the fix and bank the
                # knowledge gain. Relevance here is the team's RAW susceptibility
                # to this incident type, intentionally the un-floored value,
                # unlike Stages 2-3 which floor it via calculate_relevance().
                if (incident.incident_id in team.transformed_incidents and
                    incident.incident_id not in team.exploited_incidents):

                    relevance = team.get_susceptibility(incident.incident_type)
                    p_exploit = params.exploitation_probability * (0.5 + 0.5 * relevance)

                    if rng.random() < p_exploit:
                        team.exploited_incidents.add(incident.incident_id)

                        for dim in KNOWLEDGE_DIMENSIONS:
                            learning_amount = (
                                incident.learnable_knowledge[dim] *
                                (0.5 + 0.5 * relevance) *
                                (0.5 + 0.5 * params.documentation_quality)
                            )
                            team.learn(incident.incident_type, dim, learning_amount)

                        metrics["cumulative_learning_cost"] += params.learning_cost

        # ==============================================================
        # PHASE 3: Record Metrics
        # ==============================================================
        # Snapshot this timestep: per-subsystem incident counts, mean
        # duration/severity/cost, the cumulative learning funnel, average
        # knowledge levels, and availability samples (MTBF/MTTR/MTTD).
        for st in SubsystemType:
            metrics["incident_frequency"][st.name].append(incidents_this_step[st])

        if timestep_durations:
            metrics["incident_duration"].append(np.mean(timestep_durations))
            metrics["incident_severity"].append(np.mean(timestep_severities))
            metrics["engineering_cost"].append(sum(timestep_costs))
            metrics["cumulative_engineering_cost"] += sum(timestep_costs)
        else:
            metrics["incident_duration"].append(0.0)
            metrics["incident_severity"].append(0.0)
            metrics["engineering_cost"].append(0.0)

        # Learning stage rates — cumulative funnel across ALL incidents to date.
        # Denominator: every (team, incident) cross-team pair ever possible.
        # This reflects the true pipeline throughput, not just same-day completions.
        if all_incidents:
            total_possible = len(all_incidents) * (params.num_teams - 1)
            if total_possible > 0:
                acquired_count = sum(
                    1 for team in teams
                    for inc in all_incidents
                    if inc.incident_id in team.acquired_incidents and team.team_id != inc.source_team_id
                )
                assimilated_count = sum(
                    1 for team in teams
                    for inc in all_incidents
                    if inc.incident_id in team.assimilated_incidents and team.team_id != inc.source_team_id
                )
                transformed_count = sum(
                    1 for team in teams
                    for inc in all_incidents
                    if inc.incident_id in team.transformed_incidents and team.team_id != inc.source_team_id
                )
                exploited_count = sum(
                    1 for team in teams
                    for inc in all_incidents
                    if inc.incident_id in team.exploited_incidents and team.team_id != inc.source_team_id
                )
                metrics["acquisition_rate"].append(acquired_count / total_possible)
                metrics["assimilation_rate"].append(assimilated_count / total_possible)
                metrics["transformation_rate"].append(transformed_count / total_possible)
                metrics["exploitation_rate"].append(exploited_count / total_possible)
            else:
                metrics["acquisition_rate"].append(0.0)
                metrics["assimilation_rate"].append(0.0)
                metrics["transformation_rate"].append(0.0)
                metrics["exploitation_rate"].append(0.0)
        else:
            metrics["acquisition_rate"].append(0.0)
            metrics["assimilation_rate"].append(0.0)
            metrics["transformation_rate"].append(0.0)
            metrics["exploitation_rate"].append(0.0)

        prevention_knowledge = []
        detection_knowledge = []
        mitigation_knowledge = []
        for team in teams:
            for incident_type in IncidentType:
                prevention_knowledge.append(team.knowledge[incident_type]["prevention"])
                detection_knowledge.append(team.knowledge[incident_type]["detection"])
                mitigation_knowledge.append(team.knowledge[incident_type]["mitigation"])

        metrics["avg_prevention_knowledge"].append(np.mean(prevention_knowledge))
        metrics["avg_detection_knowledge"].append(np.mean(detection_knowledge))
        metrics["avg_mitigation_knowledge"].append(np.mean(mitigation_knowledge))

        current_mtbf = {}
        for st in SubsystemType:
            if mtbf_samples[st]:
                current_mtbf[st.name] = np.mean(mtbf_samples[st])
            # No failures yet for this subsystem: use the full window.
            else:
                current_mtbf[st.name] = params.steps
        metrics["mtbf"].append(current_mtbf)

        if mttr_samples:
            metrics["mttr"].append(np.mean(mttr_samples))
        else:
            metrics["mttr"].append(0.0)

        if mttd_samples:
            metrics["mttd"].append(np.mean(mttd_samples))
        else:
            metrics["mttd"].append(0.0)

    # ==============================================================
    # FINAL METRICS
    # ==============================================================
    # Per-subsystem availability A = MTBF / (MTBF + MTTR). MTBF is tracked in
    # days and MTTR in hours, so MTBF is converted to hours (×24) here for
    # consistent units before computing the ratio.
    final_availability = {}
    for st in SubsystemType:
        if mtbf_samples[st]:
            mtbf = np.mean(mtbf_samples[st]) * 24
        else:
            mtbf = params.steps * 24

        subsystem_incidents = [inc for inc in all_incidents if inc.subsystem == st]
        if subsystem_incidents:
            mttr = np.mean([inc.duration for inc in subsystem_incidents])
        else:
            mttr = 0.0

        if mtbf + mttr > 0:
            final_availability[st.name] = mtbf / (mtbf + mttr)
        else:
            final_availability[st.name] = 1.0

    overall_availability = np.mean(list(final_availability.values())) if final_availability else 1.0

    # Assemble the results payload: run config, time series, summary, and the
    # final knowledge state of every team.
    results = {
        "config": {
            "seed": params.seed,
            "num_teams": params.num_teams,
            "steps": params.steps,
            "learning_scenario": params.learning_scenario.name,
            "network_topology": params.network_topology,
        },
        "time_series": {
            "incident_frequency": metrics["incident_frequency"],
            "incident_duration": metrics["incident_duration"],
            "incident_severity": metrics["incident_severity"],
            "engineering_cost": metrics["engineering_cost"],
            "acquisition_rate": metrics["acquisition_rate"],
            "assimilation_rate": metrics["assimilation_rate"],
            "transformation_rate": metrics["transformation_rate"],
            "exploitation_rate": metrics["exploitation_rate"],
            "avg_prevention_knowledge": metrics["avg_prevention_knowledge"],
            "avg_detection_knowledge": metrics["avg_detection_knowledge"],
            "avg_mitigation_knowledge": metrics["avg_mitigation_knowledge"],
            "mtbf": metrics["mtbf"],
            "mttr": metrics["mttr"],
            "mttd": metrics["mttd"],
        },
        "summary": {
            "total_incidents": metrics["total_incidents"],
            "incidents_by_type": metrics["total_incidents_by_type"],
            "incidents_by_subsystem": metrics["total_incidents_by_subsystem"],
            "total_engineering_cost": metrics["cumulative_engineering_cost"],
            "total_learning_cost": metrics["cumulative_learning_cost"],
            "final_availability": final_availability,
            "overall_availability": overall_availability,
            "final_mttr": np.mean(mttr_samples) if mttr_samples else 0.0,
            "final_mttd": np.mean(mttd_samples) if mttd_samples else 0.0,
        },
        "final_knowledge": {},
    }

    for team in teams:
        results["final_knowledge"][team.team_id] = {
            "subsystem": team.subsystem.name,
            "knowledge": {
                it.name: team.knowledge[it]
                for it in IncidentType
            },
            "incidents_experienced": len(team.incidents_experienced),
            "incidents_acquired": len(team.acquired_incidents),
            "incidents_assimilated": len(team.assimilated_incidents),
            "incidents_transformed": len(team.transformed_incidents),
            "incidents_exploited": len(team.exploited_incidents),
        }

    # Per-team logging
    if params.log_per_team:
        results["team_trajectories"] = {
            "team_id": [],
            "subsystem": [],
            "cognitive_distances": [],
            "incidents_experienced": [],
            "final_prevention_knowledge": [],
            "final_detection_knowledge": [],
            "final_mitigation_knowledge": [],
        }

        for team in teams:
            results["team_trajectories"]["team_id"].append(team.team_id)
            results["team_trajectories"]["subsystem"].append(team.subsystem.name)
            results["team_trajectories"]["incidents_experienced"].append(
                len(team.incidents_experienced)
            )

            distances = []
            for other_team in teams:
                if other_team.team_id != team.team_id:
                    dist = cosine_similarity(
                        team.get_knowledge_vector(),
                        other_team.get_knowledge_vector()
                    )
                    distances.append(dist)
            results["team_trajectories"]["cognitive_distances"].append(
                np.mean(distances) if distances else 0.0
            )

            results["team_trajectories"]["final_prevention_knowledge"].append(
                np.mean([team.knowledge[it]["prevention"] for it in IncidentType])
            )
            results["team_trajectories"]["final_detection_knowledge"].append(
                np.mean([team.knowledge[it]["detection"] for it in IncidentType])
            )
            results["team_trajectories"]["final_mitigation_knowledge"].append(
                np.mean([team.knowledge[it]["mitigation"] for it in IncidentType])
            )

    # Network analysis
    results["network"] = {
        "num_edges": graph.number_of_edges(),
        "avg_degree": np.mean([d for n, d in graph.degree()]),
        "clustering_coefficient": nx.average_clustering(graph),
        "is_connected": nx.is_connected(graph),
    }
    if nx.is_connected(graph):
        results["network"]["avg_path_length"] = nx.average_shortest_path_length(graph)
    else:
        results["network"]["avg_path_length"] = float("inf")

    return results


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def compare_learning_scenarios(
    base_params: SimulationParams,
    scenarios: List[LearningScenario] = None,
    seeds: List[int] = None,
) -> Dict[str, Any]:
    """
    Compare different learning scenarios with the same base configuration.
    """
    if scenarios is None:
        scenarios = list(LearningScenario)

    if seeds is None:
        seeds = [base_params.seed]

    results = {}

    for scenario in scenarios:
        scenario_results = []

        for seed in seeds:
            # Copy every field from base_params, overriding only the seed and
            # scenario. Using dataclasses.replace keeps this in sync automatically
            # if new parameters are added to SimulationParams.
            run_params = replace(base_params, seed=seed, learning_scenario=scenario)

            result = run_simulation(run_params)
            scenario_results.append(result)

        results[scenario.name] = scenario_results

    return results


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    params = SimulationParams(
        seed=42,
        num_teams=6,
        steps=100,
        learning_scenario=LearningScenario.NEIGHBOR,
        network_topology="watts_strogatz",
        log_per_team=True,
    )

    results = run_simulation(params)

    print(f"Simulation completed: {params.steps} timesteps, {params.num_teams} teams")
    print(f"Learning scenario: {params.learning_scenario.name}")
    print(f"Network topology: {params.network_topology}")
    print(f"\nSummary:")
    print(f"  Total incidents: {results['summary']['total_incidents']}")
    print(f"  Total engineering cost: {results['summary']['total_engineering_cost']:.2f} hours")
    print(f"  Total learning cost: {results['summary']['total_learning_cost']:.2f} hours")
    print(f"  Overall availability: {results['summary']['overall_availability']:.4f}")
    print(f"\nIncidents by type:")
    for it, count in results['summary']['incidents_by_type'].items():
        if count > 0:
            print(f"  {it}: {count}")
    print(f"\nFinal availability by subsystem:")
    for st, avail in results['summary']['final_availability'].items():
        print(f"  {st}: {avail:.4f}")
