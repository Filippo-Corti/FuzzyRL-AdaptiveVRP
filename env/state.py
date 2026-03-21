from dataclasses import dataclass
from enum import Enum

PositionState = tuple[float, float]


class NodeStatusState(Enum):
    UNVISITED = 0
    ASSIGNED = 1
    VISITED = 2


class TruckStatusState(Enum):
    ACTIVE = 0
    BROKEN = 1
    RECOVERING = 2


@dataclass
class NodeState:
    id: int
    pos: PositionState
    status: NodeStatusState


@dataclass
class EdgeState:
    id: int
    pos1: PositionState
    pos2: PositionState


@dataclass
class TruckState:
    id: int
    pos: PositionState
    status: TruckStatusState
    rel_load: float


@dataclass
class DepotState:
    pos: PositionState


@dataclass
class SimulationStats:
    round: int
    orphans: int
    total_nodes: int
    total_trucks: int
    active_trucks: int
    total_distance: float
    episode_reward: float


@dataclass
class AgentState:
    memberships: dict[str, dict[str, float]]
    q_values: dict[str, float]
    chosen_action: str
    truck_id: int


@dataclass
class SimulationState:
    nodes: list[NodeState]
    edges: list[EdgeState]
    trucks: list[TruckState]
    routes: list[list[PositionState]]
    depot: DepotState
    stats: SimulationStats
    agent_state: AgentState
