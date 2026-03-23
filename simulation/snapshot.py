"""
This is the state of the simulation that is provided by the environment to the visualization package
"""

from dataclasses import dataclass
from enum import Enum

PositionSnapshot = tuple[float, float]


class NodeStatusSnapshot(Enum):
    UNVISITED = 0
    ASSIGNED = 1
    VISITED = 2


class TruckStatusSnapshot(Enum):
    ACTIVE = 0
    BROKEN = 1
    RECOVERING = 2


@dataclass
class NodeSnapshot:
    id: int
    pos: PositionSnapshot
    status: NodeStatusSnapshot


@dataclass
class TruckSnapshot:
    id: int
    pos: PositionSnapshot
    status: TruckStatusSnapshot
    rel_load: float


@dataclass
class DepotSnapshot:
    pos: PositionSnapshot


@dataclass
class SimulationStats:
    round: int
    orphans: int
    total_nodes: int
    total_trucks: int
    active_trucks: int
    total_distance: float
    episode_reward: float
    truck_turn: int
    last_action: str
    best_solution_distance: float
    last_distance: float


@dataclass
class AgentSnapshot:
    memberships: dict[str, dict[str, float]]
    q_values: dict[str, float]
    chosen_action: str
    q_table_size: int
    epsilon: float


@dataclass
class EnvironmentSnapshot:
    graph: list[NodeSnapshot]
    trucks: list[TruckSnapshot]
    depot: DepotSnapshot
    routes: list[list[PositionSnapshot]]


@dataclass
class SimulationSnapshot:
    environment: EnvironmentSnapshot
    agent: AgentSnapshot
    stats: SimulationStats
