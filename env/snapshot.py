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


@dataclass
class AgentSnapshot:
    memberships: dict[str, dict[str, float]]
    q_values: dict[str, float]
    chosen_action: str
    truck_id: int


@dataclass
class SimulationSnapshot:
    nodes: list[NodeSnapshot]
    trucks: list[TruckSnapshot]
    routes: list[list[PositionSnapshot]]
    depot: DepotSnapshot
    stats: SimulationStats
    agent_state: AgentSnapshot
