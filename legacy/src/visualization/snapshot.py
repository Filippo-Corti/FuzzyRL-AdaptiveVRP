"""Snapshot dataclasses shared by visualization and UI rendering."""

from dataclasses import dataclass

PositionSnapshot = tuple[float, float]


@dataclass
class NodeSnapshot:
    id: int
    pos: PositionSnapshot
    demand: int
    visited: bool


@dataclass
class TruckSnapshot:
    id: int
    pos: PositionSnapshot
    heading_deg: float
    load: int
    capacity: int
    routes: list[list[PositionSnapshot]]


@dataclass
class DepotSnapshot:
    pos: PositionSnapshot


@dataclass
class SimulationStats:
    round: int
    orphans: int
    total_nodes: int
    total_distance: float
    exact_cost: float | None = None


@dataclass
class AgentSnapshot:
    last_choice: PositionSnapshot | None
    epsilon: float | None


@dataclass
class EnvironmentSnapshot:
    graph: list[NodeSnapshot]
    truck: TruckSnapshot
    depot: DepotSnapshot


@dataclass
class SimulationSnapshot:
    environment: EnvironmentSnapshot
    agent: AgentSnapshot
    stats: SimulationStats
