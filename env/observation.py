from dataclasses import dataclass


@dataclass
class NodeObservation:
    """
    Observation of a single node in the environment.
    """

    id: int
    x: float
    y: float
    demand: int
    visited: bool
    depot: bool


@dataclass
class EnvObservation:
    """
    Observation of the environment state, used as input for the agent.
    """

    nodes: list[
        NodeObservation
    ]  # (id, x, y, demand) for each node, including the depot!
    truck_pos: tuple[float, float]  # (x, y) position of the truck
    truck_load: int  # current load of the truck
    truck_capacity: int  # capacity of the truck
