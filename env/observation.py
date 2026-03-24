from dataclasses import dataclass


@dataclass
class EnvObservation:
    """
    Observation of the environment state, used as input for the agent.
    """

    truck_load: float  # [0, 1] - how full the truck is
    fleet_availability: float  # [0, 1] - how many trucks are available (not broken)
    orphan_pressure: float  # [0, 1] - how many nodes are unassigned (orphans) compared to total nodes
    nearest_orphan_dist: float  # [0, 1] - distance to the nearest unassigned node (normalized by max distance in the graph)
    nearest_orphan_rel_dist: float  # [0, 2] - distance to the nearest unassigned node relative to its distance from the other trucks
    route_efficiency: float  # [0, 2] - how efficient the current route for the truck is
