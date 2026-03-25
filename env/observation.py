from dataclasses import dataclass


@dataclass
class EnvObservation:
    """
    Observation of the environment state, used as input for the agent.
    """

    # Shared
    truck_load: float  # how full the truck is
    fleet_availability: float  #  how many trucks are available (not broken)

    # Breakdown agent only
    orphan_pressure: (
        float  #  how many nodes are unassigned (orphans) compared to total nodes
    )
    nearest_orphan_dist: float  #  distance to the nearest unassigned node (normalized by max distance in the graph)
    nearest_orphan_rel_dist: float  #  distance to the nearest unassigned node relative to its distance from the other trucks
    insertion_cost: float  # breakdown agent

    # Rebalancing agent only
    removal_gain: float  # distance saved by best removal, normalised
    route_imbalance: float  # std dev of load fractions across active trucks
    route_efficiency: float  # how efficient the current route for the truck is
