from dataclasses import dataclass


@dataclass
class EnvObservation:
    """
    Observation of the environment state, used as input for the agent.
    """

    truck_load: float  # [0, 1]
    fleet_availability: float  # [0, 1]
    orphan_pressure: float  # [0, 1]
    nearest_orphan_dist: float  # [0, 1]
