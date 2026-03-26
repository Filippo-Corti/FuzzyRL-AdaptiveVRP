from enum import Enum
from typing import Iterable

from .graph import VRPNode


class TruckStatus(Enum):
    ACTIVE = 0
    BROKEN = 1


class TruckRoute:

    def __init__(self):
        self.route: list[VRPNode] = list()
        self.load: int = 0
        self.is_closed: bool = False

    def add(self, node: VRPNode):
        self.route.append(node)
        self.load += node.demand

    def close(self):
        self.is_closed = True

    def __iter__(self) -> Iterable[VRPNode]:
        return iter(self.route)


class Truck:

    def __init__(self, id: int, pos: tuple[float, float], capacity: int):
        """
        :param id: the id of the truck
        :param pos: the position of the truck, in [0,1]^2
        :param capacity: the capacity of the truck
        """
        assert 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1
        self.id = id
        self.pos = pos
        self.status = TruckStatus.ACTIVE
        self.routes: list[TruckRoute] = list()  # list of (route, load) pairs
        self.capacity = capacity

    def add(self, node: VRPNode):
        route = self.current_route
        if route.load + node.demand > self.capacity:
            raise RuntimeError("Truck capacity exceeds capacity")
        route.add(node)

    def back_to_depot(self):
        if self.routes:
            self.routes[-1].close()
        self.routes.append(TruckRoute())

    def breakdown(self):
        self.status = TruckStatus.BROKEN

    def recover(self):
        self.status = TruckStatus.ACTIVE

    def is_active(self) -> bool:
        return self.status == TruckStatus.ACTIVE

    @property
    def current_load(self) -> int:
        return self.routes[-1].load if self.routes else 0

    @property
    def current_route(self) -> TruckRoute:
        if not self.routes:
            self.routes.append(TruckRoute())
        return self.routes[-1]

    @property
    def is_full(self) -> bool:
        return self.current_load >= self.capacity

    def __iter__(self) -> Iterable[TruckRoute]:
        return iter(self.routes)

    def reset(self):
        self.pos = (0.0, 0.0)
        self.status = TruckStatus.ACTIVE
        self.routes.clear()
