from enum import Enum
from typing import Iterable


class TruckStatus(Enum):
    ACTIVE = 0
    BROKEN = 1


class Truck:

    def __init__(self, id: int, pos: tuple[float, float]):
        """
        :param id: the id of the truck
        :param pos: the position of the truck, in [0,1]^2
        """
        assert 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1
        self.id = id
        self.pos = pos
        self.status = TruckStatus.ACTIVE
        self.route: list[int] = list()

    def add_to_route(self, node_id: int, idx: int | None = None):
        assert node_id not in self.route
        if idx is None:
            self.route.append(node_id)
        else:
            self.route.insert(idx, node_id)

    def remove_from_route(self, node_id: int):
        assert node_id in self.route
        self.route.remove(node_id)

    def breakdown(self):
        self.status = TruckStatus.BROKEN

    def recover(self):
        self.status = TruckStatus.ACTIVE

    @property
    def route_size(self):
        return len(self.route)

    def __iter__(self) -> Iterable[int]:
        return iter(self.route)
