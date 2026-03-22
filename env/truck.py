from enum import Enum
from typing import Iterable


class TruckStatus(Enum):
    ACTIVE = 0
    BROKEN = 1


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
        self.route: list[int] = list()
        self.capacity = capacity

    def add_by_index(self, node_id: int, idx: int | None = None):
        """
        Adds node_id in position idx to the truck route.
        If idx is None, it appends at the end
        """
        assert node_id not in self.route
        assert not self.is_full
        if idx is None:
            self.route.append(node_id)
        else:
            self.route.insert(idx, node_id)

    def remove_by_index(self, idx: int):
        """
        Removes node in position idx from the truck route
        """
        self.route.pop(idx)

    def remove_by_id(self, node_id: int):
        """
        Removes node_id from the truck route
        """
        assert node_id in self.route
        self.route.remove(node_id)

    def add_after(self, node_id: int, prev_id: int):
        """
        Adds node_id after prev_id in the truck route
        """
        assert (
            node_id not in self.route
        ), f"Cannot add {node_id} to {self.route} because it's already there"
        assert prev_id in self.route
        assert not self.is_full
        self.route.insert(self.route.index(prev_id) + 1, node_id)

    def breakdown(self):
        self.status = TruckStatus.BROKEN

    def recover(self):
        self.status = TruckStatus.ACTIVE

    @property
    def route_size(self) -> int:
        return len(self.route)

    @property
    def load(self) -> float:
        return self.route_size / self.capacity

    @property
    def is_full(self) -> bool:
        return self.route_size >= self.capacity

    def __iter__(self) -> Iterable[int]:
        return iter(self.route)
