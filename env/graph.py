from collections.abc import Iterator


class VRPNode:
    """
    A node in the graph for the Vehicle Routing Problem
    """

    def __init__(
        self,
        id: int,
        pos: tuple[float, float],
        demand: int,
        depot: bool = False,
    ):
        """
        :param id: the id of the node
        :param pos: the position of the node, in [0,1]^2
        """
        assert 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1
        self.id = id
        self.pos = pos
        self.demand = demand
        self.visited = False
        self.depot = depot

    @property
    def x(self) -> float:
        return self.pos[0]

    @property
    def y(self) -> float:
        return self.pos[1]

    def distance_to(self, other: "VRPNode") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    @staticmethod
    def distance(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


class VRPGraph:
    """
    The **fully-connected** graph of the Vehicle Routing Problem
    """

    def __init__(self, depot: VRPNode):
        """
        :param depot: the depot node
        """
        self.nodes: dict[int, VRPNode] = {}
        self.depot = depot
        assert self.depot.depot

    def add(self, node: VRPNode):
        assert node.id not in self.nodes and node.id != self.depot.id
        self.nodes[node.id] = node

    def get_node(self, id: int) -> VRPNode:
        return self.nodes[id]

    def __iter__(self) -> Iterator[VRPNode]:
        return iter(self.nodes.values())

    @property
    def is_fully_visited(self) -> bool:
        return all(node.visited for node in self.nodes.values())

    @property
    def orphans_count(self) -> int:
        return sum(1 for node in self.nodes.values() if not node.visited)

    def unvisited_nodes(self) -> Iterator[VRPNode]:
        """
        Iterates over the unvisited nodes
        :return: the unvisited nodes
        """
        for _, node in self.nodes.items():
            if not node.visited:
                yield node

    def reset(self):
        for node in self.nodes.values():
            node.visited = False
