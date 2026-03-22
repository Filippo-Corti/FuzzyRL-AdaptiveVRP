from collections.abc import Iterator


class VRPNode:
    """
    A node in the graph for the Vehicle Routing Problem
    """

    def __init__(
        self,
        id: int,
        pos: tuple[float, float],
        assignment: int | None = None,
    ):
        """
        :param id: the id of the node
        :param pos: the position of the node, in [0,1]^2
        :param assignment: the truck id to which the node is assigned, or None
        """
        assert 0 <= pos[0] <= 1 and 0 <= pos[1] <= 1
        self.id = id
        self.pos = pos
        self.assignment = assignment

    @property
    def x(self) -> float:
        return self.pos[0]

    @property
    def y(self) -> float:
        return self.pos[1]

    @property
    def is_assigned(self) -> bool:
        return self.assignment is not None

    def distance_to(self, other: "VRPNode") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


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

    def add(self, node: VRPNode):
        assert node.id not in self.nodes and node.id != self.depot.id
        self.nodes[node.id] = node

    def get_node(self, id: int) -> VRPNode:
        return self.nodes[id]

    def __iter__(self) -> Iterator[VRPNode]:
        return iter(self.nodes.values())

    def unassigned_nodes(self) -> Iterator[VRPNode]:
        """
        Iterates over the unassigned nodes
        :return: the unassigned nodes
        """
        for _, node in self.nodes.items():
            if not node.is_assigned:
                yield node

    def assigned_nodes(self, truck_id: int | None = None) -> Iterator[VRPNode]:
        """
        Iterates over the nodes assigned to a truck, or to all trucks if truck_id is None
        :param truck_id: the id of the truck
        :return: the nodes assigned to that truck
        """
        for _, node in self.nodes.items():
            if (truck_id is not None and node.assignment == truck_id) or (
                truck_id is None and node.assignment is not None
            ):
                yield node
