import random

from env import VRPGraph, VRPNode, Truck


def parse_vrp_instance(path: str) -> tuple[VRPGraph, int]:
    with open(path) as f:
        lines = f.readlines()

    coords: dict[int, tuple[int, int]] = {}
    demands: dict[int, int] = {}
    depot_id: int = 1
    capacity: int = 0

    section = None
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue

        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())

        if line in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"):
            section = line
            continue

        if section == "NODE_COORD_SECTION":
            node_id, x, y = map(int, line.split())
            coords[node_id] = (x, y)

        elif section == "DEMAND_SECTION":
            node_id, demand = map(int, line.split())
            demands[node_id] = demand

        elif section == "DEPOT_SECTION":
            val = int(line)
            if val != -1:
                depot_id = val

    # Normalise coordinates to [0, 1]^2
    all_x = [x for x, y in coords.values()]
    all_y = [y for x, y in coords.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    def normalise(x: int, y: int) -> tuple[float, float]:
        return (
            round((x - min_x) / (max_x - min_x), 2),
            round((y - min_y) / (max_y - min_y), 2),
        )

    depot_coord = coords[depot_id]
    depot = VRPNode(id=depot_id, pos=normalise(*depot_coord), demand=0, depot=True)
    graph = VRPGraph(depot=depot)

    for node_id, (x, y) in coords.items():
        if node_id == depot_id:
            continue
        graph.add(VRPNode(id=node_id, pos=normalise(x, y), demand=demands[node_id]))

    return graph, capacity

def generate_vrp_instance(num_nodes: int) -> tuple[VRPGraph, int]:
    depot = VRPNode(id=1, pos=generate_xy(), demand=0, depot=True)
    graph = VRPGraph(depot=depot)

    for node_id in range(2, num_nodes + 2):
        pos = generate_xy()
        demand = random.randint(1, 10)
        graph.add(VRPNode(id=node_id, pos=pos, demand=demand))

    capacity = random.randint(
        int(5.5 * 5), int(5.5 * 6)
    )  # On average, demand per node is 5.5. We want the capacity to be enough to serve 4-10 nodes on average.
    return graph, capacity

def generate_xy() -> tuple[float, float]:
    return round(random.uniform(0, 1), 2), round(random.uniform(0, 1), 2)

