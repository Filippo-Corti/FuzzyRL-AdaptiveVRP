from __future__ import annotations

import math
from importlib import import_module

import torch


def compute_exact_cost_with_ortools(
    depot_xy: torch.Tensor,
    customers_xy: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    time_limit_s: int = 1,
) -> float | None:
    """Solve a CVRP instance with OR-Tools and return total route length."""
    try:
        pywrapcp = import_module("ortools.constraint_solver.pywrapcp")
        routing_enums_pb2 = import_module("ortools.constraint_solver.routing_enums_pb2")
    except Exception:
        return None

    n_customers = int(customers_xy.shape[0])

    coords: list[tuple[float, float]] = [
        (float(depot_xy[0].item()), float(depot_xy[1].item()))
    ]
    coords.extend(
        (float(customers_xy[i, 0].item()), float(customers_xy[i, 1].item()))
        for i in range(n_customers)
    )

    node_demands = [0]
    node_demands.extend(int(demands[i].item()) for i in range(n_customers))

    dist_scale = 10_000
    n_points = len(coords)
    dist_matrix: list[list[int]] = [[0 for _ in range(n_points)] for _ in range(n_points)]
    for i in range(n_points):
        xi, yi = coords[i]
        for j in range(n_points):
            if i == j:
                continue
            xj, yj = coords[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            dist_matrix[i][j] = int(round(d * dist_scale))

    num_vehicles = n_customers
    manager = pywrapcp.RoutingIndexManager(n_points, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_cb_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return node_demands[from_node]

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index,
        0,
        [int(capacity)] * num_vehicles,
        True,
        "Capacity",
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(max(1, int(time_limit_s)))

    try:
        solution = routing.SolveWithParameters(search_params)
    except Exception:
        return None

    if solution is None:
        return None

    return float(solution.ObjectiveValue()) / float(dist_scale)


def compute_nearest_neighbor_cost(
    depot_xy: torch.Tensor,
    customers_xy: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
) -> float:
    """Compute CVRP route length with a simple nearest-neighbor heuristic."""

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    depot = (float(depot_xy[0].item()), float(depot_xy[1].item()))
    customers = [
        (float(customers_xy[i, 0].item()), float(customers_xy[i, 1].item()))
        for i in range(int(customers_xy.shape[0]))
    ]
    demand_values = [float(demands[i].item()) for i in range(int(demands.shape[0]))]

    unvisited = set(range(len(customers)))
    current = depot
    remaining_cap = float(capacity)
    total_distance = 0.0

    while unvisited:
        feasible = [idx for idx in unvisited if demand_values[idx] <= remaining_cap]
        if not feasible:
            total_distance += _dist(current, depot)
            current = depot
            remaining_cap = float(capacity)
            continue

        next_idx = min(feasible, key=lambda idx: _dist(current, customers[idx]))
        total_distance += _dist(current, customers[next_idx])
        current = customers[next_idx]
        remaining_cap -= demand_values[next_idx]
        unvisited.remove(next_idx)

    if current != depot:
        total_distance += _dist(current, depot)

    return total_distance
