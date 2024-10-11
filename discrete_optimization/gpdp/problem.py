#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

# Generic pickup and delivery problem
# The General Pickup and Delivery Problem
# February 1995 Transportation Science 29(1):17-29
# https://www.researchgate.net/publication/239063487_The_General_Pickup_and_Delivery_Problem
import logging
from collections.abc import Hashable, KeysView
from typing import Any, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.tsp.problem import Point2DTspProblem
from discrete_optimization.vrp.problem import Customer2DVrpProblem

logger = logging.getLogger(__name__)


Node = Hashable
Edge = tuple[Node, Node]


class GpdpSolution(Solution):
    def __init__(
        self,
        problem: "GpdpProblem",
        trajectories: dict[int, list[Node]],
        times: dict[Node, float],
        resource_evolution: dict[Node, dict[Node, list[int]]],
    ):
        self.problem = problem
        self.trajectories = trajectories
        self.times = times
        self.resource_evolution = resource_evolution

    def copy(self) -> "GpdpSolution":
        return GpdpSolution(
            problem=self.problem,
            trajectories=self.trajectories,
            times=self.times,
            resource_evolution=self.resource_evolution,
        )

    def check_pickup_deliverable(self) -> bool:
        ok = True
        for pickup, deliverable in self.problem.list_pickup_deliverable:
            index_vehicles_pickup = set(
                [
                    [v for v, traj in self.trajectories.items() if p in traj][0]
                    for p in pickup
                ]
            )
            index_vehicles_deliverable = set(
                [
                    [v for v, traj in self.trajectories.items() if d in traj][0]
                    for d in deliverable
                ]
            )
            ok = ok and (
                len(index_vehicles_pickup) == 1 and len(index_vehicles_deliverable) == 1
            )
            vehicle_p = list(index_vehicles_pickup)[0]
            vehicle_d = list(index_vehicles_deliverable)[0]
            ok = ok and vehicle_p == vehicle_d
            index_p = [self.trajectories[vehicle_p].index(p) for p in pickup]
            index_d = [self.trajectories[vehicle_d].index(d) for d in deliverable]
            ok = ok and max(index_p) < min(index_d)
            return ok

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, GpdpProblem):
            raise ValueError("new_problem must be a GpdpProblem for GpdpSolution.")
        self.problem = new_problem


class GpdpProblem(Problem):
    MAX_VALUE = 10e9

    def __init__(
        self,
        number_vehicle: int,
        nodes_transportation: set[Node],
        nodes_origin: set[Node],
        nodes_target: set[Node],
        list_pickup_deliverable: list[tuple[list[Node], list[Node]]],
        origin_vehicle: dict[int, Node],
        target_vehicle: dict[int, Node],
        resources_set: set[str],
        capacities: dict[int, dict[str, tuple[float, float]]],
        resources_flow_node: dict[Node, dict[str, float]],
        resources_flow_edges: dict[Edge, dict[str, float]],
        distance_delta: dict[Node, dict[Node, float]],
        time_delta: dict[Node, dict[Node, float]],
        time_delta_node: Optional[dict[Node, float]] = None,
        coordinates_2d: Optional[dict[Node, tuple[float, float]]] = None,
        clusters_dict: Optional[dict[Node, Hashable]] = None,
        # For each node, returns an ID of a cluster associated. # This can modelize selective TSP
        list_pickup_deliverable_per_cluster: Optional[
            list[tuple[list[Node], list[Node]]]
        ] = None,
        mandatory_node_info: Optional[dict[Node, bool]] = None,
        # indicated for each node if the node is optional (=False) or mandatory (=True)
        cumulative_constraints: Optional[list[tuple[set[Hashable], int]]] = None,
        time_windows_nodes: Optional[
            dict[Node, tuple[Optional[int], Optional[int]]]
        ] = None,
        time_windows_cluster: Optional[
            dict[Hashable, tuple[Optional[int], Optional[int]]]
        ] = None,
        group_identical_vehicles: Optional[dict[int, list[int]]] = None,
        slack_time_bound_per_node: Optional[dict[Node, tuple[int, float]]] = None,
        node_vehicle: Optional[dict[Node, list[int]]] = None,
        compute_graph: bool = False,
    ):
        """
        number_vehicle = M in the paper
        nodes_transportation = V in the paper
        nodes_origin = M+ in the paper
        nodes_target = M- in the paper
        list_pickup_deliverable : List of Ni- and Ni+
        depart_vehicle : {vehicle: k-} in the paper
        resources_set : additional thing from the paper (we consider different ressource)
        capacities : give the capacities of vehicles and for different ressources.
        resources_flow_node: {node: {resource: q_node,resource}} (positive for pickup, negative for delivery)
        resources_flow_edges: {edge: {resource: q_edge,resource}} (optional, can modelize fuel burn during a travel)
        :param number_vehicle:
        :param nodes_transportation:
        """
        self.number_vehicle = number_vehicle
        self.nodes_transportation = nodes_transportation
        self.nodes_origin = nodes_origin
        self.nodes_target = nodes_target
        self.coordinates_2d = coordinates_2d  # can be none if it doesn't make sense.
        self.node_vehicle = node_vehicle
        self.list_pickup_deliverable = list_pickup_deliverable
        self.list_pickup_deliverable_per_cluster = list_pickup_deliverable_per_cluster
        self.origin_vehicle = origin_vehicle
        self.target_vehicle = target_vehicle
        self.resources_set = resources_set
        self.capacities = capacities
        self.resources_flow_node = resources_flow_node
        self.resources_flow_edges = resources_flow_edges
        self.distance_delta = distance_delta
        self.time_delta = time_delta

        self.all_nodes: set[Node] = set()
        self.all_nodes.update(self.nodes_origin)
        self.all_nodes.update(self.nodes_target)
        self.all_nodes.update(self.nodes_transportation)
        self.all_nodes.update([self.origin_vehicle[v] for v in self.origin_vehicle])
        self.all_nodes.update([self.target_vehicle[v] for v in self.target_vehicle])
        self.all_nodes_dict: dict[Node, dict[str, Any]] = {}
        if time_delta_node is None:
            self.time_delta_node = {n: 0.0 for n in self.all_nodes}
        else:
            self.time_delta_node = time_delta_node

        for v in self.origin_vehicle:
            self.all_nodes_dict[self.origin_vehicle[v]] = {
                "type": "origin",
                "vehicle": v,
                "time_delta_node": self.time_delta_node[self.origin_vehicle[v]],
            }
        for v in self.target_vehicle:
            self.all_nodes_dict[self.target_vehicle[v]] = {
                "type": "target",
                "vehicle": v,
                "time_delta_node": self.time_delta_node[self.target_vehicle[v]],
            }
        for n in self.nodes_transportation:
            self.all_nodes_dict[n] = {
                "type": "customer",
                "time_delta_node": self.time_delta_node[n],
            }
        for node in self.resources_flow_node:
            for res in self.resources_flow_node[node]:
                self.all_nodes_dict[node][res] = self.resources_flow_node[node][res]

        self.edges_dict: dict[Node, dict[Node, dict[str, float]]] = {}
        self.update_edges()
        self.graph: Optional[Graph] = None
        if compute_graph:
            logger.info("compute graph")
            self.compute_graph()
            logger.info("done")
        self.clusters_dict: dict[Node, Hashable]
        if clusters_dict is None:
            i = 0
            self.clusters_dict = {}
            for k in self.all_nodes_dict:
                self.clusters_dict[k] = i
                i += 1
        else:
            self.clusters_dict = clusters_dict
        self.clusters_set = set(self.clusters_dict.values())
        self.clusters_to_node: dict[Hashable, set[Node]] = {
            k: set() for k in self.clusters_set
        }
        for k in self.clusters_dict:
            self.clusters_to_node[self.clusters_dict[k]].add(k)
        if mandatory_node_info is None:
            if len(self.clusters_to_node) == len(self.all_nodes_dict):
                self.mandatory_node_info = {n: True for n in self.all_nodes_dict}
            else:
                self.mandatory_node_info = {n: False for n in self.all_nodes_dict}
        else:
            self.mandatory_node_info = mandatory_node_info
        self.cumulative_constraints: list[tuple[set[Hashable], int]]
        if cumulative_constraints is None:
            self.cumulative_constraints = []
        else:
            self.cumulative_constraints = cumulative_constraints
        self.time_windows_nodes: dict[Node, tuple[Optional[int], Optional[int]]]
        if time_windows_nodes is None:
            self.time_windows_nodes = {n: (None, None) for n in self.all_nodes_dict}
        else:
            self.time_windows_nodes = time_windows_nodes
        self.time_windows_cluster: dict[Hashable, tuple[Optional[int], Optional[int]]]
        if time_windows_cluster is None:
            self.time_windows_cluster = {k: (None, None) for k in self.clusters_set}
        else:
            self.time_windows_cluster = time_windows_cluster
        self.list_nodes = list(self.all_nodes)
        self.index_nodes: dict[Node, int] = {
            self.list_nodes[i]: i for i in range(len(self.list_nodes))
        }
        self.nodes_to_index = {
            i: self.list_nodes[i] for i in range(len(self.list_nodes))
        }

        def get_edges_for_vehicle(vehicle: int) -> KeysView[Edge]:
            if self.graph is None:
                self.compute_graph()
                if self.graph is None:
                    raise RuntimeError(
                        "self.graph must not be None after self.compute_graph()."
                    )
            if self.node_vehicle is not None:
                return [
                    e
                    for e in self.graph.get_edges()
                    if (
                        vehicle in self.node_vehicle.get(e[0], set())
                        and vehicle in self.node_vehicle.get(e[1], set())
                    )
                    or (e[0] == self.origin_vehicle[vehicle])
                    or (e[1] == self.target_vehicle[vehicle])
                ]
            else:
                return self.graph.get_edges()

        self.get_edges_for_vehicle = get_edges_for_vehicle
        if group_identical_vehicles is None:
            self.group_identical_vehicles = {i: [i] for i in range(self.number_vehicle)}
        else:
            self.group_identical_vehicles = group_identical_vehicles
        self.vehicle_to_group = {}
        for k in self.group_identical_vehicles:
            for v in self.group_identical_vehicles[k]:
                self.vehicle_to_group[v] = k
        self.any_grouping = any(
            len(self.group_identical_vehicles[k]) > 1
            for k in self.group_identical_vehicles
        )
        self.vehicles_representative = [
            self.group_identical_vehicles[k][0] for k in self.group_identical_vehicles
        ]
        if slack_time_bound_per_node is None:
            self.slack_time_bound_per_node = {
                node: (0, self.time_delta_node[node]) for node in self.time_delta_node
            }
        else:
            self.slack_time_bound_per_node = slack_time_bound_per_node

    def evaluate(self, variable: GpdpSolution) -> dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        res_distance: dict[str, float] = {}
        res_time: dict[str, float] = {}
        for v, path in variable.trajectories.items():
            res_distance[f"distance_{v}"] = self.compute_distance(path)
            if path[-1] in variable.times:
                res_time[f"time_{v}"] = variable.times[path[-1]]
            else:
                res_time[f"time_{v}"] = 0.0
        res_distance["distance_max"] = max(res_distance.values())
        res_time["time_max"] = max(res_time.values())
        res: dict[str, float] = {}
        res.update(res_distance)
        res.update(res_time)
        return res

    def compute_distance(self, path: list[Node]) -> float:
        distance = 0.0
        for i in range(len(path) - 1):
            distance += self.distance_delta[path[i]].get(
                path[i + 1], GpdpProblem.MAX_VALUE
            )
        return distance

    def evaluate_function_node(self, node_1: Node, node_2: Node) -> float:
        if self.graph is None:
            self.compute_graph()
            if self.graph is None:
                raise RuntimeError(
                    "self.graph must not be None after self.compute_graph()."
                )
        return self.graph.edges_infos_dict[(node_1, node_2)]["distance"]

    def satisfy(self, variable: GpdpSolution) -> bool:  # type: ignore  # avoid isinstance checks for efficiency
        return True

    def update_edges(self) -> None:
        for node in self.distance_delta:
            self.edges_dict[node] = {}
            for node1 in self.distance_delta[node]:
                self.edges_dict[node][node1] = {
                    "distance": self.distance_delta[node][node1],
                    "time": self.time_delta[node][node1],
                }
                resources = self.resources_flow_edges.get((node, node1), {})
                for r in resources:
                    self.edges_dict[node][node1][r] = resources[r]

    def update_graph(self) -> None:
        self.update_edges()
        self.compute_graph()

    def compute_graph(self) -> None:
        self.graph = Graph(
            nodes=[(n, self.all_nodes_dict[n]) for n in self.all_nodes_dict],
            edges=[
                (node1, node2, self.edges_dict[node1][node2])
                for node1 in self.edges_dict
                for node2 in self.edges_dict[node1]
            ],
            undirected=False,
            compute_predecessors=False,
        )

    def get_solution_type(self) -> type[Solution]:
        return GpdpSolution

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(dict_attribute_to_type={})

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc={
                "distance_max": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=-1.0
                )
            },
        )

    def __str__(self) -> str:
        s = "Routing problem : \n"
        s += str(len(self.all_nodes)) + " nodes \n"
        s += str(self.number_vehicle) + " vehicles \n"
        s += str(len(self.clusters_to_node)) + " clusters of node\n"

        return s


def build_pruned_problem(
    problem: GpdpProblem, undirected: bool = True, compute_graph: bool = False
) -> GpdpProblem:
    if problem.graph is None:
        problem.compute_graph()
        if problem.graph is None:
            raise RuntimeError(
                "self.graph must not be None after problem.compute_graph()."
            )
    kept_edges = set()
    graph_nx = nx.DiGraph()
    for node in problem.graph.neighbors_dict:
        if node not in graph_nx:
            graph_nx.add_node(node)
        neighbors = problem.graph.neighbors_dict[node]
        distance = [
            ((node, n), problem.graph.edges_infos_dict[(node, n)]["distance"])
            for n in neighbors
            if n != node
        ]
        if len(distance) > 0:
            sorted_nodes = sorted(distance, key=lambda x: x[1])
            kept_nodes = sorted_nodes[: min(len(sorted_nodes) // 2, len(sorted_nodes))]
            kept_edges.update([x[0] for x in kept_nodes])
            if undirected:
                kept_edges.update([(x[0][1], x[0][0]) for x in kept_nodes])
    for v in problem.origin_vehicle:
        kept_edges.add((problem.origin_vehicle[v], problem.target_vehicle[v]))
        kept_edges.add((problem.target_vehicle[v], problem.origin_vehicle[v]))
    for edge in kept_edges:
        if edge[0] not in graph_nx:
            graph_nx.add_node(edge[0])
        if edge[1] not in graph_nx:
            graph_nx.add_node(edge[1])
        graph_nx.add_edge(edge[0], edge[1])
    connected_components = [
        (len(c), c)
        for c in sorted(nx.weakly_connected_components(graph_nx), key=len, reverse=True)
    ]
    if len(connected_components) > 1:
        added_edges = set()
        for j in range(len(connected_components)):
            nodes_j = connected_components[j][1]
            for k in range(j + 1, len(connected_components)):
                nodes_k = connected_components[k][1]
                distance = [
                    ((n1, n2), problem.graph.edges_infos_dict[(n1, n2)]["distance"])
                    for n1 in nodes_j
                    for n2 in nodes_k
                ]
                if len(distance) > 0:
                    sorted_nodes = sorted(distance, key=lambda x: x[1])
                    kept_nodes = sorted_nodes[: min(3, len(sorted_nodes))]
                    kept_edges.update([x[0] for x in kept_nodes])
                    if undirected:
                        kept_edges.update([(x[0][1], x[0][0]) for x in kept_nodes])
                    added_edges.update([x[0] for x in kept_nodes])
                    if undirected:
                        added_edges.update([(x[0][1], x[0][0]) for x in kept_nodes])
        for edge in added_edges:
            if edge[0] not in graph_nx:
                graph_nx.add_node(edge[0])
            if edge[1] not in graph_nx:
                graph_nx.add_node(edge[1])
            graph_nx.add_edge(edge[0], edge[1])

    return GpdpProblem(
        number_vehicle=problem.number_vehicle,
        nodes_transportation=problem.nodes_transportation,
        nodes_origin=problem.nodes_origin,
        nodes_target=problem.nodes_target,
        list_pickup_deliverable=problem.list_pickup_deliverable,
        origin_vehicle=problem.origin_vehicle,
        target_vehicle=problem.target_vehicle,
        resources_set=problem.resources_set,
        capacities=problem.capacities,
        resources_flow_node=problem.resources_flow_node,
        resources_flow_edges={e: problem.resources_flow_edges[e] for e in kept_edges},
        distance_delta={
            x: {
                y: problem.distance_delta[x][y]
                for y in problem.distance_delta[x]
                if (x, y) in kept_edges
            }
            for x in problem.distance_delta
        },
        time_delta={
            x: {
                y: problem.time_delta[x][y]
                for y in problem.time_delta[x]
                if (x, y) in kept_edges
            }
            for x in problem.time_delta
        },
        time_delta_node=problem.time_delta_node,
        mandatory_node_info=problem.mandatory_node_info,
        cumulative_constraints=problem.cumulative_constraints,
        list_pickup_deliverable_per_cluster=problem.list_pickup_deliverable_per_cluster,
        clusters_dict=problem.clusters_dict,
        coordinates_2d=problem.coordinates_2d,
        compute_graph=compute_graph,
    )


def max_distance(problem: GpdpProblem) -> float:
    return max(
        [
            problem.distance_delta[i][j]
            for i in problem.distance_delta
            for j in problem.distance_delta[i]
        ]
    )


def max_time(problem: GpdpProblem) -> float:
    return max(
        [
            problem.time_delta[i][j]
            for i in problem.time_delta
            for j in problem.time_delta[i]
        ]
        + [problem.time_delta_node[j] for j in problem.time_delta_node]
    )


def build_matrix_distance(problem: GpdpProblem) -> npt.NDArray[np.float64]:
    matrix_distance = 100000 * np.ones(
        (len(problem.all_nodes_dict), len(problem.all_nodes_dict)), dtype=np.float64
    )
    for j in problem.distance_delta:
        for k in problem.distance_delta[j]:
            matrix_distance[
                problem.index_nodes[j], problem.index_nodes[k]
            ] = problem.distance_delta[j][k]
    return matrix_distance


def build_matrix_time(problem: GpdpProblem) -> npt.NDArray[np.float64]:
    matrix_time = 10000 * np.ones(
        (len(problem.all_nodes_dict), len(problem.all_nodes_dict)), dtype=np.float64
    )
    for j in problem.time_delta:
        for k in problem.time_delta[j]:
            matrix_time[
                problem.index_nodes[j], problem.index_nodes[k]
            ] = problem.time_delta[j][k]
    return matrix_time


class ProxyClass:
    @staticmethod
    def from_vrp_to_gpdp(
        vrp_problem: Customer2DVrpProblem, compute_graph: bool = False
    ) -> GpdpProblem:
        nb_vehicle = vrp_problem.vehicle_count
        nb_customers = len(vrp_problem.customers)
        all_start_index = set(vrp_problem.start_indexes)
        all_end_index = set(vrp_problem.end_indexes)
        clients_node = [
            (i, vrp_problem.customers[i])
            for i in range(nb_customers)
            if i not in all_start_index and i not in all_end_index
        ]
        virtual_debut_node = [len(clients_node) + k for k in range(nb_vehicle)]
        virtual_end_node = [
            len(clients_node) + nb_vehicle + k for k in range(nb_vehicle)
        ]
        virtual_to_initial = {
            virtual_debut_node[i]: vrp_problem.start_indexes[i]
            for i in range(len(virtual_debut_node))
        }
        virtual_to_end = {
            virtual_end_node[i]: vrp_problem.end_indexes[i]
            for i in range(len(virtual_end_node))
        }
        real_client_to_initial = {
            j: clients_node[j][0] for j in range(len(clients_node))
        }
        origin_vehicle: dict[int, Node] = {
            i: virtual_debut_node[i] for i in range(nb_vehicle)
        }
        target_vehicle: dict[int, Node] = {
            i: virtual_end_node[i] for i in range(nb_vehicle)
        }
        clients_gpdp = (
            [j for j in range(len(clients_node))]
            + virtual_debut_node
            + virtual_end_node
        )
        dictionnary_distance: dict[Node, dict[Node, float]] = {}
        time_delta: dict[Node, dict[Node, float]] = {}
        resources_flow_node: dict[Node, dict[str, float]] = {}
        coordinates: dict[Node, tuple[float, float]] = {}
        for node1 in clients_gpdp:
            dictionnary_distance[node1] = {}
            time_delta[node1] = {}
            for node2 in clients_gpdp:
                prev_node1 = None
                prev_node2 = None
                if node1 in real_client_to_initial:
                    prev_node1 = real_client_to_initial[node1]
                elif node1 in virtual_to_initial:
                    prev_node1 = virtual_to_initial[node1]
                else:  # node1 in virtual_to_end:
                    prev_node1 = virtual_to_end[node1]
                if node2 in real_client_to_initial:
                    prev_node2 = real_client_to_initial[node2]
                elif node2 in virtual_to_initial:
                    prev_node2 = virtual_to_initial[node2]
                else:  # node2 in virtual_to_end:
                    prev_node2 = virtual_to_end[node2]
                dictionnary_distance[node1][
                    node2
                ] = vrp_problem.evaluate_function_indexes(
                    index_1=prev_node1, index_2=prev_node2
                )

                time_delta[node1][node2] = dictionnary_distance[node1][node2] / 2
        for node in clients_gpdp:
            prev_node = None
            if node in real_client_to_initial:
                prev_node = real_client_to_initial[node]
            elif node in virtual_to_initial:
                prev_node = virtual_to_initial[node]
            else:  # node in virtual_to_end:
                prev_node = virtual_to_end[node]
            coordinates[node] = (
                vrp_problem.customers[prev_node].x,
                vrp_problem.customers[prev_node].y,
            )
            resources_flow_node[node] = {
                "demand": -int(vrp_problem.customers[prev_node].demand)
            }
        for v in range(nb_vehicle):
            resources_flow_node[origin_vehicle[v]]["demand"] = int(
                vrp_problem.vehicle_capacities[v]
            )
        resources_set: set[str] = {"demand"}
        nodes_transportation: set[Node] = set(real_client_to_initial.keys())
        nodes_origin: set[Node] = set(virtual_to_initial.keys())
        nodes_target: set[Node] = set(virtual_to_end.keys())
        list_pickup_deliverable: list[tuple[list[Node], list[Node]]] = []
        capacities: dict[int, dict[str, tuple[float, float]]] = {
            i: {"demand": (0.0, vrp_problem.vehicle_capacities[i])}
            for i in range(nb_vehicle)
        }
        # {Vehicle:{resource: (min_capacity, max_capacity)}}
        resources_flow_edges: dict[Edge, dict[str, float]] = {
            (x, y): {"demand": 0.0}
            for x in dictionnary_distance
            for y in dictionnary_distance[x]
        }

        return GpdpProblem(
            number_vehicle=nb_vehicle,
            nodes_transportation=nodes_transportation,
            nodes_origin=nodes_origin,
            nodes_target=nodes_target,
            list_pickup_deliverable=list_pickup_deliverable,
            origin_vehicle=origin_vehicle,
            target_vehicle=target_vehicle,
            resources_set=resources_set,
            capacities=capacities,
            resources_flow_node=resources_flow_node,
            resources_flow_edges=resources_flow_edges,
            distance_delta=dictionnary_distance,
            time_delta=time_delta,
            coordinates_2d=coordinates,
            compute_graph=compute_graph,
        )

    @staticmethod
    def from_tsp_to_gpdp(
        tsp_model: Point2DTspProblem, compute_graph: bool = False
    ) -> GpdpProblem:
        nb_vehicle = 1
        nb_customers = tsp_model.node_count
        all_start_index = {tsp_model.start_index}
        all_end_index = {tsp_model.end_index}
        clients_node = [
            (i, tsp_model.list_points[i])
            for i in range(nb_customers)
            if i not in all_start_index and i not in all_end_index
        ]
        virtual_debut_node = [len(clients_node) + k for k in range(nb_vehicle)]
        virtual_end_node = [
            len(clients_node) + nb_vehicle + k for k in range(nb_vehicle)
        ]
        virtual_to_initial = {
            virtual_debut_node[i]: tsp_model.start_index
            for i in range(len(virtual_debut_node))
        }
        virtual_to_end = {
            virtual_end_node[i]: tsp_model.end_index
            for i in range(len(virtual_debut_node))
        }
        real_client_to_initial = {
            j: clients_node[j][0] for j in range(len(clients_node))
        }
        origin_vehicle: dict[int, Node] = {
            i: virtual_debut_node[i] for i in range(nb_vehicle)
        }
        target_vehicle: dict[int, Node] = {
            i: virtual_end_node[i] for i in range(nb_vehicle)
        }
        clients_gpdp = (
            [j for j in range(len(clients_node))]
            + virtual_debut_node
            + virtual_end_node
        )
        dictionnary_distance: dict[Node, dict[Node, float]] = {}
        time_delta: dict[Node, dict[Node, float]] = {}
        resources_flow_node: dict[Node, dict[str, float]] = {}
        coordinates: dict[Node, tuple[float, float]] = {}
        for node1 in clients_gpdp:
            dictionnary_distance[node1] = {}
            time_delta[node1] = {}
            for node2 in clients_gpdp:
                prev_node1 = None
                prev_node2 = None
                if node1 in real_client_to_initial:
                    prev_node1 = real_client_to_initial[node1]
                elif node1 in virtual_to_initial:
                    prev_node1 = virtual_to_initial[node1]
                else:  # node1 in virtual_to_end:
                    prev_node1 = virtual_to_end[node1]
                if node2 in real_client_to_initial:
                    prev_node2 = real_client_to_initial[node2]
                elif node2 in virtual_to_initial:
                    prev_node2 = virtual_to_initial[node2]
                else:  # node2 in virtual_to_end:
                    prev_node2 = virtual_to_end[node2]
                dictionnary_distance[node1][
                    node2
                ] = tsp_model.evaluate_function_indexes(
                    index_1=prev_node1, index_2=prev_node2
                )

                time_delta[node1][node2] = dictionnary_distance[node1][node2] / 2
        for node in clients_gpdp:
            prev_node = None
            if node in real_client_to_initial:
                prev_node = real_client_to_initial[node]
            elif node in virtual_to_initial:
                prev_node = virtual_to_initial[node]
            else:  # node in virtual_to_end:
                prev_node = virtual_to_end[node]
            coordinates[node] = (
                tsp_model.list_points[prev_node].x,
                tsp_model.list_points[prev_node].y,
            )
            resources_flow_node[node] = {}
        for v in range(nb_vehicle):
            resources_flow_node[origin_vehicle[v]] = {}
        nodes_transportation: set[Node] = set(real_client_to_initial.keys())
        nodes_origin: set[Node] = set(virtual_to_initial.keys())
        nodes_target: set[Node] = set(virtual_to_end.keys())
        list_pickup_deliverable: list[tuple[list[Node], list[Node]]] = []
        resources_set: set[str] = set()
        capacities: dict[int, dict[str, tuple[float, float]]] = {
            i: {} for i in range(nb_vehicle)
        }
        # {Vehicle:{resource: (min_capacity, max_capacity)}}
        resources_flow_edges = {
            (x, y): {"demand": 0.0}
            for x in dictionnary_distance
            for y in dictionnary_distance[x]
        }
        return GpdpProblem(
            number_vehicle=nb_vehicle,
            nodes_transportation=nodes_transportation,
            nodes_origin=nodes_origin,
            nodes_target=nodes_target,
            list_pickup_deliverable=list_pickup_deliverable,
            origin_vehicle=origin_vehicle,
            target_vehicle=target_vehicle,
            resources_set=resources_set,
            capacities=capacities,
            resources_flow_node=resources_flow_node,
            resources_flow_edges=resources_flow_edges,
            distance_delta=dictionnary_distance,
            time_delta=time_delta,
            coordinates_2d=coordinates,
            compute_graph=compute_graph,
        )
