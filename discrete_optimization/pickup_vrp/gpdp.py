#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

# Generic pickup and delivery problem
# The General Pickup and Delivery Problem
# February 1995 Transportation Science 29(1):17-29
# https://www.researchgate.net/publication/239063487_The_General_Pickup_and_Delivery_Problem
import logging
from typing import Dict, Hashable, List, Set, Tuple, Type

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ObjectiveRegister,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.tsp.tsp_model import TSPModel
from discrete_optimization.vrp.vrp_model import VrpProblem

logger = logging.getLogger(__name__)


class GPDPSolution(Solution):
    def copy(self):
        return GPDPSolution(
            problem=self.problem,
            trajectories=self.trajectories,
            times=self.times,
            resource_evolution=self.resource_evolution,
        )

    def change_problem(self, new_problem):
        self.problem = new_problem

    def __init__(
        self,
        problem,
        trajectories: Dict[int, List[Hashable]],
        times: Dict[Hashable, int],
        resource_evolution: Dict[Hashable, Dict[Hashable, List[int]]],
    ):
        self.problem = problem
        self.trajectories = trajectories
        self.times = times
        self.resource_evolution = resource_evolution


class GPDP(Problem):
    def __init__(
        self,
        number_vehicle,
        nodes_transportation: Set[Hashable],
        nodes_origin: Set[Hashable],
        nodes_target: Set[Hashable],
        list_pickup_deliverable: List[Tuple[List[Hashable], List[Hashable]]],
        origin_vehicle: Dict[int, Hashable],
        target_vehicle: Dict[int, Hashable],
        resources_set: Set[Hashable],
        capacities: Dict[int, Dict[Hashable, Tuple[int, int]]],
        resources_flow_node: Dict[Hashable, Dict[Hashable, int]],
        resources_flow_edges: Dict[Tuple[Hashable, Hashable], Dict[Hashable, int]],
        distance_delta: Dict[Hashable, Dict[Hashable, int]],
        time_delta: Dict[Hashable, Dict[Hashable, int]],
        time_delta_node: Dict[Hashable, int] = None,
        coordinates_2d: Dict[Hashable, Tuple[float, float]] = None,
        clusters_dict: Dict[Hashable, Hashable] = None,
        # For each node, returns an ID of a cluster associated. # This can modelize selective TSP
        list_pickup_deliverable_per_cluster: List[
            Tuple[List[Hashable], List[Hashable]]
        ] = None,
        mandatory_node_info: Dict[Hashable, bool] = None,
        # indicated for each node if the node is optional (=False) or mandatory (=True)
        cumulative_constraints: List[Tuple[Set[Hashable], int]] = None,
        time_windows_nodes: Dict[Hashable, Tuple[int, int]] = None,
        time_windows_cluster: Dict[Hashable, Tuple[int, int]] = None,
        group_identical_vehicles: Dict[int, List[int]] = None,
        slack_time_bound_per_node: Dict[int, Tuple[int, int]] = None,
        node_vehicle: Dict[Hashable, List[int]] = None,
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
        self.time_delta_node = time_delta_node

        self.all_nodes = set()
        self.all_nodes.update(self.nodes_origin)
        self.all_nodes.update(self.nodes_target)
        self.all_nodes.update(self.nodes_transportation)
        self.all_nodes.update([self.origin_vehicle[v] for v in self.origin_vehicle])
        self.all_nodes.update([self.target_vehicle[v] for v in self.target_vehicle])
        self.all_nodes_dict = {}
        if self.time_delta_node is None:
            self.time_delta_node = {n: 0 for n in self.all_nodes}
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

        self.edges_dict = {}
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
        self.graph = None
        if compute_graph:
            logger.info("compute graph")
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
            logger.info("done")
        self.clusters_dict = clusters_dict
        if self.clusters_dict is None:
            i = 0
            self.clusters_dict = {}
            for k in self.all_nodes_dict:
                self.clusters_dict[k] = i
                i += 1
        self.clusters_set = set(self.clusters_dict.values())
        self.clusters_to_node = {k: set() for k in self.clusters_set}
        for k in self.clusters_dict:
            self.clusters_to_node[self.clusters_dict[k]].add(k)
        self.mandatory_node_info = mandatory_node_info
        self.cumulative_constraints = cumulative_constraints
        if self.mandatory_node_info is None:
            if len(self.clusters_to_node) == len(self.all_nodes_dict):
                self.mandatory_node_info = {n: True for n in self.all_nodes_dict}
            else:
                self.mandatory_node_info = {n: False for n in self.all_nodes_dict}
        if self.cumulative_constraints is None:
            self.cumulative_constraints = []
        self.time_windows_nodes = time_windows_nodes
        self.time_windows_cluster = time_windows_cluster
        if self.time_windows_nodes is None:
            self.time_windows_nodes = {n: (None, None) for n in self.all_nodes_dict}
        if self.time_windows_cluster is None:
            self.time_windows_cluster = {k: (None, None) for k in self.clusters_set}
        self.list_nodes = sorted(list(self.all_nodes))
        self.index_nodes = {self.list_nodes[i]: i for i in range(len(self.list_nodes))}
        self.nodes_to_index = {
            i: self.list_nodes[i] for i in range(len(self.list_nodes))
        }

        def get_edges_for_vehicle(vehicle: int):
            return self.graph.get_edges()

        self.get_edges_for_vehicle = get_edges_for_vehicle
        self.group_identical_vehicles = group_identical_vehicles
        if self.group_identical_vehicles is None:
            self.group_identical_vehicles = {i: [i] for i in range(self.number_vehicle)}
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
        self.slack_time_bound_per_node = slack_time_bound_per_node
        if self.slack_time_bound_per_node is None:
            self.slack_time_bound_per_node = {
                node: (0, self.time_delta_node[node]) for node in self.time_delta_node
            }

    def evaluate(self, variable: GPDPSolution) -> Dict[str, float]:
        return {"time": max(variable.times.values())}

    def evaluate_function_node(self, node_1, node_2):
        return self.graph.edges_infos_dict[(node_1, node_2)]["distance"]

    def satisfy(self, variable: GPDPSolution) -> bool:
        return True

    def update_edges(self):
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

    def update_graph(self):
        self.update_edges()
        self.compute_graph()

    def compute_graph(self):
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

    def get_attribute_register(self) -> EncodingRegister:
        pass

    def get_solution_type(self) -> Type[Solution]:
        return GPDPSolution

    def get_objective_register(self) -> ObjectiveRegister:
        pass

    def __str__(self):
        s = "Routing problem : \n"
        s += str(len(self.all_nodes)) + " nodes \n"
        s += str(self.number_vehicle) + " vehicles \n"
        s += str(len(self.clusters_to_node)) + " clusters of node\n"

        return s


def build_pruned_problem(problem: GPDP, undirected=True, compute_graph: bool = False):
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

    return GPDP(
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


def max_distance(problem: GPDP):
    return max(
        [
            problem.distance_delta[i][j]
            for i in problem.distance_delta
            for j in problem.distance_delta[i]
        ]
    )


def max_time(problem: GPDP):
    return max(
        [
            problem.time_delta[i][j]
            for i in problem.time_delta
            for j in problem.time_delta[i]
        ]
        + [problem.time_delta_node[j] for j in problem.time_delta_node]
    )


def build_matrix_distance(problem: GPDP):
    matrix_distance = 100000 * np.ones(
        (len(problem.all_nodes_dict), len(problem.all_nodes_dict))
    )
    for j in problem.distance_delta:
        for k in problem.distance_delta[j]:
            matrix_distance[
                problem.index_nodes[j], problem.index_nodes[k]
            ] = problem.distance_delta[j][k]
    return matrix_distance


def build_matrix_time(problem: GPDP):
    matrix_time = 10000 * np.ones(
        (len(problem.all_nodes_dict), len(problem.all_nodes_dict))
    )
    for j in problem.time_delta:
        for k in problem.time_delta[j]:
            matrix_time[
                problem.index_nodes[j], problem.index_nodes[k]
            ] = problem.time_delta[j][k]
    return matrix_time


class ProxyClass:
    @staticmethod
    def from_vrp_model_to_gpdp(vrp_model: VrpProblem, compute_graph: bool = False):
        nb_vehicle = vrp_model.vehicle_count
        nb_customers = len(vrp_model.customers)
        all_start_index = set(vrp_model.start_indexes)
        all_end_index = set(vrp_model.end_indexes)
        clients_node = [
            (i, vrp_model.customers[i])
            for i in range(nb_customers)
            if i not in all_start_index and i not in all_end_index
        ]
        virtual_debut_node = [len(clients_node) + k for k in range(nb_vehicle)]
        virtual_end_node = [
            len(clients_node) + nb_vehicle + k for k in range(nb_vehicle)
        ]
        virtual_to_initial = {
            virtual_debut_node[i]: vrp_model.start_indexes[i]
            for i in range(len(virtual_debut_node))
        }
        virtual_to_end = {
            virtual_end_node[i]: vrp_model.end_indexes[i]
            for i in range(len(virtual_debut_node))
        }
        real_client_to_initial = {
            j: clients_node[j][0] for j in range(len(clients_node))
        }
        origin_vehicle = {i: virtual_debut_node[i] for i in range(nb_vehicle)}
        target_vehicle = {i: virtual_end_node[i] for i in range(nb_vehicle)}
        clients_gpdp = (
            [j for j in range(len(clients_node))]
            + virtual_debut_node
            + virtual_end_node
        )
        dictionnary_distance = {}
        time_delta = {}
        resources_flow_node = {}
        coordinates = {}
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
                elif node1 in virtual_to_end:
                    prev_node1 = virtual_to_end[node1]
                if node2 in real_client_to_initial:
                    prev_node2 = real_client_to_initial[node2]
                elif node2 in virtual_to_initial:
                    prev_node2 = virtual_to_initial[node2]
                elif node2 in virtual_to_end:
                    prev_node2 = virtual_to_end[node2]
                dictionnary_distance[node1][
                    node2
                ] = vrp_model.evaluate_function_indexes(
                    index_1=prev_node1, index_2=prev_node2
                )

                time_delta[node1][node2] = dictionnary_distance[node1][node2] / 2
        for node in clients_gpdp:
            prev_node = None
            if node in real_client_to_initial:
                prev_node = real_client_to_initial[node]
            elif node in virtual_to_initial:
                prev_node = virtual_to_initial[node]
            elif node in virtual_to_end:
                prev_node = virtual_to_end[node]
            coordinates[node] = (
                vrp_model.customers[prev_node].x,
                vrp_model.customers[prev_node].y,
            )
            resources_flow_node[node] = {
                "demand": -int(vrp_model.customers[prev_node].demand)
            }
        for v in range(nb_vehicle):
            resources_flow_node[origin_vehicle[v]]["demand"] = int(
                vrp_model.vehicle_capacities[v]
            )
        resources_set = {"demand"}
        nodes_transportation = set(real_client_to_initial.keys())
        nodes_origin = set(virtual_to_initial.keys())
        nodes_target = set(virtual_to_end.keys())
        list_pickup_deliverable = []
        capacities = {
            i: {"demand": (0, vrp_model.vehicle_capacities[i])}
            for i in range(nb_vehicle)
        }
        # {Vehicle:{resource: (min_capacity, max_capacity)}}
        resources_flow_node = resources_flow_node
        resources_flow_edges = {
            (x, y): {"demand": 0}
            for x in dictionnary_distance
            for y in dictionnary_distance[x]
        }
        distance_delta: Dict[Hashable, Dict[Hashable, int]] = dictionnary_distance
        time_delta: Dict[Hashable, Dict[Hashable, int]] = time_delta

        return GPDP(
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
            distance_delta=distance_delta,
            time_delta=time_delta,
            coordinates_2d=coordinates,
            compute_graph=compute_graph,
        )

    @staticmethod
    def from_tsp_model_gpdp(tsp_model: TSPModel, compute_graph: bool = False):
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
        origin_vehicle = {i: virtual_debut_node[i] for i in range(nb_vehicle)}
        target_vehicle = {i: virtual_end_node[i] for i in range(nb_vehicle)}
        clients_gpdp = (
            [j for j in range(len(clients_node))]
            + virtual_debut_node
            + virtual_end_node
        )
        dictionnary_distance = {}
        time_delta = {}
        resources_flow_node = {}
        coordinates = {}
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
                elif node1 in virtual_to_end:
                    prev_node1 = virtual_to_end[node1]
                if node2 in real_client_to_initial:
                    prev_node2 = real_client_to_initial[node2]
                elif node2 in virtual_to_initial:
                    prev_node2 = virtual_to_initial[node2]
                elif node2 in virtual_to_end:
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
            elif node in virtual_to_end:
                prev_node = virtual_to_end[node]
            coordinates[node] = (
                tsp_model.list_points[prev_node].x,
                tsp_model.list_points[prev_node].y,
            )
            resources_flow_node[node] = {}
        for v in range(nb_vehicle):
            resources_flow_node[origin_vehicle[v]] = {}
        nodes_transportation = set(real_client_to_initial.keys())
        nodes_origin = set(virtual_to_initial.keys())
        nodes_target = set(virtual_to_end.keys())
        list_pickup_deliverable = []
        resources_set = set()
        capacities = {i: {} for i in range(nb_vehicle)}
        # {Vehicle:{resource: (min_capacity, max_capacity)}}
        resources_flow_node = resources_flow_node
        resources_flow_edges = {
            (x, y): {"demand": 0}
            for x in dictionnary_distance
            for y in dictionnary_distance[x]
        }
        distance_delta: Dict[Hashable, Dict[Hashable, int]] = dictionnary_distance
        time_delta: Dict[Hashable, Dict[Hashable, int]] = time_delta
        return GPDP(
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
            distance_delta=distance_delta,
            time_delta=time_delta,
            coordinates_2d=coordinates,
            compute_graph=compute_graph,
        )
