#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
import os
import random
import time
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from enum import Enum
from typing import Any, Optional, Union, cast

import networkx as nx
from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.lp_tools import (
    ConstraintType,
    GurobiCallback,
    GurobiMilpSolver,
    InequalitySense,
    MathOptCallback,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    ParametersMilp,
    VariableType,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    TupleFitness,
)
from discrete_optimization.gpdp.problem import Edge, GpdpProblem, GpdpSolution, Node
from discrete_optimization.gpdp.solvers import GpdpSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


logger = logging.getLogger(__name__)


class TemporaryResult:
    def __init__(
        self,
        flow_solution: dict[int, dict[Edge, int]],
        graph_merge: nx.DiGraph,
        graph_vehicle: dict[int, nx.DiGraph],
        obj: float,
        all_variables: Optional[dict[str, dict[Any, Any]]] = None,
    ):
        self.flow_solution = flow_solution
        self.graph_merge = graph_merge
        self.graph_vehicle = graph_vehicle
        self.obj = obj
        self.component_global: Optional[list[tuple[set[Node], int]]] = None
        self.connected_components_per_vehicle: Optional[
            dict[int, list[tuple[set[Node], int]]]
        ] = None
        self.all_variables = all_variables
        self.rebuilt_dict: Optional[dict[int, list[Node]]] = None
        self.paths_component: Optional[dict[int, dict[int, list[Node]]]] = None
        self.indexes_component: Optional[dict[int, dict[int, dict[Node, int]]]] = None


def convert_temporaryresult_to_gpdpsolution(
    temporaryresult: TemporaryResult, problem: GpdpProblem
) -> GpdpSolution:
    if temporaryresult.rebuilt_dict is None:
        raise ValueError(
            "temporaryresult.rebuilt_dict should not be None "
            "when calling convert_temporaryresult_to_gpdpsolution()"
        )
    times: dict[Node, float] = {}
    if (
        temporaryresult.all_variables is not None
        and "time_leaving" in temporaryresult.all_variables
    ):
        # We will only store the time values to visited nodes (which are not necessarly all the nodes
        # when we run cluster version of the problem
        nodes_visited_in_results = set()
        for trajectory in temporaryresult.rebuilt_dict.values():
            nodes_visited_in_results.update(trajectory)
        times = {
            n: temporaryresult.all_variables["time_leaving"][n]
            for n in nodes_visited_in_results
        }
    resource_evolution: dict[Node, dict[Node, list[int]]] = {}
    return GpdpSolution(
        problem=problem,
        trajectories=temporaryresult.rebuilt_dict,
        times=times,
        resource_evolution=resource_evolution,
    )


def retrieve_current_solution(
    get_var_value_for_current_solution: Callable[[Any], float],
    get_obj_value_for_current_solution: Callable[[], float],
    variable_decisions: dict[str, Any],
) -> tuple[dict[str, dict[Hashable, Any]], float]:
    results: dict[str, dict[Hashable, Any]] = {}
    xsolution: dict[int, dict[Edge, int]] = {
        v: {} for v in variable_decisions["variables_edges"]
    }
    obj = get_obj_value_for_current_solution()
    for vehicle in variable_decisions["variables_edges"]:
        for edge in variable_decisions["variables_edges"][vehicle]:
            value = get_var_value_for_current_solution(
                variable_decisions["variables_edges"][vehicle][edge]
            )
            if value <= 0.1:
                continue
            xsolution[vehicle][edge] = 1
    results["variables_edges"] = cast(dict[Hashable, Any], xsolution)
    for key in variable_decisions:
        if key == "variables_edges":
            continue
        results[key] = {}
        for key_2 in variable_decisions[key]:
            if isinstance(variable_decisions[key][key_2], dict):
                results[key][key_2] = {}
                for key_3 in variable_decisions[key][key_2]:
                    if isinstance(variable_decisions[key][key_2][key_3], dict):
                        results[key][key_2][key_3] = {}
                        for key_4 in variable_decisions[key][key_2][key_3]:
                            value = get_var_value_for_current_solution(
                                variable_decisions[key][key_2][key_3]
                            )
                            results[key][key_2][key_3][key_4] = value
                    else:
                        value = get_var_value_for_current_solution(
                            variable_decisions[key][key_2][key_3]
                        )
                        results[key][key_2][key_3] = value
            else:
                value = get_var_value_for_current_solution(
                    variable_decisions[key][key_2]
                )
                results[key][key_2] = value
    return results, obj


class BaseLinearFlowGpdpSolver(MilpSolver, GpdpSolver):
    problem: GpdpProblem
    warm_start: Optional[GpdpSolution] = None
    lazy: bool = False
    """Flag to know if contraints are added via a callback in a lazy way.

    Should only be True in `GurobiLazyConstraintLinearFlowGpdpSolver`.

    """

    def set_warm_start(self, solution: GpdpSolution) -> None:
        """Make the solver warm start from the given solution.

        Will be ignored if arg `warm_start` is not None in call to `solve()`.

        """
        self.warm_start = solution

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[gurobipy.Var, float]:
        """

        Not used here by set_warm_start().

        Args:
            solution:

        Returns:

        """
        raise NotImplementedError

    def one_visit_per_node(
        self,
        nodes_of_interest: Iterable[Node],
        variables_edges: dict[int, dict[Edge, Any]],
    ) -> None:
        constraint_one_visit = {}
        for node in nodes_of_interest:
            constraint_one_visit[node] = self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[x[0]][x[1]]
                    for x in self.edges_in_all_vehicles[node]
                )
                >= 1,
                name="visit_" + str(node),
            )
            constraint_one_visit[(node, "out")] = self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[x[0]][x[1]]
                    for x in self.edges_out_all_vehicles[node]
                )
                >= 1,
                name="visitout_" + str(node),
            )

    def one_visit_per_clusters(
        self,
        nodes_of_interest: Iterable[Node],
        variables_edges: dict[int, dict[Edge, Any]],
    ) -> None:
        constraint_cluster: dict[Hashable, Any] = {}
        for cluster in self.problem.clusters_to_node:
            if all(
                node in nodes_of_interest
                for node in self.problem.clusters_to_node[cluster]
            ):
                constraint_cluster[cluster] = self.add_linear_constraint(
                    self.construct_linear_sum(
                        variables_edges[x[0]][x[1]]
                        for node in self.problem.clusters_to_node[cluster]
                        for x in self.edges_in_all_vehicles[node]
                        if node in nodes_of_interest
                        and x[0] in self.problem.vehicles_representative
                    )
                    >= 1,
                    name="visit_" + str(cluster),
                )
                constraint_cluster[(cluster, "out")] = self.add_linear_constraint(
                    self.construct_linear_sum(
                        variables_edges[x[0]][x[1]]
                        for node in self.problem.clusters_to_node[cluster]
                        for x in self.edges_out_all_vehicles[node]
                        if node in nodes_of_interest
                        and x[0] in self.problem.vehicles_representative
                    )
                    == 1,
                    name="visitout_" + str(cluster),
                )

    def resources_constraint(
        self,
        variables_edges: dict[int, dict[Edge, Any]],
    ) -> dict[str, dict[str, dict[Node, Any]]]:
        resources = self.problem.resources_set
        resources_variable_coming: dict[str, dict[Node, Any]] = {
            r: {
                node: self.add_continuous_variable(
                    name="res_coming_" + str(r) + "_" + str(node),
                )
                for node in self.edges_in_all_vehicles
            }
            for r in resources
        }
        resources_variable_leaving = {
            r: {
                node: self.add_continuous_variable(
                    name="res_leaving_" + str(r) + "_" + str(node),
                )
                for node in self.edges_in_all_vehicles
            }
            for r in resources
        }
        for v in self.problem.origin_vehicle:
            for r in resources_variable_coming:
                self.add_linear_constraint(
                    resources_variable_coming[r][self.problem.origin_vehicle[v]]
                    == self.problem.resources_flow_node[self.problem.origin_vehicle[v]][
                        r
                    ]
                )
                self.add_linear_constraint(
                    resources_variable_leaving[r][self.problem.origin_vehicle[v]]
                    == self.problem.resources_flow_node[self.problem.origin_vehicle[v]][
                        r
                    ]
                )
        index = 0
        all_origin = set(self.problem.origin_vehicle.values())
        for r in resources_variable_coming:
            for node in resources_variable_coming[r]:
                for vehicle, edge in self.edges_in_all_vehicles[node]:
                    if edge[0] == edge[1]:
                        continue
                    self.add_linear_constraint_with_indicator(
                        binvar=variables_edges[vehicle][edge],
                        binval=1,
                        lhs=resources_variable_coming[r][node],
                        sense=InequalitySense.EQUAL,
                        rhs=resources_variable_leaving[r][edge[0]]
                        + self.problem.resources_flow_edges[edge][r],
                        name=f"res_coming_indicator_{index}",
                    )
                    index += 1
                if node not in all_origin:
                    self.add_linear_constraint(
                        resources_variable_leaving[r][node]
                        == resources_variable_coming[r][node]
                        + self.problem.resources_flow_node[node][r]
                    )
        return {
            "resources_variable_coming": resources_variable_coming,
            "resources_variable_leaving": resources_variable_leaving,
        }

    def simple_capacity_constraint(
        self,
        variables_edges: dict[int, dict[Edge, Any]],
    ) -> dict[str, dict[int, dict[str, Any]]]:
        # Case where we don't have to track the resource flow etc...
        # we just want to check that we respect the capacity constraint (like in VRP with capacity)
        # we admit that the resource demand/consumption are positive.
        # corresponding to negative flow values in the problem definition.
        consumption_per_vehicle: dict[int, dict[str, Any]] = {
            v: {
                r: self.add_continuous_variable(
                    lb=0,
                    ub=self.problem.capacities[v][r][1],
                    name="consumption_v_" + str(v) + "_" + str(r),
                )
                for r in self.problem.resources_set
            }
            for v in self.problem.capacities
        }
        for v in consumption_per_vehicle:
            for r in consumption_per_vehicle[v]:
                self.add_linear_constraint(
                    consumption_per_vehicle[v][r]
                    == self.construct_linear_sum(
                        variables_edges[v][e]
                        * (
                            -self.problem.resources_flow_node[e[1]][r]
                            - self.problem.resources_flow_edges[e][r]
                        )
                        for e in variables_edges[v]
                        if e[1] != self.problem.origin_vehicle[v]
                    )
                )
                # redundant :
                self.add_linear_constraint(
                    consumption_per_vehicle[v][r] <= self.problem.capacities[v][r][1],
                    name="constraint_capa_" + str(v) + "_" + str(r),
                )
        return {"consumption_per_vehicle": consumption_per_vehicle}

    def time_evolution(
        self,
        variables_edges: dict[int, dict[Edge, Any]],
    ) -> dict[str, dict[Node, Any]]:
        time_coming: dict[Node, Any] = {
            node: self.add_continuous_variable(name="time_coming_" + str(node))
            for node in self.edges_in_all_vehicles
        }
        time_leaving: dict[Node, Any] = {
            node: self.add_continuous_variable(name="time_leaving_" + str(node))
            for node in self.edges_in_all_vehicles
        }
        for v in self.problem.origin_vehicle:
            self.add_linear_constraint(time_coming[self.problem.origin_vehicle[v]] == 0)
        index = 0
        all_origin = set(self.problem.origin_vehicle.values())
        for node in time_leaving:
            for vehicle, edge in self.edges_in_all_vehicles[node]:
                if edge[0] == edge[1]:
                    continue
                self.add_linear_constraint_with_indicator(
                    binvar=variables_edges[vehicle][edge],
                    binval=1,
                    lhs=time_coming[node],
                    sense=InequalitySense.EQUAL,
                    rhs=time_leaving[edge[0]]
                    + self.problem.time_delta[edge[0]][edge[1]],
                    name=f"time_coming_{index}",
                )
                index += 1
            if node not in all_origin:
                self.add_linear_constraint(
                    time_leaving[node]
                    >= time_coming[node] + self.problem.time_delta_node[node]
                )
        return {"time_coming": time_coming, "time_leaving": time_leaving}

    def init_order_variables(self) -> None:
        variables_order = {
            node: self.add_continuous_variable(name="order_" + str(node))
            for node in self.edges_in_all_vehicles
        }
        constraints_order: dict[Node, dict[Edge, ConstraintType]] = defaultdict(dict)
        for vehicle in range(self.problem.number_vehicle):
            node_origin = self.problem.origin_vehicle[vehicle]
            constraints_order[node_origin][
                (node_origin, node_origin)
            ] = self.add_linear_constraint(
                variables_order[node_origin] == 0,
                name="order_" + str(node_origin),
            )
        self.variables_order = variables_order
        self.constraints_order = constraints_order

    def add_order_constraints(self, nodes: Iterable[Node], lazy: bool = False) -> None:
        if lazy:
            if isinstance(self, GurobiMilpSolver):
                add_constraint_fn = self.model.cbLazy
            else:
                raise NotImplementedError(
                    "Lazy constraints are implemented only for the gurobi solvers."
                )
        else:
            add_constraint_fn = self.add_linear_constraint
        constraints_order = self.constraints_order
        variables_order = self.variables_order
        nb_nodes = len(self.problem.list_nodes)
        for node in nodes:
            if node not in constraints_order:
                for vehicle, edge in self.edges_in_all_vehicles[node]:
                    if edge[0] == edge[1]:
                        continue
                    if self.subtour_use_indicator:
                        constraints_order[node][
                            edge
                        ] = self.add_linear_constraint_with_indicator(
                            binvar=self.variable_decisions["variables_edges"][vehicle][
                                edge
                            ],
                            binval=1,
                            lhs=variables_order[node],
                            sense=InequalitySense.EQUAL,
                            rhs=variables_order[edge[0]] + 1,
                            name=f"order_{node}_{edge}",
                        )
                    else:
                        constraints_order[node][edge] = add_constraint_fn(
                            variables_order[node]
                            >= variables_order[edge[0]]
                            + 1
                            - 2
                            * nb_nodes
                            * (
                                1
                                - self.variable_decisions["variables_edges"][vehicle][
                                    edge
                                ]
                            ),
                            name=f"order_{node}_{edge}",
                        )

    def init_model(self, **kwargs: Any) -> None:
        self.model = self.create_empty_model("GpdpProblem-flow")
        include_backward = kwargs.get("include_backward", True)
        include_triangle = kwargs.get("include_triangle", False)
        include_subtour = kwargs.get("include_subtour", False)
        include_resources = kwargs.get("include_resources", False)
        include_capacity = kwargs.get("include_capacity", False)
        include_time_evolution = kwargs.get("include_time_evolution", False)
        one_visit_per_node = kwargs.get("one_visit_per_node", True)
        one_visit_per_cluster = kwargs.get("one_visit_per_cluster", False)
        self.subtour_do_order = kwargs.get("subtour_do_order", False)
        self.subtour_use_indicator = kwargs.get("subtour_use_indicator", False)
        self.subtour_consider_only_first_component = kwargs.get(
            "subtour_consider_only_first_component", True
        )
        optimize_span = kwargs.get("optimize_span", False)
        unique_visit = kwargs.get("unique_visit", True)
        nb_vehicle = self.problem.number_vehicle
        name_vehicles = sorted(self.problem.origin_vehicle.keys())
        if self.problem.graph is None:
            self.problem.compute_graph()
            if self.problem.graph is None:
                raise RuntimeError(
                    "self.problem.graph cannot be None "
                    "after calling self.problem.compute_graph()."
                )
        graph = self.problem.graph
        nodes_of_interest = [
            n
            for n in graph.get_nodes()
            if n not in self.problem.nodes_origin and n not in self.problem.nodes_target
        ]
        variables_edges: dict[int, dict[Edge, Any]]
        if self.problem.any_grouping:
            variables_edges = {j: {} for j in range(nb_vehicle)}
            for k in self.problem.group_identical_vehicles:
                j = self.problem.group_identical_vehicles[k][0]
                variables_edges[j] = {
                    e: self.add_binary_variable(
                        name="flow_" + str(j) + "_" + str(e),
                        obj=graph.edges_infos_dict[e]["distance"],
                    )
                    for e in self.problem.get_edges_for_vehicle(j)
                }
                if len(self.problem.group_identical_vehicles[k]) > 0:
                    for vv in self.problem.group_identical_vehicles[k][1:]:
                        variables_edges[vv] = variables_edges[j]
        else:
            variables_edges = {
                j: {
                    e: self.add_binary_variable(
                        name="flow_" + str(j) + "_" + str(e),
                    )
                    for e in self.problem.get_edges_for_vehicle(j)
                }
                for j in range(nb_vehicle)
            }
        obj = self.construct_linear_sum(
            graph.edges_infos_dict[e]["distance"] * var
            for variables_edges_per_group in variables_edges.values()
            for e, var in variables_edges_per_group.items()
        )
        if optimize_span:
            self.spans = {
                j: self.add_integer_variable(name="span_" + str(j))
                for j in range(nb_vehicle)
            }
            self.objective = self.add_integer_variable()
            obj += 100 * self.objective
            for j in self.spans:
                self.add_linear_constraint(
                    self.spans[j]
                    == self.construct_linear_sum(
                        variables_edges[j][e] * graph.edges_infos_dict[e]["distance"]
                        for e in variables_edges[j]
                    )
                )
                self.add_linear_constraint(self.objective >= self.spans[j], name="obj")
        self.nodes_of_interest = nodes_of_interest
        self.variables_edges = variables_edges
        all_variables: dict[str, dict[Any, Any]] = {"variables_edges": variables_edges}
        constraint_loop: dict[tuple[int, Edge], Any] = {}
        self.variable_decisions = all_variables
        self.clusters_version = one_visit_per_cluster
        self.tsp_version = one_visit_per_node
        for vehicle in variables_edges:
            for e in variables_edges[vehicle]:
                if e[0] == e[1]:
                    constraint_loop[(vehicle, e)] = self.add_linear_constraint(
                        variables_edges[vehicle][e] == 0,
                        name="loop_" + str((vehicle, e)),
                    )

        (
            edges_in_all_vehicles,
            edges_out_all_vehicles,
            edges_in_per_vehicles,
            edges_out_per_vehicles,
            edges_in_all_vehicles_cluster,
            edges_out_all_vehicles_cluster,
            edges_in_per_vehicles_cluster,
            edges_out_per_vehicles_cluster,
        ) = construct_edges_in_out_dict(
            variables_edges=variables_edges, clusters_dict=self.problem.clusters_dict
        )
        self.edges_in_all_vehicles: dict[
            Node, set[tuple[int, Edge]]
        ] = edges_in_all_vehicles
        self.edges_out_all_vehicles: dict[
            Node, set[tuple[int, Edge]]
        ] = edges_out_all_vehicles
        self.edges_in_per_vehicles: dict[
            int, dict[Node, set[Edge]]
        ] = edges_in_per_vehicles
        self.edges_out_per_vehicles: dict[
            int, dict[Node, set[Edge]]
        ] = edges_out_per_vehicles
        self.edges_in_all_vehicles_cluster: dict[
            Hashable, set[tuple[int, Edge]]
        ] = edges_in_all_vehicles_cluster
        self.edges_out_all_vehicles_cluster: dict[
            Hashable, set[tuple[int, Edge]]
        ] = edges_out_all_vehicles_cluster
        self.edges_in_per_vehicles_cluster: dict[
            int, dict[Hashable, set[Edge]]
        ] = edges_in_per_vehicles_cluster
        self.edges_out_per_vehicles_cluster: dict[
            int, dict[Hashable, set[Edge]]
        ] = edges_out_per_vehicles_cluster

        constraints_out_flow: dict[tuple[int, Node], Any] = {}
        constraints_in_flow: dict[Union[tuple[int, Node], Node], Any] = {}
        constraints_flow_conservation: dict[
            Union[tuple[int, Node], tuple[int, Node, str]], Any
        ] = {}

        count_origin: dict[Node, int] = {}
        count_target: dict[Node, int] = {}
        for vehicle in range(nb_vehicle):
            node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            node_target = self.problem.target_vehicle[name_vehicles[vehicle]]
            if node_origin not in count_origin:
                count_origin[node_origin] = 0
            if node_target not in count_target:
                count_target[node_target] = 0
            count_origin[node_origin] += 1
            count_target[node_target] += 1

        for vehicle in range(nb_vehicle):
            node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            node_target = self.problem.target_vehicle[name_vehicles[vehicle]]
            same_node = node_origin == node_target
            constraints_out_flow[(vehicle, node_origin)] = self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[vehicle][edge]
                    for edge in edges_out_per_vehicles[vehicle].get(node_origin, set())
                    if edge[1] != node_origin or same_node
                )
                == count_origin[node_origin],
                name="outflow_" + str((vehicle, node_origin)),
            )  # Avoid loop
            constraints_in_flow[(vehicle, node_target)] = self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[vehicle][edge]
                    for edge in edges_in_per_vehicles[vehicle].get(node_target, set())
                    if edge[0] != node_target or same_node
                )
                == count_target[node_target],
                name="inflow_" + str((vehicle, node_target)),
            )  # Avoid loop

            constraints_out_flow[(vehicle, node_target)] = self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[vehicle][edge]
                    for edge in edges_out_per_vehicles[vehicle].get(node_target, set())
                    if edge[1] != node_target or same_node
                )
                <= 0,
                name="outflow_" + str((vehicle, node_target)),
            )

        for vehicle in range(nb_vehicle):
            origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            target = self.problem.target_vehicle[name_vehicles[vehicle]]
            self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[v][edge]
                    for v, edge in edges_out_all_vehicles[origin]
                    if self.problem.origin_vehicle[name_vehicles[v]] != origin
                )
                == 0
            )
            self.add_linear_constraint(
                self.construct_linear_sum(
                    variables_edges[v][edge]
                    for v, edge in edges_in_all_vehicles[target]
                    if self.problem.target_vehicle[name_vehicles[v]] != target
                )
                == 0
            )

        for vehicle in range(nb_vehicle):
            node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            node_target = self.problem.target_vehicle[name_vehicles[vehicle]]
            same_node = node_origin == node_target
            for node in edges_in_per_vehicles[vehicle]:
                if same_node or node not in {node_origin, node_target}:
                    constraints_flow_conservation[
                        (vehicle, node)
                    ] = self.add_linear_constraint(
                        self.construct_linear_sum(
                            variables_edges[vehicle][e]
                            for e in edges_in_per_vehicles[vehicle].get(node, set())
                            if e[1] != e[0]
                        )
                        + self.construct_linear_sum(
                            -variables_edges[vehicle][e]
                            for e in edges_out_per_vehicles[vehicle].get(node, set())
                            if e[1] != e[0]
                        )
                        == 0,
                        name="convflow_" + str((vehicle, node)),
                    )
                    if unique_visit:
                        constraints_flow_conservation[
                            (vehicle, node, "in")
                        ] = self.add_linear_constraint(
                            self.construct_linear_sum(
                                variables_edges[vehicle][e]
                                for e in edges_in_per_vehicles[vehicle].get(node, set())
                                if e[1] != e[0]
                            )
                            <= 1,
                            name="valueflow_" + str((vehicle, node)),
                        )

        if include_backward:
            constraint_tour_2length = {}
            cnt_tour = 0
            for vehicle in range(nb_vehicle):
                node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
                node_target = self.problem.target_vehicle[name_vehicles[vehicle]]
                for edge in variables_edges[vehicle]:
                    if (edge[1], edge[0]) in variables_edges[vehicle]:
                        if (edge[1], edge[0]) == edge:
                            continue
                        if edge[0] == node_origin or edge[1] == node_origin:
                            continue
                        if edge[0] == node_target or edge[1] == node_target:
                            continue
                        constraint_tour_2length[cnt_tour] = self.add_linear_constraint(
                            variables_edges[vehicle][edge]
                            + variables_edges[vehicle][(edge[1], edge[0])]
                            <= 1,
                            name="tour_" + str((vehicle, edge)),
                        )
                    cnt_tour += 1

        if include_triangle:
            constraint_triangle = {}
            cnt_triangle = 0

            for node in graph.graph_nx.nodes():
                neigh = set([n for n in nx.neighbors(graph.graph_nx, node)])
                neigh_2 = {
                    nn: neigh.intersection(
                        [n for n in nx.neighbors(graph.graph_nx, nn)]
                    )
                    for nn in neigh
                }
                for node_neigh in neigh_2:
                    if len(neigh_2[node_neigh]) >= 1:
                        for node_neigh_neigh in neigh_2[node_neigh]:
                            for vehicle in range(nb_vehicle):
                                constraint_triangle[
                                    cnt_triangle
                                ] = self.add_linear_constraint(
                                    variables_edges[vehicle][(node, node_neigh)]
                                    + variables_edges[vehicle][
                                        (node_neigh, node_neigh_neigh)
                                    ]
                                    + variables_edges[vehicle][(node_neigh_neigh, node)]
                                    <= 2,
                                    name="triangle_" + str(cnt_triangle),
                                )
                                cnt_triangle += 1

        if include_subtour or self.subtour_do_order:
            self.init_order_variables()

        if include_subtour:
            self.add_order_constraints(nodes=self.variables_order)

        for vehicle in range(nb_vehicle):
            for node in [self.problem.origin_vehicle[name_vehicles[vehicle]]]:
                if node in edges_in_all_vehicles:
                    constraints_in_flow[node] = self.add_linear_constraint(
                        self.construct_linear_sum(
                            variables_edges[x[0]][x[1]]
                            for x in edges_in_all_vehicles[node]
                        )
                        == 0,
                        name="no_inflow_start_" + str(node),
                    )

        if one_visit_per_node:
            self.one_visit_per_node(
                nodes_of_interest=nodes_of_interest,
                variables_edges=variables_edges,
            )
        if one_visit_per_cluster:
            self.one_visit_per_clusters(
                nodes_of_interest=nodes_of_interest,
                variables_edges=variables_edges,
            )
        if include_capacity:
            all_variables.update(
                self.simple_capacity_constraint(
                    variables_edges=variables_edges,
                )
            )
        if include_resources:
            all_variables.update(
                self.resources_constraint(
                    variables_edges=variables_edges,
                )
            )
        if include_time_evolution:
            all_variables.update(
                self.time_evolution(
                    variables_edges=variables_edges,
                )
            )
        self.set_model_objective(obj, minimize=True)

    def retrieve_current_temporaryresult(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> TemporaryResult:
        res, obj = retrieve_current_solution(
            get_var_value_for_current_solution=get_var_value_for_current_solution,
            get_obj_value_for_current_solution=get_obj_value_for_current_solution,
            variable_decisions=self.variable_decisions,
        )
        if self.problem.graph is None:
            raise RuntimeError(
                "self.problem.graph cannot be None "
                "when calling retrieve_ith_temporaryresult()."
            )
        temporaryresult = build_graph_solution(
            results=res, obj=obj, graph=self.problem.graph
        )
        reevaluate_result(temporaryresult, problem=self.problem)
        return temporaryresult

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> GpdpSolution:
        """

        Not used here as solve() is overriden


        """
        raise NotImplementedError()

    @abstractmethod
    def solve_one_iteration(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> list[TemporaryResult]:
        ...

    def solve_iterative(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit_subsolver: Optional[float] = 30.0,
        do_lns: bool = True,
        nb_iteration_max: int = 10,
        json_dump_folder: Optional[str] = None,
        warm_start: Optional[dict[Any, Any]] = None,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """

        Args:
            parameters_milp:
            time_limit_subsolver: the solve process of the LP subsolver stops after this time limit (in seconds).
                If None, no time limit is applied.
            do_lns:
            nb_iteration_max:
            json_dump_folder: if not None, solution will be dumped in this folder at each iteration
            warm_start:
            **kwargs:

        Returns:

        """
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()

        if json_dump_folder is not None:
            os.makedirs(json_dump_folder, exist_ok=True)
        if warm_start is None and self.warm_start is not None:
            warm_start = self.warm_start.trajectories
        if warm_start is not None:
            c = ConstraintHandlerOrWarmStart(
                linear_solver=self, problem=self.problem, do_lns=do_lns
            )
            c.adding_constraint(warm_start)
        solutions: list[TemporaryResult] = self.solve_one_iteration(
            parameters_milp=parameters_milp, time_limit=time_limit_subsolver, **kwargs
        )
        best_solution: TemporaryResult = min(solutions, key=lambda sol: sol.obj)
        if not self.lazy:
            # contraints added in `solve_one_iteration` in lazy version
            if (
                (best_solution.rebuilt_dict is None)
                or (best_solution.connected_components_per_vehicle is None)
                or (best_solution.component_global is None)
            ):
                raise RuntimeError(
                    "Temporary result attributes rebuilt_dict, component_global"
                    "and connected_components_per_vehicle cannot be None after solving."
                )
            subtour = SubtourAddingConstraint(
                problem=self.problem,
                linear_solver=self,
                cluster=self.clusters_version,
                do_order=self.subtour_do_order,
                consider_only_first_component=self.subtour_consider_only_first_component,
            )
            subtour.adding_component_constraints([best_solution])

        nb_iteration = 0
        # store solutions
        res = self.create_result_storage(
            self.convert_temporaryresults(solutions),
        )
        # earling stopping?
        stopping = callbacks_list.on_step_end(step=nb_iteration, res=res, solver=self)
        # end condition?
        if (
            max(
                [
                    len(best_solution.connected_components_per_vehicle[v])
                    for v in best_solution.connected_components_per_vehicle
                ]
            )
            == 1
        ):
            finished = True
        else:
            finished = stopping or nb_iteration > nb_iteration_max

        while not finished:
            rebuilt_dict = best_solution.rebuilt_dict
            if (json_dump_folder is not None) and all(
                rebuilt_dict[v] is not None for v in rebuilt_dict
            ):
                json.dump(
                    rebuilt_dict,
                    open(
                        os.path.join(
                            json_dump_folder,
                            "res_"
                            + str(nb_iteration)
                            + "_"
                            + str(time.time_ns())
                            + ".json",
                        ),
                        "w",
                    ),
                    indent=4,
                )
            c = ConstraintHandlerOrWarmStart(
                linear_solver=self, problem=self.problem, do_lns=do_lns
            )
            c.adding_constraint(rebuilt_dict)
            if warm_start is not None and all(
                rebuilt_dict[v] is None for v in rebuilt_dict
            ):
                c.adding_constraint(warm_start)
            solutions = self.solve_one_iteration(
                parameters_milp=parameters_milp,
                time_limit=time_limit_subsolver,
                **kwargs,
            )
            best_solution = min(solutions, key=lambda sol: sol.obj)
            if not self.lazy:
                # contraints added in `solve_one_iteration` in lazy version
                if (
                    (best_solution.rebuilt_dict is None)
                    or (best_solution.connected_components_per_vehicle is None)
                    or (best_solution.component_global is None)
                ):
                    raise RuntimeError(
                        "Temporary result attributes rebuilt_dict, component_global"
                        "and connected_components_per_vehicle cannot be None after solving."
                    )
                subtour = SubtourAddingConstraint(
                    problem=self.problem,
                    linear_solver=self,
                    cluster=self.clusters_version,
                    do_order=self.subtour_do_order,
                    consider_only_first_component=self.subtour_consider_only_first_component,
                )
                logger.debug(len(best_solution.component_global))
                subtour.adding_component_constraints([best_solution])

            nb_iteration += 1
            # store solutions
            res.extend(self.convert_temporaryresults(solutions))
            # early stopping?
            stopping = callbacks_list.on_step_end(
                step=nb_iteration, res=res, solver=self
            )
            # end condition
            if (
                max(
                    [
                        len(best_solution.connected_components_per_vehicle[v])
                        for v in best_solution.connected_components_per_vehicle
                    ]
                )
                == 1
                and not do_lns
            ):
                finished = True
            else:
                finished = stopping or nb_iteration > nb_iteration_max

        # end of solve callback
        callbacks_list.on_solve_end(res=res, solver=self)
        return res

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit_subsolver: Optional[float] = 30.0,
        do_lns: bool = True,
        nb_iteration_max: int = 10,
        json_dump_folder: Optional[str] = None,
        warm_start: Optional[dict[Any, Any]] = None,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        return self.solve_iterative(
            parameters_milp=parameters_milp,
            time_limit_subsolver=time_limit_subsolver,
            do_lns=do_lns,
            nb_iteration_max=nb_iteration_max,
            json_dump_folder=json_dump_folder,
            warm_start=warm_start,
            callbacks=callbacks,
            **kwargs,
        )

    def convert_temporaryresults(
        self, temporary_results: list[TemporaryResult]
    ) -> list[tuple[Solution, Union[float, TupleFitness]]]:
        list_solution_fits: list[tuple[Solution, Union[float, TupleFitness]]] = []
        for temporaryresult in temporary_results:
            solution = convert_temporaryresult_to_gpdpsolution(
                temporaryresult=temporaryresult, problem=self.problem
            )
            fit = self.aggreg_from_sol(solution)
            list_solution_fits.append((solution, fit))
        return list_solution_fits


class TemporaryResultGurobiCallback(GurobiCallback):
    def __init__(self, do_solver: GurobiLinearFlowGpdpSolver):
        self.do_solver = do_solver
        self.temporary_results = []

    def __call__(self, model, where) -> None:
        if where == gurobipy.GRB.Callback.MIPSOL:
            try:
                # retrieve and store new solution
                self.temporary_results.append(
                    self.do_solver.retrieve_current_temporaryresult(
                        get_var_value_for_current_solution=model.cbGetSolution,
                        get_obj_value_for_current_solution=lambda: model.cbGet(
                            gurobipy.GRB.Callback.MIPSOL_OBJ
                        ),
                    )
                )
            except Exception as e:
                # catch exceptions because gurobi ignore them and do not stop solving
                self.do_solver.early_stopping_exception = e
                model.terminate()


class GurobiLinearFlowGpdpSolver(GurobiMilpSolver, BaseLinearFlowGpdpSolver):
    model: Optional["gurobipy.Model"] = None

    def init_model(self, **kwargs):
        BaseLinearFlowGpdpSolver.init_model(self, **kwargs)
        self.model.update()
        self.model.setParam("Heuristics", 0.5)
        self.model.setParam(gurobipy.GRB.Param.Method, 2)

    def solve_one_iteration(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> list[TemporaryResult]:

        gurobi_callback = TemporaryResultGurobiCallback(do_solver=self)
        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            gurobi_callback=gurobi_callback,
            **kwargs,
        )

        return gurobi_callback.temporary_results

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit_subsolver: Optional[float] = 30.0,
        do_lns: bool = True,
        nb_iteration_max: int = 10,
        json_dump_folder: Optional[str] = None,
        warm_start: Optional[dict[Any, Any]] = None,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        return BaseLinearFlowGpdpSolver.solve(
            self,
            parameters_milp=parameters_milp,
            time_limit_subsolver=time_limit_subsolver,
            do_lns=do_lns,
            nb_iteration_max=nb_iteration_max,
            json_dump_folder=json_dump_folder,
            warm_start=warm_start,
            callbacks=callbacks,
            **kwargs,
        )

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[gurobipy.Var, float]:
        BaseLinearFlowGpdpSolver.convert_to_variable_values(self, solution=solution)

    def set_warm_start(self, solution: GpdpSolution) -> None:
        BaseLinearFlowGpdpSolver.set_warm_start(self, solution)


class TemporaryResultMathOptCallback(MathOptCallback):
    def __init__(
        self,
        do_solver: MathOptLinearFlowGpdpSolver,
        mathopt_solver_type: mathopt.SolverType,
    ):
        self.do_solver = do_solver
        self.mathopt_solver_type = mathopt_solver_type
        self.temporary_results = []

    def __call__(self, callback_data: mathopt.CallbackData) -> mathopt.CallbackResult:
        cb_sol = callback_data.solution
        try:
            # retrieve and store new solution
            get_var_value_for_current_solution = lambda var: cb_sol[var]
            if self.do_solver.has_quadratic_objective:
                get_obj_value_for_current_solution = lambda: self.do_solver.model.objective.as_quadratic_expression().evaluate(
                    cb_sol
                )
            else:
                get_obj_value_for_current_solution = lambda: self.do_solver.model.objective.as_linear_expression().evaluate(
                    cb_sol
                )
            self.temporary_results.append(
                self.do_solver.retrieve_current_temporaryresult(
                    get_var_value_for_current_solution=get_var_value_for_current_solution,
                    get_obj_value_for_current_solution=get_obj_value_for_current_solution,
                )
            )
        except Exception as e:
            # catch exceptions because gurobi ignore them and do not stop solving
            self.do_solver.early_stopping_exception = e
            stopping = True
        else:
            stopping = False
        return mathopt.CallbackResult(terminate=stopping)


class MathOptLinearFlowGpdpSolver(OrtoolsMathOptMilpSolver, BaseLinearFlowGpdpSolver):
    def solve_one_iteration(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        mathopt_solver_type: mathopt.SolverType = mathopt.SolverType.CP_SAT,
        mathopt_enable_output: bool = False,
        mathopt_model_parameters: Optional[mathopt.ModelSolveParameters] = None,
        mathopt_additional_solve_parameters: Optional[mathopt.SolveParameters] = None,
        **kwargs: Any,
    ) -> list[TemporaryResult]:

        mathopt_cb = TemporaryResultMathOptCallback(
            do_solver=self, mathopt_solver_type=mathopt_solver_type
        )
        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            mathopt_solver_type=mathopt_solver_type,
            mathopt_cb=mathopt_cb,
            mathopt_enable_output=mathopt_enable_output,
            mathopt_model_parameters=mathopt_model_parameters,
            mathopt_additional_solve_parameters=mathopt_additional_solve_parameters,
            **kwargs,
        )

        return mathopt_cb.temporary_results

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit_subsolver: Optional[float] = 30.0,
        do_lns: bool = True,
        nb_iteration_max: int = 10,
        json_dump_folder: Optional[str] = None,
        warm_start: Optional[dict[Any, Any]] = None,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        return BaseLinearFlowGpdpSolver.solve(
            self,
            parameters_milp=parameters_milp,
            time_limit_subsolver=time_limit_subsolver,
            do_lns=do_lns,
            nb_iteration_max=nb_iteration_max,
            json_dump_folder=json_dump_folder,
            warm_start=warm_start,
            callbacks=callbacks,
            **kwargs,
        )

    def convert_to_variable_values(
        self, solution: Solution
    ) -> dict[mathopt.Variable, float]:
        BaseLinearFlowGpdpSolver.convert_to_variable_values(self, solution=solution)

    def set_warm_start(self, solution: GpdpSolution) -> None:
        BaseLinearFlowGpdpSolver.set_warm_start(self, solution)


def construct_edges_in_out_dict(
    variables_edges: dict[int, dict[Edge, Any]], clusters_dict: dict[Node, Hashable]
) -> tuple[
    dict[Node, set[tuple[int, Edge]]],
    dict[Node, set[tuple[int, Edge]]],
    dict[int, dict[Node, set[Edge]]],
    dict[int, dict[Node, set[Edge]]],
    dict[Hashable, set[tuple[int, Edge]]],
    dict[Hashable, set[tuple[int, Edge]]],
    dict[int, dict[Hashable, set[Edge]]],
    dict[int, dict[Hashable, set[Edge]]],
]:
    edges_in_all_vehicles: dict[Node, set[tuple[int, Edge]]] = {}
    edges_out_all_vehicles: dict[Node, set[tuple[int, Edge]]] = {}
    edges_in_per_vehicles: dict[int, dict[Node, set[Edge]]] = {}
    edges_out_per_vehicles: dict[int, dict[Node, set[Edge]]] = {}
    edges_in_all_vehicles_cluster: dict[Hashable, set[tuple[int, Edge]]] = {}
    edges_out_all_vehicles_cluster: dict[Hashable, set[tuple[int, Edge]]] = {}
    edges_in_per_vehicles_cluster: dict[int, dict[Hashable, set[Edge]]] = {}
    edges_out_per_vehicles_cluster: dict[int, dict[Hashable, set[Edge]]] = {}

    for vehicle in variables_edges:
        edges_in_per_vehicles[vehicle] = {}
        edges_out_per_vehicles[vehicle] = {}
        edges_in_per_vehicles_cluster[vehicle] = {}
        edges_out_per_vehicles_cluster[vehicle] = {}
        for edge in variables_edges[vehicle]:
            out_ = edge[0]
            in_ = edge[1]
            if out_ not in edges_out_per_vehicles[vehicle]:
                edges_out_per_vehicles[vehicle][out_] = set()
            if in_ not in edges_in_per_vehicles[vehicle]:
                edges_in_per_vehicles[vehicle][in_] = set()
            if out_ not in edges_out_all_vehicles:
                edges_out_all_vehicles[out_] = set()
            if in_ not in edges_in_all_vehicles:
                edges_in_all_vehicles[in_] = set()
            if (
                out_ in clusters_dict
                and clusters_dict[out_] not in edges_out_per_vehicles_cluster[vehicle]
            ):
                edges_out_per_vehicles_cluster[vehicle][clusters_dict[out_]] = set()
            if (
                in_ in clusters_dict
                and clusters_dict[in_] not in edges_in_per_vehicles_cluster[vehicle]
            ):
                edges_in_per_vehicles_cluster[vehicle][clusters_dict[in_]] = set()
            if (
                out_ in clusters_dict
                and clusters_dict[out_] not in edges_out_all_vehicles_cluster
            ):
                edges_out_all_vehicles_cluster[clusters_dict[out_]] = set()
            if (
                in_ in clusters_dict
                and clusters_dict[in_] not in edges_in_all_vehicles_cluster
            ):
                edges_in_all_vehicles_cluster[clusters_dict[in_]] = set()
            edges_out_all_vehicles[out_].add((vehicle, edge))
            edges_in_all_vehicles[in_].add((vehicle, edge))
            edges_in_per_vehicles[vehicle][in_].add(edge)
            edges_out_per_vehicles[vehicle][out_].add(edge)
            if out_ in clusters_dict:
                edges_out_all_vehicles_cluster[clusters_dict[out_]].add((vehicle, edge))
                edges_out_per_vehicles_cluster[vehicle][clusters_dict[out_]].add(edge)
            if in_ in clusters_dict:
                edges_in_all_vehicles_cluster[clusters_dict[in_]].add((vehicle, edge))
                edges_in_per_vehicles_cluster[vehicle][clusters_dict[in_]].add(edge)
    return (
        edges_in_all_vehicles,
        edges_out_all_vehicles,
        edges_in_per_vehicles,
        edges_out_per_vehicles,
        edges_in_all_vehicles_cluster,
        edges_out_all_vehicles_cluster,
        edges_in_per_vehicles_cluster,
        edges_out_per_vehicles_cluster,
    )


def build_path_from_vehicle_type_flow(
    result_from_retrieve: dict[str, dict[Hashable, Any]],
    problem: GpdpProblem,
) -> dict[int, list[Node]]:
    vehicle_per_type = problem.group_identical_vehicles
    solution: dict[int, list[Node]] = {}
    flow_solution = result_from_retrieve["variables_edges"]
    for group_vehicle in vehicle_per_type:
        edges_active = flow_solution[group_vehicle]
        origin = problem.origin_vehicle[
            problem.group_identical_vehicles[group_vehicle][0]
        ]
        edges_active_origin = [e for e in edges_active if e[0] == origin]
        for i in range(len(edges_active_origin)):
            cur_vehicle = problem.group_identical_vehicles[group_vehicle][i]
            solution[cur_vehicle] = [origin, edges_active_origin[i][1]]
            cur_node = edges_active_origin[i][1]
            while cur_node != problem.target_vehicle[cur_vehicle]:
                active_edge = [e for e in edges_active if e[0] == cur_node]
                cur_node = active_edge[0][1]
                solution[cur_vehicle] += [cur_node]
    return solution


class GurobiLazyConstraintLinearFlowGpdpSolver(GurobiLinearFlowGpdpSolver):
    lazy = True

    def solve_one_iteration(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> list[TemporaryResult]:
        self.prepare_model(
            parameters_milp=parameters_milp, time_limit=time_limit, **kwargs
        )
        if self.model is None:  # for mypy
            raise RuntimeError(
                "self.model must not be None after self.prepare_model()."
            )
        if self.problem.graph is None:
            self.problem.compute_graph()
            if self.problem.graph is None:
                raise RuntimeError(
                    "self.problem.graph cannot be None "
                    "after calling self.problem.compute_graph()."
                )
        if "warm_start" in kwargs and not kwargs.get("no_warm_start", False):
            warm_start: dict[int, list[Node]] = kwargs.get("warm_start", {})
            c = ConstraintHandlerOrWarmStart(
                linear_solver=self, problem=self.problem, do_lns=False
            )
            c.adding_constraint(warm_start)
        logger.info("optimizing...")
        indexes_edges = [
            (v, e)
            for v in self.variable_decisions["variables_edges"]
            for e in self.variable_decisions["variables_edges"][v]
        ]

        def callback(model: gurobipy.Model, where: int) -> None:
            if where == gurobipy.GRB.Callback.MIPSOL:
                if self.problem.graph is None:
                    raise RuntimeError(
                        "self.problem.graph cannot be None at this point"
                    )
                solution = model.cbGetSolution(
                    [
                        self.variable_decisions["variables_edges"][x[0]][x[1]]
                        for x in indexes_edges
                    ]
                )
                flow_solution: dict[int, dict[int, int]] = {
                    v: {} for v in range(self.problem.number_vehicle)
                }
                cost = {v: 0 for v in range(self.problem.number_vehicle)}
                for k in range(len(solution)):
                    if solution[k] > 0.5:
                        flow_solution[indexes_edges[k][0]][indexes_edges[k][1]] = 1
                        cost[
                            indexes_edges[k][0]
                        ] += self.problem.graph.edges_infos_dict[indexes_edges[k][1]][
                            "distance"
                        ]
                list_temporary_results = build_graph_solutions(
                    solutions=[({"variables_edges": flow_solution}, 0)],
                    graph=self.problem.graph,
                )
                temporary_result = list_temporary_results[0]
                reevaluate_result(temporary_result, problem=self.problem)
                vehicle_keys = temporary_result.graph_vehicle.keys()
                connected_components = {
                    v: [
                        (cyc, len(cyc))
                        for cyc in nx.simple_cycles(temporary_result.graph_vehicle[v])
                    ]
                    for v in vehicle_keys
                }
                logger.debug(cost)
                logger.debug(
                    f"Cycles : {[len(connected_components[v]) for v in connected_components]}"
                )
                sorted_connected_component = {
                    v: sorted(
                        connected_components[v], key=lambda x: x[1], reverse=False
                    )
                    for v in connected_components
                }
                temporary_result.connected_components_per_vehicle = (
                    sorted_connected_component
                )

                if temporary_result.paths_component is not None:
                    for v in temporary_result.paths_component:
                        if len(temporary_result.paths_component[v]) > 1:
                            for p_ind in temporary_result.paths_component[v]:
                                pp = temporary_result.paths_component[v][p_ind] + [
                                    temporary_result.paths_component[v][p_ind][0]
                                ]
                                if self.problem.target_vehicle[v] not in pp:
                                    for vv in temporary_result.paths_component:
                                        if all(
                                            (e0, e1)
                                            in self.variable_decisions[
                                                "variables_edges"
                                            ][vv]
                                            for e0, e1 in zip(pp[:-1], pp[1:])
                                        ):
                                            keys = [
                                                (e0, e1)
                                                for e0, e1 in zip(pp[:-1], pp[1:])
                                            ]
                                            model.cbLazy(
                                                self.construct_linear_sum(
                                                    [
                                                        self.variable_decisions[
                                                            "variables_edges"
                                                        ][vv][key]
                                                        for key in keys
                                                    ]
                                                )
                                                <= len(pp) - 2
                                            )
                for v in sorted_connected_component:
                    if len(sorted_connected_component[v]) > 1:
                        for s, l in sorted_connected_component[v]:
                            keys = [(e0, e1) for e0, e1 in zip(s[:-1], s[1:])] + [
                                (s[-1], s[0])
                            ]
                            model.cbLazy(
                                self.construct_linear_sum(
                                    [
                                        self.variable_decisions["variables_edges"][v][
                                            key
                                        ]
                                        for key in keys
                                    ]
                                )
                                <= len(keys) - 1
                            )

                subtour = SubtourAddingConstraint(
                    problem=self.problem,
                    linear_solver=self,
                    lazy=True,
                    cluster=self.clusters_version,
                    do_order=self.subtour_do_order,
                    consider_only_first_component=self.subtour_consider_only_first_component,
                )
                subtour.adding_component_constraints([temporary_result])

        self.model.Params.lazyConstraints = 1
        self.model.optimize(callback)
        logger.info(f"Problem has {self.model.NumObj} objectives")
        logger.info(f"Solver found {self.model.SolCount} solutions")
        logger.info(f"Objective : {self.model.getObjective().getValue()}")

        list_temporary_results: list[TemporaryResult] = []
        for i in range(self.nb_solutions - 1, -1, -1):
            list_temporary_results.append(self.retrieve_ith_temporaryresult(i=i))

        return list_temporary_results

    def retrieve_ith_temporaryresult(self, i: int) -> TemporaryResult:
        get_var_value_for_current_solution = (
            lambda var: self.get_var_value_for_ith_solution(var=var, i=i)
        )
        get_obj_value_for_current_solution = (
            lambda: self.get_obj_value_for_ith_solution(i=i)
        )
        return self.retrieve_current_temporaryresult(
            get_var_value_for_current_solution=get_var_value_for_current_solution,
            get_obj_value_for_current_solution=get_obj_value_for_current_solution,
        )


class SubtourAddingConstraint:
    def __init__(
        self,
        problem: GpdpProblem,
        linear_solver: BaseLinearFlowGpdpSolver,
        lazy: bool = False,
        cluster: bool = False,
        do_order: bool = False,
        consider_only_first_component: bool = True,
    ):
        self.consider_only_first_component = consider_only_first_component
        self.do_order = do_order
        self.cluster = cluster
        self.problem = problem
        self.linear_solver = linear_solver
        self.lazy = lazy

    def adding_component_constraints(
        self, list_solution: list[TemporaryResult]
    ) -> None:
        if self.linear_solver.model is None:
            raise RuntimeError("self.linear_solver.model cannot be None at this point.")
        c = []
        for l in list_solution:
            if (l.paths_component is None) or (
                l.connected_components_per_vehicle is None
            ):
                raise RuntimeError(
                    "Temporary result attributes paths_component"
                    "and connected_components_per_vehicle cannot be None after solving."
                )
            for v in l.connected_components_per_vehicle:
                if not self.lazy and not self.cluster:
                    if len(l.connected_components_per_vehicle[v]) > 1:
                        ind = 0
                        for p_ind in l.paths_component[v]:
                            pp = l.paths_component[v][p_ind] + [
                                l.paths_component[v][p_ind][0]
                            ]
                            for vv in self.linear_solver.variable_decisions[
                                "variables_edges"
                            ]:
                                if all(
                                    (e0, e1)
                                    in self.linear_solver.variable_decisions[
                                        "variables_edges"
                                    ][vv]
                                    for e0, e1 in zip(pp[:-1], pp[1:])
                                ):
                                    keys = [(e0, e1) for e0, e1 in zip(pp[:-1], pp[1:])]
                                    c += [
                                        self.linear_solver.add_linear_constraint(
                                            self.linear_solver.construct_linear_sum(
                                                [
                                                    self.linear_solver.variable_decisions[
                                                        "variables_edges"
                                                    ][
                                                        vv
                                                    ][
                                                        key
                                                    ]
                                                    for key in keys
                                                ]
                                            )
                                            <= len(pp) - 2
                                        )
                                    ]

                            ind += 1
                if self.cluster:
                    connected_components = []
                    for x in l.connected_components_per_vehicle[v]:
                        s = set([self.problem.clusters_dict[k] for k in x[0]])
                        connected_components += [(s, len(s))]
                else:
                    connected_components = l.connected_components_per_vehicle[v]
                c += update_model_generic(
                    problem=self.problem,
                    lp_solver=self.linear_solver,
                    components_global=connected_components,
                    edges_in_all_vehicles=self.linear_solver.edges_in_all_vehicles,
                    edges_out_all_vehicles=self.linear_solver.edges_out_all_vehicles,
                    lazy=self.lazy,
                    cluster=self.cluster,
                    do_order=self.do_order,
                    consider_only_first_component=self.consider_only_first_component,
                )
        if isinstance(self.linear_solver, GurobiMilpSolver):
            self.linear_solver.model.update()


class ConstraintHandlerOrWarmStart:
    def __init__(
        self,
        linear_solver: BaseLinearFlowGpdpSolver,
        problem: GpdpProblem,
        do_lns: bool = True,
    ):
        self.linear_solver = linear_solver
        self.problem = problem
        self.do_lns = do_lns
        self.index_tuple = [(0, 20) for v in range(self.problem.number_vehicle)]

    def adding_constraint(self, rebuilt_dict: dict[int, list[Node]]) -> None:
        if self.linear_solver.model is None:
            raise RuntimeError("self.linear_solver.model cannot be None at this point")
        vehicle_keys = self.linear_solver.variable_decisions["variables_edges"].keys()
        edges_to_add: dict[int, set[Edge]] = {
            v: set() for v in vehicle_keys if rebuilt_dict[v] is not None
        }
        for v in rebuilt_dict:
            if rebuilt_dict[v] is None:
                continue
            edges_to_add[v].update(
                {(e0, e1) for e0, e1 in zip(rebuilt_dict[v][:-1], rebuilt_dict[v][1:])}
            )
            edges_missing = {
                (v, e)
                for e in edges_to_add[v]
                if e not in self.linear_solver.variable_decisions["variables_edges"][v]
            }
            if len(edges_missing) > 0:
                logger.warning("Some edges are missing.")

        if self.do_lns:
            try:
                self.linear_solver.remove_constraints(
                    [
                        self.linear_solver.constraint_on_edge[iedge]
                        for iedge in self.linear_solver.constraint_on_edge
                    ]
                )
            except:
                pass
        self.linear_solver.constraint_on_edge = {}
        edges_to_constraint: dict[int, set[Edge]] = {
            v: set() for v in range(self.problem.number_vehicle)
        }
        vehicle_to_not_constraints = set(
            random.sample(
                range(self.problem.number_vehicle), min(3, self.problem.number_vehicle)
            )
        )
        for v in range(self.problem.number_vehicle):
            path = [
                (e0, e1) for e0, e1 in zip(rebuilt_dict[v][:-1], rebuilt_dict[v][1:])
            ]
            if v not in vehicle_to_not_constraints:
                edges_to_constraint[v].update(
                    set(
                        random.sample(
                            list(
                                self.linear_solver.variable_decisions[
                                    "variables_edges"
                                ][v]
                            ),
                            int(
                                0.5
                                * len(
                                    self.linear_solver.variable_decisions[
                                        "variables_edges"
                                    ][v]
                                )
                            ),
                        )
                    )
                )
                for p in path:
                    if p in edges_to_constraint[v]:
                        edges_to_constraint[v].remove(p)
            else:
                edges_to_constraint[v].update(
                    set(
                        random.sample(
                            list(
                                self.linear_solver.variable_decisions[
                                    "variables_edges"
                                ][v]
                            ),
                            int(
                                0.4
                                * len(
                                    self.linear_solver.variable_decisions[
                                        "variables_edges"
                                    ][v]
                                )
                            ),
                        )
                    )
                )
                j = random.randint(0, max(0, len(path) - 50))
                for p in path:
                    if p in edges_to_constraint[v]:
                        edges_to_constraint[v].remove(p)
                for p in path[j : min(len(path), j + 350)]:
                    for sets in [
                        self.linear_solver.edges_in_per_vehicles[v].get(p[0], set()),
                        self.linear_solver.edges_in_per_vehicles[v].get(p[1], set()),
                        self.linear_solver.edges_out_per_vehicles[v].get(p[0], set()),
                        self.linear_solver.edges_out_per_vehicles[v].get(p[1], set()),
                    ]:
                        for set_ in sets:
                            if set_ in edges_to_constraint[v]:
                                edges_to_constraint[v].remove(set_)

        nb_edges_to_constraint = sum(
            [len(edges_to_constraint[v]) for v in edges_to_constraint]
        )
        nb_edges_total = sum(
            [
                len(self.linear_solver.variable_decisions["variables_edges"][v])
                for v in self.linear_solver.variable_decisions["variables_edges"]
            ]
        )
        logger.debug(f"{nb_edges_to_constraint} edges constraint over {nb_edges_total}")
        iedge = 0
        hinted_values = {}
        for v in vehicle_keys:
            if rebuilt_dict[v] is not None:
                for e in self.linear_solver.variable_decisions["variables_edges"][v]:
                    if v in edges_to_add and e in edges_to_add[v]:
                        val = 1
                    else:
                        val = 0
                    hinted_values[
                        self.linear_solver.variable_decisions["variables_edges"][v][e]
                    ] = val
                    if self.do_lns:
                        if (
                            rebuilt_dict[v] is not None
                            and v in edges_to_constraint
                            and e in edges_to_constraint[v]
                        ):
                            self.linear_solver.constraint_on_edge[
                                iedge
                            ] = self.linear_solver.add_linear_constraint(
                                self.linear_solver.variable_decisions[
                                    "variables_edges"
                                ][v][e]
                                == val,
                                name="c_" + str(v) + "_" + str(e) + "_" + str(val),
                            )
                            iedge += 1
        self.linear_solver.set_warm_start_from_values(variable_values=hinted_values)
        if isinstance(self.linear_solver, GurobiMilpSolver):
            self.linear_solver.model.update()


def build_the_cycles(
    flow_solution: dict[Edge, int],
    component: set[Node],
    start_index: Node,
    end_index: Node,
) -> tuple[list[Node], dict[Node, int]]:
    edge_of_interest: set[Edge] = {
        e for e in flow_solution if e[1] in component and e[0] in component
    }
    innn: dict[Node, Edge] = {e[1]: e for e in edge_of_interest}
    outt: dict[Node, Edge] = {e[0]: e for e in edge_of_interest}

    some_node: Node
    if start_index in outt:
        some_node = start_index
    else:
        some_node = next(e[0] for e in edge_of_interest)
    end_node = some_node if end_index not in innn else end_index
    path: list[Node] = [some_node]
    cur_edge = outt[some_node]
    indexes: dict[Node, int] = {some_node: 0}
    cur_index = 1
    while cur_edge[1] != end_node:
        path += [cur_edge[1]]
        indexes[cur_edge[1]] = cur_index
        cur_index += 1
        cur_edge = outt[cur_edge[1]]
    if end_index in innn:
        path += [end_node]
        indexes[end_node] = cur_index
    return path, indexes


def rebuild_routine(
    sorted_connected_component: list[tuple[set[Node], int]],
    paths_component: dict[int, list[Node]],
    node_to_component: dict[Node, int],
    indexes: dict[int, dict[Node, int]],
    graph: nx.DiGraph,
    edges: set[Edge],
    evaluate_function_indexes: Callable[[Node, Node], float],
    start_index: Node,
    end_index: Node,
) -> list[Node]:
    rebuilded_path = list(
        paths_component[node_to_component[start_index]]
    )  # Initial path
    component_end = node_to_component[end_index]
    component_reconnected = {node_to_component[start_index]}
    path_set = set(rebuilded_path)
    total_length_path = len(rebuilded_path)
    while len(component_reconnected) < len(sorted_connected_component):
        if (
            len(component_reconnected) == len(sorted_connected_component) - 1
            and end_index != start_index
            and node_to_component[end_index] != node_to_component[start_index]
        ):
            rebuilded_path = rebuilded_path + paths_component[component_end]
            component_reconnected.add(component_end)
        else:
            index_path: dict[Node, list[int]] = {}
            for i in range(len(rebuilded_path)):
                if rebuilded_path[i] not in index_path:
                    index_path[rebuilded_path[i]] = []
                index_path[rebuilded_path[i]] += [i]
            edge_out_of_interest = {
                e for e in edges if e[0] in path_set and e[1] not in path_set
            }
            edge_in_of_interest = {
                e for e in edges if e[0] not in path_set and e[1] in path_set
            }
            min_out_edge = None
            min_in_edge = None
            min_index_in_path = None
            min_component = None
            min_dist = float("inf")
            backup_min_dist = float("inf")
            for e in edge_out_of_interest:
                index_in = index_path[e[0]][0]
                index_in_1 = min(index_path[e[0]][0] + 1, total_length_path - 1)
                next_node_1 = rebuilded_path[index_in_1]
                component_e1 = node_to_component[e[1]]
                if (
                    component_e1 == component_end
                    and len(component_reconnected) < len(sorted_connected_component) - 1
                ):
                    continue
                index_component_e1 = indexes[component_e1][e[1]]
                index_component_e1_plus1 = index_component_e1 + 1
                if index_component_e1_plus1 >= len(paths_component[component_e1]):
                    index_component_e1_plus1 = 0
                next_node_component_e1 = paths_component[component_e1][
                    index_component_e1_plus1
                ]
                if (
                    next_node_component_e1,
                    next_node_1,
                ) in edge_in_of_interest and next_node_1 in graph[e[0]]:
                    cost = (
                        graph[e[0]][e[1]]["distance"]
                        + graph[next_node_component_e1][next_node_1]["distance"]
                        - graph[e[0]][next_node_1]["distance"]
                    )
                    if cost < min_dist:
                        min_component = node_to_component[e[1]]
                        min_out_edge = e
                        min_in_edge = (next_node_component_e1, next_node_1)
                        min_index_in_path = index_in
                        min_dist = cost
                else:
                    cost = graph[e[0]][e[1]]["distance"]
                    if cost < backup_min_dist:
                        backup_min_dist = cost
            if len(edge_out_of_interest) == 0:
                return []
            if (
                (min_component is None)
                or (min_out_edge is None)
                or (min_in_edge is None)
                or (min_index_in_path is None)
            ):
                return []
            len_this_component = len(paths_component[min_component])
            logger.debug(list(range(0, -len_this_component, -1)))
            logger.debug(f"len this component : {len_this_component}")
            logger.debug(f"out edge : {min_out_edge}")
            logger.debug(f"in edge : {min_in_edge}")
            index_of_in_component = indexes[min_component][min_out_edge[1]]
            new_component = [
                paths_component[min_component][
                    (index_of_in_component + i) % len_this_component
                ]
                for i in range(0, -len_this_component, -1)
            ]
            logger.debug(f"path component {paths_component[min_component]}")
            logger.debug(f"New component : {new_component}")
            rebuilded_path = (
                rebuilded_path[: (min_index_in_path + 1)]
                + new_component
                + rebuilded_path[(min_index_in_path + 1) :]
            )
            for e1, e2 in zip(new_component[:-1], new_component[1:]):
                if (e1, e2) not in graph.edges():
                    graph.add_edge(e1, e2, distance=evaluate_function_indexes(e1, e2))
            path_set = set(rebuilded_path)
            total_length_path = len(rebuilded_path)
            component_reconnected.add(min_component)
    if rebuilded_path.index(end_index) != len(rebuilded_path) - 1:
        rebuilded_path.remove(end_index)
        rebuilded_path += [end_index]

    return rebuilded_path


def rebuild_routine_variant(
    sorted_connected_component: list[tuple[set[Node], int]],
    paths_component: dict[int, list[Node]],
    node_to_component: dict[Node, int],
    indexes: dict[int, dict[Node, int]],
    graph: nx.DiGraph,
    edges: set[Edge],
    evaluate_function_indexes: Callable[[Node, Node], float],
    start_index: Node,
    end_index: Node,
) -> list[Node]:
    rebuilded_path = list(
        paths_component[node_to_component[start_index]]
    )  # Initial path
    component_end = node_to_component[end_index]
    component_reconnected = {node_to_component[start_index]}
    path_set = set(rebuilded_path)
    while len(component_reconnected) < len(sorted_connected_component):
        if (
            len(component_reconnected) == len(sorted_connected_component) - 1
            and end_index != start_index
            and node_to_component[end_index] != node_to_component[start_index]
        ):
            rebuilded_path = rebuilded_path + paths_component[component_end]
            component_reconnected.add(component_end)
        else:
            index_path: dict[Node, list[int]] = {}
            for i in range(len(rebuilded_path)):
                if rebuilded_path[i] not in index_path:
                    index_path[rebuilded_path[i]] = []
                index_path[rebuilded_path[i]] += [i]
            edge_out_of_interest = {
                e for e in edges if e[0] == rebuilded_path[-1] and e[1] not in path_set
            }
            min_out_edge = None
            min_component = None
            min_dist = float("inf")
            for e in edge_out_of_interest:
                component_e1 = node_to_component[e[1]]
                if (
                    component_e1 == component_end
                    and len(component_reconnected) < len(sorted_connected_component) - 1
                ):
                    continue
                cost = graph[e[0]][e[1]]["distance"]
                if cost < min_dist:
                    min_component = node_to_component[e[1]]
                    min_out_edge = e
                    min_dist = cost
            if len(edge_out_of_interest) == 0:
                return []
            if (min_component is None) or (min_out_edge is None):
                return []
            len_this_component = len(paths_component[min_component])
            logger.debug(list(range(0, -len_this_component, -1)))
            logger.debug(f"len this component : {len_this_component}")
            logger.debug(f"out edge : {min_out_edge}")
            index_of_in_component = indexes[min_component][min_out_edge[1]]
            new_component = [
                paths_component[min_component][
                    (index_of_in_component + i) % len_this_component
                ]
                for i in range(0, -len_this_component, -1)
            ]
            logger.debug(f"path component {paths_component[min_component]}")
            logger.debug(f"New component : {new_component}")
            rebuilded_path = rebuilded_path + new_component
            path_set = set(rebuilded_path)
            component_reconnected.add(min_component)
    if rebuilded_path.index(end_index) != len(rebuilded_path) - 1:
        rebuilded_path.remove(end_index)
        rebuilded_path += [end_index]
    return rebuilded_path


def reevaluate_result(
    temporary_result: TemporaryResult,
    problem: GpdpProblem,
    variant_rebuilt: bool = True,
    possible_subcycle_in_component: bool = True,
) -> TemporaryResult:
    if problem.graph is None:
        raise RuntimeError(
            "problem.graph cannot be None " "when calling reeavaluate_result()."
        )
    if variant_rebuilt:
        rout = rebuild_routine_variant
    else:
        rout = rebuild_routine
    vehicle_keys = temporary_result.graph_vehicle.keys()
    connected_components: dict[int, list[tuple[set[Node], int]]] = {
        v: [
            (set(e), len(e))
            for e in nx.weakly_connected_components(temporary_result.graph_vehicle[v])
        ]
        for v in vehicle_keys
    }
    logger.debug(
        f"Connected component : {[len(connected_components[v]) for v in connected_components]}"
    )
    sorted_connected_component: dict[int, list[tuple[set[Node], int]]] = {
        v: sorted(connected_components[v], key=lambda x: x[1], reverse=True)
        for v in connected_components
    }
    paths_component: dict[int, dict[int, list[Node]]] = {v: {} for v in vehicle_keys}
    indexes_component: dict[int, dict[int, dict[Node, int]]] = {
        v: {} for v in temporary_result.graph_vehicle
    }
    node_to_component: dict[int, dict[Node, int]] = {
        v: {} for v in temporary_result.graph_vehicle
    }
    rebuilt_dict: dict[int, list[Node]] = {}
    component_global: list[tuple[set[Node], int]] = [
        (set(e), len(e))
        for e in nx.weakly_connected_components(temporary_result.graph_merge)
    ]
    if possible_subcycle_in_component:
        for v in connected_components:
            new_connected_components_v = []
            if len(connected_components[v]) > 1:
                for c, l in connected_components[v]:
                    graph_of_interest_i: nx.DiGraph = temporary_result.graph_vehicle[
                        v
                    ].subgraph(c)
                    cycles = [
                        (set(cyc), len(cyc))
                        for cyc in nx.simple_cycles(graph_of_interest_i)
                    ]
                    if len(cycles) >= 2:
                        new_connected_components_v += cycles
                        logger.debug(cycles)
                        temporary_result.rebuilt_dict = {
                            v: [] for v in connected_components
                        }
                        temporary_result.paths_component = None
                        temporary_result.connected_components_per_vehicle = (
                            connected_components
                        )
                        temporary_result.component_global = component_global
                        return temporary_result
                    else:
                        new_connected_components_v += [(c, l)]
                connected_components[v] = new_connected_components_v

    nb_component_global = len(component_global)
    logger.debug(f"Global : {nb_component_global}")
    for v in temporary_result.graph_vehicle:
        graph_of_interest: nx.DiGraph = nx.subgraph(
            problem.graph.graph_nx,
            [e[0] for e in temporary_result.flow_solution[v]]
            + [e[1] for e in temporary_result.flow_solution[v]],
        )
        graph_of_interest = graph_of_interest.copy()
        nb = len(sorted_connected_component[v])
        for i in range(nb):
            s = sorted_connected_component[v][i]
            paths_component[v][i], indexes_component[v][i] = build_the_cycles(
                flow_solution=temporary_result.flow_solution[v],
                component=s[0],
                start_index=problem.origin_vehicle[v],
                end_index=problem.target_vehicle[v],
            )
            node_to_component[v].update({p: i for p in paths_component[v][i]})

        def f(n1: Node, n2: Node) -> float:
            return problem.evaluate_function_node(n1, n2)

        rebuilt_dict[v] = rout(
            sorted_connected_component=sorted_connected_component[v],
            paths_component=paths_component[v],
            node_to_component=node_to_component[v],
            start_index=problem.origin_vehicle[v],
            end_index=problem.target_vehicle[v],
            indexes=indexes_component[v],
            graph=graph_of_interest,
            edges=set(graph_of_interest.edges()),
            evaluate_function_indexes=f,
        )
        if rebuilt_dict[v] is not None:
            logger.debug(rebuilt_dict[v])
            logger.debug(
                (
                    "distance :",
                    sum(
                        [
                            f(e, e1)
                            for e, e1 in zip(rebuilt_dict[v][:-1], rebuilt_dict[v][1:])
                        ]
                    ),
                )
            )
    temporary_result.paths_component = paths_component
    temporary_result.indexes_component = indexes_component
    temporary_result.rebuilt_dict = rebuilt_dict
    temporary_result.component_global = component_global
    temporary_result.connected_components_per_vehicle = connected_components
    return temporary_result


def build_graph_solution(
    results: dict[str, dict[Hashable, Any]], obj: float, graph: Graph
) -> TemporaryResult:
    current_solution = results["variables_edges"]
    vehicles_keys = current_solution.keys()
    g_vehicle = {v: nx.DiGraph() for v in vehicles_keys}
    g_merge = nx.DiGraph()
    for vehicle in current_solution:
        for e in current_solution[vehicle]:
            clients = e[0], e[1]
            if clients[0] not in g_vehicle[vehicle]:
                g_vehicle[vehicle].add_node(clients[0])
            if clients[1] not in g_vehicle[vehicle]:
                g_vehicle[vehicle].add_node(clients[1])
            if clients[0] not in g_merge:
                g_merge.add_node(clients[0])
            if clients[1] not in g_merge:
                g_merge.add_node(clients[1])
            g_vehicle[vehicle].add_edge(
                clients[0],
                clients[1],
                weight=graph.edges_infos_dict[clients]["distance"],
            )
            g_merge.add_edge(
                clients[0],
                clients[1],
                weight=graph.edges_infos_dict[clients]["distance"],
            )
    return TemporaryResult(
        **{
            "graph_merge": g_merge.copy(),
            "graph_vehicle": g_vehicle,
            "flow_solution": current_solution,
            "obj": obj,
            "all_variables": results,
        }
    )


def build_graph_solutions(
    solutions: list[tuple[dict, float]], graph: Graph
) -> list[TemporaryResult]:
    n_solutions = len(solutions)
    transformed_solutions = []
    for s in range(n_solutions):
        results, obj = solutions[s]
        transformed_solutions += [
            build_graph_solution(results=results, obj=obj, graph=graph)
        ]
    return transformed_solutions


def update_model_generic(
    problem: GpdpProblem,
    lp_solver: BaseLinearFlowGpdpSolver,
    components_global: list[tuple[set[Node], int]],
    edges_in_all_vehicles: dict[Node, set[tuple[int, Edge]]],
    edges_out_all_vehicles: dict[Node, set[tuple[int, Edge]]],
    lazy: bool = False,
    cluster: bool = False,
    consider_only_first_component: bool = True,
    do_order: bool = False,
) -> list[Any]:
    if lp_solver.model is None:
        raise RuntimeError("self.lp_solver.model cannot be None at this point.")
    if lazy:
        if isinstance(lp_solver, GurobiMilpSolver):
            add_constraint_fn = lp_solver.model.cbLazy
        else:
            raise NotImplementedError(
                "Lazy constraints are implemented only for the gurobi solvers."
            )
    else:
        add_constraint_fn = lp_solver.add_linear_constraint
    if cluster:
        cluster_label_fn = lambda node: problem.clusters_dict[node]
    else:
        cluster_label_fn = lambda node: node
    len_component_global = len(components_global)
    list_constraints: list[Any] = []
    if len_component_global > 1:
        logger.debug(f"Nb component : {len_component_global}")
        if consider_only_first_component:
            components_global_to_consider = components_global[:1]
        else:
            components_global_to_consider = components_global

        for s in components_global_to_consider:
            edge_in_of_interest = [
                e
                for n in s[0]
                for e in edges_in_all_vehicles.get(n, set())
                if cluster_label_fn(e[1][0]) not in s[0]
                and cluster_label_fn(e[1][1]) in s[0]
            ]
            edge_out_of_interest = [
                e
                for n in s[0]
                for e in edges_out_all_vehicles.get(n, set())
                if cluster_label_fn(e[1][0]) in s[0]
                and cluster_label_fn(e[1][1]) not in s[0]
            ]
            if not any(
                cluster_label_fn(problem.target_vehicle[v]) in s[0]
                for v in problem.origin_vehicle
            ):
                list_constraints += [
                    add_constraint_fn(
                        lp_solver.construct_linear_sum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_out_of_interest
                            ]
                        )
                        >= 1,
                    )
                ]
            if not any(
                problem.origin_vehicle[v] in s[0] for v in problem.origin_vehicle
            ):
                list_constraints += [
                    add_constraint_fn(
                        lp_solver.construct_linear_sum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_in_of_interest
                            ]
                        )
                        >= 1,
                    )
                ]
    if do_order and len_component_global > 1:
        lp_solver.add_order_constraints(
            nodes=[int(node) for node in min(components_global, key=lambda x: x[1])[0]],
            lazy=lazy,
        )

    if isinstance(lp_solver, GurobiMilpSolver):
        lp_solver.model.update()
    return list_constraints
