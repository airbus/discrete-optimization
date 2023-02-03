#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import mip
import networkx as nx

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lp_tools import ParametersMilp, PymipMilpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    TupleFitness,
)
from discrete_optimization.pickup_vrp.gpdp import GPDP, Edge, Node
from discrete_optimization.pickup_vrp.solver.lp_solver import (
    TemporaryResult,
    build_graph_solution,
    build_path_from_vehicle_type_flow,
    construct_edges_in_out_dict,
    convert_temporaryresult_to_gpdpsolution,
    reevaluate_result,
)

logger = logging.getLogger(__name__)


class MipModelException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return self.message


def retrieve_ith_solution(
    i: int, model: mip.Model, variable_decisions: Dict[str, Any]
) -> Tuple[Dict[str, Dict[Hashable, Any]], float]:
    results: Dict[str, Dict[Hashable, Any]] = {}
    xsolution: Dict[int, Dict[Edge, int]] = {
        v: {} for v in variable_decisions["variables_edges"]
    }
    obj = float(model.objective_values[i])
    for vehicle in variable_decisions["variables_edges"]:
        for edge in variable_decisions["variables_edges"][vehicle]:
            value = variable_decisions["variables_edges"][vehicle][edge].xi(i)
            if value <= 0.1:
                continue
            xsolution[vehicle][edge] = 1
    results["variables_edges"] = cast(Dict[Hashable, Any], xsolution)
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
                            value = variable_decisions[key][key_2][key_3].xi(i)
                            results[key][key_2][key_3][key_4] = value
                    else:
                        value = variable_decisions[key][key_2][key_3].xi(i)
                        results[key][key_2][key_3] = value
            else:
                value = variable_decisions[key][key_2].xi(i)
                results[key][key_2] = value
    return results, obj


class LinearFlowSolver(PymipMilpSolver):
    def __init__(
        self,
        problem: GPDP,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=problem,
            params_objective_function=params_objective_function,
        )
        self.model: Optional[mip.Model] = None
        self.constraint_on_edge: Dict[int, Any] = {}
        self.variable_order: Dict[Node, Any] = {}

    def one_visit_per_node(
        self,
        model: mip.Model,
        nodes_of_interest: Iterable[Node],
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> None:
        constraint_one_visit = {}
        for node in nodes_of_interest:
            constraint_one_visit[node] = model.add_constr(
                mip.quicksum(
                    [variables_edges[x[0]][x[1]] for x in edges_in_all_vehicles[node]]
                )
                == 1,
                name="visit_" + str(node),
            )
            constraint_one_visit[(node, "out")] = model.add_constr(
                mip.quicksum(
                    [variables_edges[x[0]][x[1]] for x in edges_out_all_vehicles[node]]
                )
                == 1,
                name="visitout_" + str(node),
            )

    def one_visit_per_clusters(
        self,
        model: mip.Model,
        nodes_of_interest: Iterable[Node],
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> None:
        constraint_cluster: Dict[Hashable, Any] = {}
        for cluster in self.problem.clusters_to_node:
            constraint_cluster[cluster] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[x[0]][x[1]]
                        for node in self.problem.clusters_to_node[cluster]
                        for x in edges_in_all_vehicles[node]
                        if node in nodes_of_interest
                        and x[0] in self.problem.vehicles_representative
                    ]
                )
                >= 1,
                name="visit_" + str(cluster),
            )
            constraint_cluster[(cluster, "out")] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[x[0]][x[1]]
                        for node in self.problem.clusters_to_node[cluster]
                        for x in edges_out_all_vehicles[node]
                        if node in nodes_of_interest
                        and x[0] in self.problem.vehicles_representative
                    ]
                )
                == 1,
                name="visitout_" + str(cluster),
            )

    def resources_constraint(
        self,
        model: mip.Model,
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> Dict[str, Dict[str, Dict[Node, Any]]]:
        resources = self.problem.resources_set
        resources_variable_coming = {
            r: {
                node: model.add_var(
                    var_type=mip.CONTINUOUS,
                    name="res_coming_" + str(r) + "_" + str(node),
                )
                for node in edges_in_all_vehicles
            }
            for r in resources
        }
        resources_variable_leaving = {
            r: {
                node: model.add_var(
                    var_type=mip.CONTINUOUS,
                    name="res_leaving_" + str(r) + "_" + str(node),
                )
                for node in edges_in_all_vehicles
            }
            for r in resources
        }
        for v in self.problem.origin_vehicle:
            for r in resources_variable_coming:
                model.add_constr(
                    resources_variable_coming[r][self.problem.origin_vehicle[v]]
                    == self.problem.resources_flow_node[self.problem.origin_vehicle[v]][
                        r
                    ]
                )
                model.add_constr(
                    resources_variable_leaving[r][self.problem.origin_vehicle[v]]
                    == self.problem.resources_flow_node[self.problem.origin_vehicle[v]][
                        r
                    ]
                )
        all_origin = set(self.problem.origin_vehicle.values())
        for r in resources_variable_coming:
            for node in resources_variable_coming[r]:
                for vehicle, edge in edges_in_all_vehicles[node]:
                    if edge[0] == edge[1]:
                        continue
                    model.add_constr(
                        resources_variable_coming[r][node]
                        >= resources_variable_leaving[r][edge[0]]
                        + self.problem.resources_flow_edges[edge][r]
                        - 100000 * (1 - variables_edges[vehicle][edge])
                    )
                if node not in all_origin:
                    model.add_constr(
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
        model: mip.Model,
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        # Case where we don't have to track the resource flow etc...
        # we just want to check that we respect the capacity constraint (like in VRP with capacity)
        # we admit that the resource demand/consumption are positive.
        # corresponding to negative flow values in the problem definition.
        consumption_per_vehicle: Dict[int, Dict[str, Any]] = {
            v: {
                r: model.add_var(
                    var_type=mip.CONTINUOUS,
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
                model.add_constr(
                    consumption_per_vehicle[v][r]
                    == mip.quicksum(
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
                model.add_constr(
                    consumption_per_vehicle[v][r] <= self.problem.capacities[v][r][1],
                    name="constraint_capa_" + str(v) + "_" + str(r),
                )
        return {"consumption_per_vehicle": consumption_per_vehicle}

    def time_evolution(
        self,
        model: mip.Model,
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> Dict[str, Dict[Node, Any]]:
        time_coming = {
            node: model.add_var(
                var_type=mip.CONTINUOUS, name="time_coming_" + str(node)
            )
            for node in edges_in_all_vehicles
        }
        time_leaving = {
            node: model.add_var(
                var_type=mip.CONTINUOUS, name="time_leaving_" + str(node)
            )
            for node in edges_in_all_vehicles
        }
        for v in self.problem.origin_vehicle:
            model.add_constr(time_coming[self.problem.origin_vehicle[v]] == 0)
        all_origin = set(self.problem.origin_vehicle.values())
        for node in time_leaving:
            for vehicle, edge in edges_in_all_vehicles[node]:
                if edge[0] == edge[1]:
                    continue
                model.add_constr(
                    time_coming[node]
                    >= time_leaving[edge[0]]
                    + self.problem.time_delta[edge[0]][edge[1]]
                    - 100000 * (1 - variables_edges[vehicle][edge])
                )
            if node not in all_origin:
                model.add_constr(
                    time_leaving[node]
                    - time_coming[node]
                    - self.problem.time_delta_node[node]
                    >= 0
                )
        return {"time_coming": time_coming, "time_leaving": time_leaving}

    def init_model(self, **kwargs: Any) -> None:
        solver_name = kwargs.get("solver_name", mip.CBC)
        model: mip.Model = mip.Model(
            "GPDP-flow", sense=mip.MINIMIZE, solver_name=solver_name
        )
        include_backward = kwargs.get("include_backward", True)
        include_triangle = kwargs.get("include_triangle", False)
        include_subtour = kwargs.get("include_subtour", False)
        include_resources = kwargs.get("include_resources", False)
        include_capacity = kwargs.get("include_capacity", False)
        include_time_evolution = kwargs.get("include_time_evolution", False)
        one_visit_per_node = kwargs.get("one_visit_per_node", True)
        one_visit_per_cluster = kwargs.get("one_visit_per_cluster", False)
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
        variables_edges: Dict[int, Dict[Edge, Any]]
        if self.problem.any_grouping:
            variables_edges = {j: {} for j in range(nb_vehicle)}
            for k in self.problem.group_identical_vehicles:
                j = self.problem.group_identical_vehicles[k][0]
                variables_edges[j] = {
                    e: model.add_var(
                        var_type=mip.BINARY,
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
                    e: model.add_var(
                        var_type=mip.BINARY,
                        name="flow_" + str(j) + "_" + str(e),
                        obj=graph.edges_infos_dict[e]["distance"],
                    )
                    for e in self.problem.get_edges_for_vehicle(j)
                }
                for j in range(nb_vehicle)
            }
        self.nodes_of_interest = nodes_of_interest
        self.variables_edges = variables_edges
        all_variables: Dict[str, Dict[Any, Any]] = {"variables_edges": variables_edges}
        constraint_loop = {}
        for vehicle in variables_edges:
            for e in variables_edges[vehicle]:
                if e[0] == e[1]:
                    constraint_loop[(vehicle, e)] = model.add_constr(
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

        constraints_out_flow: Dict[Tuple[int, Node], Any] = {}
        constraints_in_flow: Dict[Union[Tuple[int, Node], Node], Any] = {}
        constraints_flow_conservation: Dict[
            Union[Tuple[int, Node], Tuple[int, Node, str]], Any
        ] = {}

        count_origin: Dict[Node, int] = {}
        count_target: Dict[Node, int] = {}
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
            constraints_out_flow[(vehicle, node_origin)] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[vehicle][edge]
                        for edge in edges_out_per_vehicles[vehicle].get(
                            node_origin, set()
                        )
                        if edge[1] != node_origin or same_node
                    ]
                )
                == count_origin[node_origin],
                name="outflow_" + str((vehicle, node_origin)),
            )
            constraints_in_flow[(vehicle, node_target)] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[vehicle][edge]
                        for edge in edges_in_per_vehicles[vehicle].get(
                            node_target, set()
                        )
                        if edge[0] != node_target or same_node
                    ]
                )
                == count_target[node_target],
                name="inflow_" + str((vehicle, node_target)),
            )  # Avoid loop
            constraints_out_flow[(vehicle, node_target)] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[vehicle][edge]
                        for edge in edges_out_per_vehicles[vehicle].get(
                            node_target, set()
                        )
                        if edge[1] != node_target or same_node
                    ]
                )
                <= 0,
                name="outflow_" + str((vehicle, node_target)),
            )

        for vehicle in range(nb_vehicle):
            origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            target = self.problem.target_vehicle[name_vehicles[vehicle]]
            model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[v][edge]
                        for v, edge in edges_out_all_vehicles[origin]
                        if self.problem.origin_vehicle[name_vehicles[v]] != origin
                    ]
                )
                == 0
            )
            model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[v][edge]
                        for v, edge in edges_in_all_vehicles[target]
                        if self.problem.target_vehicle[name_vehicles[v]] != target
                    ]
                )
                == 0
            )

        for vehicle in range(nb_vehicle):
            node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            node_target = self.problem.target_vehicle[name_vehicles[vehicle]]
            same_node = node_origin == node_target
            for node in edges_in_per_vehicles[vehicle]:
                if same_node or node not in {node_origin, node_target}:
                    constraints_flow_conservation[(vehicle, node)] = model.add_constr(
                        mip.quicksum(
                            [
                                variables_edges[vehicle][e]
                                for e in edges_in_per_vehicles[vehicle].get(node, set())
                                if e[1] != e[0]
                            ]
                            + [
                                -variables_edges[vehicle][e]
                                for e in edges_out_per_vehicles[vehicle].get(
                                    node, set()
                                )
                                if e[1] != e[0]
                            ]
                        )
                        == 0,
                        name="convflow_" + str((vehicle, node)),
                    )
                    constraints_flow_conservation[
                        (vehicle, node, "in")
                    ] = model.add_constr(
                        mip.quicksum(
                            [
                                variables_edges[vehicle][e]
                                for e in edges_in_per_vehicles[vehicle].get(node, set())
                                if e[1] != e[0]
                            ]
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
                        constraint_tour_2length[cnt_tour] = model.add_constr(
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
                                constraint_triangle[cnt_triangle] = model.add_constr(
                                    variables_edges[vehicle][(node, node_neigh)]
                                    + variables_edges[vehicle][
                                        (node_neigh, node_neigh_neigh)
                                    ]
                                    + variables_edges[vehicle][(node_neigh_neigh, node)]
                                    <= 2,
                                    name="triangle_" + str(cnt_triangle),
                                )
                                cnt_triangle += 1

        if include_subtour:
            variables_order = {
                node: model.add_var(
                    var_type=mip.CONTINUOUS,
                    lb=0,
                    ub=len(self.problem.all_nodes) + 1,
                    name="order_" + str(node),
                )
                for node in edges_in_all_vehicles
            }
            constraints_order = {}
            for vehicle in range(nb_vehicle):
                node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
                constraints_order[node_origin] = model.add_constr(
                    variables_order[node_origin] == 0, name="order_" + str(node_origin)
                )
            use_big_m = True
            use_indicator = False
            for node in variables_order:
                if node not in constraints_order:
                    for vehicle, edge in edges_in_all_vehicles[node]:
                        if use_big_m:
                            constraints_order[node] = model.add_constr(
                                variables_order[node]
                                >= variables_order[edge[0]]
                                + 1
                                - 1000 * (1 - variables_edges[vehicle][edge]),
                                name="order_" + str(node),
                            )
                        if use_indicator:
                            raise MipModelException(
                                "indicator constraint dont exist in pymip"
                            )

        for vehicle in range(nb_vehicle):
            for node in [self.problem.origin_vehicle[name_vehicles[vehicle]]]:
                if node in edges_in_all_vehicles:
                    constraints_in_flow[node] = model.add_constr(
                        mip.quicksum(
                            [
                                variables_edges[x[0]][x[1]]
                                for x in edges_in_all_vehicles[node]
                            ]
                        )
                        == 0,
                        name="no_inflow_start_" + str(node),
                    )

        if one_visit_per_node:
            self.one_visit_per_node(
                model=model,
                nodes_of_interest=nodes_of_interest,
                variables_edges=variables_edges,
                edges_in_all_vehicles=edges_in_all_vehicles,
                edges_out_all_vehicles=edges_out_all_vehicles,
            )
        if one_visit_per_cluster:
            self.one_visit_per_clusters(
                model=model,
                nodes_of_interest=nodes_of_interest,
                variables_edges=variables_edges,
                edges_in_all_vehicles=edges_in_all_vehicles,
                edges_out_all_vehicles=edges_out_all_vehicles,
            )
        if include_capacity:
            all_variables.update(
                self.simple_capacity_constraint(
                    model=model,
                    variables_edges=variables_edges,
                    edges_in_all_vehicles=edges_in_all_vehicles,
                    edges_out_all_vehicles=edges_out_all_vehicles,
                )
            )
        if include_resources:
            all_variables.update(
                self.resources_constraint(
                    model=model,
                    variables_edges=variables_edges,
                    edges_in_all_vehicles=edges_in_all_vehicles,
                    edges_out_all_vehicles=edges_out_all_vehicles,
                )
            )
        if include_time_evolution:
            all_variables.update(
                self.time_evolution(
                    model=model,
                    variables_edges=variables_edges,
                    edges_in_all_vehicles=edges_in_all_vehicles,
                    edges_out_all_vehicles=edges_out_all_vehicles,
                )
            )

        self.edges_in_all_vehicles: Dict[
            Node, Set[Tuple[int, Edge]]
        ] = edges_in_all_vehicles
        self.edges_out_all_vehicles: Dict[
            Node, Set[Tuple[int, Edge]]
        ] = edges_out_all_vehicles
        self.edges_in_per_vehicles: Dict[
            int, Dict[Node, Set[Edge]]
        ] = edges_in_per_vehicles
        self.edges_out_per_vehicles: Dict[
            int, Dict[Node, Set[Edge]]
        ] = edges_out_per_vehicles
        self.edges_in_all_vehicles_cluster: Dict[
            Hashable, Set[Tuple[int, Edge]]
        ] = edges_in_all_vehicles_cluster
        self.edges_out_all_vehicles_cluster: Dict[
            Hashable, Set[Tuple[int, Edge]]
        ] = edges_out_all_vehicles_cluster
        self.edges_in_per_vehicles_cluster: Dict[
            int, Dict[Hashable, Set[Edge]]
        ] = edges_in_per_vehicles_cluster
        self.edges_out_per_vehicles_cluster: Dict[
            int, Dict[Hashable, Set[Edge]]
        ] = edges_out_per_vehicles_cluster
        self.variable_decisions = all_variables
        self.model = model
        self.model.sense = mip.MINIMIZE
        self.solver_name = solver_name
        self.clusters_version = one_visit_per_cluster

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        """

        Not used here as GurobiMilpSolver.solve() is overriden


        """
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, self.nb_solutions)
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for s in range(n_solutions):
            temporaryresult = self.retrieve_ith_temporaryresult(i=s)
            solution = convert_temporaryresult_to_gpdpsolution(
                temporaryresult=temporaryresult, problem=self.problem
            )
            fit = self.aggreg_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            limit_store=False,
            mode_optim=self.params_objective_function.sense_function,
        )

    def retrieve_ith_temporaryresult(self, i: int) -> TemporaryResult:
        res, obj = retrieve_ith_solution(
            i=i, model=self.model, variable_decisions=self.variable_decisions
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

    def solve_one_iteration(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> List[TemporaryResult]:
        self.optimize_model(parameters_milp=parameters_milp, **kwargs)
        list_temporary_results: List[TemporaryResult] = []
        for i in range(self.nb_solutions):
            list_temporary_results.append(self.retrieve_ith_temporaryresult(i=i))

        return list_temporary_results

    def solve_iterative(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> List[TemporaryResult]:
        if self.model is None:
            self.init_model(**kwargs)
            if self.model is None:  # for mypy
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        finished = False
        do_lns = kwargs.get("do_lns", True)
        reinit_model_at_each_iteration = kwargs.get(
            "reinit", self.solver_name == mip.CBC
        )
        nb_iteration_max = kwargs.get("nb_iteration_max", 10)
        solutions: List[TemporaryResult] = self.solve_one_iteration(
            parameters_milp=parameters_milp, **kwargs
        )
        first_solution: TemporaryResult = solutions[0]
        if (
            (first_solution.rebuilt_dict is None)
            or (first_solution.connected_components_per_vehicle is None)
            or (first_solution.component_global is None)
        ):
            raise RuntimeError(
                "Temporary result attributes rebuilt_dict, component_global"
                "and connected_components_per_vehicle cannot be None after solving."
            )
        if reinit_model_at_each_iteration:
            self.model.reset()
            self.init_model(**kwargs)
        subtour: Union[SubtourAddingConstraint, SubtourAddingConstraintCluster]
        if self.clusters_version:
            subtour = SubtourAddingConstraintCluster(
                problem=self.problem, linear_solver=self
            )
        else:
            subtour = SubtourAddingConstraint(problem=self.problem, linear_solver=self)
        if (
            max(
                [
                    len(first_solution.connected_components_per_vehicle[v])
                    for v in first_solution.connected_components_per_vehicle
                ]
            )
            == 1
        ):
            return solutions
        list_constraints_tuples: List[Any] = []
        list_constraints_tuples += subtour.adding_component_constraints(
            [first_solution]
        )
        all_solutions = solutions
        nb_iteration = 0
        while not finished:
            rebuilt_dict = first_solution.rebuilt_dict
            c = ConstraintHandlerOrWarmStart(
                linear_solver=self, problem=self.problem, do_lns=do_lns
            )
            c.adding_constraint(rebuilt_dict)
            solutions = self.solve_one_iteration(
                parameters_milp=parameters_milp, **kwargs
            )
            first_solution = solutions[0]
            if (
                (first_solution.rebuilt_dict is None)
                or (first_solution.connected_components_per_vehicle is None)
                or (first_solution.component_global is None)
            ):
                raise RuntimeError(
                    "Temporary result attributes rebuilt_dict, component_global"
                    "and connected_components_per_vehicle cannot be None after solving."
                )
            all_solutions += solutions
            if self.clusters_version:
                subtour = SubtourAddingConstraintCluster(
                    problem=self.problem, linear_solver=self
                )
            else:
                subtour = SubtourAddingConstraint(
                    problem=self.problem, linear_solver=self
                )
            if reinit_model_at_each_iteration:
                self.model.reset()
                self.init_model(**kwargs)
                self.reapply_constraint(list_constraints_tuples)
                self.constraint_on_edge = {}
            list_constraints_tuples += subtour.adding_component_constraints(
                [first_solution]
            )
            if (
                max(
                    [
                        len(first_solution.connected_components_per_vehicle[v])
                        for v in first_solution.connected_components_per_vehicle
                    ]
                )
                == 1
                and not do_lns
            ):
                return all_solutions
            nb_iteration += 1
            finished = nb_iteration > nb_iteration_max
        return all_solutions

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        temporaryresults = self.solve_iterative(
            parameters_milp=parameters_milp,
            **kwargs,
        )
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, len(temporaryresults))
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for s in range(n_solutions):
            solution = convert_temporaryresult_to_gpdpsolution(
                temporaryresult=temporaryresults[s], problem=self.problem
            )
            fit = self.aggreg_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )

    def reapply_constraint(
        self, list_constraint_tuple: List[Tuple[Set[Tuple[int, int]], int]]
    ) -> None:
        if self.model is None:
            raise RuntimeError(
                "self.model must not be None when calling reapply_constraint()."
            )
        for edge, val in list_constraint_tuple:
            self.model.add_constr(
                mip.quicksum(
                    [
                        self.variable_decisions["variables_edges"][e[0]][e[1]]
                        for e in edge
                    ]
                )
                >= val
            )


# adapted for vehicle type optim and directed acyclic graph. (so typically fleet rotation optim)
class LinearFlowSolverVehicleType(PymipMilpSolver):
    def __init__(
        self,
        problem: GPDP,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        self.model: Optional[mip.Model] = None
        self.constraint_on_edge: Dict[int, Any] = {}
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=problem,
            params_objective_function=params_objective_function,
        )

    def one_visit_per_node(
        self,
        model: mip.Model,
        nodes_of_interest: Iterable[Node],
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> None:
        constraint_one_visit = {}
        for node in nodes_of_interest:
            constraint_one_visit[node] = model.add_constr(
                mip.quicksum(
                    [variables_edges[x[0]][x[1]] for x in edges_in_all_vehicles[node]]
                )
                == 1,
                name="visit_" + str(node),
            )
            constraint_one_visit[(node, "out")] = model.add_constr(
                mip.quicksum(
                    [variables_edges[x[0]][x[1]] for x in edges_out_all_vehicles[node]]
                )
                == 1,
                name="visitout_" + str(node),
            )

    def one_visit_per_clusters(
        self,
        model: mip.Model,
        nodes_of_interest: Iterable[Node],
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> None:
        constraint_cluster = {}
        for cluster in self.problem.clusters_to_node:
            constraint_cluster[cluster] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[x[0]][x[1]]
                        for node in self.problem.clusters_to_node[cluster]
                        for x in edges_in_all_vehicles[node]
                        if node in nodes_of_interest
                    ]
                )
                >= 1,
                name="visit_" + str(cluster),
            )
            constraint_cluster[(cluster, "out")] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[x[0]][x[1]]
                        for node in self.problem.clusters_to_node[cluster]
                        for x in edges_out_all_vehicles[node]
                        if node in nodes_of_interest
                    ]
                )
                == 1,
                name="visitout_" + str(cluster),
            )

    def simple_capacity_constraint(
        self,
        model: mip.Model,
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        # Case where we don't have to track the resource flow etc...
        # we just want to check that we respect the capacity constraint (like in VRP with capacity)
        # we admit that the resource demand/consumption are positive.
        # corresponding to negative flow values in the problem definition.
        consumption_per_vehicle = {
            v: {
                r: model.add_var(
                    var_type=mip.CONTINUOUS,
                    lb=0,
                    ub=len(self.problem.group_identical_vehicles[v])
                    * self.problem.capacities[
                        self.problem.group_identical_vehicles[v][0]
                    ][r][1],
                    name="consumption_v_" + str(v) + "_" + str(r),
                )
                for r in self.problem.resources_set
            }
            for v in self.problem.group_identical_vehicles
        }
        for v in consumption_per_vehicle:
            for r in consumption_per_vehicle[v]:
                model.add_constr(
                    consumption_per_vehicle[v][r]
                    == mip.quicksum(
                        variables_edges[v][e]
                        * (
                            -self.problem.resources_flow_node[e[1]][r]
                            - self.problem.resources_flow_edges[e][r]
                        )
                        for e in variables_edges[v]
                        if e[1]
                        != self.problem.origin_vehicle[
                            self.problem.group_identical_vehicles[v][0]
                        ]
                    )
                )
        return {"consumption_per_vehicle": consumption_per_vehicle}

    def time_evolution(
        self,
        model: mip.Model,
        variables_edges: Dict[int, Dict[Edge, Any]],
        edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
        edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    ) -> Dict[str, Dict[Node, Any]]:
        time_coming = {
            node: model.add_var(
                var_type=mip.CONTINUOUS, name="time_coming_" + str(node)
            )
            for node in edges_in_all_vehicles
        }
        time_leaving = {
            node: model.add_var(
                var_type=mip.CONTINUOUS, name="time_leaving_" + str(node)
            )
            for node in edges_in_all_vehicles
        }
        for v in self.problem.origin_vehicle:
            model.add_constr(time_coming[self.problem.origin_vehicle[v]] == 0)
        all_origin = set(self.problem.origin_vehicle.values())
        for node in time_leaving:
            for vehicle, edge in edges_in_all_vehicles[node]:
                model.add_constr(
                    time_coming[node]
                    >= time_leaving[edge[0]]
                    + self.problem.time_delta[edge[0]][edge[1]]
                    - 100000 * (1 - variables_edges[vehicle][edge])
                )
            if node not in all_origin:
                model.add_constr(
                    time_leaving[node]
                    - time_coming[node]
                    - self.problem.time_delta_node[node]
                    >= 0
                )
        return {"time_coming": time_coming, "time_leaving": time_leaving}

    def init_model(self, **kwargs: Any) -> None:
        model: mip.Model = mip.Model(
            "GPDP-flow", sense=mip.MINIMIZE, solver_name=mip.CBC
        )
        include_capacity = kwargs.get("include_capacity", False)
        include_time_evolution = kwargs.get("include_time_evolution", False)
        one_visit_per_node = kwargs.get("one_visit_per_node", True)
        one_visit_per_cluster = kwargs.get("one_visit_per_cluster", False)
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
        variables_edges: Dict[int, Dict[Edge, Any]]
        variables_edges = {j: {} for j in range(nb_vehicle)}
        for k in self.problem.group_identical_vehicles:
            j = self.problem.group_identical_vehicles[k][0]
            variables_edges[k] = {
                e: model.add_var(
                    var_type=mip.BINARY,
                    name="flow_" + str(k) + "_" + str(e),
                    obj=graph.edges_infos_dict[e]["distance"],
                )
                for e in self.problem.get_edges_for_vehicle(j)
            }
        self.nodes_of_interest = nodes_of_interest
        self.variables_edges = variables_edges
        all_variables: Dict[str, Dict[Any, Any]] = {"variables_edges": variables_edges}
        constraint_loop: Dict[Tuple[int, Edge], Any] = {}
        for group_vehicle in variables_edges:
            for e in variables_edges[group_vehicle]:
                if e[0] == e[1]:
                    constraint_loop[(group_vehicle, e)] = model.add_constr(
                        variables_edges[group_vehicle][e] == 0,
                        name="loop_" + str((group_vehicle, e)),
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

        constraints_out_flow: Dict[Tuple[int, Node], Any] = {}
        constraints_in_flow: Dict[Tuple[int, Node], Any] = {}
        constraints_flow_conservation: Dict[
            Union[Tuple[int, Node], Tuple[int, Node, str]], Any
        ] = {}

        count_origin = {}
        count_target = {}
        for vehicle in range(nb_vehicle):
            node_origin = self.problem.origin_vehicle[name_vehicles[vehicle]]
            node_target = self.problem.target_vehicle[name_vehicles[vehicle]]
            if node_origin not in count_origin:
                count_origin[node_origin] = 0
            if node_target not in count_target:
                count_target[node_target] = 0
            count_origin[node_origin] += 1
            count_target[node_target] += 1

        for group_vehicle in self.problem.group_identical_vehicles:
            representative_vehicle = self.problem.group_identical_vehicles[
                group_vehicle
            ][0]
            node_origin = self.problem.origin_vehicle[
                name_vehicles[representative_vehicle]
            ]
            node_target = self.problem.target_vehicle[
                name_vehicles[representative_vehicle]
            ]
            same_node = node_origin == node_target
            constraints_out_flow[(group_vehicle, node_origin)] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[group_vehicle][edge]
                        for edge in edges_out_per_vehicles[group_vehicle].get(
                            node_origin, set()
                        )
                        if edge[1] != node_origin or same_node
                    ]
                )
                <= count_origin[node_origin],
                name="outflow_" + str((group_vehicle, node_origin)),
            )  # Avoid loop
            constraints_in_flow[(group_vehicle, node_target)] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[group_vehicle][edge]
                        for edge in edges_in_per_vehicles[group_vehicle].get(
                            node_target, set()
                        )
                        if edge[0] != node_target or same_node
                    ]
                )
                <= count_target[node_target],
                name="inflow_" + str((group_vehicle, node_target)),
            )  # Avoid loop

            constraints_out_flow[(group_vehicle, node_target)] = model.add_constr(
                mip.quicksum(
                    [
                        variables_edges[group_vehicle][edge]
                        for edge in edges_out_per_vehicles[group_vehicle].get(
                            node_target, set()
                        )
                        if edge[1] != node_target or same_node
                    ]
                )
                <= 0,
                name="outflow_" + str((group_vehicle, node_target)),
            )

        for group_vehicle in self.problem.group_identical_vehicles:
            representative_vehicle = self.problem.group_identical_vehicles[
                group_vehicle
            ][0]
            node_origin = self.problem.origin_vehicle[
                name_vehicles[representative_vehicle]
            ]
            node_target = self.problem.target_vehicle[
                name_vehicles[representative_vehicle]
            ]
            same_node = node_origin == node_target
            for node in edges_in_per_vehicles[group_vehicle]:
                if same_node or node not in {node_origin, node_target}:
                    constraints_flow_conservation[
                        (group_vehicle, node)
                    ] = model.add_constr(
                        mip.quicksum(
                            [
                                variables_edges[group_vehicle][e]
                                for e in edges_in_per_vehicles[group_vehicle].get(
                                    node, set()
                                )
                                if e[1] != e[0]
                            ]
                            + [
                                -variables_edges[group_vehicle][e]
                                for e in edges_out_per_vehicles[group_vehicle].get(
                                    node, set()
                                )
                                if e[1] != e[0]
                            ]
                        )
                        == 0,
                        name="convflow_" + str((group_vehicle, node)),
                    )
                    constraints_flow_conservation[
                        (group_vehicle, node, "in")
                    ] = model.add_constr(
                        mip.quicksum(
                            [
                                variables_edges[group_vehicle][e]
                                for e in edges_in_per_vehicles[group_vehicle].get(
                                    node, set()
                                )
                                if e[1] != e[0]
                            ]
                        )
                        <= 1,
                        name="valueflow_" + str((group_vehicle, node)),
                    )

        if one_visit_per_node:
            self.one_visit_per_node(
                model=model,
                nodes_of_interest=nodes_of_interest,
                variables_edges=variables_edges,
                edges_in_all_vehicles=edges_in_all_vehicles,
                edges_out_all_vehicles=edges_out_all_vehicles,
            )
        if one_visit_per_cluster:
            self.one_visit_per_clusters(
                model=model,
                nodes_of_interest=nodes_of_interest,
                variables_edges=variables_edges,
                edges_in_all_vehicles=edges_in_all_vehicles,
                edges_out_all_vehicles=edges_out_all_vehicles,
            )
        if include_capacity:
            all_variables.update(
                self.simple_capacity_constraint(
                    model=model,
                    variables_edges=variables_edges,
                    edges_in_all_vehicles=edges_in_all_vehicles,
                    edges_out_all_vehicles=edges_out_all_vehicles,
                )
            )

        if include_time_evolution:
            all_variables.update(
                self.time_evolution(
                    model=model,
                    variables_edges=variables_edges,
                    edges_in_all_vehicles=edges_in_all_vehicles,
                    edges_out_all_vehicles=edges_out_all_vehicles,
                )
            )
        model.threads = 4
        self.edges_in_all_vehicles: Dict[
            Node, Set[Tuple[int, Edge]]
        ] = edges_in_all_vehicles
        self.edges_out_all_vehicles: Dict[
            Node, Set[Tuple[int, Edge]]
        ] = edges_out_all_vehicles
        self.edges_in_per_vehicles: Dict[
            int, Dict[Node, Set[Edge]]
        ] = edges_in_per_vehicles
        self.edges_out_per_vehicles: Dict[
            int, Dict[Node, Set[Edge]]
        ] = edges_out_per_vehicles
        self.edges_in_all_vehicles_cluster: Dict[
            Hashable, Set[Tuple[int, Edge]]
        ] = edges_in_all_vehicles_cluster
        self.edges_out_all_vehicles_cluster: Dict[
            Hashable, Set[Tuple[int, Edge]]
        ] = edges_out_all_vehicles_cluster
        self.edges_in_per_vehicles_cluster: Dict[
            int, Dict[Hashable, Set[Edge]]
        ] = edges_in_per_vehicles_cluster
        self.edges_out_per_vehicles_cluster: Dict[
            int, Dict[Hashable, Set[Edge]]
        ] = edges_out_per_vehicles_cluster
        self.variable_decisions = all_variables
        self.model = model
        self.clusters_version = one_visit_per_cluster
        self.tsp_version = one_visit_per_node

    def retrieve_ith_temporaryresult(self, i: int) -> TemporaryResult:
        res, obj = retrieve_ith_solution(
            i=i, model=self.model, variable_decisions=self.variable_decisions
        )
        path_dict = build_path_from_vehicle_type_flow(
            result_from_retrieve=res, problem=self.problem
        )
        res["path_dict"] = cast(Dict[Hashable, Any], path_dict)

        return TemporaryResult()  # type: ignore # TO BE COMPLETED

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, self.nb_solutions)
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for s in range(n_solutions):
            temporaryresult = self.retrieve_ith_temporaryresult(i=s)
            solution = convert_temporaryresult_to_gpdpsolution(
                temporaryresult=temporaryresult, problem=self.problem
            )
            fit = self.aggreg_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            limit_store=False,
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve_one_iteration(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> List[TemporaryResult]:
        self.optimize_model(parameters_milp=parameters_milp, **kwargs)
        list_temporary_results: List[TemporaryResult] = []
        for i in range(self.nb_solutions):
            list_temporary_results.append(self.retrieve_ith_temporaryresult(i=i))

        return list_temporary_results

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        # Probably missing a solve_iterative here
        temporaryresults = self.solve_one_iteration(
            parameters_milp=parameters_milp, **kwargs
        )
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, self.nb_solutions)
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for s in range(n_solutions):
            solution = convert_temporaryresult_to_gpdpsolution(
                temporaryresult=temporaryresults[s], problem=self.problem
            )
            fit = self.aggreg_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )


class SubtourAddingConstraint:
    def __init__(
        self, problem: GPDP, linear_solver: LinearFlowSolver, lazy: bool = False
    ):
        self.problem = problem
        self.linear_solver = linear_solver
        self.lazy = lazy

    def adding_component_constraints(
        self, list_solution: List[TemporaryResult]
    ) -> List[Any]:
        c = []
        for l in list_solution:
            if l.connected_components_per_vehicle is None:
                raise RuntimeError(
                    "Temporary result attributes "
                    "connected_components_per_vehicle cannot be None after solving."
                )
            for v in l.connected_components_per_vehicle:

                if self.lazy:
                    c += update_model_lazy(
                        self.problem,
                        self.linear_solver,
                        l.connected_components_per_vehicle[v],
                        self.linear_solver.edges_in_all_vehicles,
                        self.linear_solver.edges_out_all_vehicles,
                    )[1]
                else:
                    c += update_model(
                        self.problem,
                        self.linear_solver,
                        l.connected_components_per_vehicle[v],
                        self.linear_solver.edges_in_all_vehicles,
                        self.linear_solver.edges_out_all_vehicles,
                    )[1]
        return c


class SubtourAddingConstraintCluster:
    def __init__(self, problem: GPDP, linear_solver: LinearFlowSolver):
        self.problem = problem
        self.linear_solver = linear_solver

    def adding_component_constraints(
        self, list_solution: List[TemporaryResult]
    ) -> List[Any]:
        c = []
        for l in list_solution:
            if l.connected_components_per_vehicle is None:
                raise RuntimeError(
                    "Temporary result attributes "
                    "connected_components_per_vehicle cannot be None after solving."
                )
            for v in l.connected_components_per_vehicle:
                connected_components = l.connected_components_per_vehicle[v]
                conn = []
                for x in connected_components:
                    s = set([self.problem.clusters_dict[k] for k in x[0]])
                    conn += [(s, len(s))]
                c += update_model_cluster_tsp(
                    self.problem,
                    self.linear_solver,
                    conn,
                    self.linear_solver.edges_in_all_vehicles_cluster,
                    self.linear_solver.edges_out_all_vehicles_cluster,
                )
        return c


class ConstraintHandlerOrWarmStart:
    def __init__(
        self,
        linear_solver: LinearFlowSolver,
        problem: GPDP,
        do_lns: bool = True,
        remove_constr: bool = True,
    ):
        self.linear_solver = linear_solver
        self.problem = problem
        self.do_lns = do_lns
        self.remove_constr = remove_constr

    def adding_constraint(self, rebuilt_dict: Dict[int, List[Node]]) -> None:
        if self.linear_solver.model is None:
            raise RuntimeError("self.linear_solver.model cannot be None at this point")

        vehicle_keys = self.linear_solver.variable_decisions["variables_edges"].keys()
        edges_to_add: Dict[int, Set[Edge]] = {
            v: set() for v in vehicle_keys if rebuilt_dict[v] is not None
        }
        for v in rebuilt_dict:
            if rebuilt_dict[v] is None:
                continue
            edges_to_add[v].update(
                {(e0, e1) for e0, e1 in zip(rebuilt_dict[v][:-1], rebuilt_dict[v][1:])}
            )
            logger.info(f"edges to add {edges_to_add}")
            edges_missing = {
                (v, e)
                for e in edges_to_add[v]
                if e not in self.linear_solver.variable_decisions["variables_edges"][v]
            }
            logger.info(f"missing : {edges_missing}")
            if len(edges_missing) > 0:
                logger.warning("Some edges are missing.")
        if self.do_lns:
            for iedge in self.linear_solver.constraint_on_edge:
                try:
                    self.linear_solver.model.remove(
                        self.linear_solver.constraint_on_edge[iedge]
                    )
                except IndexError as e:
                    pass
        self.linear_solver.constraint_on_edge = {}
        edges_to_constraint: Dict[int, Set[Edge]] = {
            v: set() for v in range(self.problem.number_vehicle)
        }
        vehicle_to_not_constraints = set(
            random.sample(
                range(self.problem.number_vehicle), min(2, self.problem.number_vehicle)
            )
        )
        for v in range(self.problem.number_vehicle):
            if v not in vehicle_to_not_constraints:
                edges_to_constraint[v].update(
                    set(
                        random.sample(
                            set(
                                self.linear_solver.variable_decisions[
                                    "variables_edges"
                                ][v]
                            ),
                            int(
                                0.9
                                * len(
                                    self.linear_solver.variable_decisions[
                                        "variables_edges"
                                    ][v]
                                )
                            ),
                        )
                    )
                )
            else:
                edges_to_constraint[v].update(
                    set(
                        random.sample(
                            set(
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
        logger.info(
            (
                sum([len(edges_to_constraint[v]) for v in edges_to_constraint]),
                " edges constraint over ",
                sum(
                    [
                        len(self.linear_solver.variable_decisions["variables_edges"][v])
                        for v in self.linear_solver.variable_decisions[
                            "variables_edges"
                        ]
                    ]
                ),
            )
        )
        iedge = 0
        start_list = []
        for v in vehicle_keys:
            logger.debug(f"Rebuild dict = , {rebuilt_dict[v]}")
            for edge in self.linear_solver.variable_decisions["variables_edges"][v]:
                val = 0.0
                if v in edges_to_add and edge in edges_to_add[v]:
                    start_list.append(
                        (
                            self.linear_solver.variable_decisions["variables_edges"][v][
                                edge
                            ],
                            1.0,
                        )
                    )
                    val = 1.0
                else:
                    start_list.append(
                        (
                            self.linear_solver.variable_decisions["variables_edges"][v][
                                edge
                            ],
                            0.0,
                        )
                    )
                if self.do_lns:

                    if (
                        rebuilt_dict[v] is not None
                        and v in edges_to_constraint
                        and edge in edges_to_constraint[v]
                    ):
                        self.linear_solver.constraint_on_edge[
                            iedge
                        ] = self.linear_solver.model.add_constr(
                            self.linear_solver.variable_decisions["variables_edges"][v][
                                edge
                            ]
                            == val,
                            name="c_" + str(v) + "_" + str(edge) + "_" + str(val),
                        )
                        iedge += 1
        self.linear_solver.model.start = start_list


def update_model_cluster_tsp(
    problem: GPDP,
    lp_solver: LinearFlowSolver,
    components_global: List[Tuple[Set[Hashable], int]],
    edges_in_all_vehicles_cluster: Dict[Hashable, Set[Tuple[int, Edge]]],
    edges_out_all_vehicles_cluster: Dict[Hashable, Set[Tuple[int, Edge]]],
) -> List[Any]:
    if lp_solver.model is None:
        raise RuntimeError("self.lp_solver.model cannot be None at this point.")
    len_component_global = len(components_global)
    list_constraints = []
    if len_component_global > 1:
        logger.debug(f"Nb component : {len_component_global}")
        for s in components_global:
            edge_in_of_interest = [
                e
                for cluster in s[0]
                for e in edges_in_all_vehicles_cluster[cluster]
                if problem.clusters_dict[e[1][0]] not in s[0]
                and problem.clusters_dict[e[1][1]] in s[0]
            ]
            edge_out_of_interest = [
                e
                for cluster in s[0]
                for e in edges_out_all_vehicles_cluster[cluster]
                if problem.clusters_dict[e[1][0]] in s[0]
                and problem.clusters_dict[e[1][1]] not in s[0]
            ]
            if not any(
                problem.clusters_dict[problem.target_vehicle[v]] in s[0]
                for v in problem.origin_vehicle
            ):
                list_constraints += [
                    lp_solver.model.add_constr(
                        mip.quicksum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_out_of_interest
                            ]
                        )
                        >= 1,
                        name="component_out_" + str(s)[:10],
                    )
                ]
            if not any(
                problem.clusters_dict[problem.origin_vehicle[v]] in s[0]
                for v in problem.origin_vehicle
            ):
                list_constraints += [
                    lp_solver.model.add_constr(
                        mip.quicksum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_in_of_interest
                            ]
                        )
                        >= 1,
                        name="component_in_" + str(s)[:10],
                    )
                ]
    return list_constraints


def update_model(
    problem: GPDP,
    lp_solver: LinearFlowSolver,
    components_global: List[Tuple[Set[Node], int]],
    edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
) -> Tuple[List[Any], List[Tuple[List[Tuple[int, Edge]], int]]]:
    if lp_solver.model is None:
        raise RuntimeError("self.lp_solver.model cannot be None at this point.")
    len_component_global = len(components_global)
    list_constraints = []
    list_constraints_tuple: List[Tuple[List[Tuple[int, Edge]], int]] = []
    if len_component_global > 1:
        logger.debug(f"Nb component : {len_component_global}")
        for s in components_global:
            edge_in_of_interest = [
                e
                for n in s[0]
                for e in edges_in_all_vehicles[n]
                if e[1][0] not in s[0] and e[1][1] in s[0]
            ]
            edge_out_of_interest = [
                e
                for n in s[0]
                for e in edges_out_all_vehicles[n]
                if e[1][0] in s[0] and e[1][1] not in s[0]
            ]
            if not any(
                problem.target_vehicle[v] in s[0] for v in problem.origin_vehicle
            ):
                list_constraints_tuple += [(edge_out_of_interest, 1)]
                list_constraints += [
                    lp_solver.model.add_constr(
                        mip.quicksum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_out_of_interest
                            ]
                        )
                        >= 1,
                        name="component_out_" + str(s)[:10],
                    )
                ]
            if not any(
                problem.origin_vehicle[v] in s[0] for v in problem.origin_vehicle
            ):
                list_constraints_tuple += [(edge_in_of_interest, 1)]
                list_constraints += [
                    lp_solver.model.add_constr(
                        mip.quicksum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_in_of_interest
                            ]
                        )
                        >= 1,
                        name="component_in_" + str(s)[:10],
                    )
                ]
    if len_component_global > 1:
        constraints_order = {}
        try:
            variable_order = lp_solver.model.variable_order

        except:
            variable_order = {
                node: lp_solver.model.add_var(
                    var_type=mip.CONTINUOUS, name="order_" + str(node)
                )
                for node in edges_in_all_vehicles
            }
            for vehicle in range(lp_solver.problem.number_vehicle):
                node_origin = lp_solver.problem.origin_vehicle[vehicle]
                constraints_order[node_origin] = lp_solver.model.add_constr(
                    variable_order[node_origin] == 0, name="order_" + str(node_origin)
                )
            lp_solver.variable_order = variable_order
        c = max(components_global, key=lambda x: x[1])
        for s in [c]:
            use_big_m = True
            for node in s[0]:
                if node not in constraints_order:
                    for vehicle, edge in edges_in_all_vehicles[node]:
                        if edge[0] == edge[1]:
                            continue
                        if use_big_m:
                            constraints_order[node] = lp_solver.model.add_constr(
                                variable_order[node]
                                >= variable_order[edge[0]]
                                + 1
                                - 1000
                                * (
                                    1
                                    - lp_solver.variable_decisions["variables_edges"][
                                        vehicle
                                    ][edge]
                                ),
                                name="order_" + str(node),
                            )
    return list_constraints, list_constraints_tuple


def update_model_lazy(
    problem: GPDP,
    lp_solver: LinearFlowSolver,
    components_global: List[Tuple[Set[Node], int]],
    edges_in_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    edges_out_all_vehicles: Dict[Node, Set[Tuple[int, Edge]]],
    do_constraint_on_order: bool = False,
) -> List[Any]:
    if lp_solver.model is None:
        raise RuntimeError("self.lp_solver.model cannot be None at this point.")
    len_component_global = len(components_global)
    list_constraints = []
    if len_component_global > 1:
        logger.debug(f"Nb component : {len_component_global}")
        for s in components_global:
            edge_in_of_interest = [
                e
                for n in s[0]
                for e in edges_in_all_vehicles[n]
                if e[1][0] not in s[0] and e[1][1] in s[0]
            ]
            edge_out_of_interest = [
                e
                for n in s[0]
                for e in edges_out_all_vehicles[n]
                if e[1][0] in s[0] and e[1][1] not in s[0]
            ]
            if not any(
                problem.target_vehicle[v] in s[0] for v in problem.origin_vehicle
            ):
                list_constraints += [
                    lp_solver.model.add_lazy_constr(
                        mip.quicksum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_out_of_interest
                            ]
                        )
                        >= 1
                    )
                ]
            if not any(
                problem.origin_vehicle[v] in s[0] for v in problem.origin_vehicle
            ):
                list_constraints += [
                    lp_solver.model.add_lazy_constr(
                        mip.quicksum(
                            [
                                lp_solver.variable_decisions["variables_edges"][e[0]][
                                    e[1]
                                ]
                                for e in edge_in_of_interest
                            ]
                        )
                        >= 1
                    )
                ]
    if len_component_global > 1 and do_constraint_on_order:
        constraints_order = {}
        try:
            variable_order = lp_solver.variable_order

        except:
            variable_order = {
                node: lp_solver.model.add_var(
                    var_type=mip.CONTINUOUS, name="order_" + str(node)
                )
                for node in edges_in_all_vehicles
            }
            for vehicle in range(lp_solver.problem.number_vehicle):
                node_origin = lp_solver.problem.origin_vehicle[vehicle]
                constraints_order[node_origin] = lp_solver.model.add_constr(
                    variable_order[node_origin] == 0
                )
            lp_solver.variable_order = variable_order
        c = max(components_global, key=lambda x: x[1])
        for s in [c]:
            use_big_m = True
            for node in s[0]:
                if node not in constraints_order:
                    for vehicle, edge in edges_in_all_vehicles[node]:
                        if edge[0] == edge[1]:
                            continue
                        if use_big_m:
                            constraints_order[node] = lp_solver.model.add_lazy_constr(
                                variable_order[node]
                                >= variable_order[edge[0]]
                                + 1
                                - 1000
                                * (
                                    1
                                    - lp_solver.variable_decisions["variables_edges"][
                                        vehicle
                                    ][edge]
                                )
                            )
    return list_constraints
