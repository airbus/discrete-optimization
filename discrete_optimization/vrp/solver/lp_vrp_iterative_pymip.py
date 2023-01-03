#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
from mip import BINARY, CBC, MINIMIZE, Model, Var, xsum

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.generic_tools.mip.pymip_tools import MyModelMilp
from discrete_optimization.vrp.solver.lp_vrp_iterative import (
    build_graph_pruned_vrp,
    build_the_cycles,
    build_warm_edges_and_update_graph,
    compute_start_end_flows_info,
    plot_solve,
    reevaluate_solutions,
    update_graph,
)
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import (
    BasicCustomer,
    Customer2D,
    VrpProblem,
    VrpProblem2D,
    VrpSolution,
    compute_length,
    length,
)

logger = logging.getLogger(__name__)

Node = Tuple[int, int]
Edge = Tuple[Node, Node]


def init_model_lp(
    g: nx.DiGraph,
    edges: Set[Edge],
    edges_in_customers: Dict[int, Set[Edge]],
    edges_out_customers: Dict[int, Set[Edge]],
    edges_in_merged_graph: Dict[Node, Set[Edge]],
    edges_out_merged_graph: Dict[Node, Set[Edge]],
    edges_warm_set: Set[Edge],
    fraction: float,
    start_indexes: List[int],
    end_indexes: List[int],
    vehicle_count: int,
    vehicle_capacity: List[float],
    do_lns: bool = False,
    include_backward: bool = True,
    include_triangle: bool = False,
    solver_name: str = CBC,
) -> Tuple[
    MyModelMilp,
    Dict[Edge, Var],
    Dict[Union[str, int], Any],
    Dict[str, Any],
    Dict[int, Any],
]:
    tsp_model = MyModelMilp("VRP-master", sense=MINIMIZE, solver_name=solver_name)
    x_var: Dict[Edge, Var] = {}  # decision variables on edges
    constraint_on_edge: Dict[int, Any] = {}
    edges_to_constraint: Set[Edge] = set()
    if do_lns:
        edges_to_constraint = set(
            random.sample(list(edges), int(fraction * len(edges)))
        )
        for iedge in constraint_on_edge:
            tsp_model.remove(constraint_on_edge[iedge])
    iedge = 0
    start: List[Tuple[Var, int]] = []
    for e in edges:
        x_var[e] = tsp_model.add_var(
            var_type=BINARY, obj=g[e[0]][e[1]]["weight"], name="x_" + str(e)
        )
        val = 0
        if e in edges_warm_set:
            start += [(x_var[e], 1)]
            val = 1
        else:
            start += [(x_var[e], 0)]
        if e in edges_to_constraint:
            constraint_on_edge[iedge] = tsp_model.add_constr(
                x_var[e] == val, name=str((e, iedge))
            )
            iedge += 1
    tsp_model.start = start
    constraint_tour_2length: Dict[int, Any] = {}
    cnt_tour = 0
    if include_backward:
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                if (edge[1], edge[0]) == edge:
                    continue
                if (
                    edge[0][1] == start_indexes[edge[0][0]]
                    or edge[1][1] == start_indexes[edge[0][0]]
                ):
                    continue
                if (
                    edge[0][1] == end_indexes[edge[0][0]]
                    or edge[1][1] == end_indexes[edge[0][0]]
                ):
                    continue
                constraint_tour_2length[cnt_tour] = tsp_model.add_constr(
                    x_var[edge] + x_var[(edge[1], edge[0])] <= 1,
                    name="tour_" + str(edge),
                )
                cnt_tour += 1
    if include_triangle:
        constraint_triangle: Dict[int, Any] = {}
        for node in g.nodes():
            neigh = set([n for n in nx.neighbors(g, node)])
            neigh_2 = {
                nn: neigh.intersection([n for n in nx.neighbors(g, nn)]) for nn in neigh
            }
            for node_neigh in neigh_2:
                if len(neigh_2[node_neigh]) >= 1:
                    for node_neigh_neigh in neigh_2[node_neigh]:
                        constraint_triangle[cnt_tour] = tsp_model.add_constr(
                            x_var[(node, node_neigh)]
                            + x_var[(node_neigh, node_neigh_neigh)]
                            + x_var[(node_neigh_neigh, node)]
                            <= 2
                        )
    constraint_flow_in: Dict[Union[str, int], Any] = {}
    constraint_flow_out: Dict[str, Any] = {}
    start_to_i, end_to_i = compute_start_end_flows_info(start_indexes, end_indexes)
    for s in start_to_i:
        for vehicle in start_to_i[s]["vehicle"]:
            constraint_flow_out["start_v_" + str(vehicle)] = tsp_model.add_constr(
                xsum([x_var[e] for e in edges_out_customers[s] if e[0][0] == vehicle])
                == 1,
                name="start_v_" + str(vehicle),
            )
    for s in end_to_i:
        for vehicle in end_to_i[s]["vehicle"]:
            constraint_flow_in["end_v_" + str(vehicle)] = tsp_model.add_constr(
                xsum([x_var[e] for e in edges_in_customers[s] if e[0][0] == vehicle])
                == 1,
                name="end_v_" + str(vehicle),
            )
    for customer in edges_in_customers:
        if customer in end_to_i or customer in start_to_i:
            # Already dealt by previous constraints
            continue
        else:
            constraint_flow_in[customer] = tsp_model.add_constr(
                xsum([x_var[e] for e in edges_in_customers[customer]]) == 1,
                name="in_" + str(customer),
            )
    c_flow: Dict[Node, Any] = {}
    for n in edges_in_merged_graph:
        if start_indexes[n[0]] == end_indexes[n[0]] or n[1] not in [
            start_indexes[n[0]],
            end_indexes[n[0]],
        ]:
            c_flow[n] = tsp_model.add_constr(
                xsum(
                    [x_var[e] for e in edges_in_merged_graph[n]]
                    + [-x_var[e] for e in edges_out_merged_graph[n]]
                )
                == 0,
                name="flow_" + str(n),
            )
    for v in range(vehicle_count):
        tsp_model.add_constr(
            xsum([g[e[0]][e[1]]["demand"] * x_var[e] for e in edges if e[0][0] == v])
            <= vehicle_capacity[v],
            name="capa_" + str(v),
        )
    return tsp_model, x_var, constraint_flow_in, constraint_flow_out, constraint_on_edge


class VRPIterativeLP_Pymip(SolverVrp):
    vrp_model: VrpProblem2D
    edges: Set[Edge]
    edges_in_customers: Dict[int, Set[Edge]]
    edges_out_customers: Dict[int, Set[Edge]]
    edges_in_merged_graph: Dict[Node, Set[Edge]]
    edges_out_merged_graph: Dict[Node, Set[Edge]]
    edges_warm_set: Set[Edge]

    def __init__(
        self,
        vrp_model: VrpProblem2D,
        params_objective_function: ParamsObjectiveFunction,
    ):
        SolverVrp.__init__(self, vrp_model=vrp_model)
        self.model: Optional[MyModelMilp] = None
        self.x_var: Optional[Dict[Edge, Var]] = None
        self.constraint_on_edge: Optional[Dict[int, Any]] = None
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.vrp_model, params_objective_function=params_objective_function
        )

    def init_model(self, **kwargs: Any) -> None:
        (
            g,
            g_empty,
            edges_in_customers,
            edges_out_customers,
            edges_in_merged_graph,
            edges_out_merged_graph,
        ) = build_graph_pruned_vrp(self.vrp_model)

        initial_solution = kwargs.get("initial_solution", None)
        if initial_solution is None:
            solution = self.vrp_model.get_dummy_solution()
        else:
            vehicle_tours_b = initial_solution
            solution = VrpSolution(
                problem=self.vrp_model,
                list_start_index=self.vrp_model.start_indexes,
                list_end_index=self.vrp_model.end_indexes,
                list_paths=vehicle_tours_b,
                length=None,
                lengths=None,
                capacities=None,
            )
        edges = set(g.edges())
        edges_warm, edges_warm_set = build_warm_edges_and_update_graph(
            vrp_problem=self.vrp_model,
            vrp_solution=solution,
            graph=g,
            edges=edges,
            edges_in_merged_graph=edges_in_merged_graph,
            edges_out_merged_graph=edges_out_merged_graph,
            edges_in_customers=edges_in_customers,
            edges_out_customers=edges_out_customers,
        )
        do_lns = kwargs.get("do_lns", False)
        fraction = kwargs.get("fraction_lns", 0.9)
        solver_name = kwargs.get("solver_name", CBC)
        (
            tsp_model,
            x_var,
            constraint_flow_in,
            constraint_flow_out,
            constraint_on_edge,
        ) = init_model_lp(
            g=g,
            edges=edges,
            edges_in_customers=edges_in_customers,
            edges_out_customers=edges_out_customers,
            edges_in_merged_graph=edges_in_merged_graph,
            edges_out_merged_graph=edges_out_merged_graph,
            edges_warm_set=edges_warm_set,
            start_indexes=self.vrp_model.start_indexes,
            end_indexes=self.vrp_model.end_indexes,
            do_lns=do_lns,
            fraction=fraction,
            vehicle_count=self.vrp_model.vehicle_count,
            vehicle_capacity=self.vrp_model.vehicle_capacities,
            solver_name=solver_name,
        )
        self.model = tsp_model
        self.x_var = x_var
        self.constraint_on_edge = constraint_on_edge
        self.graph = g
        self.edges = edges
        self.edges_in_customers = edges_in_customers
        self.edges_out_customers = edges_out_customers
        self.edges_in_merged_graph = edges_in_merged_graph
        self.edges_out_merged_graph = edges_out_merged_graph
        self.edges_warm_set = edges_warm_set

    def solve(self, **kwargs: Any) -> ResultStorage:
        solver_name = kwargs.get("solver_name", CBC)
        do_lns = kwargs.get("do_lns", False)
        fraction = kwargs.get("fraction_lns", 0.9)
        nb_iteration_max = kwargs.get("nb_iteration_max", 20)
        if self.model is None or self.x_var is None or self.constraint_on_edge is None:
            kwargs["solver_name"] = solver_name
            kwargs["do_lns"] = do_lns
            kwargs["fraction_lns"] = fraction
            self.init_model(**kwargs)
            if (
                self.model is None
                or self.x_var is None
                or self.constraint_on_edge is None
            ):
                raise RuntimeError(
                    "model, x_var and constraint_on_edge attributes "
                    "should not be None after init_model()"
                )
        logger.info("optimizing...")
        limit_time_s = kwargs.get("limit_time_s", 10)
        self.model.optimize(max_seconds=limit_time_s)
        objective = self.model.objective_value
        # Query number of multiple objectives, and number of solutions
        finished = False
        solutions: List[Dict[int, Set[Edge]]] = []
        cost: List[float] = []
        nb_components: List[int] = []
        iteration = 0
        rebuilt_solution: List[Dict[int, List[Node]]] = []
        rebuilt_obj: List[float] = []
        best_solution_rebuilt_index = 0
        best_solution_objective_rebuilt = float("inf")
        vehicle_count = self.vrp_model.vehicle_count
        customers = self.vrp_model.customers
        edges_in_customers = self.edges_in_customers
        edges_out_customers = self.edges_out_customers
        edges_in_merged_graph = self.edges_in_merged_graph
        edges_out_merged_graph = self.edges_out_merged_graph
        edges = self.edges
        edges_warm_set = self.edges_warm_set
        g = self.graph
        while not finished:
            solutions_ll = retrieve_solutions(self.model, self.x_var, vehicle_count, g)
            solutions += [solutions_ll[0][-1]]
            cost += [objective]
            (
                x_solution,
                rebuilt_dict,
                obj,
                components,
                components_global,
                component_all,
                component_global_all,
            ) = reevaluate_solutions(
                solutions=solutions_ll,
                vehicle_count=vehicle_count,
                g=g,
                vrp_problem=self.vrp_model,
            )
            for comp in component_global_all:
                update_model_2(
                    model=self.model,
                    x_var=self.x_var,
                    components_global=comp,
                    edges_in_customers=edges_in_customers,
                    edges_out_customers=edges_out_customers,
                )

            nb_components += [len(components_global)]
            rebuilt_solution += [rebuilt_dict]
            rebuilt_obj += [obj]
            logger.debug(f"Objective rebuilt : {rebuilt_obj[-1]}")
            if obj < best_solution_objective_rebuilt:
                best_solution_objective_rebuilt = obj
                best_solution_rebuilt_index = iteration
            iteration += 1
            if len(component_global_all[0]) > 1:
                edges_to_add: Set[Edge] = set()
                for v in rebuilt_dict:
                    edges_to_add.update(
                        {
                            (e0, e1)
                            for e0, e1 in zip(rebuilt_dict[v][:-1], rebuilt_dict[v][1:])
                        }
                    )
                edges_missing = {e for e in edges_to_add if e not in edges}

                if len(edges_missing) > 0:
                    (
                        g,
                        edges,
                        edges_in_customers,
                        edges_out_customers,
                        edges_in_merged_graph,
                        edges_out_merged_graph,
                    ) = update_graph(
                        g=g,
                        edges=edges,
                        edges_in_customers=edges_in_customers,
                        edges_out_customers=edges_out_customers,
                        edges_in_merged_graph=edges_in_merged_graph,
                        edges_out_merged_graph=edges_out_merged_graph,
                        edges_missing=edges_missing,
                        customers=customers,
                    )
                    self.model = None
                    (
                        tsp_model,
                        x_var,
                        constraint_flow_in,
                        constraint_flow_out,
                        constraint_on_edge,
                    ) = init_model_lp(
                        g=g,
                        edges=edges,
                        edges_in_customers=edges_in_customers,
                        edges_out_customers=edges_out_customers,
                        edges_in_merged_graph=edges_in_merged_graph,
                        edges_out_merged_graph=edges_out_merged_graph,
                        edges_warm_set=edges_warm_set,
                        start_indexes=self.vrp_model.start_indexes,
                        end_indexes=self.vrp_model.end_indexes,
                        do_lns=do_lns,
                        fraction=fraction,
                        vehicle_count=self.vrp_model.vehicle_count,
                        vehicle_capacity=self.vrp_model.vehicle_capacities,
                        solver_name=solver_name,
                    )
                    self.model = tsp_model
                    self.x_var = x_var
                    self.constraint_on_edge = constraint_on_edge
                    self.graph = g
                    self.edges = edges
                    self.edges_in_customers = edges_in_customers
                    self.edges_out_customers = edges_out_customers
                    self.edges_in_merged_graph = edges_in_merged_graph
                    self.edges_out_merged_graph = edges_out_merged_graph
                    self.edges_warm_set = edges_warm_set
                    for iedge in self.constraint_on_edge:
                        self.model.remove(self.constraint_on_edge[iedge])
                    self.model.update()
                    self.constraint_on_edge = {}
                edges_to_constraint = set(self.x_var.keys())
                if do_lns:
                    for iedge in self.constraint_on_edge:
                        self.model.remove(self.constraint_on_edge[iedge])
                    self.model.update()
                    self.constraint_on_edge = {}
                    edges_to_constraint = set()
                    vehicle = set(
                        random.sample(range(vehicle_count), min(4, vehicle_count))
                    )
                    edges_to_constraint.update(
                        set([e for e in edges if e[0][0] not in vehicle])
                    )
                    logger.debug(
                        (
                            len(edges_to_constraint),
                            " edges constraint over ",
                            len(edges),
                        )
                    )
                iedge = 0
                x_var = self.x_var
                start = []
                if all((e in edges) for e in edges_to_add):
                    for e in x_var:
                        val = 0
                        if e in edges_to_add:
                            start += [(x_var[e], 1)]
                            val = 1
                        else:
                            start += [(x_var[e], 0)]
                        if e in edges_to_constraint:
                            if do_lns:
                                self.constraint_on_edge[iedge] = self.model.add_constr(
                                    x_var[e] == val, name=str((e, iedge))
                                )
                                iedge += 1
                    self.model.update()
                else:
                    pass
                self.model.start = start
                self.model.optimize(max_seconds=limit_time_s)
                objective = self.model.objective_value
            else:
                finished = True
            finished = finished or iteration >= nb_iteration_max

        plot = kwargs.get("plot", True)
        if plot:
            plot_solve(
                solutions=solutions,
                customers=customers,
                rebuilt_solution=rebuilt_solution,
                cost=cost,
                rebuilt_obj=rebuilt_obj,
            )
        logger.debug(f"Best obj : {best_solution_objective_rebuilt}")
        solution = VrpSolution(
            problem=self.vrp_model,
            list_start_index=self.vrp_model.start_indexes,
            list_end_index=self.vrp_model.end_indexes,
            list_paths=[
                [x[1] for x in rebuilt_solution[best_solution_rebuilt_index][l][1:-1]]
                for l in sorted(rebuilt_solution[best_solution_rebuilt_index])
            ],
            length=None,
            lengths=None,
            capacities=None,
        )
        _ = self.vrp_model.evaluate(solution)
        fit = self.aggreg_sol(solution)
        return ResultStorage(
            list_solution_fits=[(solution, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )


def retrieve_solutions(
    model: Model, x_var: Dict[Edge, Var], vehicle_count: int, g: nx.DiGraph
) -> List[Tuple[nx.DiGraph, Dict[int, nx.DiGraph], Dict[int, Set[Edge]]]]:
    nSolutions = model.num_solutions
    solutions: List[Tuple[nx.Digraph, Dict[int, nx.DiGraph], Dict[int, Set[Edge]]]] = []
    for s in range(nSolutions):
        # Set which solution we will query from now on
        g_empty = {v: nx.DiGraph() for v in range(vehicle_count)}
        g_merge = nx.DiGraph()
        x_solution: Dict[int, Set[Edge]] = {v: set() for v in range(vehicle_count)}
        for e in x_var:
            value = x_var[e].xi(s)
            if value is None:
                raise RuntimeError("Solution not available.")
            if value >= 0.5:
                vehicle = e[0][0]
                x_solution[vehicle].add(e)
                clients = e[0], e[1]
                if clients[0] not in g_empty[vehicle]:
                    g_empty[vehicle].add_node(clients[0])
                if clients[1] not in g_empty[vehicle]:
                    g_empty[vehicle].add_node(clients[1])
                if clients[0][1] not in g_merge:
                    g_merge.add_node(clients[0][1])
                if clients[1][1] not in g_merge:
                    g_merge.add_node(clients[1][1])
                g_empty[vehicle].add_edge(
                    clients[0], clients[1], weight=g[e[0]][e[1]]["weight"]
                )
                g_merge.add_edge(
                    clients[0][1], clients[1][1], weight=g[e[0]][e[1]]["weight"]
                )
        solutions += [
            (
                g_merge.copy(),
                g_empty,
                x_solution.copy(),
            )
        ]
    return solutions


def update_model_2(
    model: MyModelMilp,
    x_var: Dict[Edge, Var],
    components_global: List[Tuple[Set[Node], int]],
    edges_in_customers: Dict[int, Set[Edge]],
    edges_out_customers: Dict[int, Set[Edge]],
) -> None:
    len_component_global = len(components_global)
    if len_component_global > 1:
        logger.debug(f"Nb component : {len_component_global}")
        for s in components_global:
            customers_component: Set[int] = {customer for vehicle, customer in s[0]}
            edge_in_of_interest = [
                e
                for customer in customers_component
                for e in edges_in_customers[customer]
                if e[0][1] not in s[0] and e[1][1] in customers_component
            ]
            edge_out_of_interest = [
                e
                for customer in customers_component
                for e in edges_out_customers[customer]
                if e[1][1] not in s[0] and e[0][1] in customers_component
            ]
            model.add_constr(xsum([x_var[e] for e in edge_in_of_interest]) >= 1)
            model.add_constr(xsum([x_var[e] for e in edge_out_of_interest]) >= 1)
    model.update()
