#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from ortools.linear_solver import pywraplp

from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.tsp.problem import (
    Point,
    Point2D,
    Point2DTspProblem,
    TspProblem,
    TspSolution,
)
from discrete_optimization.tsp.solvers import TspSolver
from discrete_optimization.tsp.utils import (
    build_matrice_distance,
    build_matrice_distance_np,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, Model, quicksum


logger = logging.getLogger(__name__)

Node = int
Edge = tuple[Node, Node]


def build_graph_pruned(
    tsp_model: Point2DTspProblem,
) -> tuple[
    nx.DiGraph,
    nx.DiGraph,
    dict[int, set[tuple[int, int]]],
    dict[int, set[tuple[int, int]]],
]:
    nodeCount = tsp_model.node_count
    points = tsp_model.list_points
    sd, d = build_matrice_distance_np(nodeCount, points)
    g = nx.DiGraph()
    g.add_nodes_from([i for i in range(nodeCount)])
    shape = sd.shape[0]
    edges_in: dict[int, set[tuple[int, int]]] = {i: set() for i in range(nodeCount)}
    edges_out: dict[int, set[tuple[int, int]]] = {i: set() for i in range(nodeCount)}

    def length_ij(i: int, j: int) -> float:
        return tsp_model.evaluate_function_indexes(i, j)

    for i in range(shape):
        nodes_to_add: Iterable[int] = sd[i, 1:50]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n))
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
        nodes_to_add = range(i, min(i + 5, nodeCount))
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n))
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
        nodes_to_add = [tsp_model.end_index]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n))
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
    g_empty = nx.DiGraph()
    g_empty.add_nodes_from([i for i in range(nodeCount)])
    return g, g_empty, edges_in, edges_out


def build_graph_complete(
    tsp_model: TspProblem,
) -> tuple[nx.DiGraph, nx.DiGraph, dict[int, set[Edge]], dict[int, set[Edge]],]:
    nodeCount = tsp_model.node_count
    mat = build_matrice_distance(nodeCount, tsp_model.evaluate_function_indexes)
    sd: npt.NDArray[np.int_] = np.argsort(mat, axis=1)
    g = nx.DiGraph()
    g.add_nodes_from([i for i in range(nodeCount)])
    shape = sd.shape[0]
    edges_in: dict[int, set[Edge]] = {i: set() for i in range(nodeCount)}
    edges_out: dict[int, set[Edge]] = {i: set() for i in range(nodeCount)}

    def length_ij(i: int, j: int) -> float:
        return mat[i, j]

    for i in range(shape):
        nodes_to_add: Iterable[Node] = sd[i, 1:]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n))
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
        nodes_to_add = [tsp_model.end_index]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n))
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
    g_empty = nx.DiGraph()
    g_empty.add_nodes_from([i for i in range(nodeCount)])
    return g, g_empty, edges_in, edges_out


class MILPSolver(Enum):
    GUROBI = 0
    CBC = 1


class LPIterativeTspSolver(TspSolver):
    def __init__(
        self,
        problem: TspProblem,
        graph_builder: Callable[
            [TspProblem],
            tuple[
                nx.DiGraph,
                nx.DiGraph,
                dict[int, set[Edge]],
                dict[int, set[Edge]],
            ],
        ],
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.node_count = self.problem.node_count
        self.list_points = self.problem.list_points
        self.start_index = self.problem.start_index
        self.end_index = self.problem.end_index
        self.graph_builder = graph_builder
        self.g: nx.DiGraph
        self.edges: set[Edge]
        self.method: MILPSolver
        self.variables: dict[str, dict[Edge, Any]]
        self.aggreg_from_sol: Callable[[Solution], float]
        self.aggreg_from_dict: Callable[[dict[str, float]], float]
        if (
            self.params_objective_function.objective_handling
            == ObjectiveHandling.MULTI_OBJ
        ):
            raise NotImplementedError(
                "LPIterativeTspSolver can only handle single or aggregated objective."
            )

    def init_model(self, method: MILPSolver = MILPSolver.CBC, **kwargs: Any) -> None:
        if method == MILPSolver.GUROBI:
            self.init_model_gurobi()
            self.method = method
        else:
            self.init_model_cbc()
            self.method = method

    def init_model_gurobi(self, **kwargs: Any) -> None:
        g, g_empty, edges_in, edges_out = self.graph_builder(self.problem)
        tsp_model = Model("TSP-master")
        edges: set[Edge] = set(g.edges())
        self.edges = edges
        self.g = g
        x_var: dict[Edge, Any] = {}  # decision variables on edges
        dummy_sol = self.problem.get_dummy_solution()
        path: list[Node] = (
            [self.problem.start_index]
            + dummy_sol.permutation
            + [self.problem.end_index]
        )
        edges_to_add: set[Edge] = {
            (node0, node1) for node0, node1 in zip(path[:-1], path[1:])
        }
        flow_in: dict[Node, set[Edge]] = {}
        flow_out: dict[Node, set[Edge]] = {}
        for e in edges:
            x_var[e] = tsp_model.addVar(
                vtype=GRB.BINARY, obj=g[e[0]][e[1]]["weight"], name="x_" + str(e)
            )
            if e[0] not in flow_out:
                flow_out[e[0]] = set()
            if e[1] not in flow_in:
                flow_in[e[1]] = set()
            flow_in[e[1]].add(e)
            flow_out[e[0]].add(e)
        if all((e in edges) for e in edges_to_add):
            for e in edges:
                if e in edges_to_add:
                    x_var[e].start = 1
                    x_var[e].varhintval = 1
                else:
                    x_var[e].start = 0
                    x_var[e].varhintval = 0
        constraint_tour_2length = {}
        cnt_tour = 0
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                constraint_tour_2length[cnt_tour] = tsp_model.addLConstr(
                    x_var[edge] + x_var[(edge[1], edge[0])] <= 1,
                    name="Tour_" + str(cnt_tour),
                )
                cnt_tour += 1
        tsp_model.update()
        constraint_flow: dict[Union[Node, tuple[Node, str], tuple[Node, int]], Any] = {}
        for n in flow_in:
            if n != self.problem.start_index and n != self.problem.end_index:
                constraint_flow[n] = tsp_model.addLConstr(
                    quicksum(
                        [x_var[i] for i in flow_in[n]]
                        + [-x_var[i] for i in flow_out[n]]
                    )
                    == 0,
                    name="flow_" + str(n),
                )
            if n != self.problem.start_index:
                constraint_flow[(n, "sub")] = tsp_model.addLConstr(
                    quicksum([x_var[i] for i in flow_in[n]]) == 1,
                    name="flowin_" + str(n),
                )
            if n == self.problem.start_index:
                constraint_flow[(n, 0)] = tsp_model.addLConstr(
                    quicksum([x_var[i] for i in flow_out[n]]) == 1,
                    name="flowoutsource_" + str(n),
                )
                if n != self.problem.end_index:
                    constraint_flow[(n, 1)] = tsp_model.addLConstr(
                        quicksum([x_var[i] for i in flow_in[n]]) == 0,
                        name="flowinsource_" + str(n),
                    )
            if n == self.problem.end_index:
                constraint_flow[(n, 0)] = tsp_model.addLConstr(
                    quicksum([x_var[i] for i in flow_in[n]]) == 1,
                    name="flowinsink_" + str(n),
                )
                if n != self.problem.start_index:
                    constraint_flow[(n, 1)] = tsp_model.addLConstr(
                        quicksum([x_var[i] for i in flow_out[n]]) == 0,
                        name="flowoutsink_" + str(n),
                    )
        tsp_model.setParam("TimeLimit", 1000)
        tsp_model.modelSense = GRB.MINIMIZE
        tsp_model.setParam(GRB.Param.Threads, 8)
        tsp_model.setParam(GRB.Param.PoolSolutions, 10000)
        tsp_model.setParam(GRB.Param.Method, -1)
        tsp_model.setParam("MIPGapAbs", 0.001)
        tsp_model.setParam("MIPGap", 0.001)
        tsp_model.setParam("Heuristics", 0.1)

        self.model = tsp_model
        self.variables = {"x": x_var}

    def init_model_cbc(self, **kwargs: Any) -> None:
        g, g_empty, edges_in, edges_out = self.graph_builder(self.problem)
        tsp_model = pywraplp.Solver(
            "TSP-master", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )
        edges = set(g.edges())
        self.edges = edges
        self.g = g
        x_var = {}  # decision variables on edges
        dummy_sol = self.problem.get_dummy_solution()
        path = (
            [self.problem.start_index]
            + dummy_sol.permutation
            + [self.problem.end_index]
        )
        edges_to_add = {(e0, e1) for e0, e1 in zip(path[:-1], path[1:])}
        flow_in: dict[Node, set[Edge]] = {}
        flow_out: dict[Node, set[Edge]] = {}
        for e in edges:
            x_var[e] = tsp_model.BoolVar("x_" + str(e))
            if e[0] not in flow_out:
                flow_out[e[0]] = set()
            if e[1] not in flow_in:
                flow_in[e[1]] = set()
            flow_in[e[1]].add(e)
            flow_out[e[0]].add(e)
        if all((e in edges) for e in edges_to_add):
            for e in edges:
                if e in edges_to_add:
                    tsp_model.SetHint([x_var[e]], [1])
                else:
                    tsp_model.SetHint([x_var[e]], [0])
        constraint_tour_2length = {}
        cnt_tour = 0
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                constraint_tour_2length[cnt_tour] = tsp_model.Add(
                    x_var[edge] + x_var[(edge[1], edge[0])] <= 1
                )
                cnt_tour += 1
        constraint_flow: dict[Union[Node, tuple[Node, str], tuple[Node, int]], Any] = {}
        for n in flow_in:
            if n != self.problem.start_index and n != self.problem.end_index:
                constraint_flow[n] = tsp_model.Add(
                    tsp_model.Sum(
                        [x_var[i] for i in flow_in[n]]
                        + [-x_var[i] for i in flow_out[n]]
                    )
                    == 0
                )
            if n != self.problem.start_index:
                constraint_flow[(n, "sub")] = tsp_model.Add(
                    tsp_model.Sum([x_var[i] for i in flow_in[n]]) == 1
                )
            if n == self.problem.start_index:
                constraint_flow[(n, 0)] = tsp_model.Add(
                    tsp_model.Sum([x_var[i] for i in flow_out[n]]) == 1
                )
                if n != self.problem.end_index:
                    constraint_flow[(n, 1)] = tsp_model.Add(
                        tsp_model.Sum([x_var[i] for i in flow_in[n]]) == 0
                    )
            if n == self.problem.end_index:
                constraint_flow[(n, 0)] = tsp_model.Add(
                    tsp_model.Sum([x_var[i] for i in flow_in[n]]) == 1
                )
                if n != self.problem.start_index:
                    constraint_flow[(n, 1)] = tsp_model.Add(
                        tsp_model.Sum([x_var[i] for i in flow_out[n]]) == 0
                    )
        value = tsp_model.Sum([x_var[i] * g[i[0]][i[1]]["weight"] for i in x_var])
        tsp_model.Minimize(value)
        self.model = tsp_model
        self.variables = {"x": x_var}
        self.model.SetTimeLimit(60000)

    def retrieve_results_cbc(self) -> tuple[nx.DiGraph, set[Edge]]:
        g_empty = nx.DiGraph()
        g_empty.add_nodes_from([i for i in range(self.node_count)])
        x_solution: set[Edge] = set()
        x_var = self.variables["x"]
        for e in x_var:
            value = x_var[e].solution_value()
            if value >= 0.5:
                x_solution.add(e)
                g_empty.add_edge(e[0], e[1], weight=1)
        return g_empty, x_solution

    def retrieve_results_gurobi(self) -> tuple[nx.DiGraph, set[Edge]]:
        g_empty = nx.DiGraph()
        g_empty.add_nodes_from([i for i in range(self.node_count)])
        x_solution: set[Edge] = set()
        x_var = self.variables["x"]
        for e in x_var:
            value = x_var[e].getAttr("X")
            if value >= 0.5:
                x_solution.add(e)
                g_empty.add_edge(e[0], e[1], weight=1)
        return g_empty, x_solution

    def solve(self, **kwargs: Any) -> ResultStorage:
        nb_iteration_max = kwargs.get("nb_iteration_max", 20)
        plot = kwargs.get("plot", True)
        plot_folder: Optional[str] = kwargs.get("plot_folder", None)
        if plot_folder is not None:
            os.makedirs(plot_folder, exist_ok=True)
        tsp_model = self.model
        logger.info("optimizing...")
        objective: float
        if self.method == MILPSolver.GUROBI:
            tsp_model.optimize()
            nSolutions = tsp_model.SolCount
            nObjectives = tsp_model.NumObj
            objective = tsp_model.getObjective().getValue()
            logger.info(f"Problem has {nObjectives} objectives")
            logger.info(f"Gurobi found {nSolutions} solutions")
            status = tsp_model.getAttr("Status")
        else:
            self.model.Solve()
            res = self.model.Solve()
            resdict = {
                0: "OPTIMAL",
                1: "FEASIBLE",
                2: "INFEASIBLE",
                3: "UNBOUNDED",
                4: "ABNORMAL",
                5: "MODEL_INVALID",
                6: "NOT_SOLVED",
            }
            logger.debug(f"Result : {resdict[res]}")
            objective = self.model.Objective().Value()
        finished = False
        solutions: list[set[Edge]] = []
        cost: list[float] = []
        nb_components: list[int] = []
        iteration = 0
        rebuilt_solution: list[list[int]] = []
        rebuilt_obj: list[float] = []
        best_solution_rebuilt_index = 0
        best_solution_rebuilt = float("inf")
        while not finished:
            if self.method == MILPSolver.GUROBI:
                g_empty, x_solution = self.retrieve_results_gurobi()
            if self.method == MILPSolver.CBC:
                g_empty, x_solution = self.retrieve_results_cbc()
            connected_components: list[tuple[set[Node], int]] = [
                (set(e), len(e)) for e in nx.weakly_connected_components(g_empty)
            ]
            logger.debug(f"Connected component : {len(connected_components)}")
            sorted_connected_component: list[tuple[set[Node], int]] = sorted(
                connected_components, key=lambda x: x[1], reverse=True
            )
            nb_components += [len(sorted_connected_component)]
            cost += [objective]
            solutions += [x_solution.copy()]
            paths_component: dict[int, list[int]] = {}
            indexes_component: dict[int, dict[int, int]] = {}
            node_to_component: dict[int, int] = {}
            nb_component = len(sorted_connected_component)
            x_var = self.variables["x"]
            for i in range(nb_component):
                s = sorted_connected_component[i]
                paths_component[i], indexes_component[i] = build_the_cycles(
                    x_solution=x_solution,
                    component=s[0],
                    graph=self.g,
                    start_index=self.start_index,
                    end_index=self.end_index,
                )
                node_to_component.update({p: i for p in paths_component[i]})
                edge_in_of_interest = [
                    e for e in self.edges if e[1] in s[0] and e[0] not in s[0]
                ]
                edge_out_of_interest = [
                    e for e in self.edges if e[0] in s[0] and e[1] not in s[0]
                ]
                if self.method == MILPSolver.GUROBI:
                    tsp_model.addLConstr(
                        quicksum([x_var[e] for e in edge_in_of_interest]) >= 1
                    )
                    tsp_model.addLConstr(
                        quicksum([x_var[e] for e in edge_out_of_interest]) >= 1
                    )
                if self.method == MILPSolver.CBC:
                    tsp_model.Add(
                        tsp_model.Sum([x_var[e] for e in edge_in_of_interest]) >= 1
                    )
                    tsp_model.Add(
                        tsp_model.Sum([x_var[e] for e in edge_out_of_interest]) >= 1
                    )
            logger.debug((len(node_to_component), self.node_count))
            logger.debug(len(x_solution))
            rebuilt, objective_dict = rebuild_tsp_routine(
                sorted_connected_component,
                paths_component,
                node_to_component,
                indexes_component,
                self.g,
                self.edges,
                self.node_count,
                self.list_points,
                self.problem.evaluate_function_indexes,
                self.problem,
                self.start_index,
                self.end_index,
            )
            objective = self.aggreg_from_dict(objective_dict)
            rebuilt_solution += [rebuilt]
            rebuilt_obj += [objective]
            if objective < best_solution_rebuilt:
                best_solution_rebuilt = objective
                best_solution_rebuilt_index = iteration
            if len(sorted_connected_component) > 1:
                edges_to_add = {(e0, e1) for e0, e1 in zip(rebuilt[:-1], rebuilt[1:])}
                if all((e in self.edges) for e in edges_to_add):
                    for e in x_var:
                        if e in edges_to_add:
                            if self.method == MILPSolver.GUROBI:
                                x_var[e].start = 1
                                x_var[e].varhintval = 1
                            elif self.method == MILPSolver.CBC:
                                tsp_model.SetHint([x_var[e]], [1])
                        else:
                            if self.method == MILPSolver.GUROBI:
                                x_var[e].start = 0
                                x_var[e].varhintval = 0
                            elif self.method == MILPSolver.CBC:
                                tsp_model.SetHint([x_var[e]], [1])
                else:
                    logger.debug([e for e in edges_to_add if e not in self.edges])
                if self.method == MILPSolver.GUROBI:
                    tsp_model.update()
                    tsp_model.optimize()
                if self.method == MILPSolver.CBC:
                    tsp_model.Solve()
                iteration += 1
            else:
                finished = True
            finished = finished or iteration >= nb_iteration_max
            if self.method == MILPSolver.GUROBI:
                objective = tsp_model.getObjective().getValue()
            elif self.method == MILPSolver.CBC:
                objective = self.model.Objective().Value()
            logger.debug(f"Objective : {objective}")
        if plot or plot_folder is not None:
            self.plot_solve(
                solutions=solutions,
                rebuilt_solution=rebuilt_solution,
                cost=cost,
                nb_components=nb_components,
                rebuilt_obj=rebuilt_obj,
                show=plot,
                plot_folder=plot_folder,
            )
        logger.debug(f"Best solution : {best_solution_rebuilt}")
        logger.debug(rebuilt_obj[best_solution_rebuilt_index])
        path = rebuilt_solution[best_solution_rebuilt_index]
        var_tsp = TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=path[1:-1],
            lengths=None,
            length=None,
        )
        fit = self.aggreg_from_sol(var_tsp)
        return self.create_result_storage(
            [(var_tsp, fit)],
        )

    def plot_solve(
        self,
        solutions: list[set[Edge]],
        rebuilt_solution: list[list[int]],
        cost: list[float],
        nb_components: list[int],
        rebuilt_obj: list[float],
        show: bool = True,
        plot_folder: Optional[str] = None,
    ) -> None:
        # implemented only for list of 2d points
        if not all(isinstance(point, Point2D) for point in self.list_points):
            raise NotImplementedError(
                "plot_solve() is only implemented for list of Point2D"
            )
        else:
            list_points = cast(Sequence[Point2D], self.list_points)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for i in range(len(solutions)):
            ll = []
            ax[0].clear()
            ax[1].clear()
            for e in solutions[i]:
                ll.append(
                    ax[0].plot(
                        [list_points[e[0]].x, list_points[e[1]].x],
                        [list_points[e[0]].y, list_points[e[1]].y],
                        color="b",
                    )
                )
            ax[1].plot(
                [list_points[n].x for n in rebuilt_solution[i]],
                [list_points[n].y for n in rebuilt_solution[i]],
                color="orange",
            )
            ax[0].set_title(
                "iter "
                + str(i)
                + " obj="
                + str(int(cost[i]))
                + " nbcomp="
                + str(nb_components[i])
            )
            ax[1].set_title("iter " + str(i) + " obj=" + str(int(rebuilt_obj[i])))
            if plot_folder is not None:
                fig.savefig(os.path.join(plot_folder, "tsp_" + str(i) + ".png"))
            if show:
                plt.draw()
                plt.pause(1)

        if show:
            plt.show()


def build_the_cycles(
    x_solution: set[Edge],
    component: set[Node],
    graph: nx.DiGraph,
    start_index: Node,
    end_index: Node,
) -> tuple[list[Node], dict[Node, int]]:
    edge_of_interest = {
        e for e in x_solution if e[1] in component and e[0] in component
    }
    innn: dict[Node, Edge] = {e[1]: e for e in edge_of_interest}
    outt: dict[Node, Edge] = {e[0]: e for e in edge_of_interest}
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


def rebuild_tsp_routine(
    sorted_connected_component: Sequence[tuple[set[Node], int]],
    paths_component: dict[int, list[Node]],
    node_to_component: dict[Node, int],
    indexes: dict[int, dict[Node, int]],
    graph: nx.DiGraph,
    edges: set[Edge],
    nodeCount: int,
    list_points: Sequence[Point],
    evaluate_function_indexes: Callable[[int, int], float],
    tsp_model: TspProblem,
    start_index: Node = 0,
    end_index: Node = 0,
) -> tuple[list[Node], dict[str, float]]:
    rebuilded_path: list[Node] = list(paths_component[node_to_component[start_index]])
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
            index_path = {rebuilded_path[i]: i for i in range(len(rebuilded_path))}
            edge_out_of_interest = {
                e for e in edges if e[0] in path_set and e[1] not in path_set
            }
            edge_in_of_interest = {
                e for e in edges if e[0] not in path_set and e[1] in path_set
            }
            min_out_edge = None
            min_index_in_path: Optional[int] = None
            min_component: Optional[int] = None
            min_dist = float("inf")
            backup_min_out_edge = None
            backup_min_in_edge = None
            backup_min_index_in_path: Optional[int] = None
            backup_min_component: Optional[int] = None
            backup_min_dist = float("inf")
            for e in edge_out_of_interest:
                index_in = index_path[e[0]]
                if index_in == total_length_path - 1:
                    continue
                index_in_1 = index_path[e[0]] + 1
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
                if (next_node_component_e1, next_node_1) in edge_in_of_interest:
                    cost = (
                        graph[e[0]][e[1]]["weight"]
                        + graph[next_node_component_e1][next_node_1]["weight"]
                        - graph[e[0]][next_node_1]["weight"]
                    )
                    if cost < min_dist:
                        min_component = node_to_component[e[1]]
                        min_out_edge = e
                        min_index_in_path = index_in
                        min_dist = cost
                else:
                    cost = graph[e[0]][e[1]]["weight"]
                    if cost < backup_min_dist:
                        backup_min_component = node_to_component[e[1]]
                        backup_min_out_edge = e
                        backup_min_in_edge = (next_node_component_e1, next_node_1)
                        backup_min_index_in_path = index_in
                        backup_min_dist = cost

            if (
                min_out_edge is None
                or min_component is None
                or min_index_in_path is None
            ):
                if (
                    backup_min_component is None
                    or backup_min_out_edge is None
                    or backup_min_in_edge is None
                    or backup_min_index_in_path is None
                ):
                    # for mypy to realize that we must have define backup values at this point
                    raise RuntimeError("backup values cannot be None now.")
                e = backup_min_in_edge
                graph.add_edge(e[0], e[1], weight=evaluate_function_indexes(e[0], e[1]))
                graph.add_edge(e[1], e[0], weight=evaluate_function_indexes(e[1], e[0]))
                min_out_edge = backup_min_out_edge
                min_index_in_path = backup_min_index_in_path
                min_component = backup_min_component
            len_this_component = len(paths_component[min_component])
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
                    graph.add_edge(e1, e2, weight=evaluate_function_indexes(e1, e2))
            path_set = set(rebuilded_path)
            total_length_path = len(rebuilded_path)
            component_reconnected.add(min_component)
    var = TspSolution(
        problem=tsp_model,
        start_index=start_index,
        end_index=end_index,
        permutation=rebuilded_path[1:-1],
        lengths=None,
        length=None,
    )
    fit = tsp_model.evaluate(var)
    logger.debug(f"ObjRebuilt={fit}")
    return rebuilded_path, fit
