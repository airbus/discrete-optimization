#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import networkx as nx
import pytest

from discrete_optimization.coloring.coloring_model import ColoringProblem
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.qiskit_tools import qiskit_available
from discrete_optimization.generic_tools.quantum_solvers import (
    solve,
    solve_coloring,
    solvers_map_coloring,
    solvers_map_mis,
    solvers_map_tsp,
)
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.tsp.tsp_model import Point2D, TSPModel2D

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not qiskit_available, reason="You need Qiskit modules to this test."
)
@pytest.mark.parametrize("solver_class", solvers_map_coloring)
def test_solvers_coloring(solver_class):
    nodes = [(1, {}), (2, {}), (3, {}), (4, {})]
    edges = [(1, 2, {}), (1, 3, {}), (2, 4, {})]
    nb_colors = 2
    coloring_model: ColoringProblem = ColoringProblem(Graph(nodes=nodes, edges=edges))
    results = solve_coloring(
        method=solver_class,
        problem=coloring_model,
        nb_color=nb_colors,
        **solvers_map_coloring[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


@pytest.mark.skipif(
    not qiskit_available, reason="You need Qiskit modules to this test."
)
@pytest.mark.parametrize("solver_class", solvers_map_mis)
def test_solvers_mis(solver_class):
    graph = nx.Graph()

    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(2, 6)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    graph.add_edge(4, 6)
    mis_model: MisProblem = MisProblem(graph)
    results = solve(
        method=solver_class, problem=mis_model, **solvers_map_mis[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


@pytest.mark.skipif(
    not qiskit_available, reason="You need Qiskit modules to this test."
)
@pytest.mark.parametrize("solver_class", solvers_map_tsp)
def test_solver_TSP(solver_class):

    p1 = Point2D(0, 0)
    p2 = Point2D(-1, 1)
    p3 = Point2D(1, -1)
    p4 = Point2D(1, 1)
    p5 = Point2D(1, -2)
    tspProblem: TSPModel2D = TSPModel2D(
        [p1, p2, p3, p4, p5], 5, start_index=0, end_index=4
    )
    results = solve(
        method=solver_class, problem=tspProblem, **solvers_map_mis[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


if __name__ == "__main__":
    test_solvers_coloring()
    test_solvers_mis()
