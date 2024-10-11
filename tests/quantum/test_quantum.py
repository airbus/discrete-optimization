#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import networkx as nx
import pytest

from discrete_optimization.coloring.problem import ColoringProblem
from discrete_optimization.facility.problem import (
    Customer,
    Facility,
    Facility2DProblem,
    Point,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.qiskit_tools import qiskit_available
from discrete_optimization.generic_tools.quantum_solvers import (
    solve,
    solve_coloring,
    solvers_map_coloring,
    solvers_map_facility,
    solvers_map_knapsack,
    solvers_map_mis,
    solvers_map_tsp,
)
from discrete_optimization.knapsack.problem import Item, KnapsackProblem
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.tsp.problem import Point2D, Point2DTspProblem

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not qiskit_available, reason="You need Qiskit modules to this test."
)
@pytest.mark.parametrize("solver_class", solvers_map_coloring)
def test_solvers_coloring(solver_class):
    nodes = [(1, {}), (2, {}), (3, {}), (4, {})]
    edges = [(1, 2, {}), (1, 3, {}), (2, 4, {})]
    nb_colors = 2
    coloring_problem: ColoringProblem = ColoringProblem(Graph(nodes=nodes, edges=edges))
    results = solve_coloring(
        method=solver_class,
        problem=coloring_problem,
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
@pytest.mark.parametrize("solver_class", solvers_map_facility)
def test_solvers_facility(solver_class):
    f1 = Facility(0, 2, 5, Point(1, 1))
    f2 = Facility(1, 1, 2, Point(-1, -1))

    c1 = Customer(0, 2, Point(2, 2))
    c2 = Customer(1, 5, Point(0, -1))

    facilityProblem = Facility2DProblem(2, 2, [f1, f2], [c1, c2])
    results = solve(
        method=solver_class,
        problem=facilityProblem,
        **solvers_map_facility[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


@pytest.mark.skipif(
    not qiskit_available, reason="You need Qiskit modules to this test."
)
@pytest.mark.parametrize("solver_class", solvers_map_tsp)
def test_solver_tsp(solver_class):

    p1 = Point2D(0, 0)
    p2 = Point2D(-1, 1)
    p3 = Point2D(1, -1)
    p4 = Point2D(1, 1)
    p5 = Point2D(1, -2)
    tspProblem: Point2DTspProblem = Point2DTspProblem(
        [p1, p2, p3, p4, p5], 5, start_index=0, end_index=4
    )
    results = solve(
        method=solver_class, problem=tspProblem, **solvers_map_tsp[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


@pytest.mark.skipif(
    not qiskit_available, reason="You need Qiskit modules to this test."
)
@pytest.mark.parametrize("solver_class", solvers_map_knapsack)
def test_solver_Knapsack(solver_class):

    max_capacity = 10

    i1 = Item(0, 4, 2)
    i2 = Item(1, 5, 2)
    i3 = Item(2, 4, 3)
    i4 = Item(3, 2, 1)
    i5 = Item(4, 5, 3)
    i6 = Item(5, 2, 1)
    knapsackProblem = KnapsackProblem([i1, i2, i3, i4, i5, i6], max_capacity)
    results = solve(
        method=solver_class,
        problem=knapsackProblem,
        **solvers_map_knapsack[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()


if __name__ == "__main__":
    test_solvers_coloring()
    test_solvers_mis()
    test_solver_tsp()
    test_solver_Knapsack()
