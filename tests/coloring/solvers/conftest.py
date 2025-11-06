#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import fixture, param_fixture

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.greedy import GreedyColoringSolver

with_subset_nodes = param_fixture("with_subset_nodes", [False, True])
with_coloring_constraint = param_fixture("with_coloring_constraint", [False, True])


@fixture
def problem(with_subset_nodes, with_coloring_constraint) -> ColoringProblem:
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    problem = parse_file(small_example)
    nb_subset_nodes = 5
    if with_subset_nodes:
        subset_nodes = {problem.nodes_name[i] for i in range(nb_subset_nodes)}
        problem = ColoringProblem(
            graph=problem.graph,
            subset_nodes=subset_nodes,
            constraints_coloring=problem.constraints_coloring,
        )
    if with_coloring_constraint:
        constraints_coloring = ColoringConstraints({problem.nodes_name[0]: 1})
        problem = ColoringProblem(
            graph=problem.graph,
            subset_nodes=problem.subset_nodes,
            constraints_coloring=constraints_coloring,
        )
    return problem


@fixture
def start_solution(problem, with_coloring_constraint) -> ColoringSolution:
    greedy_solver = GreedyColoringSolver(problem)
    start_solution: ColoringSolution = greedy_solver.solve().get_best_solution()
    if with_coloring_constraint:
        # permute color to satisfy coloring constraint color(node0) = 1
        colors = start_solution.colors
        old_color0 = colors[0]
        expected_color0 = 1
        unexisting_color = -1
        if old_color0 != expected_color0:
            colors = [unexisting_color if c == old_color0 else c for c in colors]
            colors = [old_color0 if c == expected_color0 else c for c in colors]
            colors = [expected_color0 if c == unexisting_color else c for c in colors]
            start_solution.colors = colors
        assert problem.satisfy(start_solution)
    return start_solution
