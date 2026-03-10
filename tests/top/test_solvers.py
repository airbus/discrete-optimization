#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.top.parser import get_data_available, parse_file
from discrete_optimization.top.solvers.cpsat import CpsatTopSolver
from discrete_optimization.top.solvers.dp import DpTopSolver
from discrete_optimization.top.solvers.optal import OptalTopSolver, optalcp_available
from discrete_optimization.top.solvers.ortools import OrtoolsTopSolver

solvers = [CpsatTopSolver, DpTopSolver, OptalTopSolver, OrtoolsTopSolver]


@pytest.mark.parametrize("solver_class", solvers)
def test_several_solvers(solver_class):
    if solver_class == OptalTopSolver:
        if not optalcp_available:
            return
    files, files_dict = get_data_available()
    file = [f for f in files if "p2.2.a.txt" in f][0]
    problem = parse_file(file)
    solver = solver_class(problem)
    solver.init_model(scaling=100)
    res = solver.solve(time_limit=10)
    sol = res[-1][0]
    if solvers != OptalTopSolver:
        assert problem.satisfy(sol)
