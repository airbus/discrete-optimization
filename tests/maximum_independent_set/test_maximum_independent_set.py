#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.mis_parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.mis_solvers import (
    solve,
    solvers_map,
    toulbar_available,
)
from discrete_optimization.maximum_independent_set.solvers.mis_gurobi import (
    MisMilpSolver,
    MisQuadraticSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_kamis import (
    MisKamisSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_toulbar import (
    MisToulbarSolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True

kamis_available = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers(solver_class):
    if (
        solver_class == MisMilpSolver or solver_class == MisQuadraticSolver
    ) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    if solver_class == MisToulbarSolver and not toulbar_available:
        pytest.skip("You need Toulbar2 to test this solver.")
    if solver_class == MisKamisSolver and not kamis_available:
        pytest.skip("You need Kamis to test this solver.")
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    results = solve(
        method_solver=solver_class, problem=mis_model, **solvers_map[solver_class][1]
    )
    sol, fit = results.get_best_solution_fit()
    assert mis_model.satisfy(sol)


def test_solver_cpsat():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10

    solver = MisOrtoolsSolver(mis_model)
    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )

    # test warm start
    start_solution = mis_model.get_dummy_solution()

    # first solution is not start_solution
    assert result_storage[0][0].chosen != start_solution.chosen

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].chosen == start_solution.chosen


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_solver_gurobi():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    solver = MisMilpSolver(mis_model)
    result_storage = solver.solve()

    # test warm start
    start_solution = mis_model.get_dummy_solution()

    # first solution is not start_solution
    assert result_storage[0][0].chosen != start_solution.chosen

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve()
    assert result_storage[0][0].chosen == start_solution.chosen
