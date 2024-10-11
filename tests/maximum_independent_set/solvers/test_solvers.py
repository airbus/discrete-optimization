#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest
from ortools.math_opt.python import mathopt

from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.gurobi import (
    GurobiMisSolver,
    GurobiQuadraticMisSolver,
)
from discrete_optimization.maximum_independent_set.solvers.kamis import KamisMisSolver
from discrete_optimization.maximum_independent_set.solvers.mathopt import (
    MathOptMisSolver,
    MathOptQuadraticMisSolver,
)
from discrete_optimization.maximum_independent_set.solvers.toulbar import (
    ToulbarMisSolver,
)
from discrete_optimization.maximum_independent_set.solvers_map import solve, solvers_map

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True
kamis_available = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers(solver_class):
    if (
        solver_class == GurobiMisSolver or solver_class == GurobiQuadraticMisSolver
    ) and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    if solver_class == ToulbarMisSolver and not toulbar_available:
        pytest.skip("You need Toulbar2 to test this solver.")
    if solver_class == KamisMisSolver and not kamis_available:
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

    solver = CpSatMisSolver(mis_model)
    result_storage = solver.solve(
        time_limit=10,
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
        time_limit=10,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].chosen == start_solution.chosen


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_solver_gurobi():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    solver = GurobiMisSolver(mis_model)
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


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_solver_quad_gurobi():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    solver = GurobiQuadraticMisSolver(mis_model)
    result_storage = solver.solve()

    # test warm start
    start_solver = GurobiMisSolver(mis_model)
    start_solution = start_solver.solve().get_best_solution()

    # first solution is not start_solution
    assert result_storage[0][0].chosen != start_solution.chosen

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve()
    assert result_storage[0][0].chosen == start_solution.chosen


def test_solver_mathopt():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    solver = MathOptMisSolver(mis_model)
    kwargs = dict(
        mathopt_solver_type=mathopt.SolverType.CP_SAT, mathopt_enable_output=True
    )
    result_storage = solver.solve(**kwargs)

    sol, fit = result_storage.get_best_solution_fit(satisfying=mis_model)
    assert isinstance(sol, MisSolution)

    # test warm start with no assert (sometimes mathopt start from a feasible solution
    # but provides only the next optimized solution in results)
    solver.set_warm_start(sol)


def test_solver_quad_mathopt():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)

    solver = MathOptQuadraticMisSolver(mis_model)
    kwargs = dict(
        mathopt_solver_type=mathopt.SolverType.GSCIP, mathopt_enable_output=True
    )
    result_storage = solver.solve(**kwargs)

    sol, fit = result_storage.get_best_solution_fit(satisfying=mis_model)
    assert isinstance(sol, MisSolution)

    # test warm start with no assert (sometimes mathopt start from a feasible solution
    # but provides only the next optimized solution in results)
    solver.set_warm_start(sol)
