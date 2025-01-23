#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.toulbar import (
    ToulbarMisSolver,
)

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_mis_toulbar():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = ToulbarMisSolver(problem=mis_model)
    solver.init_model()
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert mis_model.satisfy(sol)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_mis_toulbar_ws():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver_ws = CpSatMisSolver(problem=mis_model)
    sol_ws = solver_ws.solve(time_limit=5)[-1][0]
    solver = ToulbarMisSolver(problem=mis_model)
    solver.init_model()
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        time_limit=20,
    )
    sol = res[0][0]
    print(mis_model.evaluate(sol))
    print(mis_model.satisfy(sol))
