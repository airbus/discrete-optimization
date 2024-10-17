#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.dp import (
    DpMisSolver,
    DpModeling,
    dp,
)


@pytest.mark.parametrize("modeling", [DpModeling.ORDER, DpModeling.ANY_ORDER])
def test_mis_dp(modeling):
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DpMisSolver(problem=mis_model)
    res = solver.solve(solver=dp.LNBS, modeling=modeling, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert mis_model.satisfy(sol)


@pytest.mark.parametrize("modeling", [DpModeling.ORDER, DpModeling.ANY_ORDER])
def test_dip_solver_ws(modeling):
    small_example = [f for f in get_data_available() if "1tc.1024" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver_ws = CpSatMisSolver(problem=mis_model)
    sol_ws = solver_ws.solve(time_limit=5)[-1][0]
    solver = DpMisSolver(problem=mis_model)
    solver.init_model(modeling=modeling)
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        use_callback=True,
        time_limit=100,
    )
    sol = res[0][0]
    print(mis_model.evaluate(sol))
    print(mis_model.satisfy(sol))
    assert sol.chosen == sol_ws.chosen
