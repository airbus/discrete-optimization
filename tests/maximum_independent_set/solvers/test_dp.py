#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.dp import (
    DpMisSolver,
    DpModeling,
    dp,
)


def test_mis_dp():
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    mis_model: MisProblem = dimacs_parser_nx(small_example)
    solver = DpMisSolver(problem=mis_model)
    res = solver.solve(solver=dp.LNBS, time_limit=20)
    sol, fit = res.get_best_solution_fit()
    assert mis_model.satisfy(sol)
