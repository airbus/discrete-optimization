#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.singlemachine.solvers.dp import DpWTSolver


@pytest.mark.parametrize("add_dominated_transition", [False, True])
def test_dp(problem, add_dominated_transition):
    solver = DpWTSolver(problem)
    solver.init_model(add_dominated_transition=add_dominated_transition)
    res = solver.solve(
        solver="LNBS", callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    sol = res.get_best_solution()
    assert problem.satisfy(sol)

    # warm-start
    ref = problem.get_dummy_solution()
    assert not sol.schedule == ref.schedule  # different sols before warmstart
    solver.set_warm_start(ref)
    res = solver.solve(
        solver="DDLNS", callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    sol2, _ = res[0]
    assert sol2.schedule == ref.schedule  # same sols after warmstart
