#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.singlemachine.solvers.cpsat import CpsatWTSolver


def test_cpsat(problem):
    solver = CpsatWTSolver(problem)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    sol = res.get_best_solution()
    assert problem.satisfy(sol)

    # warm-start
    ref = problem.get_dummy_solution()
    assert not sol.schedule == ref.schedule  # different sols before warmstart
    solver.set_warm_start(ref)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs=dict(
            log_search_progress=True, fix_variables_to_their_hinted_value=True
        ),
    )
    sol2, _ = res[0]

    assert sol2.schedule == ref.schedule  # same sols after warmstart
