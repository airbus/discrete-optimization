#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.tsptw.solvers.ortools_routing import OrtoolsTspTwSolver


def test_ortools(problem):
    solver = OrtoolsTspTwSolver(problem=problem)
    solver.init_model()
    sol = solver.solve(
        gpdp_solver_kwargs=dict(callbacks=[NbIterationStopper(nb_iteration_max=59)]),
        time_limit=5,
    ).get_best_solution()
    assert problem.satisfy(sol)
