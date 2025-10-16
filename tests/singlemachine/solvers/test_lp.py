#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.singlemachine.solvers.lp import MathOptSingleMachineSolver


def test_mathopt(problem):
    solver = MathOptSingleMachineSolver(problem)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
