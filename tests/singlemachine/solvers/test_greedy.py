#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.singlemachine.solvers.greedy import (
    GreedySingleMachineSolver,
    GreedySingleMachineWSPT,
)


def test_greedy(problem):
    solver = GreedySingleMachineSolver(problem)
    res = solver.solve()
    sol = res.get_best_solution()
    assert problem.satisfy(sol)


def test_greedy_wspt(problem):
    solver = GreedySingleMachineWSPT(problem)
    res = solver.solve()
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
