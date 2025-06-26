#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver


def test_greedy(problem):
    solver = GreedyBinPackSolver(problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)
