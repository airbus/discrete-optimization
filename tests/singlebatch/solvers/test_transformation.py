#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)
from discrete_optimization.ovensched.solvers.greedy import GreedyOvenSchedulingSolver
from discrete_optimization.singlebatch.transformations.to_ovensched import (
    SinglebatchToOvenschedTransformation,
)


def test_via_ovensched(small_problem):
    solver = TransformationSolver(
        transformation=SinglebatchToOvenschedTransformation(),
        source_problem=small_problem,
        solver_brick=SubBrick(GreedyOvenSchedulingSolver, {}),
    )
    res = solver.solve()
    sol = res[-1][0]
    print(small_problem.evaluate(sol))
    assert small_problem.satisfy(sol)
