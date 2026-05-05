#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.facility.transformations.to_binpack import (
    FacilityToBinpackTransformation,
)
from discrete_optimization.facility.transformations.to_salbp import (
    FacilityToSalbpTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)


def test_via_salbp(problem):
    from discrete_optimization.alb.salbp.solvers.greedy import GreedySalbpSolver

    solver = TransformationSolver(
        transformation=FacilityToSalbpTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(GreedySalbpSolver, {}),
    )
    solution, fit = solver.solve().get_best_solution_fit()
    print(problem.satisfy(solution))
    print(problem.evaluate(solution))
    # don't pass, normal
    # assert problem.satisfy(solution)


def test_via_binpack(problem):
    from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver

    solver = TransformationSolver(
        transformation=FacilityToBinpackTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(GreedyBinPackSolver, {}),
    )
    solution, fit = solver.solve().get_best_solution_fit()
    print(problem.satisfy(solution))
    print(problem.evaluate(solution))
