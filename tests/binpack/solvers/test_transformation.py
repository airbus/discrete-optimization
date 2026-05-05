#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.binpack.transformations.to_facility import (
    BinpackToFacilityTransformation,
)
from discrete_optimization.binpack.transformations.to_rcpsp import (
    BinpackToRcpspTransformation,
)
from discrete_optimization.binpack.transformations.to_salbp import (
    BinpackToSalbpTransformation,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)


def test_via_facility(problem):
    from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver

    # remove incompatible
    problem.incompatible_items = set()
    solver = TransformationSolver(
        transformation=BinpackToFacilityTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(GreedyFacilitySolver, {}),
    )
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)


def test_via_rcpsp(problem):
    from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver

    # remove incompatible
    solver = TransformationSolver(
        transformation=BinpackToRcpspTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(PileRcpspSolver, {}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_via_salbp(problem):
    from discrete_optimization.alb.salbp.solvers.greedy import GreedySalbpSolver

    # remove incompatible
    problem.incompatible_items = set()
    solver = TransformationSolver(
        transformation=BinpackToSalbpTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(GreedySalbpSolver, {}),
    )
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
