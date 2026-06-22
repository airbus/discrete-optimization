#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)
from discrete_optimization.shop.jsp.solvers.cpsat_auto import CpSatAutoJspSolver
from discrete_optimization.singlemachine.transformations.to_jsp import (
    SingleMachineToJspTransformation,
)


def test_via_jsp(problem):
    solver = TransformationSolver(
        transformation=SingleMachineToJspTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(CpSatAutoJspSolver, {}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
