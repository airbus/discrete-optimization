#  Copyright (c) 2026s AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.coloring.transformations.to_generic import (
    ColoringToGenericSchedulingTransformation,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver


def test_via_generic_scheduling(
    problem, generic_solver_cls, generic_solver_kwargs, generic_solver_check_solution
):
    p = ParametersCp.default()
    solver = TransformationSolver(
        transformation=ColoringToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            generic_solver_cls,
            {
                "time_limit": 20,
                "parameters_cp": p,
                **generic_solver_kwargs,
            },
        ),
        source_problem=problem,
    )
    res = solver.solve()
    if generic_solver_check_solution:
        sol = res[-1][0]
        assert problem.satisfy(sol)
