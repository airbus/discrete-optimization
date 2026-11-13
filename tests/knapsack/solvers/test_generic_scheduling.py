#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.knapsack.transformations.to_generic import (
    KnapsackToGenericSchedulingTransformation,
)


def test_generic_knapsack(
    problem, generic_solver_cls, generic_solver_kwargs, generic_solver_check_solution
):
    p = ParametersCp.default()
    solver = TransformationSolver(
        transformation=KnapsackToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            generic_solver_cls,
            {
                "time_limit": 10,
                "parameters_cp": p,
                **generic_solver_kwargs,
            },
        ),
        source_problem=problem,
    )
    result_storage = solver.solve(callbacks=[NbIterationStopper(1)])
    sol, fit = result_storage.get_best_solution_fit()
    if generic_solver_check_solution:
        assert problem.satisfy(sol)
