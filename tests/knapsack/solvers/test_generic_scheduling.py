#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from pytest_cases import fixture

from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto_impl import (
    GenericSchedulingAutoCpSatImplSolver,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.transformations.to_generic import (
    KnapsackToGenericSchedulingTransformation,
)


@fixture
def solver(problem):
    p = ParametersCp.default()
    solver = TransformationSolver(
        transformation=KnapsackToGenericSchedulingTransformation(),
        solver_brick=SubBrick(
            GenericSchedulingAutoCpSatImplSolver,
            {
                "time_limit": 10,
                "parameters_cp": p,
            },
        ),
        source_problem=problem,
    )
    return solver


def test_generic_knapsack(problem, solver, start_solution):
    result_storage = solver.solve(time_limit=10)
    sol: KnapsackSolution
    sol, fit = result_storage.get_best_solution_fit()
    assert problem.satisfy(sol)
