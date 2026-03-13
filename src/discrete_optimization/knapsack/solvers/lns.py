#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.lns_tools import InitialSolution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.problem import KnapsackProblem
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver


class InitialKnapsackMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialKnapsackSolution(InitialSolution):
    hyperparameters = [
        EnumHyperparameter(
            name="initial_method",
            enum=InitialKnapsackMethod,
        ),
    ]

    def __init__(
        self,
        problem: KnapsackProblem,
        initial_method: InitialKnapsackMethod,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        self.initial_method = initial_method
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=params_objective_function
        )

    def get_starting_solution(self) -> ResultStorage:
        if self.initial_method == InitialKnapsackMethod.GREEDY:
            greedy_solver = GreedyBestKnapsackSolver(
                self.problem, params_objective_function=self.params_objective_function
            )
            return greedy_solver.solve()
        else:
            solution = self.problem.get_dummy_solution()
            fit = self.aggreg_from_sol(solution)
            return ResultStorage(
                list_solution_fits=[(solution, fit)],
                mode_optim=self.params_objective_function.sense_function,
            )
