#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from collections.abc import Iterable
from enum import Enum
from typing import Any

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.lns_mip import (
    InitialSolution,
    OrtoolsMathOptConstraintHandler,
)
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers.greedy import (
    GreedyBestKnapsackSolver,
    ResultStorage,
)
from discrete_optimization.knapsack.solvers.lp import MathOptKnapsackSolver


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
        params_objective_function: ParamsObjectiveFunction,
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


class MathOptKnapsackConstraintHandler(OrtoolsMathOptConstraintHandler):
    def __init__(self, problem: KnapsackProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: MathOptKnapsackSolver,
        result_storage: ResultStorage,
        **kwargs: Any
    ) -> Iterable[Any]:
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a KnapsackSolution."
            )
        solver.set_warm_start(current_solution)

        x_var = solver.variable_decision["x"]
        lns_constraint = []
        for c in range(self.problem.nb_items):
            if c in subpart_item:
                lns_constraint.append(
                    solver.add_linear_constraint(
                        x_var[c] == current_solution.list_taken[c], name=str(c)
                    )
                )
        return lns_constraint
