#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import random
from typing import Any, Optional, Set

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
    create_subknapsack_model,
)
from discrete_optimization.knapsack.knapsack_solvers import solve
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class KnapsackDecomposedSolver(SolverKnapsack):
    """
    This solver is based on the current observation. From a given knapsack model and one current solution,
    if we decide to freeze the decision variable for a subset of items, the remaining problem to solve is also a
    knapsack problem, with fewer items and smaller capacity.
    A solution to this subproblem can be found by any knapsack solver and a full solution to the original problem,
    can be rebuilt.
    KnapsackDecomposedSolver is a basic iterative solver that starts from a given solution, then freeze random items,
    solve subproblem with a custom root solver, rebuild original solution and repeat the process.
    """

    def __init__(
        self,
        knapsack_model: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.knapsack_model, params_objective_function=params_objective_function
        )

    def rebuild_sol(
        self,
        sol: KnapsackSolution,
        original_knapsack_model: KnapsackModel,
        original_solution: KnapsackSolution,
        indexes_to_remove: Set[int],
    ):
        """
        Rebuild a knapsack solution object from a partial solution.
        :param sol: solution to a sub-knapsack problem
        :param original_knapsack_model: original knapsack model to solve
        :param original_solution: original base solution
        :param indexes_to_remove: indexes of item removed when building the sub-knapsack problem.
        :return: A new solution object for the original problem.
        """
        list_taken = [0 for i in range(original_knapsack_model.nb_items)]
        for i in indexes_to_remove:
            list_taken[i] = original_solution.list_taken[i]
        for j in range(len(sol.list_taken)):
            original_index = original_knapsack_model.index_to_index_list[
                sol.problem.list_items[j].index
            ]
            list_taken[original_index] = sol.list_taken[j]
        solution = KnapsackSolution(
            problem=original_knapsack_model, list_taken=list_taken
        )
        return solution

    def solve(self, **kwargs: Any) -> ResultStorage:
        initial_solver = kwargs.get("initial_solver", GreedyBest)
        sub_solver = kwargs.get("root_solver", GreedyBest)
        nb_iteration = kwargs.get("nb_iteration", 100)
        proportion_to_remove = kwargs.get("proportion_to_remove", 0.7)
        initial_results = solve(
            method=initial_solver, knapsack_model=self.knapsack_model, **kwargs
        )
        results_storage = ResultStorage(
            list_solution_fits=initial_results.list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )
        logger.info(
            f"Initial solution fitness : {results_storage.get_best_solution_fit()[1]}"
        )
        all_indexes = set(range(self.knapsack_model.nb_items))
        for j in range(nb_iteration):
            sol, fit = results_storage.get_best_solution_fit()
            indexes_to_remove = set(
                random.sample(
                    all_indexes,
                    int(proportion_to_remove * self.knapsack_model.nb_items),
                )
            )
            sub_model = create_subknapsack_model(
                knapsack_model=self.knapsack_model,
                solution=sol,
                indexes_to_remove=indexes_to_remove,
            )
            res = solve(method=sub_solver, knapsack_model=sub_model, **kwargs)
            best_sol, fit = res.get_best_solution_fit()
            reb_sol = self.rebuild_sol(
                sol=best_sol,
                original_solution=sol,
                original_knapsack_model=self.knapsack_model,
                indexes_to_remove=indexes_to_remove,
            )
            fit = self.aggreg_sol(reb_sol)
            logger.info(f"Iteration {j}/{nb_iteration} : --- Current fitness {fit}")
            results_storage.list_solution_fits += [(reb_sol, fit)]
        return results_storage
