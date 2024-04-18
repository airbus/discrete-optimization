#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Type

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrickHyperparameter,
    SubBrickKwargsHyperparameter,
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
from discrete_optimization.knapsack.solvers.cp_solvers import (
    CPKnapsackMZN,
    CPKnapsackMZN2,
)
from discrete_optimization.knapsack.solvers.dyn_prog_knapsack import KnapsackDynProg
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.knapsack_asp_solver import KnapsackASPSolver
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack
from discrete_optimization.knapsack.solvers.lp_solvers import (
    KnapsackORTools,
    LPKnapsack,
    LPKnapsackCBC,
    LPKnapsackGurobi,
)

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


subsolvers = [
    KnapsackORTools,
    LPKnapsack,
    LPKnapsackCBC,
    LPKnapsackGurobi,
    KnapsackASPSolver,
    KnapsackDynProg,
    CPKnapsackMZN,
    CPKnapsackMZN2,
]


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

    hyperparameters = [
        FloatHyperparameter(
            name="proportion_to_remove", low=0.0, high=1.0, default=0.7
        ),
        IntegerHyperparameter(name="nb_iteration", low=0, high=int(10e6), default=100),
        SubBrickHyperparameter(
            name="initial_solver", choices=subsolvers, default=GreedyBest
        ),
        SubBrickKwargsHyperparameter(
            name="initial_solver_kwargs", subbrick_hyperparameter="initial_solver"
        ),
        SubBrickHyperparameter(
            name="root_solver", choices=subsolvers, default=GreedyBest
        ),
        SubBrickKwargsHyperparameter(
            name="root_solver_kwargs", subbrick_hyperparameter="root_solver"
        ),
    ]

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

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        kwargs = self.complete_with_default_hyperparameters(kwargs)
        initial_solver: Type[SolverKnapsack] = kwargs["initial_solver"]
        if kwargs["initial_solver_kwargs"] is None:
            initial_solver_kwargs: Dict[str, Any] = {}
        else:
            initial_solver_kwargs = kwargs["initial_solver_kwargs"]
        sub_solver: Type[SolverKnapsack] = kwargs["root_solver"]
        if kwargs["root_solver_kwargs"] is None:
            sub_solver_kwargs: Dict[str, Any] = {}
        else:
            sub_solver_kwargs = kwargs["root_solver_kwargs"]
        nb_iteration = kwargs["nb_iteration"]
        proportion_to_remove = kwargs["proportion_to_remove"]

        initial_results = solve(
            method=initial_solver, problem=self.problem, **initial_solver_kwargs
        )
        results_storage = ResultStorage(
            list_solution_fits=initial_results.list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )
        logger.info(
            f"Initial solution fitness : {results_storage.get_best_solution_fit()[1]}"
        )
        all_indexes = set(range(self.problem.nb_items))
        for j in range(nb_iteration):
            sol, fit = results_storage.get_best_solution_fit()
            indexes_to_remove = set(
                random.sample(
                    list(all_indexes),
                    int(proportion_to_remove * self.problem.nb_items),
                )
            )
            sub_model = create_subknapsack_model(
                knapsack_model=self.problem,
                solution=sol,
                indexes_to_remove=indexes_to_remove,
            )
            res = solve(method=sub_solver, problem=sub_model, **sub_solver_kwargs)
            best_sol, fit = res.get_best_solution_fit()
            reb_sol = self.rebuild_sol(
                sol=best_sol,
                original_solution=sol,
                original_knapsack_model=self.problem,
                indexes_to_remove=indexes_to_remove,
            )
            fit = self.aggreg_from_sol(reb_sol)
            logger.info(f"Iteration {j}/{nb_iteration} : --- Current fitness {fit}")
            results_storage.list_solution_fits += [(reb_sol, fit)]

            stopping = callbacks_list.on_step_end(
                step=j, res=results_storage, solver=self
            )
            if stopping:
                break

        # end of solve callback
        callbacks_list.on_solve_end(res=results_storage, solver=self)
        return results_storage
