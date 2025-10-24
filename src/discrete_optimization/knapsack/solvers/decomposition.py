#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import random
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import (
    TrivialSolverFromSolution,
    WarmstartMixin,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.problem import (
    KnapsackProblem,
    KnapsackSolution,
    create_subknapsack_problem,
)
from discrete_optimization.knapsack.solvers import KnapsackSolver
from discrete_optimization.knapsack.solvers.asp import AspKnapsackSolver
from discrete_optimization.knapsack.solvers.cp_mzn import (
    Cp2KnapsackSolver,
    CpKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.dp import ExactDpKnapsackSolver
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver
from discrete_optimization.knapsack.solvers.lp import (
    CbcKnapsackSolver,
    GurobiKnapsackSolver,
    OrtoolsKnapsackSolver,
)
from discrete_optimization.knapsack.solvers_map import solve

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


subsolvers = [
    OrtoolsKnapsackSolver,
    CbcKnapsackSolver,
    GurobiKnapsackSolver,
    AspKnapsackSolver,
    ExactDpKnapsackSolver,
    CpKnapsackSolver,
    Cp2KnapsackSolver,
]


class DecomposedKnapsackSolver(KnapsackSolver, WarmstartMixin):
    """
    This solver is based on the current observation. From a given knapsack model and one current solution,
    if we decide to freeze the decision variable for a subset of items, the remaining problem to solve is also a
    knapsack problem, with fewer items and smaller capacity.
    A solution to this subproblem can be found by any knapsack solver and a full solution to the original problem,
    can be rebuilt.
    DecomposedKnapsackSolver is a basic iterative solver that starts from a given solution, then freeze random items,
    solve subproblem with a custom root solver, rebuild original solution and repeat the process.
    """

    hyperparameters = [
        FloatHyperparameter(
            name="proportion_to_remove", low=0.0, high=1.0, default=0.7
        ),
        IntegerHyperparameter(name="nb_iteration", low=0, high=int(10e6), default=100),
        SubBrickHyperparameter(
            name="initial_solver",
            choices=subsolvers,
            default=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        ),
        SubBrickHyperparameter(
            name="root_solver",
            choices=subsolvers,
            default=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        ),
    ]

    initial_solution: Optional[KnapsackSolution] = None
    """Initial solution used for warm start."""

    def rebuild_sol(
        self,
        sol: KnapsackSolution,
        original_knapsack_problem: KnapsackProblem,
        original_solution: KnapsackSolution,
        indexes_to_remove: set[int],
    ):
        """
        Rebuild a knapsack solution object from a partial solution.
        :param sol: solution to a sub-knapsack problem
        :param original_knapsack_problem: original knapsack model to solve
        :param original_solution: original base solution
        :param indexes_to_remove: indexes of item removed when building the sub-knapsack problem.
        :return: A new solution object for the original problem.
        """
        list_taken = [0 for i in range(original_knapsack_problem.nb_items)]
        for i in indexes_to_remove:
            list_taken[i] = original_solution.list_taken[i]
        for j in range(len(sol.list_taken)):
            original_index = original_knapsack_problem.index_to_index_list[
                sol.problem.list_items[j].index
            ]
            list_taken[original_index] = sol.list_taken[j]
        solution = KnapsackSolution(
            problem=original_knapsack_problem, list_taken=list_taken
        )
        return solution

    def set_warm_start(self, solution: KnapsackSolution) -> None:
        """Make the solver warm start from the given solution.

        Will be ignored if arg `initial_solver` is set and not None in call to `solve()`.

        """
        self.initial_solution = solution

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        # wrap all callbacks in a single one
        callbacks_list = CallbackList(callbacks=callbacks)
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        # manage warm start if self.initial_solution is set
        if self.initial_solution is not None:
            kwargs["initial_solver"] = SubBrick(
                cls=TrivialSolverFromSolution,
                kwargs=dict(solution=self.initial_solution),
            )

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        initial_solver: SubBrick = kwargs["initial_solver"]
        initial_solver_cls: type[KnapsackSolver] = initial_solver.cls
        initial_solver_kwargs = initial_solver.kwargs

        root_solver: SubBrick = kwargs["root_solver"]
        root_solver_cls: type[KnapsackSolver] = root_solver.cls
        root_solver_kwargs = root_solver.kwargs

        nb_iteration = kwargs["nb_iteration"]
        proportion_to_remove = kwargs["proportion_to_remove"]

        initial_results = solve(
            method=initial_solver_cls, problem=self.problem, **initial_solver_kwargs
        )
        results_storage = self.create_result_storage(
            initial_results.list_solution_fits,
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
            sub_model = create_subknapsack_problem(
                knapsack_problem=self.problem,
                solution=sol,
                indexes_to_remove=indexes_to_remove,
            )
            res = solve(method=root_solver_cls, problem=sub_model, **root_solver_kwargs)
            best_sol, fit = res.get_best_solution_fit()
            reb_sol = self.rebuild_sol(
                sol=best_sol,
                original_solution=sol,
                original_knapsack_problem=self.problem,
                indexes_to_remove=indexes_to_remove,
            )
            fit = self.aggreg_from_sol(reb_sol)
            logger.info(f"Iteration {j}/{nb_iteration} : --- Current fitness {fit}")
            results_storage.append((reb_sol, fit))

            stopping = callbacks_list.on_step_end(
                step=j, res=results_storage, solver=self
            )
            if stopping:
                break

        # end of solve callback
        callbacks_list.on_solve_end(res=results_storage, solver=self)
        return results_storage
