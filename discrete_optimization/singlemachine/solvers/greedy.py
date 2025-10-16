#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


class GreedySingleMachineSolver(SolverDO):
    """
    Greedy solver inserting jobs only using their initial index.
    """

    problem: WeightedTardinessProblem

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        sol = self.problem.get_dummy_solution()
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        cb.on_solve_end(res, self)
        return res


class GreedySingleMachineWSPT(SolverDO):
    """
    Greedy solver for the Single Machine Weighted Tardiness problem based on the
    Weighted Shortest Processing Time (WSPT) heuristic.

    It sorts jobs in increasing order of (processing_time / weight).
    """

    problem: WeightedTardinessProblem

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        # Create a list of job indices to be sorted
        jobs = list(range(self.problem.num_jobs))

        # Calculate the WSPT ratio for each job
        wspt_ratios = [
            self.problem.processing_times[i] / self.problem.weights[i] for i in jobs
        ]
        # Sort jobs based on the WSPT ratio (ascending)
        sorted_jobs = sorted(jobs, key=lambda i: wspt_ratios[i])
        # The sorted list is our new permutation
        permutation = sorted_jobs
        # Create a solution object from the permutation
        sol = WTSolution(problem=self.problem, permutation=permutation)
        # Evaluate the solution
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        cb.on_solve_end(res, self)
        return res
