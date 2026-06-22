#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Basic greedy solver for CommonShopProblem
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import (
    ResultStorage,
    SolverDO,
)
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem


class GreedyShopSolver(SolverDO):
    problem: CommonShopProblem

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        schedule_dict = {}
        sched_per_job = {i: [] for i in range(len(self.problem.list_jobs))}
        schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}

        queue = list(self.problem.tasks_list)
        while queue:
            i, j = queue.pop(0)
            default_recipe = self.problem.list_jobs[i].subjobs[j].recipes[0]
            machine = default_recipe.machine_index
            proc_time = default_recipe.processing_time
            next_start = 0
            if len(sched_per_job[i]) >= 1:
                next_start = sched_per_job[i][-1][1]
            if len(schedule_per_machine[machine]) > 0:
                next_start = max(next_start, schedule_per_machine[machine][-1][1])
            sched_per_job[i].append((next_start, next_start + proc_time))
            schedule_per_machine[machine].append((next_start, next_start + proc_time))
            schedule_dict[i, j] = (next_start, next_start + proc_time)
        sol = AnyShopSolution(
            problem=self.problem,
            schedule=[
                [schedule_dict[i, j] for j in range(self.problem.nb_subjob_per_job[i])]
                for i in range(self.problem.n_jobs)
            ],
        )
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        cb.on_solve_end(res, self)
        return res
