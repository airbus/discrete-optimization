#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import (
    ParamsObjectiveFunction,
    SolverDO,
)
from discrete_optimization.salbp.problem import SalbpProblem, SalbpSolution


class GreedySalbpSolver(SolverDO):
    problem: SalbpProblem

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
    ):
        super().__init__(problem, params_objective_function)
        self.graph_precedence = self.problem.get_graph_precedence()

    def solve(self, callbacks: list[Callback] = None, **kwargs):
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        precedences = self.graph_precedence.predecessors_dict
        priority_task = self.problem.tasks_list
        allocation_dict = {}
        scheduled = set()
        workload_per_station = defaultdict(lambda: 0)
        while len(scheduled) < self.problem.number_of_tasks:
            next_one = next(
                (
                    p
                    for p in priority_task
                    if all(
                        predecessor in scheduled
                        for predecessor in precedences.get(p, {})
                    )
                    and p not in scheduled
                )
            )
            if len(precedences.get(next_one, {})) == 0:
                minimal_start = 0
            else:
                minimal_start = max([allocation_dict[p] for p in precedences[next_one]])
            task_time = self.problem.task_times[next_one]
            next_time = next(
                j
                for j in range(minimal_start, self.problem.number_of_tasks)
                if workload_per_station[j] + task_time <= self.problem.cycle_time
            )
            allocation_dict[next_one] = next_time
            workload_per_station[next_time] += task_time
            scheduled.add(next_one)
        sol = SalbpSolution(
            problem=self.problem,
            allocation_to_station=[allocation_dict[t] for t in self.problem.tasks_list],
        )
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        cb.on_solve_end(res, self)
        return res
