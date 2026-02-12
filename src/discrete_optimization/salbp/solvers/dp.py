#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import math

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, dp
from discrete_optimization.salbp.problem import SalbpProblem, SalbpSolution


class DpSalbpSolver(DpSolver):
    problem: SalbpProblem

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.transition_dict = {}

    def init_model(self, **kwargs):
        """Model adapted from :
        https://github.com/domain-independent-dp/didp-rs/blob/main/didppy/examples/salbp-1.ipynb"""
        times = [self.problem.task_times[t] for t in self.problem.tasks]
        c = self.problem.cycle_time
        # The weight in the first term of the second bound.
        weight_2_1 = [
            1 if times[i] > c / 2 else 0 for i in range(self.problem.number_of_tasks)
        ]
        # The weight in the second term of the second bound.
        weight_2_2 = [
            1 / 2 if times[i] == c / 2 else 0
            for i in range(self.problem.number_of_tasks)
        ]
        # The weight in the third bound (truncated to three decimal points).
        weight_3 = [
            1.0
            if times[i] > c * 2 / 3
            else 2 / 3 // 0.001 * 1000
            if times[i] == c * 2 / 3
            else 0.5
            if times[i] > c / 3
            else 1 / 3 // 0.001 * 1000
            if times[i] == c / 3
            else 0.0
            for i in range(self.problem.number_of_tasks)
        ]
        self.model = dp.Model()
        n_tasks = self.problem.number_of_tasks
        task = self.model.add_object_type(number=n_tasks)
        unscheduled = self.model.add_set_var(
            object_type=task, target=list(range(n_tasks))
        )
        idle_time = self.model.add_int_resource_var(target=0, less_is_better=False)
        processing_time = self.model.add_int_table(times)
        predecessor_list = [set() for i in range(n_tasks)]
        for p in self.problem.predecessors:
            predecessor_list[self.problem.tasks_to_index[p]].update(
                set(
                    [
                        self.problem.tasks_to_index[pred]
                        for pred in self.problem.predecessors[p]
                    ]
                )
            )
        predecessors = self.model.add_set_table(predecessor_list, object_type=task)
        for i in range(n_tasks):
            schedule = dp.Transition(
                name=f"schedule {i}",
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (unscheduled, unscheduled.remove(i)),
                    (idle_time, idle_time - processing_time[i]),
                ],
                preconditions=[
                    unscheduled.contains(i),
                    unscheduled.isdisjoint(predecessors[i]),
                    processing_time[i] <= idle_time,
                ],
            )
            self.model.add_transition(schedule)
            self.transition_dict[f"schedule {i}"] = ("schedule", i)

        open_new = dp.Transition(
            name="open a new station",
            cost=1 + dp.IntExpr.state_cost(),
            effects=[(idle_time, c)],
            preconditions=[
                ~unscheduled.contains(i)
                | ~unscheduled.isdisjoint(predecessors[i])
                | (processing_time[i] > idle_time)
                for i in range(n_tasks)
            ],
        )
        self.model.add_transition(open_new, forced=True)
        self.transition_dict["open a new station"] = ("new_one", None)

        self.model.add_base_case([unscheduled.is_empty()])
        self.model.add_dual_bound(
            math.ceil((processing_time[unscheduled] - idle_time) / c)
        )
        weight_2_1_table = self.model.add_int_table(weight_2_1)
        weight_2_2_table = self.model.add_float_table(weight_2_2)
        self.model.add_dual_bound(
            weight_2_1_table[unscheduled]
            + math.ceil(weight_2_2_table[unscheduled])
            - (idle_time >= c / 2).if_then_else(1, 0)
        )

        weight_3_table = self.model.add_float_table(weight_3)
        self.model.add_dual_bound(
            math.ceil(weight_3_table[unscheduled])
            - (idle_time >= c / 3).if_then_else(1, 0)
        )

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        allocation = [0 for _ in range(self.problem.number_of_tasks)]
        current_station = 0
        for i, t in enumerate(sol.transitions):
            transition = self.transition_dict[t.name]
            if transition[0] == "new_one":
                current_station += 1
            else:
                j_task = transition[1]
                allocation[j_task] = current_station
        return SalbpSolution(problem=self.problem, allocation_to_station=allocation)
