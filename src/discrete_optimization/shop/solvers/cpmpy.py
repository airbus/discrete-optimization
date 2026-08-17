#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

import cpmpy as cp

from discrete_optimization.generic_tools.cpmpy_tools import CpmpySolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem


class CpmpyShopSolver(CpmpySolver):
    hyperparameters = [
        CategoricalHyperparameter(
            name="link_mode_to_duration", choices=[True, False], default=False
        )
    ]
    problem: CommonShopProblem

    def __init__(
        self,
        problem: CommonShopProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self._max_time = None

    def get_makespan_upper_bound(self) -> int:
        if self._max_time is None:
            return self.problem.get_makespan_upper_bound()
        else:
            return min(self._max_time, self.problem.get_makespan_upper_bound())

    def init_model(self, **kwargs: Any) -> None:
        # optional parameters
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        link_mode_to_duration = kwargs["link_mode_to_duration"]
        self._max_time: int | None = kwargs.get(
            "max_time", None
        )  # update the upper bound for makespan
        self.model = cp.Model()
        # Decision `start[j]`: integer start time for each job `j`
        start = cp.intvar(
            0,
            self.get_makespan_upper_bound(),
            shape=self.problem.n_all_jobs,
            name="start",
        )
        # Decision `end[j]`: integer end time for each job `j`
        end = cp.intvar(
            0,
            self.get_makespan_upper_bound(),
            shape=self.problem.n_all_jobs,
            name="end",
        )
        max_dur = self.problem.get_max_duration_of_tasks()
        duration = cp.intvar(
            lb=0, ub=max_dur, name="duration", shape=self.problem.n_all_jobs
        )
        modes = {}
        for t in self.problem.tasks_list:
            for m in self.problem.get_task_modes(t):
                modes[(t, m)] = cp.boolvar(1, name=f"mode_{t}_{m}")
                if link_mode_to_duration:
                    self.model += [
                        modes[(t, m)].implies(
                            duration[self.problem.get_index_from_task(t)]
                            == self.problem.get_task_mode_duration(t, m)
                        )
                    ]
            self.model += [
                cp.sum([modes[(t, m)] for m in self.problem.get_task_modes(t)]) == 1
            ]
        self.variables["start"] = start
        self.variables["end"] = end
        self.variables["modes"] = modes
        successors = self.problem.get_precedence_constraints()
        for pred in successors:
            for succ in successors[pred]:
                self.model += [
                    start[self.problem.get_index_from_task(succ)]
                    >= end[self.problem.get_index_from_task(pred)]
                ]
        for set_task in self.problem.get_no_overlap():
            self.model += [
                cp.NoOverlap(
                    start=[
                        start[self.problem.get_index_from_task(task)]
                        for task in set_task
                    ],
                    end=[
                        end[self.problem.get_index_from_task(task)] for task in set_task
                    ],
                    duration=[
                        duration[self.problem.get_index_from_task(task)]
                        for task in set_task
                    ],
                )
            ]
        for machine in self.problem.cumulative_resources_list:
            task_mode = [
                ((t, m), self.problem.get_task_mode_duration(t, m))
                for t, m in modes
                if self.problem.get_cumulative_resource_consumption(machine, t, m) > 0
            ]
            self.model += [
                cp.NoOverlapOptional(
                    start=[
                        start[self.problem.get_index_from_task(tm[0])]
                        for tm, _ in task_mode
                    ],
                    duration=[d for _, d in task_mode],
                    end=[
                        end[self.problem.get_index_from_task(tm[0])]
                        for tm, _ in task_mode
                    ],
                    is_present=[modes[tm] for tm, _ in task_mode],
                )
            ]
        self.model.minimize(cp.max(end))

    def retrieve_current_solution(self) -> AnyShopSolution:
        starts = self.variables["start"].value()
        ends = self.variables["end"].value()
        schedule = [
            [
                (
                    starts[self.problem.get_index_from_task((i, k))],
                    ends[self.problem.get_index_from_task((i, k))],
                )
                for k in range(self.problem.nb_subjob_per_job[i])
            ]
            for i in range(self.problem.n_jobs)
        ]
        machine_index = [
            [None for _ in range(self.problem.nb_subjob_per_job[i])]
            for i in range(self.problem.n_jobs)
        ]
        recipe_index = [
            [None for _ in range(self.problem.nb_subjob_per_job[i])]
            for i in range(self.problem.n_jobs)
        ]
        for t, m in self.variables["modes"]:
            val = self.variables["modes"][(t, m)].value()
            if val == 1:
                recipe_index[t[0]][t[1]] = m
                machine_index[t[0]][t[1]] = self.problem.mode2machine[t][m]

        return AnyShopSolution(
            problem=self.problem,
            schedule=schedule,
            machine_index=machine_index,
            recipe_index=recipe_index,
        )
