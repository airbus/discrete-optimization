#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

import didppy as dp

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.shop.osp.problem import OpenShopProblem, OpenShopSolution


class DpOspSolver(DpSolver, WarmstartMixin):
    transitions: dict
    transitions_name_to_index: dict
    time_per_machine: list
    time_per_job: list
    problem: OpenShopProblem

    def init_model(self, **kwargs: Any) -> None:
        self.transitions = {}
        self.transitions_name_to_index = {}
        self.model = dp.Model()
        tasks = self.problem.tasks_list
        durations = [self.problem.get_task_duration(task) for task in tasks]
        machines = [self.problem.mode2machine[task][0] for task in tasks]
        nb_tasks = len(tasks)
        jobs = self.model.add_object_type(nb_tasks)
        done = self.model.add_set_var(object_type=jobs, target=set())
        undone = self.model.add_set_var(
            object_type=jobs, target={i for i in range(nb_tasks)}
        )
        time_per_machine = [
            self.model.add_int_resource_var(target=0, less_is_better=True)
            for m in range(self.problem.n_machines)
        ]
        time_per_job = [
            self.model.add_int_var(target=0) for m in range(self.problem.n_jobs)
        ]
        global_time = self.model.add_int_resource_var(target=0, less_is_better=True)
        remaining_work_per_machine = [
            self.model.add_int_resource_var(
                target=sum(
                    [durations[i] for i in range(len(durations)) if machines[i] == m]
                ),
                less_is_better=False,
            )
            for m in range(self.problem.n_machines)
        ]

        for index in range(len(tasks)):
            machine = machines[index]
            duration = durations[index]
            original_index_job = tasks[index][0]
            insert_time = dp.max(
                time_per_job[original_index_job], time_per_machine[machine]
            )
            new_global_time = dp.max(global_time, insert_time + duration)
            insert = dp.Transition(
                name=f"sched_{index}",
                cost=(new_global_time - global_time) + dp.IntExpr.state_cost(),
                effects=[
                    (done, done.add(index)),
                    (undone, undone.remove(index)),
                    (time_per_job[original_index_job], insert_time + duration),
                    (time_per_machine[machine], insert_time + duration),
                    (global_time, new_global_time),
                    (
                        remaining_work_per_machine[machine],
                        remaining_work_per_machine[machine] - duration,
                    ),
                ],
                preconditions=[undone.contains(index)],
            )
            self.model.add_transition(insert)
            self.transitions[f"sched_{index}"] = insert
            self.transitions_name_to_index[f"sched_{index}"] = index
        self.model.add_base_case([undone.is_empty()])
        self.model.add_dual_bound(0)
        # for m in range(self.problem.n_machines):
        #    self.model.add_dual_bound(time_per_machine[m]+remaining_work_per_machine[m]-
        #                              global_time)
        self.time_per_machine = time_per_machine
        self.time_per_job = time_per_job

    def retrieve_solution(self, sol: dp.Solution) -> OpenShopSolution:
        schedules = {}
        schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}
        state = self.model.target_state
        for transition in sol.transitions:
            state = transition.apply(state, self.model)
            index = self.transitions_name_to_index[transition.name]
            task = self.problem.tasks_list[index]
            m = (
                self.problem.list_jobs[task[0]]
                .subjobs[task[1]]
                .recipes[0]
                .machine_index
            )
            dur = (
                self.problem.list_jobs[task[0]]
                .subjobs[task[1]]
                .recipes[0]
                .processing_time
            )
            time_per_job = state[self.time_per_job[task[0]]]
            start = time_per_job - dur
            end = time_per_job
            schedule_per_machine[m].append((start, end))
            schedules[task] = (start, end)
        sol = OpenShopSolution(
            problem=self.problem,
            schedule=[
                [schedules[(i, j)] for j in range(self.problem.nb_subjob_per_job[i])]
                for i in range(self.problem.n_jobs)
            ],
        )
        return sol

    def set_warm_start(self, solution: OpenShopSolution) -> None:
        initial_solution = []
        flatten_schedule = [
            (i, solution.get_start_time(self.problem.tasks_list[i]))
            for i in range(len(self.problem.tasks_list))
        ]
        sorted_flatten = sorted(flatten_schedule, key=lambda x: x[1])
        for index, _ in sorted_flatten:
            initial_solution.append(self.transitions[f"sched_{index}"])
        self.initial_solution = initial_solution
