#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re
from typing import Any

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, dp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)


class DpFjspSolver(DpSolver, WarmstartMixin):
    problem: FJobShopProblem
    hyperparameters = DpSolver.hyperparameters + [
        CategoricalHyperparameter(
            name="add_penalty_on_inefficiency", choices=[True, False], default=True
        )
    ]

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        add_penalty = kwargs["add_penalty_on_inefficiency"]
        model = dp.Model()
        jobs = []
        durations = []
        machines = []
        job_id = []
        cur_sub_job_per_jobs = {i: 0 for i in range(self.problem.n_jobs)}
        index = {}
        len_ = 0
        while len_ < self.problem.n_all_jobs:
            for i in range(self.problem.n_jobs):
                if cur_sub_job_per_jobs[i] < len(self.problem.list_jobs[i].sub_jobs):
                    jobs.append((i, cur_sub_job_per_jobs[i]))
                    durations.append(
                        [
                            x.processing_time
                            for x in self.problem.list_jobs[i].sub_jobs[
                                cur_sub_job_per_jobs[i]
                            ]
                        ]
                    )
                    machines.append(
                        [
                            x.machine_id
                            for x in self.problem.list_jobs[i].sub_jobs[
                                cur_sub_job_per_jobs[i]
                            ]
                        ]
                    )
                    job_id.append(i)
                    index[(i, cur_sub_job_per_jobs[i])] = len_
                    cur_sub_job_per_jobs[i] += 1
                    len_ += 1
        precedence_by_index = [set() for i in range(self.problem.n_all_jobs)]
        successors_by_index = [set() for i in range(self.problem.n_all_jobs)]
        for i in range(self.problem.n_jobs):
            for j in range(1, len(self.problem.list_jobs[i].sub_jobs)):
                ind = index[(i, j)]
                ind_pred = index[(i, j - 1)]
                precedence_by_index[ind].add(ind_pred)
                successors_by_index[ind_pred].add(ind)

        task = model.add_object_type(number=self.problem.n_all_jobs)
        job = model.add_object_type(number=self.problem.n_jobs)
        to_job = model.add_int_table(job_id)
        done = model.add_set_var(object_type=task, target=set())
        undone = model.add_set_var(
            object_type=task, target=range(self.problem.n_all_jobs)
        )
        cur_time_per_machine = [
            model.add_int_var(target=0) for m in range(self.problem.n_machines)
        ]
        workload_per_machine = [
            model.add_int_var(target=0) for m in range(self.problem.n_machines)
        ]
        cur_time_per_job = [
            model.add_int_var(target=0) for m in range(self.problem.n_jobs)
        ]
        finish = model.add_int_var(0)
        cur_time_total = model.add_int_resource_var(target=0, less_is_better=True)
        model.add_base_case([finish == 1, undone.is_empty()])
        self.transitions = {}
        for i in range(len(jobs)):
            jid = job_id[i]
            m_s = machines[i]
            dur_s = durations[i]
            for choice_machine in range(len(m_s)):
                m = m_s[choice_machine]
                dur = dur_s[choice_machine]
                if add_penalty:
                    c = (
                        1
                        * (
                            dp.max(
                                cur_time_per_machine[m] + dur,
                                cur_time_per_job[jid] + dur,
                            )
                            - workload_per_machine[m]
                            - dur
                            - (cur_time_per_machine[m] - workload_per_machine[m])
                        )
                        + 100
                        * (
                            dp.max(
                                cur_time_total,
                                dp.max(
                                    cur_time_per_machine[m] + dur,
                                    cur_time_per_job[jid] + dur,
                                ),
                            )
                            - cur_time_total
                        )
                    ) + dp.IntExpr.state_cost()
                else:
                    c = (
                        dp.max(
                            cur_time_total,
                            dp.max(
                                cur_time_per_machine[m] + dur,
                                cur_time_per_job[jid] + dur,
                            ),
                        )
                        - cur_time_total
                    ) + dp.IntExpr.state_cost()
                sched = dp.Transition(
                    name=f"sched_{i}_{choice_machine}",
                    cost=c,
                    effects=[
                        (done, done.add(i)),
                        (undone, undone.remove(i)),
                        (workload_per_machine[m], workload_per_machine[m] + dur),
                        (
                            cur_time_per_job[jid],
                            dp.max(
                                cur_time_per_machine[m] + dur,
                                cur_time_per_job[jid] + dur,
                            ),
                        ),
                        (
                            cur_time_per_machine[m],
                            dp.max(
                                cur_time_per_machine[m] + dur,
                                cur_time_per_job[jid] + dur,
                            ),
                        ),
                        (
                            cur_time_total,
                            dp.max(
                                cur_time_total,
                                dp.max(
                                    cur_time_per_machine[m] + dur,
                                    cur_time_per_job[jid] + dur,
                                ),
                            ),
                        ),
                    ],
                    preconditions=[undone.contains(i)]
                    + [done.contains(j) for j in precedence_by_index[i]],
                )
                model.add_transition(sched)
                self.transitions[(i, choice_machine)] = sched
        finish = dp.Transition(
            name="finish_",
            effects=[(finish, 1)],
            # cost=cur_time_total+dp.IntExpr.state_cost(),
            cost=dp.IntExpr.state_cost(),
            preconditions=[done.len() == self.problem.n_all_jobs],
        )
        model.add_transition(finish)
        self.transitions["finish_"] = finish
        # model.add_dual_bound()
        self.jobs = jobs
        self.prec = precedence_by_index
        self.index = index
        self.machines = machines
        self.duration = durations
        self.model = model
        self.cur_time_per_machine = cur_time_per_machine
        self.cur_time_per_job = cur_time_per_job

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}
        schedules = {}
        state = self.model.target_state

        for transition in sol.transitions:
            state = transition.apply(state, self.model)
            if "finish" not in transition.name:
                t_number, choice = extract_ints(transition.name)
                m = self.machines[t_number][choice]
                j = self.jobs[t_number]
                start = 0
                if len(schedule_per_machine[m]) > 0:
                    start = max(start, schedule_per_machine[m][-1][1])
                if j[1] > 0:
                    start = max(start, schedules[(j[0], j[1] - 1)][1])
                end = start + self.duration[t_number][choice]
                schedule_per_machine[m].append((start, end))
                schedules[j] = (start, end, m)
        sol = FJobShopSolution(
            problem=self.problem,
            schedule=[
                [
                    schedules[(i, j)]
                    for j in range(len(self.problem.list_jobs[i].sub_jobs))
                ]
                for i in range(self.problem.n_jobs)
            ],
        )
        return sol

    def set_warm_start(self, solution: FJobShopSolution) -> None:
        initial_solution = []
        flatten_schedule = [
            (i, solution.schedule[self.jobs[i][0]][self.jobs[i][1]])
            for i in range(len(self.jobs))
        ]
        sorted_flatten = sorted(flatten_schedule, key=lambda x: (x[1][0], x[1][1]))
        for index, (start, end, machine) in sorted_flatten:
            m_ind = self.machines[index].index(machine)
            initial_solution.append(self.transitions[(index, m_ind)])
        initial_solution.append(self.transitions["finish_"])
        self.initial_solution = initial_solution
