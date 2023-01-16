#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple, Type, Union

import numpy as np
import scipy.stats as ss

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TupleFitness,
    TypeAttribute,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp_multiskill.fast_function_ms_rcpsp import (
    sgs_fast_ms,
    sgs_fast_ms_partial_schedule,
    sgs_fast_ms_preemptive,
    sgs_fast_ms_preemptive_partial_schedule,
    sgs_fast_ms_preemptive_some_special_constraints,
)

logger = logging.getLogger(__name__)


def tree():
    return defaultdict(tree)


class ScheduleGenerationScheme(Enum):
    SERIAL_SGS = 0
    PARALLEL_SGS = 1


class TaskDetails:
    def __init__(self, start, end, resource_units_used: List[int]):
        self.start = start
        self.end = end
        self.resource_units_used = resource_units_used

    def __str__(self):
        return (
            "Start :"
            + str(self.start)
            + " End : "
            + str(self.end)
            + " Resource "
            + str(self.resource_units_used)
        )


class TaskDetailsPreemptive:
    def __init__(
        self,
        starts: List[int],
        ends: List[int],
        resource_units_used: List[List[Hashable]],
    ):
        self.starts = starts
        self.ends = ends
        self.resource_units_used = resource_units_used

    def __str__(self):
        return (
            "Start :"
            + str(self.starts)
            + " End : "
            + str(self.ends)
            + " Resource "
            + str(self.resource_units_used)
        )


class MS_RCPSPSolution(Solution):
    def __init__(
        self,
        problem: Problem,
        modes: Dict[Hashable, int],
        schedule: Dict[
            Hashable, Dict[str, Union[int, List[int]]]
        ],  # (task: {"start_time": start, "end_time": }}
        employee_usage: Dict[Hashable, Dict[Hashable, Set[str]]],
    ):  # {task: {employee: Set(skills})}):
        self.problem: MS_RCPSPModel = problem
        self.modes = modes
        self.schedule = schedule
        self.employee_usage = employee_usage

    def get_number_of_part(self, task):
        return 1

    def get_start_times_list(self, task):
        return [self.schedule.get(task, {"start_time": None})["start_time"]]

    def get_end_times_list(self, task):
        return [self.schedule.get(task, {"end_time": None})["end_time"]]

    def employee_used(self, task):
        if task not in self.employee_usage:
            return [[] for i in range(self.get_number_of_part(task))]
        else:
            return [
                [
                    e
                    for e in self.employee_usage[task]
                    if len(self.employee_usage[task][e]) > 0
                ]
            ]

    def copy(self):
        return MS_RCPSPSolution(
            problem=self.problem,
            modes=deepcopy(self.modes),
            schedule=deepcopy(self.schedule),
            employee_usage=deepcopy(self.employee_usage),
        )

    def change_problem(self, new_problem: Problem):
        self.problem = new_problem

    def get_start_time(self, task):
        return self.schedule.get(task, {"start_time": None})["start_time"]

    def get_end_time(self, task):
        return self.schedule.get(task, {"end_time": None})["end_time"]

    def get_active_time(self, task):
        return list(range(self.get_start_time(task), self.get_end_time(task)))

    def get_max_end_time(self):
        return max([self.get_end_time(x) for x in self.schedule])


class MS_RCPSPSolution_Preemptive(MS_RCPSPSolution):
    def __init__(
        self,
        problem: Problem,
        modes: Dict[Hashable, int],
        schedule: Dict[Hashable, Dict[str, List[int]]],
        employee_usage: Dict[Hashable, List[Dict[Hashable, Set[str]]]],
    ):  # {task: {employee: Set(skills})}):
        super().__init__(problem, modes, schedule, None)
        self.employee_usage = employee_usage

    def copy(self):
        return MS_RCPSPSolution_Preemptive(
            problem=self.problem,
            modes=deepcopy(self.modes),
            schedule=deepcopy(self.schedule),
            employee_usage=deepcopy(self.employee_usage),
        )

    def get_start_time(self, task):
        return self.schedule.get(task, {"starts": [None]})["starts"][0]

    def get_start_times_list(self, task):
        return self.schedule.get(task, {"starts": [None]})["starts"]

    def get_end_time(self, task):
        return self.schedule.get(task, {"ends": [None]})["ends"][-1]

    def get_end_times_list(self, task):
        return self.schedule.get(task, {"ends": [None]})["ends"]

    def get_active_time(self, task):
        l = []
        for s, e in zip(self.schedule[task]["starts"], self.schedule[task]["ends"]):
            l += list(range(s, e))
        return l

    def get_nb_task_preemption(self):
        return len([t for t in self.schedule if len(self.schedule[t]["starts"]) > 1])

    def total_number_of_cut(self):
        return sum([self.get_number_of_part(task) - 1 for task in self.schedule])

    def get_number_of_part(self, task):
        return len(self.schedule.get(task, {"starts": []})["starts"])

    def get_min_duration_subtask(self):
        return min(
            [
                e - s
                for t in self.schedule
                for e, s in zip(self.schedule[t]["ends"], self.schedule[t]["starts"])
                if len(self.schedule[t]["starts"]) > 1
            ],
            default=None,
        )

    def get_max_preempted(self):
        return max([len(self.schedule[t]["starts"]) for t in self.schedule])

    def get_task_preempted(self):
        return [t for t in self.schedule if len(self.schedule[t]["starts"]) > 1]

    def employee_used(self, task):
        if task not in self.employee_usage:
            return [[] for i in range(self.get_number_of_part(task))]
        else:
            return [
                [
                    e
                    for e in self.employee_usage[task][i]
                    if len(self.employee_usage[task][i][e]) > 0
                ]
                for i in range(self.get_number_of_part(task))
            ]


def schedule_solution_to_variant(solution: MS_RCPSPSolution):
    s: MS_RCPSPSolution = solution
    priority_list_task = sorted(s.schedule, key=lambda x: s.schedule[x]["start_time"])
    priority_list_task.remove(s.problem.source_task)
    priority_list_task.remove(s.problem.sink_task)
    workers = []
    for i in solution.problem.tasks_list_non_dummy:
        w = []
        if len(s.employee_usage.get(i, {})) > 0:
            w = [w for w in s.employee_usage.get(i)]
        w += [wi for wi in s.problem.employees if wi not in w]
        workers += [w]
    solution = MS_RCPSPSolution_Variant(
        problem=s.problem,
        priority_list_task=[
            solution.problem.index_task_non_dummy[p] for p in priority_list_task
        ],
        priority_worker_per_task=workers,
        modes_vector=[s.modes[i] for i in s.problem.tasks_list_non_dummy],
    )
    return solution


class MS_RCPSPSolution_Variant(MS_RCPSPSolution):
    def __init__(
        self,
        problem: Problem,
        modes_vector: Optional[List[int]] = None,
        modes_vector_from0: Optional[List[int]] = None,
        priority_list_task: Optional[List[int]] = None,
        priority_worker_per_task: Optional[List[List[Hashable]]] = None,
        modes: Dict[int, int] = None,
        schedule: Dict[int, Dict[str, int]] = None,
        employee_usage: Dict[int, Dict[int, Set[str]]] = None,
        fast: bool = True,
    ):  # {task: {employee: Set(skills})}):
        super().__init__(problem, modes, schedule, employee_usage)
        self.priority_list_task = priority_list_task
        self.modes_vector = modes_vector
        self.modes_vector_from0 = modes_vector_from0
        if priority_worker_per_task is None:
            self.priority_worker_per_task = None
        elif any(
            isinstance(i, list) for i in priority_worker_per_task
        ):  # if arg is a nested list
            self.priority_worker_per_task = priority_worker_per_task
        else:  # if arg is a single list
            self.priority_worker_per_task = (
                self.problem.convert_fixed_priority_worker_per_task_from_permutation(
                    priority_worker_per_task
                )
            )

        if self.modes_vector is None and self.modes_vector_from0 is None:
            if "fixed_modes" in self.problem.__dict__:
                self.modes_vector = self.problem.fixed_modes
                self.modes_vector_from0 = [x - 1 for x in self.problem.fixed_modes]
            else:
                self.modes_vector = [
                    self.modes[x] for x in self.problem.tasks_list_non_dummy
                ]
                self.modes_vector_from0 = [x - 1 for x in self.modes_vector]
        if self.modes_vector is None and self.modes_vector_from0 is not None:
            self.modes_vector = [x + 1 for x in self.modes_vector_from0]
        if self.modes_vector_from0 is None:
            self.modes_vector_from0 = [x - 1 for x in self.modes_vector]
        if self.priority_list_task is None:
            if self.schedule is None:
                self.priority_list_task = self.problem.fixed_permutation
            else:
                sorted_task = [
                    self.problem.index_task_non_dummy[i]
                    for i in sorted(
                        self.schedule, key=lambda x: self.schedule[x]["start_time"]
                    )
                    if i in self.problem.index_task_non_dummy
                ]
                self.priority_list_task = sorted_task
        if self.priority_worker_per_task is None:
            if self.employee_usage is None:
                self.priority_worker_per_task = (
                    self.problem.fixed_priority_worker_per_task
                )
            else:
                workers = []
                for i in sorted(self.problem.tasks)[1:-1]:
                    w = []
                    if len(self.employee_usage.get(i, {})) > 0:
                        w = [w for w in self.employee_usage.get(i)]
                    w += [wi for wi in self.problem.employees if wi not in w]
                    workers += [w]
                self.priority_worker_per_task = workers
        self._schedule_to_recompute = self.schedule is None
        self.fast = fast
        if self._schedule_to_recompute:
            self.do_recompute(self.fast)

    def do_recompute(self, fast=True):
        if not fast:
            (
                rcpsp_schedule,
                modes_extended,
                employee_usage,
                modes_dict,
            ) = sgs_multi_skill(solution=self)
            self.schedule = rcpsp_schedule
            self.modes = modes_dict
            self.employee_usage = employee_usage
            self._schedule_to_recompute = False
        else:
            #  check modes not out-of-bound
            if max(self.modes_vector_from0) >= self.problem.max_number_of_mode:
                rcpsp_schedule = None
                skills_usage = None
                unfeasible_non_renewable_resources = True
            else:
                self.employee_usage = {}
                (
                    rcpsp_schedule,
                    skills_usage,
                    unfeasible_non_renewable_resources,
                ) = self.problem.func_sgs(
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        permutation_do=self.priority_list_task,
                    ),  # permutation_task=array(task)->task index
                    priority_worker_per_task=priority_worker_per_task_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        priority_worker_per_task=self.priority_worker_per_task,
                    ),  # array(task, worker)
                    modes_array=np.array([0] + self.modes_vector_from0 + [0]),
                )
            self.update_infos_from_numba_output(
                rcpsp_schedule=rcpsp_schedule,
                skills_usage=skills_usage,
                unfeasible_non_renewable_resources=unfeasible_non_renewable_resources,
            )

    def run_sgs_partial(
        self,
        current_t,
        completed_tasks: Dict[Hashable, TaskDetails],
        scheduled_tasks_start_times: Dict[Hashable, TaskDetails],
        fast=True,
    ):
        if not fast:
            (
                rcpsp_schedule,
                modes_extended,
                employee_usage,
                modes_dict,
            ) = sgs_multi_skill_partial_schedule(
                solution=self,
                current_t=current_t,
                completed_tasks=completed_tasks,
                scheduled_tasks_start_times=scheduled_tasks_start_times,
            )
            self.schedule = rcpsp_schedule
            self.modes = modes_dict
            self.employee_usage = employee_usage
            self._schedule_to_recompute = False
        else:
            #  check modes not out-of-bound
            if max(self.modes_vector_from0) >= self.problem.max_number_of_mode:
                rcpsp_schedule = None
                skills_usage = None
                unfeasible_non_renewable_resources = True
            else:
                self.employee_usage = {}
                (
                    scheduled_task_indicator,
                    scheduled_tasks_start_times_vector,
                    scheduled_tasks_end_times_vector,
                    worker_used,
                ) = build_partial_vectors(
                    problem=self.problem,
                    completed_tasks=completed_tasks,
                    scheduled_tasks_start_times=scheduled_tasks_start_times,
                )
                (
                    rcpsp_schedule,
                    skills_usage,
                    unfeasible_non_renewable_resources,
                ) = self.problem.func_sgs_partial(
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        permutation_do=self.priority_list_task,
                    ),
                    priority_worker_per_task=priority_worker_per_task_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        priority_worker_per_task=self.priority_worker_per_task,
                    ),
                    modes_array=np.array([0] + self.modes_vector_from0 + [0]),
                    scheduled_task_indicator=scheduled_task_indicator,
                    scheduled_start_task_times=scheduled_tasks_start_times_vector,
                    scheduled_end_task_times=scheduled_tasks_end_times_vector,
                    worker_used=worker_used,
                    current_time=current_t,
                )
            self.update_infos_from_numba_output(
                rcpsp_schedule=rcpsp_schedule,
                skills_usage=skills_usage,
                unfeasible_non_renewable_resources=unfeasible_non_renewable_resources,
            )

    def update_infos_from_numba_output(
        self, rcpsp_schedule, skills_usage, unfeasible_non_renewable_resources
    ):
        if unfeasible_non_renewable_resources:
            self.schedule = {
                t: {"start_time": 99999, "end_time": 99999}
                for t in self.problem.tasks_list
            }
            return
        self.schedule = {}
        for k in rcpsp_schedule:
            self.schedule[self.problem.tasks_list[k]] = {
                "start_time": rcpsp_schedule[k][0],
                "end_time": rcpsp_schedule[k][1],
            }
        for act_id in skills_usage:
            non_z = np.nonzero(skills_usage[act_id])
            for i, j in zip(non_z[0], non_z[1]):
                if self.problem.tasks_list[act_id] not in self.employee_usage:
                    self.employee_usage[self.problem.tasks_list[act_id]] = {}
                if (
                    self.problem.employees_list[i]
                    not in self.employee_usage[self.problem.tasks_list[act_id]]
                ):
                    self.employee_usage[self.problem.tasks_list[act_id]][
                        self.problem.employees_list[i]
                    ] = set()
                self.employee_usage[self.problem.tasks_list[act_id]][
                    self.problem.employees_list[i]
                ].add(self.problem.skills_list[j])
        self.modes = {
            self.problem.tasks_list_non_dummy[i]: self.modes_vector[i]
            for i in range(self.problem.n_jobs_non_dummy)
        }
        self.modes[self.problem.sink_task] = 1
        self.modes[self.problem.source_task] = 1
        self._schedule_to_recompute = False

    def copy(self):
        return MS_RCPSPSolution_Variant(
            problem=self.problem,
            priority_list_task=deepcopy(self.priority_list_task),
            modes_vector=deepcopy(self.modes_vector),
            priority_worker_per_task=deepcopy(self.priority_worker_per_task),
            modes=deepcopy(self.modes),
            schedule=deepcopy(self.schedule),
            employee_usage=deepcopy(self.employee_usage),
            fast=self.fast,
        )


class MS_RCPSPSolution_Preemptive_Variant(MS_RCPSPSolution_Preemptive):
    def __init__(
        self,
        problem: Problem,
        modes_vector: Optional[List[int]] = None,
        modes_vector_from0: Optional[List[int]] = None,
        priority_list_task: Optional[List[int]] = None,
        priority_worker_per_task: Optional[List[List[Hashable]]] = None,
        modes: Dict[int, int] = None,
        schedule: Dict[Hashable, Dict[str, List[int]]] = None,
        employee_usage: Dict[Hashable, List[Dict[Hashable, Set[str]]]] = None,
        fast: bool = True,
    ):  # {task: {employee: Set(skills})}):
        super().__init__(problem, modes, schedule, employee_usage)
        self.priority_list_task = priority_list_task
        self.modes_vector = modes_vector
        self.modes_vector_from0 = modes_vector_from0
        if priority_worker_per_task is None:
            self.priority_worker_per_task = None
        elif any(
            isinstance(i, list) for i in priority_worker_per_task
        ):  # if arg is a nested list
            self.priority_worker_per_task = priority_worker_per_task
        else:  # if arg is a single list
            self.priority_worker_per_task = (
                self.problem.convert_fixed_priority_worker_per_task_from_permutation(
                    priority_worker_per_task
                )
            )

        if self.modes_vector is None and self.modes_vector_from0 is None:
            self.modes_vector = self.problem.fixed_modes
            self.modes_vector_from0 = [x - 1 for x in self.problem.fixed_modes]
        if self.modes_vector is None and self.modes_vector_from0 is not None:
            self.modes_vector = [x + 1 for x in self.modes_vector_from0]
        if self.modes_vector_from0 is None:
            self.modes_vector_from0 = [x - 1 for x in self.modes_vector]
        if self.priority_list_task is None:
            self.priority_list_task = self.problem.fixed_permutation
        if self.priority_worker_per_task is None:
            self.priority_worker_per_task = self.problem.fixed_priority_worker_per_task
        self._schedule_to_recompute = True
        self.fast = fast
        if self.schedule is None:
            self.do_recompute(self.fast)

    def do_recompute(self, fast=True):
        if not fast:
            (
                rcpsp_schedule,
                modes_extended,
                employee_usage,
                modes_dict,
            ) = sgs_multi_skill_preemptive(solution=self)
            self.schedule = rcpsp_schedule
            self.modes = modes_dict
            self.employee_usage = employee_usage
            self._schedule_to_recompute = False
        else:
            self.employee_usage = {}
            if max(self.modes_vector_from0) >= self.problem.max_number_of_mode:
                starts_dict = {}
                ends_dict = {}
                skills_usage = None
                unfeasible_non_renewable_resources = True
            else:
                (
                    starts_dict,
                    ends_dict,
                    skills_usage,
                    unfeasible_non_renewable_resources,
                ) = self.problem.func_sgs(
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        permutation_do=self.priority_list_task,
                    ),
                    # permutation_task=array(task)->task index
                    priority_worker_per_task=priority_worker_per_task_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        priority_worker_per_task=self.priority_worker_per_task,
                    ),
                    # array(task, worker)
                    modes_array=np.array([0] + self.modes_vector_from0 + [0]),
                )
            self.update_from_numba_output(
                starts_dict, ends_dict, skills_usage, unfeasible_non_renewable_resources
            )

    def update_from_numba_output(
        self, starts_dict, ends_dict, skills_usage, unfeasible_non_renewable_resources
    ):
        if unfeasible_non_renewable_resources:
            self.schedule = {
                t: {"starts": [99999], "ends": [99999]} for t in self.problem.tasks_list
            }
            return
        self.schedule = {}
        for k in starts_dict:
            self.schedule[self.problem.tasks_list[k]] = {
                "starts": starts_dict[k],
                "ends": ends_dict[k],
            }
        for act_id in skills_usage:
            self.employee_usage[self.problem.tasks_list[act_id]] = [
                {}
                for k in range(
                    len(self.schedule[self.problem.tasks_list[act_id]]["starts"])
                )
            ]
            non_z = np.nonzero(skills_usage[act_id])
            for i, j, k in zip(non_z[0], non_z[1], non_z[2]):
                if (
                    self.problem.employees_list[j]
                    not in self.employee_usage[self.problem.tasks_list[act_id]][i]
                ):
                    self.employee_usage[self.problem.tasks_list[act_id]][i][
                        self.problem.employees_list[j]
                    ] = set()
                self.employee_usage[self.problem.tasks_list[act_id]][i][
                    self.problem.employees_list[j]
                ].add(self.problem.skills_list[k])
        self.modes = {
            self.problem.tasks_list_non_dummy[i]: self.modes_vector[i]
            for i in range(self.problem.n_jobs_non_dummy)
        }
        self.modes[self.problem.sink_task] = 1
        self.modes[self.problem.source_task] = 1
        self._schedule_to_recompute = False

    def run_sgs_partial(
        self,
        current_t,
        completed_tasks: Dict[Hashable, TaskDetailsPreemptive],
        scheduled_tasks_start_times: Dict[Hashable, TaskDetailsPreemptive],
        fast: bool = True,
    ):
        if not fast:
            (
                rcpsp_schedule,
                modes_extended,
                employee_usage,
                modes_dict,
            ) = sgs_multi_skill_preemptive_partial_schedule(
                solution=self,
                current_t=current_t,
                completed_tasks=completed_tasks,
                scheduled_tasks_start_times=scheduled_tasks_start_times,
            )
            self.schedule = rcpsp_schedule
            self.modes = modes_dict
            self.employee_usage = employee_usage
            self._schedule_to_recompute = False
        else:
            if max(self.modes_vector_from0) >= self.problem.max_number_of_mode:
                starts_dict = {}
                ends_dict = {}
                skills_usage = None
                unfeasible_non_renewable_resources = True
            else:
                (
                    scheduled_task_indicator,
                    scheduled_tasks_start_times_array,
                    scheduled_tasks_end_times_array,
                    nb_subparts,
                    worker_used,
                ) = build_partial_vectors_preemptive(
                    problem=self.problem,
                    completed_tasks=completed_tasks,
                    scheduled_tasks_start_times=scheduled_tasks_start_times,
                )
                (
                    starts_dict,
                    ends_dict,
                    skills_usage,
                    unfeasible_non_renewable_resources,
                ) = self.problem.func_sgs_partial(
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        permutation_do=self.priority_list_task,
                    ),
                    priority_worker_per_task=priority_worker_per_task_do_to_permutation_sgs_fast(
                        rcpsp_problem=self.problem,
                        priority_worker_per_task=self.priority_worker_per_task,
                    ),
                    modes_array=np.array([0] + self.modes_vector_from0 + [0]),
                    scheduled_task_indicator=scheduled_task_indicator,
                    scheduled_start_task_times=scheduled_tasks_start_times_array,
                    scheduled_end_task_times=scheduled_tasks_end_times_array,
                    nb_subparts=nb_subparts,
                    worker_used=worker_used,
                    current_time=current_t,
                )
            self.update_from_numba_output(
                starts_dict, ends_dict, skills_usage, unfeasible_non_renewable_resources
            )

    def copy(self):
        return MS_RCPSPSolution_Preemptive_Variant(
            problem=self.problem,
            priority_list_task=deepcopy(self.priority_list_task),
            modes_vector=deepcopy(self.modes_vector),
            priority_worker_per_task=deepcopy(self.priority_worker_per_task),
            modes=deepcopy(self.modes),
            schedule=deepcopy(self.schedule),
            employee_usage=deepcopy(self.employee_usage),
            fast=self.fast,
        )


def schedule_solution_preemptive_to_variant(solution: MS_RCPSPSolution_Preemptive):
    s: MS_RCPSPSolution_Preemptive = solution
    priority_list_task = sorted(s.schedule, key=lambda x: s.get_start_time(x))
    priority_list_task.remove(s.problem.source_task)
    priority_list_task.remove(s.problem.sink_task)
    workers = []
    for i in s.problem.tasks_list_non_dummy:
        w = []
        if len(s.employee_usage.get(i, [{}])[0]) > 0:
            w = [w for w in s.employee_usage.get(i, [{}])[0]]
        w += [wi for wi in s.problem.employees if wi not in w]
        workers += [w]
    solution = MS_RCPSPSolution_Preemptive_Variant(
        problem=s.problem,
        priority_list_task=[
            s.problem.index_task_non_dummy[p] for p in priority_list_task
        ],
        priority_worker_per_task=workers,
        modes_vector=[s.modes[i] for i in s.problem.tasks_list_non_dummy],
    )
    return solution


def sgs_multi_skill(solution: MS_RCPSPSolution_Variant):
    problem: MS_RCPSPModel = solution.problem
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    unfeasible_in_horizon = False
    new_horizon = problem.horizon * problem.horizon_multiplier
    resource_avail_in_time = {}
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.array(
            problem.resources_availability[res][: new_horizon + 1]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.array(
            problem.employees[i].calendar_employee[: new_horizon + 1], dtype=bool
        )
    minimum_starting_time = {}
    for act in problem.tasks_list:
        minimum_starting_time[act] = 0
        if problem.do_special_constraints:
            if act in problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    problem.special_constraints.start_times_window[act][0]
                    if problem.special_constraints.start_times_window[act][0]
                    is not None
                    else 0
                )
    perm_extended = [
        problem.tasks_list_non_dummy[x] for x in solution.priority_list_task
    ]

    perm_extended.insert(0, problem.source_task)
    perm_extended.append(problem.sink_task)

    modes_dict = problem.build_mode_dict(solution.modes_vector)
    employee_usage = {}
    scheduled = set()
    unfeasible_skills = False
    while (
        len(perm_extended) > 0
        and not unfeasible_non_renewable_resources
        and not unfeasible_in_horizon
    ):
        act_id = None
        for id_successor in perm_extended:
            respected = True
            for pred in problem.predecessors.get(id_successor, set()):
                if pred not in scheduled:
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            mode = modes_dict[act_id]
            range_time = range(
                current_min_time,
                current_min_time + problem.mode_details[act_id][mode]["duration"],
            )
            if (
                current_min_time + problem.mode_details[act_id][mode]["duration"]
                >= problem.horizon
            ):
                unfeasible_in_horizon = True
                break
            for t in range_time:
                for res in resource_avail_in_time.keys():
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0):
                            valid = False
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                required_skills = {
                    s: problem.mode_details[act_id][mode][s]
                    for s in problem.mode_details[act_id][mode]
                    if s in problem.skills_set
                    and problem.mode_details[act_id][mode][s] > 0
                }
                worker_ids = None
                if len(required_skills) > 0:
                    worker_ids = [
                        worker
                        for worker in worker_avail_in_time
                        if all(worker_avail_in_time[worker][t] for t in range_time)
                        if any(
                            s in problem.employees[worker].dict_skill
                            and problem.employees[worker].dict_skill[s].skill_value > 0
                            for s in required_skills
                        )
                    ]
                    if problem.one_unit_per_task_max:
                        good = False
                        ws = []
                        for worker in worker_ids:
                            if any(
                                problem.employees[worker].dict_skill[s].skill_value
                                < required_skills[s]
                                for s in required_skills
                            ):
                                continue
                            else:
                                good = True
                                ws += [worker]
                                break
                        valid = good
                        if good:
                            worker_ids = ws
                    else:
                        if not all(
                            sum(
                                [
                                    problem.employees[worker].dict_skill[s].skill_value
                                    for worker in worker_ids
                                    if s in problem.employees[worker].dict_skill
                                ]
                            )
                            >= required_skills[s]
                            for s in required_skills
                        ):
                            valid = False
            if not valid:
                current_min_time += 1
            if current_min_time > new_horizon:
                unfeasible_in_horizon = True
                break
        if not unfeasible_non_renewable_resources and not unfeasible_in_horizon:
            end_t = (
                current_min_time
                + problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                - 1
            )
            for res in resource_avail_in_time.keys():
                resource_avail_in_time[res][
                    current_min_time : end_t + 1
                ] -= problem.mode_details[act_id][modes_dict[act_id]].get(res, 0)
                if res in problem.non_renewable_resources:
                    resource_avail_in_time[res][end_t + 1 :] -= problem.mode_details[
                        act_id
                    ][modes_dict[act_id]][res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
            if worker_ids is not None:
                priority_list_this_task = solution.priority_worker_per_task[
                    problem.index_task_non_dummy[act_id]
                ]
                worker_used = []
                current_skills = {s: 0.0 for s in required_skills}
                skills_fulfilled = False
                for w in priority_list_this_task:
                    if w in worker_ids and any(
                        problem.employees[w]
                        .dict_skill.get(s, SkillDetail(0, 0, 0))
                        .skill_value
                        > 0
                        for s in required_skills
                    ):
                        worker_used += [w]
                        for s in problem.employees[w].dict_skill:
                            if s in current_skills:
                                current_skills[s] += (
                                    problem.employees[w].dict_skill[s].skill_value
                                )
                                if act_id not in employee_usage:
                                    employee_usage[act_id] = {}
                                if w not in employee_usage[act_id]:
                                    employee_usage[act_id][w] = set()
                                employee_usage[act_id][w].add(s)
                        worker_avail_in_time[w][
                            current_min_time : current_min_time
                            + problem.mode_details[act_id][modes_dict[act_id]][
                                "duration"
                            ]
                        ] = False
                    if all(
                        current_skills[s] >= required_skills[s] for s in required_skills
                    ):
                        skills_fulfilled = True
                        break
                if not skills_fulfilled:
                    unfeasible_skills = True
                    logger.warning(
                        "You probably didnt give the right worker named in priority_worker_per_task"
                    )
                    break
            if unfeasible_skills:
                break
            activity_end_times[act_id] = (
                current_min_time
                + problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            perm_extended.remove(act_id)
            scheduled.add(act_id)
            for s in problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
        worker_ids = None
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    if unfeasible_non_renewable_resources or unfeasible_in_horizon or unfeasible_skills:
        last_act_id = max(problem.successors.keys())
        rcpsp_schedule[last_act_id] = {
            "start_time": 99999999,
            "end_time": 9999999,
        }
    return rcpsp_schedule, [], employee_usage, modes_dict


def sgs_multi_skill_preemptive(solution: MS_RCPSPSolution_Preemptive_Variant):
    problem: MS_RCPSPModel = solution.problem
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    unfeasible_in_horizon = False
    new_horizon = problem.horizon * problem.horizon_multiplier
    resource_avail_in_time = {}
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.array(
            problem.resources_availability[res][: new_horizon + 1]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.array(
            problem.employees[i].calendar_employee[: new_horizon + 1], dtype=bool
        )
    minimum_starting_time = {}
    for act in problem.tasks_list:
        minimum_starting_time[act] = 0
        if problem.do_special_constraints:
            if act in problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    problem.special_constraints.start_times_window[act][0]
                    if problem.special_constraints.start_times_window[act][0]
                    is not None
                    else 0
                )
    perm_extended = [
        problem.tasks_list_non_dummy[x] for x in solution.priority_list_task
    ]
    perm_extended.insert(0, problem.source_task)
    perm_extended.append(problem.sink_task)
    modes_dict = problem.build_mode_dict(solution.modes_vector)
    employee_usage = {}
    scheduled = set()
    unfeasible_skills = False
    expected_durations_task = {
        k: problem.mode_details[k][modes_dict[k]]["duration"] for k in modes_dict
    }

    def find_workers(required_skills, task, current_min_time):
        valid = True
        if len(required_skills) > 0:
            worker_ids = [
                worker
                for worker in solution.priority_worker_per_task[
                    problem.index_task_non_dummy[task]
                ]
                if worker_avail_in_time[worker][current_min_time]
                if any(
                    s in problem.employees[worker].dict_skill
                    and problem.employees[worker].dict_skill[s].skill_value > 0
                    for s in required_skills
                )
            ]
            if problem.one_unit_per_task_max:
                good = False
                ws = []
                for worker in worker_ids:
                    if any(
                        s not in problem.employees[worker].dict_skill
                        for s in required_skills
                    ) or any(
                        problem.employees[worker].dict_skill[s].skill_value
                        < required_skills[s]
                        for s in required_skills
                    ):
                        continue
                    else:
                        good = True
                        ws += [worker]
                        break
                valid = good
                if good:
                    worker_ids = ws
            else:
                if not all(
                    sum(
                        [
                            problem.employees[worker].dict_skill[s].skill_value
                            for worker in worker_ids
                        ]
                    )
                    >= required_skills[s]
                    for s in required_skills
                ):
                    valid = False
            if valid:
                priority_list_this_task = solution.priority_worker_per_task[
                    problem.index_task_non_dummy[task]
                ]
                worker_used = []
                current_skills = {s: 0.0 for s in required_skills}
                skills_fulfilled = False
                for w in priority_list_this_task:
                    if w in worker_ids:
                        worker_used += [w]
                        for s in problem.employees[w].dict_skill:
                            if s in current_skills:
                                current_skills[s] += (
                                    problem.employees[w].dict_skill[s].skill_value
                                )
                        if all(
                            current_skills[s] >= required_skills[s]
                            for s in required_skills
                        ):
                            skills_fulfilled = True
                            break
                return valid, skills_fulfilled, worker_used
            else:
                return valid, False, []
        return True, True, []

    schedules = {}
    while (
        len(perm_extended) > 0
        and not unfeasible_non_renewable_resources
        and not unfeasible_in_horizon
    ):
        act_id = None
        for id_successor in perm_extended:
            respected = True
            for pred in problem.predecessors.get(id_successor, set()):
                if pred not in scheduled:
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        cur_duration = 0
        starts = []
        ends = []
        workers = []
        mode = modes_dict[act_id]
        required_skills = {
            s: problem.mode_details[act_id][mode][s]
            for s in problem.mode_details[act_id][mode]
            if s in problem.skills_set and problem.mode_details[act_id][mode][s] > 0
        }
        while not valid:
            valid = True
            reached_t = None
            if expected_durations_task[act_id] == 0:
                starts += [current_min_time]
                ends += [current_min_time]
                workers += [[]]
                cur_duration += ends[-1] - starts[-1]
            else:
                reached_end = True
                for t in range(
                    current_min_time,
                    current_min_time + expected_durations_task[act_id] - cur_duration,
                ):
                    if t == current_min_time:
                        valid, skills_fulfilled, worker_used = find_workers(
                            required_skills=required_skills,
                            task=act_id,
                            current_min_time=current_min_time,
                        )
                        if not valid or not skills_fulfilled:
                            reached_end = False
                            break
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    if any(
                        resource_avail_in_time[res][t]
                        < problem.mode_details[act_id][modes_dict[act_id]].get(res, 0)
                        for res in problem.resources_list
                    ):
                        reached_end = False
                        break
                    else:
                        if any(not worker_avail_in_time[w][t] for w in worker_used):
                            reached_end = False
                            break
                        else:
                            reached_t = t
                if reached_t is not None:
                    starts += [current_min_time]
                    ends += [reached_t + 1]
                    workers += [worker_used]
                    cur_duration += ends[-1] - starts[-1]
            valid = cur_duration == expected_durations_task[act_id]
            if not valid:
                current_min_time = next(
                    t
                    for t in range(
                        reached_t + 2
                        if reached_t is not None
                        else current_min_time + 1,
                        new_horizon,
                    )
                    if all(
                        resource_avail_in_time[res][t]
                        >= problem.mode_details[act_id][modes_dict[act_id]].get(res, 0)
                        for res in problem.resources_list
                    )
                )
            if current_min_time > new_horizon:
                unfeasible_in_horizon = True
                break
        if not unfeasible_non_renewable_resources and not unfeasible_in_horizon:
            end_t = ends[-1]
            for s, e, ws, j in zip(starts, ends, workers, range(len(starts))):
                for res in resource_avail_in_time.keys():
                    resource_avail_in_time[res][s:e] -= problem.mode_details[act_id][
                        modes_dict[act_id]
                    ].get(res, 0)
                    if res in problem.non_renewable_resources and e == end_t:
                        resource_avail_in_time[res][
                            end_t + 1 :
                        ] -= problem.mode_details[act_id][modes_dict[act_id]][res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                current_skills = {skill: 0.0 for skill in required_skills}
                for w in ws:
                    for skill in problem.employees[w].dict_skill:
                        if skill in current_skills:
                            current_skills[skill] += (
                                problem.employees[w].dict_skill[skill].skill_value
                            )
                            if act_id not in employee_usage:
                                employee_usage[act_id] = [
                                    {} for k in range(len(starts))
                                ]
                            if w not in employee_usage[act_id][j]:
                                employee_usage[act_id][j][w] = set()
                            employee_usage[act_id][j][w].add(skill)
                            worker_avail_in_time[w][s:e] = False
                    if all(
                        current_skills[skill] >= required_skills[skill]
                        for skill in required_skills
                    ):
                        skills_fulfilled = True
                        break
            activity_end_times[act_id] = ends[-1]
            schedules[act_id] = {"starts": starts, "ends": ends}
            perm_extended.remove(act_id)
            scheduled.add(act_id)
            for s in problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule = schedules
    if unfeasible_non_renewable_resources or unfeasible_in_horizon or unfeasible_skills:
        rcpsp_schedule_feasible = False
        last_act_id = max(problem.successors.keys())
        rcpsp_schedule[last_act_id] = {
            "starts": [99999999],
            "ends": [9999999],
        }
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, [], employee_usage, modes_dict


def sgs_multi_skill_preemptive_partial_schedule(
    solution: MS_RCPSPSolution_Preemptive_Variant,
    current_t,
    completed_tasks: Dict[Hashable, TaskDetailsPreemptive],
    scheduled_tasks_start_times: Dict[Hashable, TaskDetailsPreemptive],
):
    problem: MS_RCPSPModel = solution.problem
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    unfeasible_in_horizon = False
    new_horizon = problem.horizon * problem.horizon_multiplier
    resource_avail_in_time = {}
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.array(
            problem.resources_availability[res][: new_horizon + 1]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.array(
            problem.employees[i].calendar_employee[: new_horizon + 1], dtype=bool
        )
    minimum_starting_time = {}
    for act in problem.tasks_list:
        minimum_starting_time[act] = current_t
        if problem.do_special_constraints:
            if act in problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    max(
                        problem.special_constraints.start_times_window[act][0],
                        current_t,
                    )
                    if problem.special_constraints.start_times_window[act][0]
                    is not None
                    else current_t
                )
    perm_extended = [
        problem.tasks_list_non_dummy[x] for x in solution.priority_list_task
    ]
    perm_extended.insert(0, problem.source_task)
    perm_extended.append(problem.sink_task)
    modes_dict = problem.build_mode_dict(solution.modes_vector)
    employee_usage = {}
    scheduled = set()
    unfeasible_skills = False
    expected_durations_task = {
        k: problem.mode_details[k][modes_dict[k]]["duration"] for k in modes_dict
    }
    schedules = {}

    for c in completed_tasks:
        start_time = completed_tasks[c].starts
        end_time = completed_tasks[c].ends
        workers = completed_tasks[c].resource_units_used
        employee_usage[c] = [{} for k in range(len(start_time))]
        for s, e, ws, j in zip(start_time, end_time, workers, range(len(start_time))):
            for res in resource_avail_in_time.keys():
                resource_avail_in_time[res][s:e] -= problem.mode_details[c][
                    modes_dict[c]
                ].get(res, 0)
                if res in problem.non_renewable_resources and e == end_time[-1]:
                    resource_avail_in_time[res][
                        end_time[-1] + 1 :
                    ] -= problem.mode_details[c][modes_dict[c]][res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
            for unit in ws:
                worker_avail_in_time[unit][s:e] = False
            if len(ws) > 0:
                for unit in ws:
                    employee_usage[c][j][unit] = set(
                        problem.employees[unit].dict_skill.keys()
                    ).intersection(
                        set(
                            [
                                s
                                for s in problem.skills_set
                                if problem.mode_details[c][modes_dict[c]].get(s, 0) > 0
                            ]
                        )
                    )
        scheduled.add(c)
        perm_extended.remove(c)
        schedules[c] = {"starts": start_time, "ends": end_time}
        for succ in problem.successors.get(c, set()):
            minimum_starting_time[succ] = max(minimum_starting_time[succ], end_time[-1])
    for c in scheduled_tasks_start_times:
        start_time = scheduled_tasks_start_times[c].starts
        end_time = scheduled_tasks_start_times[c].ends
        workers = scheduled_tasks_start_times[c].resource_units_used
        employee_usage[c] = [{} for k in range(len(start_time))]
        for s, e, ws, j in zip(start_time, end_time, workers, range(len(start_time))):
            for res in resource_avail_in_time.keys():
                resource_avail_in_time[res][s:e] -= problem.mode_details[c][
                    modes_dict[c]
                ].get(res, 0)
                if res in problem.non_renewable_resources and e == end_time[-1]:
                    resource_avail_in_time[res][
                        end_time[-1] + 1 :
                    ] -= problem.mode_details[c][modes_dict[c]][res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
            for unit in ws:
                worker_avail_in_time[unit][s:e] = False
            if len(ws) > 0:
                for unit in ws:
                    employee_usage[c][j][unit] = set(
                        problem.employees[unit].dict_skill.keys()
                    ).intersection(
                        set(
                            [
                                s
                                for s in problem.skills_set
                                if problem.mode_details[c][modes_dict[c]].get(s, 0) > 0
                            ]
                        )
                    )

        scheduled.add(c)
        perm_extended.remove(c)
        schedules[c] = {"starts": start_time, "ends": end_time}
        for succ in problem.successors.get(c, set()):
            minimum_starting_time[succ] = max(minimum_starting_time[succ], end_time[-1])

    def find_workers(required_skills, task, current_min_time):
        valid = True
        if len(required_skills) > 0:
            worker_ids = [
                worker
                for worker in worker_avail_in_time
                if worker_avail_in_time[worker][current_min_time]
            ]
            if problem.one_unit_per_task_max:
                good = False
                ws = []
                for worker in worker_ids:
                    if any(
                        s not in problem.employees[worker].dict_skill
                        for s in required_skills
                    ) or any(
                        problem.employees[worker].dict_skill[s].skill_value
                        < required_skills[s]
                        for s in required_skills
                    ):
                        continue
                    else:
                        good = True
                        ws += [worker]
                        break
                valid = good
                if good:
                    worker_ids = ws
            else:
                if not all(
                    sum(
                        [
                            problem.employees[worker].dict_skill[s].skill_value
                            for worker in worker_ids
                            if s in problem.employees[worker].dict_skill
                        ]
                    )
                    >= required_skills[s]
                    for s in required_skills
                ):
                    valid = False
            if valid:
                priority_list_this_task = solution.priority_worker_per_task[
                    problem.index_task_non_dummy[task]
                ]
                worker_used = []
                current_skills = {s: 0.0 for s in required_skills}
                skills_fulfilled = False
                for w in priority_list_this_task:
                    if w in worker_ids:
                        worker_used += [w]
                        for s in problem.employees[w].dict_skill:
                            if s in current_skills:
                                current_skills[s] += (
                                    problem.employees[w].dict_skill[s].skill_value
                                )
                        if all(
                            current_skills[s] >= required_skills[s]
                            for s in required_skills
                        ):
                            skills_fulfilled = True
                            break
                return valid, skills_fulfilled, worker_used
            else:
                return valid, False, []
        return True, True, []

    while (
        len(perm_extended) > 0
        and not unfeasible_non_renewable_resources
        and not unfeasible_in_horizon
    ):
        act_id = None
        for id_successor in perm_extended:
            respected = True
            for pred in problem.predecessors.get(id_successor, set()):
                if pred not in scheduled:
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        cur_duration = 0
        starts = []
        ends = []
        workers = []
        mode = modes_dict[act_id]
        required_skills = {
            s: problem.mode_details[act_id][mode][s]
            for s in problem.mode_details[act_id][mode]
            if s in problem.skills_set and problem.mode_details[act_id][mode][s] > 0
        }
        while not valid:
            valid = True
            reached_t = None
            if expected_durations_task[act_id] == 0:
                starts += [current_min_time]
                ends += [current_min_time]
                workers += [[]]
                cur_duration += ends[-1] - starts[-1]
            else:
                reached_end = True
                for t in range(
                    current_min_time,
                    current_min_time + expected_durations_task[act_id] - cur_duration,
                ):
                    if t == current_min_time:
                        valid, skills_fulfilled, worker_used = find_workers(
                            required_skills=required_skills,
                            task=act_id,
                            current_min_time=current_min_time,
                        )
                        if not valid or not skills_fulfilled:
                            reached_end = False
                            break
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    if any(
                        resource_avail_in_time[res][t]
                        < problem.mode_details[act_id][modes_dict[act_id]].get(res, 0)
                        for res in problem.resources_list
                    ):
                        reached_end = False
                        break
                    else:
                        if any(not worker_avail_in_time[w][t] for w in worker_used):
                            reached_end = False
                            break
                        else:
                            reached_t = t
                if reached_t is not None:
                    starts += [current_min_time]
                    ends += [reached_t + 1]
                    workers += [worker_used]
                    cur_duration += ends[-1] - starts[-1]
            valid = cur_duration == expected_durations_task[act_id]
            if not valid:
                current_min_time = next(
                    t
                    for t in range(
                        reached_t + 2
                        if reached_t is not None
                        else current_min_time + 1,
                        new_horizon,
                    )
                    if all(
                        resource_avail_in_time[res][t]
                        >= problem.mode_details[act_id][modes_dict[act_id]].get(res, 0)
                        for res in problem.resources_list
                    )
                )
            if current_min_time > new_horizon:
                unfeasible_in_horizon = True
                break
        if not unfeasible_non_renewable_resources and not unfeasible_in_horizon:
            end_t = ends[-1]
            for s, e, ws, j in zip(starts, ends, workers, range(len(starts))):
                for res in resource_avail_in_time.keys():
                    resource_avail_in_time[res][s:e] -= problem.mode_details[act_id][
                        modes_dict[act_id]
                    ].get(res, 0)
                    if res in problem.non_renewable_resources and e == end_t:
                        resource_avail_in_time[res][
                            end_t + 1 :
                        ] -= problem.mode_details[act_id][modes_dict[act_id]][res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                current_skills = {skill: 0.0 for skill in required_skills}
                for w in ws:
                    for skill in problem.employees[w].dict_skill:
                        if skill in current_skills:
                            current_skills[skill] += (
                                problem.employees[w].dict_skill[skill].skill_value
                            )
                            if act_id not in employee_usage:
                                employee_usage[act_id] = [
                                    {} for k in range(len(starts))
                                ]
                            if w not in employee_usage[act_id][j]:
                                employee_usage[act_id][j][w] = set()
                            employee_usage[act_id][j][w].add(skill)
                            worker_avail_in_time[w][s:e] = False
                    if all(
                        current_skills[skill] >= required_skills[skill]
                        for skill in required_skills
                    ):
                        skills_fulfilled = True
                        break
            activity_end_times[act_id] = ends[-1]
            schedules[act_id] = {"starts": starts, "ends": ends}
            perm_extended.remove(act_id)
            scheduled.add(act_id)
            for s in problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
        worker_ids = None
    rcpsp_schedule = schedules
    if unfeasible_non_renewable_resources or unfeasible_in_horizon or unfeasible_skills:
        rcpsp_schedule_feasible = False
        last_act_id = max(problem.successors.keys())
        rcpsp_schedule[last_act_id] = {
            "starts": [99999999],
            "ends": [9999999],
        }
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, [], employee_usage, modes_dict


def sgs_multi_skill_partial_schedule(
    solution: MS_RCPSPSolution_Variant,
    current_t,
    completed_tasks: Dict[Hashable, TaskDetails],
    scheduled_tasks_start_times: Dict[Hashable, TaskDetails],
):
    problem: MS_RCPSPModel = solution.problem
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    unfeasible_in_horizon = False
    new_horizon = problem.horizon * problem.horizon_multiplier
    resource_avail_in_time = {}
    for res in problem.resources_set:
        resource_avail_in_time[res] = np.array(
            problem.resources_availability[res][: new_horizon + 1]
        )
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = np.array(
            problem.employees[i].calendar_employee[: new_horizon + 1], dtype=bool
        )
    perm_extended = [
        problem.tasks_list_non_dummy[x] for x in solution.priority_list_task
    ]
    perm_extended.insert(0, problem.source_task)
    perm_extended.append(problem.sink_task)
    modes_dict = problem.build_mode_dict(solution.modes_vector)
    employee_usage = {}
    scheduled = set()
    minimum_starting_time = {}
    rcpsp_schedule = {}
    for act in problem.tasks_list:
        minimum_starting_time[act] = current_t
        if act in problem.special_constraints.start_times_window:
            minimum_starting_time[act] = (
                max(problem.special_constraints.start_times_window[act][0], current_t)
                if problem.special_constraints.start_times_window[act][0] is not None
                else current_t
            )

    for c in completed_tasks:
        start_time = completed_tasks[c].start
        end_time = completed_tasks[c].end
        for res in resource_avail_in_time.keys():
            resource_avail_in_time[res][start_time:end_time] -= problem.mode_details[c][
                modes_dict[c]
            ].get(res, 0)
            if res in problem.non_renewable_resources:
                resource_avail_in_time[res][end_time:] -= problem.mode_details[c][
                    modes_dict[c]
                ][res]
                if resource_avail_in_time[res][-1] < 0:
                    unfeasible_non_renewable_resources = True
        for unit in completed_tasks[c].resource_units_used:
            worker_avail_in_time[unit][start_time:end_time] = False
        if len(completed_tasks[c].resource_units_used) > 0:
            employee_usage[c] = {}
            for unit in completed_tasks[c].resource_units_used:
                employee_usage[c][unit] = set(
                    problem.employees[unit].dict_skill.keys()
                ).intersection(
                    set(
                        [
                            s
                            for s in problem.skills_set
                            if problem.mode_details[c][modes_dict[c]].get(s, 0) > 0
                        ]
                    )
                )
        scheduled.add(c)
        perm_extended.remove(c)
        rcpsp_schedule[c] = {"start_time": start_time, "end_time": end_time}
        for succ in problem.successors.get(c, set()):
            minimum_starting_time[succ] = max(minimum_starting_time[succ], end_time)
    for c in scheduled_tasks_start_times:
        start_time = scheduled_tasks_start_times[c].start
        end_time = scheduled_tasks_start_times[c].end
        rcpsp_schedule[c] = {"start_time": start_time, "end_time": end_time}
        for res in resource_avail_in_time.keys():
            resource_avail_in_time[res][start_time:end_time] -= problem.mode_details[c][
                modes_dict[c]
            ].get(res, 0)
            if res in problem.non_renewable_resources:
                resource_avail_in_time[res][end_time:] -= problem.mode_details[c][
                    modes_dict[c]
                ][res]
                if resource_avail_in_time[res][-1] < 0:
                    unfeasible_non_renewable_resources = True
        for unit in scheduled_tasks_start_times[c].resource_units_used:
            worker_avail_in_time[unit][start_time:end_time] = False
        if len(scheduled_tasks_start_times[c].resource_units_used) > 0:
            employee_usage[c] = {}
            for unit in scheduled_tasks_start_times[c].resource_units_used:
                employee_usage[c][unit] = set(
                    problem.employees[unit].dict_skill.keys()
                ).intersection(
                    set(
                        [
                            s
                            for s in problem.skills_set
                            if problem.mode_details[c][modes_dict[c]].get(s, 0) > 0
                        ]
                    )
                )
        scheduled.add(c)
        perm_extended.remove(c)
        minimum_starting_time[c] = start_time
        for succ in problem.successors.get(c, set()):
            minimum_starting_time[succ] = max(minimum_starting_time[succ], end_time)
    unfeasible_skills = False
    while (
        len(perm_extended) > 0
        and not unfeasible_non_renewable_resources
        and not unfeasible_in_horizon
    ):
        act_id = None
        for id_successor in perm_extended:
            respected = True
            for pred in problem.predecessors.get(id_successor, set()):
                if pred not in scheduled:
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            mode = modes_dict[act_id]
            range_time = range(
                current_min_time,
                current_min_time + problem.mode_details[act_id][mode]["duration"],
            )
            if (
                current_min_time + problem.mode_details[act_id][mode]["duration"]
                >= problem.horizon
            ):
                unfeasible_in_horizon = True
                break
            for t in range_time:
                for res in resource_avail_in_time.keys():
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0):
                            valid = False
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                required_skills = {
                    s: problem.mode_details[act_id][mode][s]
                    for s in problem.mode_details[act_id][mode]
                    if s in problem.skills_set
                    and problem.mode_details[act_id][mode][s] > 0
                }
                worker_ids = None
                if len(required_skills) > 0:
                    worker_ids = [
                        worker
                        for worker in worker_avail_in_time
                        if all(worker_avail_in_time[worker][t] for t in range_time)
                    ]
                    if problem.one_unit_per_task_max:
                        good = False
                        ws = []
                        for worker in worker_ids:
                            if any(
                                s not in problem.employees[worker].dict_skill
                                for s in required_skills
                            ) or any(
                                problem.employees[worker].dict_skill[s].skill_value
                                < required_skills[s]
                                for s in required_skills
                            ):
                                continue
                            else:
                                good = True
                                ws += [worker]
                                break
                        valid = good
                        if good:
                            worker_ids = ws
                    else:
                        if not all(
                            sum(
                                [
                                    problem.employees[worker].dict_skill[s].skill_value
                                    for worker in worker_ids
                                    if s in problem.employees[worker].dict_skill
                                ]
                            )
                            >= required_skills[s]
                            for s in required_skills
                        ):
                            valid = False
            if not valid:
                current_min_time += 1
            if current_min_time > new_horizon:
                unfeasible_in_horizon = True
                break
        if not unfeasible_non_renewable_resources and not unfeasible_in_horizon:
            end_t = (
                current_min_time
                + problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                - 1
            )
            for res in resource_avail_in_time.keys():
                resource_avail_in_time[res][
                    current_min_time : end_t + 1
                ] -= problem.mode_details[act_id][modes_dict[act_id]].get(res, 0)
                if res in problem.non_renewable_resources:
                    resource_avail_in_time[res][end_t + 1 :] -= problem.mode_details[
                        act_id
                    ][modes_dict[act_id]][res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
            if worker_ids is not None:
                priority_list_this_task = solution.priority_worker_per_task[
                    problem.index_task_non_dummy[act_id]
                ]
                worker_used = []
                current_skills = {s: 0.0 for s in required_skills}
                skills_fulfilled = False
                for w in priority_list_this_task:
                    if w in worker_ids:
                        worker_used += [w]
                        for s in problem.employees[w].dict_skill:
                            if s in current_skills:
                                current_skills[s] += (
                                    problem.employees[w].dict_skill[s].skill_value
                                )
                                if act_id not in employee_usage:
                                    employee_usage[act_id] = {}
                                if w not in employee_usage[act_id]:
                                    employee_usage[act_id][w] = set()
                                employee_usage[act_id][w].add(s)
                        worker_avail_in_time[w][
                            current_min_time : current_min_time
                            + problem.mode_details[act_id][modes_dict[act_id]][
                                "duration"
                            ]
                        ] = False
                    if all(
                        current_skills[s] >= required_skills[s] for s in required_skills
                    ):
                        skills_fulfilled = True
                        break
                if not skills_fulfilled:
                    logger.warning(
                        "You probably didnt give the right worker named in priority_worker_per_task"
                    )
                    unfeasible_skills = True
                    break
            if unfeasible_skills:
                break
            activity_end_times[act_id] = (
                current_min_time
                + problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            perm_extended.remove(act_id)
            scheduled.add(act_id)
            for s in problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
        worker_ids = None
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    if unfeasible_non_renewable_resources or unfeasible_in_horizon or unfeasible_skills:
        rcpsp_schedule_feasible = False
        last_act_id = max(problem.successors.keys())
        if last_act_id not in rcpsp_schedule.keys():
            rcpsp_schedule[last_act_id] = {
                "start_time": 99999999,
                "end_time": 9999999,
            }
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, [], employee_usage, modes_dict


class SkillDetail:
    skill_value: float
    efficiency_ratio: float
    experience: float

    def __init__(self, skill_value: float, efficiency_ratio: float, experience: float):
        self.skill_value = skill_value
        self.efficiency_ratio = efficiency_ratio
        self.experience = experience

    def copy(self):
        return SkillDetail(
            skill_value=self.skill_value,
            efficiency_ratio=self.efficiency_ratio,
            experience=self.experience,
        )

    def __str__(self):
        return (
            "skill-value:"
            + str(self.skill_value)
            + " efficiency:"
            + str(self.efficiency_ratio)
            + " experience:"
            + str(self.experience)
        )


class Employee:
    dict_skill: Dict[str, SkillDetail]
    calendar_employee: List[bool]

    def __init__(
        self,
        dict_skill: Dict[str, SkillDetail],
        calendar_employee: List[bool],
        salary: float = 0.0,
    ):
        self.salary = salary
        self.dict_skill = dict_skill
        self.calendar_employee = calendar_employee

    def copy(self):
        return Employee(
            dict_skill={s: self.dict_skill[s].copy() for s in self.dict_skill},
            calendar_employee=list(self.calendar_employee),
        )

    def get_non_zero_skills(self):
        return [s for s in self.dict_skill if self.dict_skill[s].skill_value > 0]

    def to_json(self, with_calendar: bool = True):
        if with_calendar:
            return {
                "dict_skill": {
                    s: {
                        attr: getattr(self.dict_skill[s], attr)
                        for attr in ["skill_value", "efficiency_ratio", "experience"]
                    }
                    for s in self.dict_skill
                },
                "calendar_employee": list(self.calendar_employee),
            }
        else:
            return {
                "dict_skill": {
                    s: {
                        attr: getattr(self.dict_skill[s], attr)
                        for attr in ["skill_value", "efficiency_ratio", "experience"]
                    }
                    for s in self.dict_skill
                },
                "calendar_employee": [True],
            }

    def get_skill_level(self, s):
        return self.dict_skill.get(s, SkillDetail(0, 0, 0)).skill_value


def intersect(i1, i2):
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


class MS_RCPSPModel(Problem):
    sgs: ScheduleGenerationScheme
    skills_set: Set[str]
    resources_set: Set[str]
    non_renewable_resources: Set[str]
    resources_availability: Dict[str, List[int]]
    employees: Dict[Hashable, Employee]
    employees_availability: List[int]
    n_jobs_non_dummy: int
    mode_details: Dict[Hashable, Dict[int, Dict[str, int]]]
    successors: Dict[Hashable, List[Hashable]]
    partial_preemption_data: Dict[Hashable, Dict[int, Dict[str, bool]]]
    resource_blocking_data: List[Tuple[List[Hashable], Set[str]]]
    # List task 1, task 2, resource : resource should be blocked between start of task 1 and end of task 2
    # {task_id: {mode: {ressource: is_releasable during preemption }
    strictly_disjunctive_subtasks: bool
    # only used in preemptive mode, specifies that subtasks of tasks should be strictly disjunctive or not (i.e (st1, end1), (st2, end2), in strictly disjunctive case, st2>end1+1)

    def __init__(
        self,
        skills_set: Set[str],
        resources_set: Set[str],
        non_renewable_resources: Set[str],
        resources_availability: Dict[str, List[int]],
        employees: Dict[Hashable, Employee],
        mode_details: Dict[Hashable, Dict[int, Dict[str, int]]],
        successors: Dict[Hashable, List[Hashable]],
        horizon,
        employees_availability: List[int] = None,
        tasks_list: List[Hashable] = None,
        employees_list: List[Hashable] = None,
        horizon_multiplier=1,
        sink_task: Optional[Hashable] = None,
        source_task: Optional[Hashable] = None,
        one_unit_per_task_max: bool = False,
        preemptive: bool = False,
        preemptive_indicator: Dict[Hashable, bool] = None,
        special_constraints: SpecialConstraintsDescription = None,
        partial_preemption_data: Dict[Hashable, Dict[int, Dict[str, bool]]] = None,
        always_releasable_resources: Set[str] = None,
        never_releasable_resources: Set[str] = None,
        resource_blocking_data: List[Tuple[List[Hashable], Set[str]]] = None,
        strictly_disjunctive_subtasks: bool = True,
    ):
        self.skills_set = skills_set
        self.skills_list = sorted(self.skills_set)

        self.resources_set = resources_set
        self.resources_list = list(self.resources_set)
        self.non_renewable_resources = non_renewable_resources
        self.resources_availability = resources_availability
        self.employees = employees
        self.employees_availability = employees_availability
        self.mode_details = mode_details
        self.successors = successors

        self.n_jobs = len(self.mode_details)
        self.n_jobs_non_dummy = self.n_jobs - 2
        self.horizon = horizon
        self.horizon_multiplier = horizon_multiplier
        self.nb_tasks = len(self.mode_details)
        self.tasks = list(self.mode_details.keys())
        self.tasks_list = tasks_list
        if self.tasks_list is None:
            self.tasks_list = self.tasks
        self.employees_list = employees_list
        if self.employees_list is None:
            self.employees_list = list(self.employees.keys())
        self.index_task = {self.tasks_list[i]: i for i in range(self.n_jobs)}
        self.source_task = source_task
        if source_task is None:
            self.source_task = min(
                self.tasks_list
            )  # tasks id should be comparable in this case
        self.sink_task = sink_task
        if sink_task is None:
            self.sink_task = max(
                self.tasks_list
            )  # tasks id should be comparable in this case
        self.tasks_list_non_dummy = [
            t for t in self.tasks_list if t not in {self.source_task, self.sink_task}
        ]
        self.index_task_non_dummy = {
            self.tasks_list_non_dummy[i]: i for i in range(self.n_jobs_non_dummy)
        }
        self.one_unit_per_task_max = one_unit_per_task_max

        self.is_multimode = (
            max(
                [
                    len(self.mode_details[key1].keys())
                    for key1 in self.mode_details.keys()
                ]
            )
            > 1
        )
        self.is_calendar = False
        if any(
            isinstance(self.resources_availability[res], Iterable)
            for res in self.resources_availability
        ):
            self.is_calendar = (
                max(
                    [
                        len(
                            set(self.resources_availability[res])
                            if isinstance(self.resources_availability[res], Iterable)
                            else {self.resources_availability[res]}
                        )
                        for res in self.resources_availability
                    ]
                )
                > 1
            )

        self.max_resource_capacity = {}
        for r in self.resources_availability:
            if isinstance(self.resources_availability[r], Iterable):
                self.max_resource_capacity[r] = max(self.resources_availability[r])
            else:
                self.max_resource_capacity[r] = self.resources_availability[r]
        self.max_number_of_mode = max(
            [len(self.mode_details[x]) for x in self.mode_details]
        )
        self.preemptive = preemptive
        self.preemptive_indicator = preemptive_indicator
        if self.preemptive_indicator is None:
            if self.preemptive:
                self.preemptive_indicator = {t: True for t in self.tasks_list}
            else:
                self.preemptive_indicator = {t: False for t in self.tasks_list}
        self.index_employee = {
            self.employees_list[i]: i for i in range(len(self.employees_list))
        }
        self.nb_employees = len(self.employees_list)
        self.special_constraints = special_constraints
        self.do_special_constraints = special_constraints is not None
        if self.special_constraints is None:
            self.special_constraints = SpecialConstraintsDescription()
        self.predecessors_dict = {task: [] for task in self.successors}
        for task in self.successors:
            for stask in self.successors[task]:
                self.predecessors_dict[stask] += [task]
        if self.do_special_constraints:
            for t1, t2 in self.special_constraints.start_at_end:
                if t2 not in self.successors[t1]:
                    self.successors[t1].append(t2)
            for t1, t2, off in self.special_constraints.start_at_end_plus_offset:
                if t2 not in self.successors[t1]:
                    self.successors[t1].append(t2)
            for t1, t2 in self.special_constraints.start_together:
                for predt1 in self.predecessors_dict[t1]:
                    if t2 not in self.successors[predt1]:
                        self.successors[predt1] += [t2]
                for predt2 in self.predecessors_dict[t2]:
                    if t1 not in self.successors[predt2]:
                        self.successors[predt2] += [t1]
        self.graph = self.compute_graph()
        self.predecessors = self.graph.predecessors_dict
        self.total_number_step_available = {
            employee: sum(self.employees[employee].calendar_employee[: self.horizon])
            for employee in self.employees
        }

        self.partial_preemption_data = partial_preemption_data
        self.always_releasable_resources = always_releasable_resources
        self.never_releasable_resources = never_releasable_resources
        if all(
            f is None
            for f in [
                self.partial_preemption_data,
                self.always_releasable_resources,
                self.never_releasable_resources,
            ]
        ):
            self.always_releasable_resources = self.resources_set
            self.never_releasable_resources = set()
            self.partial_preemption_data = {
                t: {
                    m: {r: True for r in self.resources_set}
                    for m in self.mode_details[t]
                }
                for t in self.mode_details
            }
        if (
            self.always_releasable_resources is not None
            and self.never_releasable_resources is not None
            and len(self.always_releasable_resources)
            + len(self.never_releasable_resources)
            == len(self.resources_set)
        ):
            if self.partial_preemption_data is None:
                self.partial_preemption_data = {
                    t: {
                        m: {
                            r: r in self.always_releasable_resources
                            for r in self.resources_set
                        }
                        for m in self.mode_details[t]
                    }
                    for t in self.mode_details
                }
        if (
            self.partial_preemption_data is not None
            and self.always_releasable_resources is None
            and self.never_releasable_resources is None
        ):
            self.always_releasable_resources = set()
            self.never_releasable_resources = set()
            for r in self.resources_set:
                if all(
                    self.partial_preemption_data[t][m].get(r, True)
                    for t in self.partial_preemption_data
                    for m in self.partial_preemption_data[t]
                ):
                    self.always_releasable_resources.add(r)
        self.strictly_disjunctive_subtasks = strictly_disjunctive_subtasks
        self.func_sgs, self.func_sgs_partial = create_np_data_and_jit_functions(
            rcpsp_problem=self
        )
        self.resource_blocking_data = resource_blocking_data
        if self.resource_blocking_data is None:
            self.resource_blocking_data = []

    def get_resource_available(self, res, time):
        return self.resources_availability[res][time]

    def get_tasks_list(self):
        return self.tasks_list

    def get_resource_names(self):
        return self.resources_list

    def is_preemptive(self):
        return self.preemptive

    def is_multiskill(self):
        return True

    def get_resource_availability_array(self, res):
        if self.is_varying_resource():
            return self.resources_availability[res]
        else:
            return self.resources_availability[res]

    def update_functions(self):
        self.func_sgs, self.func_sgs_partial = create_np_data_and_jit_functions(
            rcpsp_problem=self
        )

    def update_function(self):
        self.update_functions()

    def is_rcpsp_multimode(self):
        return self.is_multimode

    def is_varying_resource(self):
        return self.is_calendar

    def includes_special_constraint(self):
        return self.do_special_constraints

    def build_multimode_rcpsp_calendar_representative(self):
        # put skills as ressource.
        if len(self.resources_list) == 0:
            skills_availability = {s: [0] * int(self.horizon) for s in self.skills_set}
        else:
            skills_availability = {
                s: [0] * len(self.resources_availability[self.resources_list[0]])
                for s in self.skills_set
            }
        for emp in self.employees:
            for j in range(len(self.employees[emp].calendar_employee)):
                if self.employees[emp].calendar_employee[
                    min(j, len(self.employees[emp].calendar_employee) - 1)
                ]:
                    for s in self.employees[emp].dict_skill:
                        skills_availability[s][
                            min(j, len(skills_availability[s]) - 1)
                        ] += (self.employees[emp].dict_skill[s].skill_value)
        res_availability = deepcopy(self.resources_availability)
        for s in skills_availability:
            res_availability[s] = [int(x) for x in skills_availability[s]]
        mode_details = deepcopy(self.mode_details)
        for task in mode_details:
            for mode in mode_details[task]:
                for r in self.resources_set:
                    if r not in mode_details[task][mode]:
                        mode_details[task][mode][r] = int(0)
                for s in self.skills_set:
                    if s not in mode_details[task][mode]:
                        mode_details[task][mode][s] = int(0)
        rcpsp_model = RCPSPModel(
            resources=res_availability,
            non_renewable_resources=list(self.non_renewable_resources),
            mode_details=mode_details,
            tasks_list=self.tasks_list,
            source_task=self.source_task,
            sink_task=self.sink_task,
            successors=self.successors,
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
            name_task={i: str(i) for i in self.tasks},
        )
        return rcpsp_model

    def return_index_task(self, task, offset=0):
        return self.index_task[task] + offset

    def compute_graph(self) -> Graph:
        nodes = [
            (
                n,
                {
                    mode: self.mode_details[n][mode]["duration"]
                    for mode in self.mode_details[n]
                },
            )
            for n in self.tasks_list
        ]
        edges = []
        for n in self.successors:
            for succ in self.successors[n]:
                edges += [(n, succ, {})]
        return Graph(nodes, edges, False)

    def build_mode_dict(self, rcpsp_modes_from_solution):
        modes_dict = {
            self.tasks_list_non_dummy[i]: rcpsp_modes_from_solution[i]
            for i in range(self.n_jobs_non_dummy)
        }
        modes_dict[self.source_task] = 1
        modes_dict[self.sink_task] = 1
        return modes_dict

    def get_modes_dict(self, rcpsp_solution: MS_RCPSPSolution):
        return rcpsp_solution.modes

    def evaluate_function(self, rcpsp_sol: MS_RCPSPSolution):
        makespan = rcpsp_sol.get_end_time(self.sink_task)
        return makespan

    @abstractmethod
    def evaluate_from_encoding(self, int_vector, encoding_name):
        ...

    def evaluate(self, rcpsp_sol: MS_RCPSPSolution) -> Dict[str, float]:
        obj_makespan = self.evaluate_function(rcpsp_sol)
        d = {"makespan": obj_makespan}
        if self.includes_special_constraint():
            penalty = evaluate_constraints(
                solution=rcpsp_sol, constraints=self.special_constraints
            )
            d["constraint_penalty"] = penalty
        return d

    def evaluate_mobj(self, rcpsp_sol: MS_RCPSPSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(rcpsp_sol))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]) -> TupleFitness:
        return TupleFitness(np.array([-dict_values["makespan"]]), 1)

    def satisfy(self, variable: Solution) -> bool:
        if isinstance(variable, MS_RCPSPSolution_Preemptive):
            return self.satisfy_preemptive(variable)
        return self.satisfy_classic(variable)

    def satisfy_classic(self, rcpsp_sol: MS_RCPSPSolution) -> bool:
        # check the skills :
        if len(rcpsp_sol.schedule) != self.nb_tasks:
            return False
        for task in self.tasks:
            mode = rcpsp_sol.modes[task]
            required_skills = {
                s: self.mode_details[task][mode][s]
                for s in self.mode_details[task][mode]
                if s in self.skills_set and self.mode_details[task][mode][s] > 0
            }
            # Skills for the given task are used
            if len(required_skills) > 0:
                for skill in required_skills:
                    employees_used = [
                        self.employees[emp].dict_skill[skill].skill_value
                        for emp in rcpsp_sol.employee_usage[task]
                        if skill in rcpsp_sol.employee_usage[task][emp]
                    ]
                    if sum(employees_used) < required_skills[skill]:
                        logger.debug("1")
                        return False
            if task in rcpsp_sol.employee_usage:
                employee_used = [
                    emp
                    for emp in rcpsp_sol.employee_usage[task]
                    if len(rcpsp_sol.employee_usage[task][emp]) > 0
                ]
                # employee available at this time
                if len(employee_used) > 0:
                    for e in employee_used:
                        if not all(
                            self.employees[e].calendar_employee[t]
                            for t in range(
                                rcpsp_sol.schedule[task]["start_time"],
                                rcpsp_sol.schedule[task]["end_time"],
                            )
                        ):
                            logger.debug(f"Task : {task}")
                            logger.debug(f"Employee : {e}")
                            logger.debug(
                                (
                                    e,
                                    [
                                        self.employees[e].calendar_employee[t]
                                        for t in range(
                                            rcpsp_sol.schedule[task]["start_time"],
                                            rcpsp_sol.schedule[task]["end_time"],
                                        )
                                    ],
                                )
                            )
                            logger.warning("Problem with employee availability")
                            return False
        overlaps = [
            (t1, t2)
            for t1 in self.tasks
            for t2 in self.tasks
            if self.index_task[t2] > self.index_task[t1]
            and intersect(
                (
                    rcpsp_sol.schedule[t1]["start_time"],
                    rcpsp_sol.schedule[t1]["end_time"],
                ),
                (
                    rcpsp_sol.schedule[t2]["start_time"],
                    rcpsp_sol.schedule[t2]["end_time"],
                ),
            )
            is not None
            and rcpsp_sol.schedule[t1]["end_time"]
            > rcpsp_sol.schedule[t1]["start_time"]
            and rcpsp_sol.schedule[t2]["end_time"]
            > rcpsp_sol.schedule[t2]["start_time"]
        ]
        for t1, t2 in overlaps:
            if any(
                k in rcpsp_sol.employee_usage.get(t2, {})
                for k in rcpsp_sol.employee_usage.get(t1, {})
            ):
                logger.debug("Worker working on 2 task the same time")
                logger.debug(
                    [
                        k
                        for k in rcpsp_sol.employee_usage.get(t1, {})
                        if k in rcpsp_sol.employee_usage.get(t2, {})
                    ]
                )
                logger.debug(("Tasks ", t1, t2))
                return False
        # ressource usage respected
        makespan = rcpsp_sol.schedule[self.sink_task]["end_time"]
        for t in range(makespan):
            resource_usage = {}
            for res in self.resources_set:
                resource_usage[res] = 0
            for act_id in self.tasks:
                start = rcpsp_sol.schedule[act_id]["start_time"]
                end = rcpsp_sol.schedule[act_id]["end_time"]
                mode = rcpsp_sol.modes[act_id]
                if start <= t and t < end:
                    for res in self.resources_set:  # self.mode_details[act_id][mode]:
                        resource_usage[res] += self.mode_details[act_id][mode].get(
                            res, 0
                        )
            for res in self.resources_set:
                if resource_usage[res] > self.resources_availability[res][t]:
                    logger.debug(
                        (
                            "Time step resource violation: time: ",
                            t,
                            "res",
                            res,
                            "res_usage: ",
                            resource_usage[res],
                            "res_avail: ",
                            self.resources_availability[res][t],
                        )
                    )
                    return False

        # Check for non-renewable resource violation
        for res in self.non_renewable_resources:
            usage = 0
            for act_id in self.tasks:
                mode = rcpsp_sol.modes[act_id]
                usage += self.mode_details[act_id][mode][res]
            if usage > self.resources_availability[res][0]:
                logger.debug(
                    (
                        "Non-renewable res",
                        res,
                        "res_usage: ",
                        usage,
                        "res_avail: ",
                        self.resources_availability[res][0],
                    )
                )
                return False
        # Check precedences / successors
        for act_id in list(self.successors.keys()):
            for succ_id in self.successors[act_id]:
                start_succ = rcpsp_sol.schedule[succ_id]["start_time"]
                end_pred = rcpsp_sol.schedule[act_id]["end_time"]
                if start_succ < end_pred:
                    logger.debug(
                        (
                            "Precedence relationship broken: ",
                            act_id,
                            "end at ",
                            end_pred,
                            "while ",
                            succ_id,
                            "start at",
                            start_succ,
                        )
                    )
                    return False
        return True

    def satisfy_preemptive(self, rcpsp_sol: MS_RCPSPSolution_Preemptive) -> bool:
        # check the skills :
        if len(rcpsp_sol.schedule) != self.nb_tasks:
            return False
        for task in self.tasks:
            mode = rcpsp_sol.modes[task]
            required_skills = {
                s: self.mode_details[task][mode][s]
                for s in self.mode_details[task][mode]
                if s in self.skills_set and self.mode_details[task][mode][s] > 0
            }
            # Skills for the given task are used
            if (
                len(required_skills) > 0
                and self.mode_details[task][mode]["duration"] > 0
            ):
                for skill in required_skills:
                    for i in range(rcpsp_sol.get_number_of_part(task)):
                        employees_used = [
                            self.employees[emp].dict_skill[skill].skill_value
                            for emp in rcpsp_sol.employee_usage[task][i]
                            if skill in rcpsp_sol.employee_usage[task][i][emp]
                        ]
                        if sum(employees_used) < required_skills[skill]:
                            logger.debug(f"Not enough skills to do task : {task}")
                            return False
            if task in rcpsp_sol.employee_usage:
                for i in range(rcpsp_sol.get_number_of_part(task)):
                    employee_used = [
                        emp
                        for emp in rcpsp_sol.employee_usage[task][i]
                        if len(rcpsp_sol.employee_usage[task][i][emp]) > 0
                    ]
                    if len(employee_used) > 0:
                        for e in employee_used:
                            if not all(
                                self.employees[e].calendar_employee[t]
                                for t in range(
                                    rcpsp_sol.schedule[task]["starts"][i],
                                    rcpsp_sol.schedule[task]["ends"][i],
                                )
                            ):
                                logger.debug(f"Task : {task}")
                                logger.debug(f"Employee : {e}")
                                logger.debug(
                                    (
                                        rcpsp_sol.schedule[task]["starts"][i],
                                        rcpsp_sol.schedule[task]["ends"][i],
                                        [
                                            self.employees[e].calendar_employee[t]
                                            for t in range(
                                                rcpsp_sol.schedule[task]["starts"][i],
                                                rcpsp_sol.schedule[task]["ends"][i],
                                            )
                                        ],
                                    )
                                )
                                logger.warning("Problem with employee availability")
                                return False
        for employee in self.employees:
            usage = np.zeros((rcpsp_sol.get_end_time(self.sink_task) + 1))
            task_employees = set(
                [
                    (t, j)
                    for t in rcpsp_sol.employee_usage
                    for j in range(rcpsp_sol.get_number_of_part(t))
                    if employee in rcpsp_sol.employee_usage[t]
                ]
            )
            for t, j in task_employees:
                s = rcpsp_sol.schedule[t]["starts"][j]
                e = rcpsp_sol.schedule[t]["ends"][j]
                usage[s:e] += 1
                if np.max(usage) >= 2:
                    logger.debug(f"Two task at same time for worker {employee}")
                    return False
        makespan = rcpsp_sol.get_end_time(self.sink_task)
        for t in range(makespan):
            resource_usage = {}
            for res in self.resources_set:
                resource_usage[res] = 0
            for act_id in self.tasks:
                starts = rcpsp_sol.schedule[act_id]["starts"]
                ends = rcpsp_sol.schedule[act_id]["ends"]
                mode = rcpsp_sol.modes[act_id]
                for start, end in zip(starts, ends):
                    if start <= t < end:
                        for res in self.resources_set:
                            resource_usage[res] += self.mode_details[act_id][mode].get(
                                res, 0
                            )
            for res in self.resources_set:
                if resource_usage[res] > self.resources_availability[res][t]:
                    logger.debug(
                        (
                            "Time step resource violation: time: ",
                            t,
                            "res",
                            res,
                            "res_usage: ",
                            resource_usage[res],
                            "res_avail: ",
                            self.resources_availability[res][t],
                        )
                    )
                    return False
        # Check for non-renewable resource violation
        for res in self.non_renewable_resources:
            usage = 0
            for act_id in self.tasks:
                mode = rcpsp_sol.modes[act_id]
                usage += self.mode_details[act_id][mode][res]
            if usage > self.resources_availability[res][0]:
                logger.debug(
                    (
                        "Non-renewable res",
                        res,
                        "res_usage: ",
                        usage,
                        "res_avail: ",
                        self.resources_availability[res][0],
                    )
                )
                return False
        # Check precedences / successors
        for act_id in list(self.successors.keys()):
            for succ_id in self.successors[act_id]:
                start_succ = rcpsp_sol.get_start_time(succ_id)
                end_pred = rcpsp_sol.get_end_time(act_id)
                if start_succ < end_pred:
                    logger.debug(
                        (
                            "Precedence relationship broken: ",
                            act_id,
                            "end at ",
                            end_pred,
                            "while ",
                            succ_id,
                            "start at",
                            start_succ,
                        )
                    )
                    return False
        return True

    def __str__(self):
        val = "Multiskill RCPSP model\n"
        val += "Ressource : " + str(self.resources_list)
        val += "\nMultimode : " + str(self.is_multimode)
        val += "\nVarying ressource : " + str(self.is_varying_resource())
        val += "\nPreemptive : " + str(self.preemptive)
        val += "\nSpecial constraints : " + str(self.do_special_constraints)
        return val

    def get_solution_type(self) -> Type[Solution]:
        return MS_RCPSPSolution

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
            "modes": {"name": "modes", "type": [Dict[int, int]]},
            "schedule": {
                "name": "schedule",
                "type": [Dict[int, Dict[str, int]]],
            },
            "employee_usage": {
                "name": "employee_usage",
                "type": [Dict[int, Dict[int, Set[str]]]],
            },
        }
        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0)
        }
        handling = ObjectiveHandling.SINGLE
        if self.includes_special_constraint():
            dict_objective["constraint_penalty"] = ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100.0
            )
            handling = ObjectiveHandling.AGGREGATE
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=handling,
            dict_objective_to_doc=dict_objective,
        )

    def copy(self):
        return MS_RCPSPModel(
            skills_set=self.skills_set,
            resources_set=self.resources_set,
            non_renewable_resources=self.non_renewable_resources,
            resources_availability=self.resources_availability,
            employees=self.employees,
            employees_availability=self.employees_availability,
            mode_details=self.mode_details,
            successors=self.successors,
            horizon=self.horizon,
            sink_task=self.sink_task,
            source_task=self.source_task,
            horizon_multiplier=self.horizon_multiplier,
            partial_preemption_data=self.partial_preemption_data,
            always_releasable_resources=self.always_releasable_resources,
            never_releasable_resources=self.never_releasable_resources,
            resource_blocking_data=self.resource_blocking_data,
            strictly_disjunctive_subtasks=self.strictly_disjunctive_subtasks,
        )

    def to_variant_model(self):
        return MS_RCPSPModel_Variant(
            skills_set=self.skills_set,
            resources_set=self.resources_set,
            non_renewable_resources=self.non_renewable_resources,
            resources_availability=self.resources_availability,
            employees=self.employees,
            employees_availability=self.employees_availability,
            mode_details=self.mode_details,
            successors=self.successors,
            horizon=self.horizon,
            sink_task=self.sink_task,
            source_task=self.source_task,
            horizon_multiplier=self.horizon_multiplier,
            one_unit_per_task_max=self.one_unit_per_task_max,
            preemptive=self.preemptive,
            preemptive_indicator=self.preemptive_indicator,
            special_constraints=None
            if not self.do_special_constraints
            else self.special_constraints,
            partial_preemption_data=self.partial_preemption_data,
            always_releasable_resources=self.always_releasable_resources,
            never_releasable_resources=self.never_releasable_resources,
            resource_blocking_data=self.resource_blocking_data,
            strictly_disjunctive_subtasks=self.strictly_disjunctive_subtasks,
        )

    def get_dummy_solution(self):
        return None

    def get_max_resource_capacity(self, res):
        return self.max_resource_capacity[res]


class MS_RCPSPModel_Variant(MS_RCPSPModel):
    def __init__(
        self,
        skills_set: Set[str],
        resources_set: Set[str],
        non_renewable_resources: Set[str],
        resources_availability: Dict[str, List[int]],
        employees: Dict[Hashable, Employee],
        employees_availability: List[int],
        mode_details: Dict[Hashable, Dict[int, Dict[str, int]]],
        successors: Dict[Hashable, List[Hashable]],
        horizon,
        tasks_list: List[Hashable] = None,
        employees_list: List[Hashable] = None,
        horizon_multiplier=1,
        sink_task: Optional[Hashable] = None,
        source_task: Optional[Hashable] = None,
        one_unit_per_task_max: bool = False,
        preemptive: bool = False,
        preemptive_indicator: Dict[Hashable, bool] = None,
        special_constraints: SpecialConstraintsDescription = None,
        partial_preemption_data: Dict[Hashable, Dict[int, Dict[str, bool]]] = None,
        always_releasable_resources: Set[str] = None,
        never_releasable_resources: Set[str] = None,
        resource_blocking_data: List[Tuple[List[Hashable], Set[str]]] = None,
        strictly_disjunctive_subtasks: bool = True,
    ):
        MS_RCPSPModel.__init__(
            self,
            skills_set=skills_set,
            resources_set=resources_set,
            non_renewable_resources=non_renewable_resources,
            resources_availability=resources_availability,
            employees=employees,
            employees_availability=employees_availability,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            tasks_list=tasks_list,
            employees_list=employees_list,
            horizon_multiplier=horizon_multiplier,
            sink_task=sink_task,
            source_task=source_task,
            one_unit_per_task_max=one_unit_per_task_max,
            preemptive=preemptive,
            preemptive_indicator=preemptive_indicator,
            special_constraints=special_constraints,
            partial_preemption_data=partial_preemption_data,
            always_releasable_resources=always_releasable_resources,
            never_releasable_resources=never_releasable_resources,
            resource_blocking_data=resource_blocking_data,
            strictly_disjunctive_subtasks=strictly_disjunctive_subtasks,
        )
        self.fixed_modes = None
        self.fixed_permutation = None
        self.fixed_priority_worker_per_task = None

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {}
        mode_arity = [
            len(self.mode_details[task]) for task in self.tasks_list_non_dummy
        ]
        max_number_modes = max(mode_arity)
        dict_register["priority_list_task"] = {
            "name": "priority_list_task",
            "type": [TypeAttribute.PERMUTATION, TypeAttribute.PERMUTATION_RCPSP],
            "range": range(self.n_jobs_non_dummy),
            "n": self.n_jobs_non_dummy,
        }
        dict_register["priority_worker_per_task_perm"] = {
            "name": "priority_worker_per_task",
            "type": [TypeAttribute.PERMUTATION, TypeAttribute.PERMUTATION_RCPSP],
            "range": range(self.n_jobs_non_dummy * len(self.employees.keys())),
            "n": self.n_jobs_non_dummy * len(self.employees.keys()),
        }
        dict_register["priority_worker_per_task"] = {
            "name": "priority_worker_per_task",
            "type": [List[List[int]]],
        }
        dict_register["modes_vector"] = {
            "name": "modes_vector",
            "n": self.n_jobs_non_dummy,
            "arity": max_number_modes,
            "low": 1,
            "up": mode_arity,
            "arities": mode_arity,
            "type": [
                TypeAttribute.LIST_INTEGER,
                TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY,
            ],
        }

        dict_register["modes_arity_fix"] = {
            "name": "modes_vector",
            "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY],
            "n": self.n_jobs_non_dummy,
            "low": 1,
            "up": mode_arity,
            "arities": mode_arity,
        }
        dict_register["modes_arity_fix_from_0"] = {
            "name": "modes_vector_from0",
            "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY],
            "n": self.n_jobs_non_dummy,
            "low": 0,
            "up": [x - 1 for x in mode_arity],
            "arities": mode_arity,
        }

        dict_register["schedule"] = {
            "name": "schedule",
            "type": [Dict[int, Dict[str, int]]],
        }
        dict_register["employee_usage"] = {
            "name": "employee_usage",
            "type": [Dict[int, Dict[int, Set[str]]]],
        }
        return EncodingRegister(dict_register)

    def get_dummy_solution(self, preemptive: Optional[bool] = None):
        preemptive = self.preemptive if preemptive is None else preemptive
        if not preemptive:
            return MS_RCPSPSolution_Variant(
                problem=self,
                priority_list_task=[i for i in range(self.n_jobs_non_dummy)],
                modes_vector=[1 for i in range(self.n_jobs_non_dummy)],
                priority_worker_per_task=[
                    [w for w in self.employees] for i in range(self.n_jobs_non_dummy)
                ],
                fast=True,
            )
        else:
            return MS_RCPSPSolution_Preemptive_Variant(
                problem=self,
                priority_list_task=[i for i in range(self.n_jobs_non_dummy)],
                modes_vector=[1 for i in range(self.n_jobs_non_dummy)],
                priority_worker_per_task=[
                    [w for w in self.employees] for i in range(self.n_jobs_non_dummy)
                ],
                fast=True,
            )

    def evaluate_function(self, rcpsp_sol: MS_RCPSPSolution_Variant):
        try:
            if rcpsp_sol._schedule_to_recompute:
                rcpsp_sol.do_recompute(rcpsp_sol.fast)
        except:
            pass
        return super().evaluate_function(rcpsp_sol)

    def set_fixed_attributes(self, encoding_str: str, sol: MS_RCPSPSolution_Variant):
        att = self.get_attribute_register().dict_attribute_to_type[encoding_str]["name"]
        if att == "modes_vector" or att == "modes_vector_from0":
            self.set_fixed_modes(sol.modes_vector)
            logger.debug(f"self.fixed_modes: {self.fixed_modes}")
        elif att == "priority_worker_per_task":
            self.set_fixed_priority_worker_per_task(sol.priority_worker_per_task)
            logger.debug(
                f"self.fixed_priority_worker_per_task: {self.fixed_priority_worker_per_task}"
            )
        elif att == "priority_list_task":
            self.set_fixed_task_permutation(sol.priority_list_task)
            logger.debug(f"self.fixed_permutation: {self.fixed_permutation}")

    def set_fixed_modes(self, fixed_modes):
        self.fixed_modes = fixed_modes

    def set_fixed_task_permutation(self, fixed_permutation):
        self.fixed_permutation = fixed_permutation

    def set_fixed_priority_worker_per_task(self, fixed_priority_worker_per_task):
        self.fixed_priority_worker_per_task = fixed_priority_worker_per_task

    def set_fixed_priority_worker_per_task_from_permutation(self, permutation):
        self.fixed_priority_worker_per_task = (
            self.convert_fixed_priority_worker_per_task_from_permutation(permutation)
        )

    def convert_fixed_priority_worker_per_task_from_permutation(self, permutation):
        priority_worker_per_task_corrected = []
        for i in range(self.n_jobs_non_dummy):
            tmp = []
            for j in range(len(self.employees.keys())):
                tmp.append(permutation[i * len(self.employees.keys()) + j])
            tmp_corrected = [int(x) for x in ss.rankdata(tmp)]
            priority_worker_per_task_corrected.append(tmp_corrected)
        return priority_worker_per_task_corrected

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == "priority_list_task":
            # change the permutation in the solution with int_vector and set the modes with self.fixed_modes
            rcpsp_sol = MS_RCPSPSolution_Variant(
                problem=self,
                priority_list_task=int_vector,
                modes_vector=self.fixed_modes,
                priority_worker_per_task=self.fixed_priority_worker_per_task,
            )
        elif encoding_name == "modes_vector":
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = int_vector
            rcpsp_sol = MS_RCPSPSolution_Variant(
                problem=self,
                priority_list_task=self.fixed_permutation,
                modes_vector=modes_corrected,
                priority_worker_per_task=self.fixed_priority_worker_per_task,
            )
        elif encoding_name == "modes_vector_from0":
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = [x + 1 for x in int_vector]
            rcpsp_sol = MS_RCPSPSolution_Variant(
                problem=self,
                priority_list_task=self.fixed_permutation,
                modes_vector=modes_corrected,
                priority_worker_per_task=self.fixed_priority_worker_per_task,
            )
        elif encoding_name == "priority_worker_per_task":
            # change the resource permutation priority lists in the solution from int_vector and set the permutation
            # with self.fixed_permutation and the modes with self.fixed_modes
            priority_worker_per_task_corrected = []
            for i in range(self.n_jobs_non_dummy):
                tmp = []
                for j in range(len(self.employees.keys())):
                    tmp.append(int_vector[i * len(self.employees.keys()) + j])
                tmp_corrected = [int(x) for x in ss.rankdata(tmp)]
                priority_worker_per_task_corrected.append(tmp_corrected)
            rcpsp_sol = MS_RCPSPSolution_Variant(
                problem=self,
                priority_list_task=self.fixed_permutation,
                modes_vector=self.fixed_modes,
                priority_worker_per_task=priority_worker_per_task_corrected,
            )
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def get_solution_type(self) -> Type[Solution]:
        if not self.preemptive:
            return MS_RCPSPSolution_Variant
        else:
            return MS_RCPSPSolution_Preemptive_Variant


def permutation_do_to_permutation_sgs_fast(
    rcpsp_problem: MS_RCPSPModel, permutation_do
):
    perm_extended = [
        rcpsp_problem.index_task[rcpsp_problem.tasks_list_non_dummy[x]]
        for x in permutation_do
    ]
    perm_extended.insert(0, rcpsp_problem.index_task[rcpsp_problem.source_task])
    perm_extended.append(rcpsp_problem.index_task[rcpsp_problem.sink_task])
    return np.array(perm_extended, dtype=np.int32)


def priority_worker_per_task_do_to_permutation_sgs_fast(
    rcpsp_problem: MS_RCPSPModel, priority_worker_per_task
):
    p = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.nb_employees), dtype=int)
    p[0, :] = np.arange(0, rcpsp_problem.nb_employees, 1)
    p[-1, :] = p[0, :]
    for i in range(len(priority_worker_per_task)):
        p[i + 1, :] = np.array(
            [rcpsp_problem.index_employee[e] for e in priority_worker_per_task[i]]
        )
    return p


def build_partial_vectors(
    problem: MS_RCPSPModel,
    completed_tasks: Dict[Hashable, TaskDetails],
    scheduled_tasks_start_times: Dict[Hashable, TaskDetails],
):
    scheduled_task_indicator = np.zeros(problem.n_jobs)
    scheduled_tasks_start_times_vector = np.zeros(problem.n_jobs, dtype=np.int32)
    scheduled_tasks_end_times_vector = np.zeros(problem.n_jobs, dtype=np.int32)
    worker_used = np.zeros((problem.n_jobs, problem.nb_employees), dtype=int)
    for dict_data in [completed_tasks, scheduled_tasks_start_times]:
        for t in dict_data:
            scheduled_task_indicator[problem.index_task[t]] = 1
            scheduled_tasks_start_times_vector[problem.index_task[t]] = dict_data[
                t
            ].start
            scheduled_tasks_end_times_vector[problem.index_task[t]] = dict_data[t].end
            for w in dict_data[t].resource_units_used:
                if w in problem.index_employee:
                    worker_used[problem.index_task[t], problem.index_employee[w]] = 1
    return (
        scheduled_task_indicator,
        scheduled_tasks_start_times_vector,
        scheduled_tasks_end_times_vector,
        worker_used,
    )


def build_partial_vectors_preemptive(
    problem: MS_RCPSPModel,
    completed_tasks: Dict[Hashable, TaskDetailsPreemptive],
    scheduled_tasks_start_times: Dict[Hashable, TaskDetailsPreemptive],
):
    scheduled_task_indicator = np.zeros(problem.n_jobs)
    scheduled_tasks_start_times_array = np.zeros((problem.n_jobs, 10), dtype=np.int32)
    scheduled_tasks_end_times_array = np.zeros((problem.n_jobs, 10), dtype=np.int32)
    nb_subparts = np.zeros(problem.n_jobs, dtype=int)
    worker_used = np.zeros((problem.n_jobs, 10, problem.nb_employees), dtype=int)
    for dict_data in [completed_tasks, scheduled_tasks_start_times]:
        for t in dict_data:
            scheduled_task_indicator[problem.index_task[t]] = 1
            nb_subparts[problem.index_task[t]] = len(dict_data[t].starts)
            scheduled_tasks_start_times_array[
                problem.index_task[t], : nb_subparts[problem.index_task[t]]
            ] = np.array(dict_data[t].starts)
            scheduled_tasks_end_times_array[
                problem.index_task[t], : nb_subparts[problem.index_task[t]]
            ] = np.array(dict_data[t].ends)
            for j in range(len(dict_data[t].resource_units_used)):
                for w in dict_data[t].resource_units_used[j]:
                    worker_used[problem.index_task[t], j, problem.index_employee[w]] = 1
    return (
        scheduled_task_indicator,
        scheduled_tasks_start_times_array,
        scheduled_tasks_end_times_array,
        nb_subparts,
        worker_used,
    )


def create_fake_tasks_multiskills(
    rcpsp_problem: Union[MS_RCPSPModel, MS_RCPSPSolution_Variant]
):
    ressources_arrays = {
        r: np.array(rcpsp_problem.resources_availability[r])
        for r in rcpsp_problem.resources_list
    }
    max_capacity = {r: np.max(ressources_arrays[r]) for r in ressources_arrays}
    fake_tasks = []
    for r in ressources_arrays:
        delta = ressources_arrays[r][:-1] - ressources_arrays[r][1:]
        index_non_zero = np.nonzero(delta)[0]
        if ressources_arrays[r][0] < max_capacity[r]:
            consume = {
                r: int(max_capacity[r] - ressources_arrays[r][0]),
                "duration": int(index_non_zero[0] + 1),
                "start": 0,
            }
            fake_tasks += [consume]
        for j in range(len(index_non_zero) - 1):
            ind = index_non_zero[j]
            value = ressources_arrays[r][ind + 1]
            if value != max_capacity[r]:
                consume = {
                    r: int(max_capacity[r] - value),
                    "duration": int(index_non_zero[j + 1] - ind),
                    "start": int(ind + 1),
                }
                fake_tasks += [consume]
    unit_arrays = {
        j: np.array(rcpsp_problem.employees[j].calendar_employee, dtype=np.int32)
        for j in rcpsp_problem.employees_list
    }
    max_capacity = {r: np.max(unit_arrays[r]) for r in unit_arrays}
    fake_tasks_unit = []
    for r in unit_arrays:
        delta = unit_arrays[r][:-1] - unit_arrays[r][1:]
        index_non_zero = np.nonzero(delta)[0]
        if unit_arrays[r][0] < max_capacity[r]:
            consume = {
                r: int(max_capacity[r] - unit_arrays[r][0]),
                "duration": int(index_non_zero[0] + 1),
                "start": 0,
            }
            fake_tasks_unit += [consume]
        for j in range(len(index_non_zero) - 1):
            ind = index_non_zero[j]
            value = unit_arrays[r][ind + 1]
            if value != max_capacity[r]:
                consume = {
                    r: int(max_capacity[r] - value),
                    "duration": int(index_non_zero[j + 1] - ind),
                    "start": int(ind + 1),
                }
                fake_tasks_unit += [consume]
    return fake_tasks, fake_tasks_unit


def cluster_employees_to_resource_types(ms_rcpsp_problem: MS_RCPSPModel):
    skills_representation_str = dict()
    skills_dict = dict()
    for employee in ms_rcpsp_problem.employees:
        skills = [
            s
            for s in sorted(ms_rcpsp_problem.employees[employee].dict_skill.keys())
            if ms_rcpsp_problem.employees[employee].dict_skill[s].skill_value > 0
        ]
        str_representation = "-".join(
            [
                str(s) + "-" + str(ms_rcpsp_problem.employees[employee].dict_skill[s])
                for s in skills
            ]
        )
        if str_representation not in skills_representation_str:
            skills_representation_str[str_representation] = set()
            skills_dict[str_representation] = {
                s: ms_rcpsp_problem.employees[employee].dict_skill[s] for s in skills
            }
        skills_representation_str[str_representation].add(employee)
    return skills_representation_str, skills_dict


def create_np_data_and_jit_functions(
    rcpsp_problem: Union[MS_RCPSPModel, MS_RCPSPModel_Variant]
):
    consumption_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int32,
    )
    is_releasable_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int32,
    )
    skills_need = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.skills_list),
        ),
        dtype=np.int32,
    )
    duration_array = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.max_number_of_mode), dtype=np.int32
    )

    predecessors = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int32
    )
    successors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int32)
    horizon = rcpsp_problem.horizon
    ressource_available = np.zeros(
        (len(rcpsp_problem.resources_list), horizon), dtype=np.int32
    )
    worker_available = np.zeros(
        (len(rcpsp_problem.employees_list), horizon), dtype=np.int32
    )
    ressource_renewable = np.ones((len(rcpsp_problem.resources_list)), dtype=bool)
    worker_skills = np.zeros(
        (len(rcpsp_problem.employees_list), len(rcpsp_problem.skills_list)),
        dtype=np.int32,
    )
    minimum_starting_time_array = np.zeros(rcpsp_problem.n_jobs, dtype=int)
    consider_partial_preemptive = False
    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        task_mode_details = rcpsp_problem.mode_details[task]
        index_mode = 0

        for mode in sorted(task_mode_details):
            for k in range(len(rcpsp_problem.resources_list)):
                resource = rcpsp_problem.resources_list[k]
                consumption_array[i, index_mode, k] = task_mode_details[mode].get(
                    resource, 0
                )
                if rcpsp_problem.partial_preemption_data[task][mode].get(
                    resource, True
                ):
                    is_releasable_array[i, index_mode, k] = 1
                else:
                    is_releasable_array[i, index_mode, k] = 0
                if not is_releasable_array[i, index_mode, k]:
                    consider_partial_preemptive = True
            for s in range(len(rcpsp_problem.skills_list)):
                skills_need[i, index_mode, s] = rcpsp_problem.mode_details[task][
                    mode
                ].get(rcpsp_problem.skills_list[s], 0)
            duration_array[i, index_mode] = rcpsp_problem.mode_details[task][mode][
                "duration"
            ]
            index_mode += 1
    if rcpsp_problem.includes_special_constraint():
        for t in rcpsp_problem.special_constraints.start_times_window:
            if rcpsp_problem.special_constraints.start_times_window[t][0] is not None:
                minimum_starting_time_array[
                    rcpsp_problem.index_task[t]
                ] = rcpsp_problem.special_constraints.start_times_window[t][0]
    task_index = {rcpsp_problem.tasks_list[i]: i for i in range(rcpsp_problem.n_jobs)}
    for k in range(len(rcpsp_problem.resources_list)):
        ressource_available[k, :] = rcpsp_problem.resources_availability[
            rcpsp_problem.resources_list[k]
        ][:horizon]
        if rcpsp_problem.resources_list[k] in rcpsp_problem.non_renewable_resources:
            ressource_renewable[k] = False
    for emp in range(len(rcpsp_problem.employees_list)):
        worker_available[emp, :] = np.array(
            rcpsp_problem.employees[
                rcpsp_problem.employees_list[emp]
            ].calendar_employee,
            dtype=np.int32,
        )[:horizon]
        for s in range(len(rcpsp_problem.skills_list)):
            worker_skills[emp, s] = (
                rcpsp_problem.employees[rcpsp_problem.employees_list[emp]]
                .dict_skill.get(
                    rcpsp_problem.skills_list[s],
                    SkillDetail(skill_value=0, efficiency_ratio=0, experience=0),
                )
                .skill_value
            )
    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        for s in rcpsp_problem.successors[task]:
            index_s = task_index[s]
            predecessors[index_s, i] = 1
            successors[i, index_s] = 1

    if rcpsp_problem.includes_special_constraint():
        start_at_end_plus_offset = np.zeros(
            (len(rcpsp_problem.special_constraints.start_at_end_plus_offset), 3),
            dtype=int,
        )
        start_after_nunit = np.zeros(
            (len(rcpsp_problem.special_constraints.start_after_nunit), 3), dtype=int
        )
        j = 0
        for t1, t2, off in rcpsp_problem.special_constraints.start_at_end_plus_offset:
            start_at_end_plus_offset[j, 0] = rcpsp_problem.index_task[t1]
            start_at_end_plus_offset[j, 1] = rcpsp_problem.index_task[t2]
            start_at_end_plus_offset[j, 2] = off
            j += 1
        j = 0
        for t1, t2, off in rcpsp_problem.special_constraints.start_after_nunit:
            start_after_nunit[j, 0] = rcpsp_problem.index_task[t1]
            start_after_nunit[j, 1] = rcpsp_problem.index_task[t2]
            start_after_nunit[j, 2] = off
            j += 1
    if rcpsp_problem.preemptive:
        preemptive_tag = np.ones(rcpsp_problem.n_jobs, dtype=np.int32)
        for t in rcpsp_problem.preemptive_indicator:
            preemptive_tag[rcpsp_problem.index_task[t]] = (
                1 if rcpsp_problem.preemptive_indicator[t] else 0
            )

    # modes_array,          # modes=array(task)->0, 1...
    # consumption_array,    # consumption_array=array3D(task, mode, res),
    # skills_needs,         # array(task, mode, skill)
    # duration_array,       # array(task, mode) -> d
    # predecessors,         # array(task, task) -> bool
    # successors,           # array(task, task)->bool
    # horizon,              # int
    # ressource_available,  # array(res, times)->int
    # ressource_renewable,  # array(res)->bool
    # worker_available,     # array(workers, int)->bool
    # worker_skills         # array(workers, skills)->int
    if rcpsp_problem.preemptive:
        if rcpsp_problem.includes_special_constraint() and (
            start_after_nunit.shape[0] > 0 or start_at_end_plus_offset.shape[0] > 0
        ):
            func_sgs = partial(
                sgs_fast_ms_preemptive_some_special_constraints,
                consumption_array=consumption_array,
                skills_needs=skills_need,
                worker_skills=worker_skills,
                worker_available=worker_available,
                duration_array=duration_array,
                minimum_starting_time_array=minimum_starting_time_array,
                start_at_end_plus_offset=start_at_end_plus_offset,
                start_after_nunit=start_after_nunit,
                predecessors=predecessors,
                preemptive_tag=preemptive_tag,
                successors=successors,
                horizon=horizon,
                ressource_available=ressource_available,
                ressource_renewable=ressource_renewable,
                one_unit_per_task=rcpsp_problem.one_unit_per_task_max,
                is_releasable=is_releasable_array,
                consider_partial_preemptive=consider_partial_preemptive,
                strictly_disjunctive_subtasks=rcpsp_problem.strictly_disjunctive_subtasks,
            )
        else:
            func_sgs = partial(
                sgs_fast_ms_preemptive,
                consumption_array=consumption_array,
                skills_needs=skills_need,
                worker_skills=worker_skills,
                worker_available=worker_available,
                duration_array=duration_array,
                minimum_starting_time_array=minimum_starting_time_array,
                predecessors=predecessors,
                preemptive_tag=preemptive_tag,
                successors=successors,
                horizon=horizon,
                ressource_available=ressource_available,
                ressource_renewable=ressource_renewable,
                one_unit_per_task=rcpsp_problem.one_unit_per_task_max,
                is_releasable=is_releasable_array,
                consider_partial_preemptive=consider_partial_preemptive,
                strictly_disjunctive_subtasks=rcpsp_problem.strictly_disjunctive_subtasks,
            )
        func_sgs_partial = partial(
            sgs_fast_ms_preemptive_partial_schedule,
            consumption_array=consumption_array,
            skills_needs=skills_need,
            worker_skills=worker_skills,
            worker_available=worker_available,
            duration_array=duration_array,
            minimum_starting_time_array=minimum_starting_time_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            preemptive_tag=preemptive_tag,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            one_unit_per_task=rcpsp_problem.one_unit_per_task_max,
            is_releasable=is_releasable_array,
            consider_partial_preemptive=consider_partial_preemptive,
        )
    else:
        func_sgs = partial(
            sgs_fast_ms,
            consumption_array=consumption_array,
            skills_needs=skills_need,
            worker_skills=worker_skills,
            worker_available=worker_available,
            duration_array=duration_array,
            minimum_starting_time_array=minimum_starting_time_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            one_unit_per_task=rcpsp_problem.one_unit_per_task_max,
        )
        func_sgs_partial = partial(
            sgs_fast_ms_partial_schedule,
            consumption_array=consumption_array,
            skills_needs=skills_need,
            worker_skills=worker_skills,
            worker_available=worker_available,
            duration_array=duration_array,
            minimum_starting_time_array=minimum_starting_time_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            one_unit_per_task=rcpsp_problem.one_unit_per_task_max,
        )
    return func_sgs, func_sgs_partial


def employee_usage(
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
    problem: MS_RCPSPModel,
):
    makespan = solution.get_max_end_time()
    employee_usage_matrix = np.zeros((problem.nb_employees, makespan + 1))
    employees_usage_dict = {emp: set() for emp in problem.employees_list}
    for task in problem.tasks_list:
        employees = solution.employee_used(task=task)
        starts = solution.get_start_times_list(task=task)
        ends = solution.get_end_times_list(task=task)
        for i in range(len(employees)):
            for e in employees[i]:
                employees_usage_dict[e].add((task, i))
                employee_usage_matrix[
                    problem.index_employee[e], starts[i] : ends[i]
                ] = 1
    sum_usage = np.sum(employee_usage_matrix, axis=1)
    return employee_usage_matrix, sum_usage, employees_usage_dict


def evaluate_constraints(
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
    constraints: SpecialConstraintsDescription,
):
    list_constraints_not_respected = compute_constraints_details(solution, constraints)
    return sum([x[-1] for x in list_constraints_not_respected])


def compute_constraints_details(
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
    constraints: SpecialConstraintsDescription,
):
    start_together = constraints.start_together
    start_at_end = constraints.start_at_end
    start_at_end_plus_offset = constraints.start_at_end_plus_offset
    start_after_nunit = constraints.start_after_nunit
    disjunctive = constraints.disjunctive_tasks
    list_constraints_not_respected = []
    for (t1, t2) in start_together:
        time1 = solution.get_start_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            list_constraints_not_respected += [
                ("start_together", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2) in start_at_end:
        time1 = solution.get_end_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            list_constraints_not_respected += [
                ("start_at_end", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2, off) in start_at_end_plus_offset:
        time1 = solution.get_end_time(t1) + off
        time2 = solution.get_start_time(t2)
        b = time2 >= time1
        if not b:
            list_constraints_not_respected += [
                ("start_at_end_plus_offset", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for (t1, t2, off) in start_after_nunit:
        time1 = solution.get_start_time(t1) + off
        time2 = solution.get_start_time(t2)
        b = time2 >= time1
        if not b:
            list_constraints_not_respected += [
                ("start_after_nunit", t1, t2, time1, time2, abs(time2 - time1))
            ]
    for t1, t2 in disjunctive:
        b = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if b is not None:
            list_constraints_not_respected += [
                ("disjunctive", t1, t2, None, None, b[1] - b[0])
            ]
    for t in constraints.start_times_window:
        if constraints.start_times_window[t][0] is not None:
            if solution.get_start_time(t) < constraints.start_times_window[t][0]:
                list_constraints_not_respected += [
                    (
                        "start_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.start_times_window[t][0]
                        - solution.get_start_time(t),
                    )
                ]

        if constraints.start_times_window[t][1] is not None:
            if solution.get_start_time(t) > constraints.start_times_window[t][1]:
                list_constraints_not_respected += [
                    (
                        "start_window_1",
                        t,
                        t,
                        None,
                        None,
                        -constraints.start_times_window[t][1]
                        + solution.get_start_time(t),
                    )
                ]

    for t in constraints.end_times_window:
        if constraints.end_times_window[t][0] is not None:
            if solution.get_end_time(t) < constraints.end_times_window[t][0]:
                list_constraints_not_respected += [
                    (
                        "end_window_0",
                        t,
                        t,
                        None,
                        None,
                        constraints.end_times_window[t][0] - solution.get_end_time(t),
                    )
                ]

        if constraints.end_times_window[t][1] is not None:
            if solution.get_end_time(t) > constraints.end_times_window[t][1]:
                list_constraints_not_respected += [
                    (
                        "end_window_1",
                        t,
                        t,
                        None,
                        None,
                        -constraints.end_times_window[t][1] + solution.get_end_time(t),
                    )
                ]

    return list_constraints_not_respected


def start_together_problem_description(
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
    constraints: SpecialConstraintsDescription,
):
    start_together = constraints.start_together
    list_constraints_not_respected = []
    for (t1, t2) in start_together:
        time1 = solution.get_start_time(t1)
        time2 = solution.get_start_time(t2)
        b = time1 == time2
        if not b:
            employees_used_t1_first_part = set(solution.employee_used(t1)[0])
            employees_used_t2_first_part = set(solution.employee_used(t2)[0])
            intersection_employees = employees_used_t1_first_part.intersection(
                employees_used_t2_first_part
            )
            if len(intersection_employees) > 0:
                logger.debug(f"starting together constraint between task {t1} and {t2}")
                logger.debug(f"{intersection_employees} employees working on both.. !")
            list_constraints_not_respected += [
                (
                    "start_together",
                    t1,
                    t2,
                    time1,
                    time2,
                    abs(time2 - time1),
                    intersection_employees,
                )
            ]
    return list_constraints_not_respected


def check_solution(
    problem: Union[MS_RCPSPModel],
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
    relax_the_start_at_end: bool = True,
):
    start_together = problem.special_constraints.start_together
    start_at_end = problem.special_constraints.start_at_end
    start_at_end_plus_offset = problem.special_constraints.start_at_end_plus_offset
    start_after_nunit = problem.special_constraints.start_after_nunit
    disjunctive = problem.special_constraints.disjunctive_tasks
    for (t1, t2) in start_together:
        if not relax_the_start_at_end:
            b = solution.get_start_time(t1) == solution.get_start_time(t2)
            if not b:
                return False
    for (t1, t2) in start_at_end:
        if relax_the_start_at_end:
            b = solution.get_start_time(t2) >= solution.get_end_time(t1)
        else:
            b = solution.get_start_time(t2) == solution.get_end_time(t1)
        if not b:
            return False
    for (t1, t2, off) in start_at_end_plus_offset:
        b = solution.get_start_time(t2) >= solution.get_end_time(t1) + off
        if not b:
            return False
    for (t1, t2, off) in start_after_nunit:
        b = solution.get_start_time(t2) >= solution.get_start_time(t1) + off
        if not b:
            return False
    for t1, t2 in disjunctive:
        b = intersect(
            [solution.get_start_time(t1), solution.get_end_time(t1)],
            [solution.get_start_time(t2), solution.get_end_time(t2)],
        )
        if b is not None:
            return False
    for t in problem.special_constraints.start_times_window:
        if problem.special_constraints.start_times_window[t][0] is not None:
            if (
                solution.get_start_time(t)
                < problem.special_constraints.start_times_window[t][0]
            ):
                return False
        if problem.special_constraints.start_times_window[t][1] is not None:
            if (
                solution.get_start_time(t)
                > problem.special_constraints.start_times_window[t][1]
            ):
                return False
    for t in problem.special_constraints.end_times_window:
        if problem.special_constraints.end_times_window[t][0] is not None:
            if (
                solution.get_end_time(t)
                < problem.special_constraints.end_times_window[t][0]
            ):
                return False
        if problem.special_constraints.end_times_window[t][1] is not None:
            if (
                solution.get_end_time(t)
                > problem.special_constraints.end_times_window[t][1]
            ):
                return False
    return True


def compute_skills_missing_problem(
    problem: MS_RCPSPModel,
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
):
    problems = []
    for task in problem.tasks:
        mode = solution.modes[task]
        required_skills = {
            s: problem.mode_details[task][mode][s]
            for s in problem.mode_details[task][mode]
            if s in problem.skills_set and problem.mode_details[task][mode][s] > 0
        }
        # Skills for the given task are used
        if len(required_skills) > 0:
            for skill in required_skills:
                employees_used = [
                    problem.employees[emp].dict_skill[skill].skill_value
                    for emp in solution.employee_usage[task]
                    if skill in solution.employee_usage[task][emp]
                ]

                if sum(employees_used) < required_skills[skill]:
                    problems += [
                        (
                            "task",
                            task,
                            skill,
                            sum(employees_used),
                            required_skills[skill],
                        )
                    ]
            if task in solution.employee_usage:
                for emp in solution.employee_usage[task]:
                    set_sk = set(
                        [
                            s
                            for s in problem.employees[emp].dict_skill
                            if problem.employees[emp].dict_skill[s].skill_value > 0
                        ]
                    )
                    if len(set_sk.intersection(set(required_skills))) == 0:
                        problems += [
                            ("employee", emp, task, set(required_skills), set_sk)
                        ]
    return problems


def compute_ressource_array_preemptive(
    problem: MS_RCPSPModel,
    solution: Union[MS_RCPSPSolution, MS_RCPSPSolution_Preemptive],
):
    resource_avail_in_time = {}
    makespan = solution.get_end_time(problem.sink_task)
    for res in problem.resources_list:
        resource_avail_in_time[res] = np.copy(
            problem.get_resource_availability_array(res)
        )
    modes = solution.modes
    for act_id in problem.tasks_list:
        starts = solution.get_start_times_list(act_id)
        ends = solution.get_end_times_list(act_id)
        mode = modes[act_id]
        for res in resource_avail_in_time:
            need = problem.mode_details[act_id][mode].get(res, 0)
            if need > 0:
                if problem.partial_preemption_data[act_id][mode][res]:
                    for i in range(len(starts)):
                        resource_avail_in_time[res][starts[i] : ends[i]] -= need
                else:
                    resource_avail_in_time[res][starts[0] : ends[-1]] -= need
    return resource_avail_in_time


def compute_overskill(problem: MS_RCPSPModel, solution: MS_RCPSPSolution_Preemptive):
    overskill = {}
    for task in problem.tasks_list:
        overskill[task] = {}
        mode = solution.modes[task]
        required_skills = {
            s: problem.mode_details[task][mode][s]
            for s in problem.mode_details[task][mode]
            if s in problem.skills_set and problem.mode_details[task][mode][s] > 0
        }
        # Skills for the given task are used
        if (
            len(required_skills) > 0
            and problem.mode_details[task][mode]["duration"] > 0
        ):
            for skill in required_skills:
                for i in range(solution.get_number_of_part(task)):
                    skills_provided = [
                        problem.employees[emp].dict_skill[skill].skill_value
                        for emp in solution.employee_usage[task][i]
                        if skill in solution.employee_usage[task][i][emp]
                    ]
                    if sum(skills_provided) < required_skills[skill]:
                        logger.debug(f"Not enough skills to do task : {task}")
                        return False
                    if sum(skills_provided) > required_skills[skill]:
                        if skill not in overskill[task]:
                            overskill[task][skill] = {}
                        overskill[task][skill][i] = (
                            sum(skills_provided),
                            required_skills[skill],
                        )
    return overskill
