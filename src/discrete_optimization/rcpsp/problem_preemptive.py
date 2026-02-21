#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_rcpsp_tools.attribute_type import (
    ListIntegerRcpsp,
    PermutationRcpsp,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TupleFitness,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import EncodingRegister
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.rcpsp.fast_function import (
    compute_mean_ressource,
    sgs_fast_partial_schedule_preemptive,
    sgs_fast_partial_schedule_preemptive_minduration,
    sgs_fast_preemptive,
    sgs_fast_preemptive_minduration,
)

logger = logging.getLogger(__name__)


def tree():
    return defaultdict(tree)


class ScheduleGenerationScheme(Enum):
    SERIAL_SGS = 0
    PARALLEL_SGS = 1


class PreemptiveRcpspSolution(Solution):
    rcpsp_permutation: Union[list[int], np.ndarray]
    rcpsp_schedule: dict[Hashable, dict[str, list[int]]]
    #  {task_id: {'starts': [start_times], 'ends':  [end_times], 'resources': list_of_resource_ids}
    #   for task_id in self.problem.tasks_list}
    rcpsp_modes: list[int]  # [mode_id[i] for i in range(self.problem.n_jobs_non_dummy)]
    standardised_permutation: Union[list[int], np.ndarray]
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        rcpsp_permutation: list[int] | np.ndarray | None = None,
        rcpsp_schedule: dict[Hashable, dict[str, list[int]]] | None = None,
        rcpsp_modes: list[int] | None = None,
        rcpsp_schedule_feasible: bool | None = None,
        standardised_permutation: list[int] | np.ndarray | None = None,
    ):
        super().__init__(problem=problem)
        self.rcpsp_permutation = rcpsp_permutation
        self.rcpsp_schedule = rcpsp_schedule
        self._schedule_to_recompute = rcpsp_schedule is None
        self.rcpsp_modes = rcpsp_modes
        self.rcpsp_schedule_feasible = rcpsp_schedule_feasible
        self.standardised_permutation = standardised_permutation

        if self.rcpsp_modes is None:
            self.rcpsp_modes = [1 for i in range(self.problem.n_jobs_non_dummy)]
        if self.rcpsp_permutation is None:
            if self.rcpsp_schedule is not None:
                self.standardised_permutation = (
                    self.generate_permutation_from_schedule()
                )
                self.rcpsp_permutation = deepcopy(self.standardised_permutation)
                self._schedule_to_recompute = False

        if rcpsp_schedule is None:
            self.generate_schedule_from_permutation_serial_sgs()

        if self.standardised_permutation is None:
            self.standardised_permutation = self.generate_permutation_from_schedule()

    def get_nb_task_preemption(self):
        return len(
            [
                t
                for t in self.rcpsp_schedule
                if len(self.rcpsp_schedule[t]["starts"]) > 1
            ]
        )

    def total_number_of_cut(self):
        return sum([self.get_number_of_part(task) - 1 for task in self.rcpsp_schedule])

    def get_min_duration_subtask(self):
        return min(
            [
                e - s
                for t in self.rcpsp_schedule
                for e, s in zip(
                    self.rcpsp_schedule[t]["ends"], self.rcpsp_schedule[t]["starts"]
                )
                if len(self.rcpsp_schedule[t]["starts"]) > 1
            ],
            default=None,
        )

    def get_number_of_part(self, task):
        return len(self.rcpsp_schedule.get(task, {"starts": []})["starts"])

    def get_max_preempted(self):
        return max([len(self.rcpsp_schedule[t]["starts"]) for t in self.rcpsp_schedule])

    def get_task_preempted(self):
        return [
            t for t in self.rcpsp_schedule if len(self.rcpsp_schedule[t]["starts"]) > 1
        ]

    def get_start_time(self, task):
        return self.rcpsp_schedule.get(task, {"starts": [None]})["starts"][0]

    def get_start_times_list(self, task):
        return self.rcpsp_schedule.get(task, {"starts": [None]})["starts"]

    def get_end_time(self, task):
        return self.rcpsp_schedule.get(task, {"ends": [None]})["ends"][-1]

    def get_end_times_list(self, task):
        return self.rcpsp_schedule.get(task, {"ends": [None]})["ends"]

    def get_max_end_time(self):
        return max([self.get_end_time(x) for x in self.rcpsp_schedule])

    def get_active_time(self, task):
        l = []
        for s, e in zip(
            self.rcpsp_schedule[task]["starts"], self.rcpsp_schedule[task]["ends"]
        ):
            l += list(range(s, e))
        return l

    def change_problem(self, new_problem: Problem):
        self.__init__(
            problem=new_problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
        )

    def __setattr__(self, key, value):
        super.__setattr__(self, key, value)
        if key == "rcpsp_permutation":
            self._schedule_to_recompute = True

    def copy(self):
        return PreemptiveRcpspSolution(
            problem=self.problem,
            rcpsp_permutation=deepcopy(self.rcpsp_permutation),
            rcpsp_modes=deepcopy(self.rcpsp_modes),
            rcpsp_schedule=deepcopy(self.rcpsp_schedule),
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
        )

    def lazy_copy(self):
        return PreemptiveRcpspSolution(
            problem=self.problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
            rcpsp_schedule=self.rcpsp_schedule,
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
        )

    def __str__(self):
        if self.rcpsp_schedule is None:
            sched_str = "None"
        else:
            sched_str = str(self.rcpsp_schedule)
        val = "RCPSP solution (rcpsp_schedule): " + sched_str
        return val

    def generate_permutation_from_schedule(self):
        sorted_task = [
            self.problem.index_task[i] - 1
            for i in sorted(
                self.rcpsp_schedule, key=lambda x: self.rcpsp_schedule[x]["starts"][0]
            )
        ]
        sorted_task.remove(-1)
        sorted_task.remove(max(sorted_task))
        return sorted_task

    def compute_mean_resource_reserve(self, fast=True):
        if not fast:
            return compute_mean_resource_reserve(
                solution=self, rcpsp_problem=self.problem
            )
        else:
            if not self.rcpsp_schedule_feasible:
                return 0.0
            last_activity = self.problem.sink_task
            makespan = self.rcpsp_schedule[last_activity]["ends"]
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                return 0.0
            else:
                return self.problem.compute_mean_resource(
                    horizon=makespan,
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,
                    start_array=np.array(
                        [
                            self.rcpsp_schedule[t]["starts"][0]
                            for t in self.problem.tasks_list
                        ]
                    ),
                    end_array=np.array(
                        [
                            self.rcpsp_schedule[t]["ends"][0]
                            for t in self.problem.tasks_list
                        ]
                    ),
                )

    def generate_schedule_from_permutation_serial_sgs(self, do_fast=True):
        if do_fast:
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                starts_dict, ends_dict, unfeasible = {}, {}, True
            else:
                starts_dict, ends_dict, unfeasible = self.problem.func_sgs(
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        self.problem, self.rcpsp_permutation
                    ),
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,
                )
                self.rcpsp_schedule_feasible = not unfeasible
                self.rcpsp_schedule = {
                    self.problem.sink_task: {
                        "starts": [99999999],
                        "ends": [99999999],
                    }
                }
            for k in starts_dict:
                self.rcpsp_schedule[self.problem.tasks_list[k]] = {
                    "starts": list(starts_dict[k]),
                    "ends": list(ends_dict[k]),
                }
            self._schedule_to_recompute = False
        else:
            schedule, feasible = generate_schedule_from_permutation_serial_sgs(
                solution=self, rcpsp_problem=self.problem
            )
            self.rcpsp_schedule = schedule
            self.rcpsp_schedule_feasible = feasible
            self._schedule_to_recompute = False

    def generate_schedule_from_permutation_serial_sgs_2(
        self,
        current_t=0,
        completed_tasks: Optional[set[Hashable]] = None,
        partial_schedule: Optional[dict[Hashable, dict[str, list[int]]]] = None,
        do_fast=True,
    ):
        if completed_tasks is None:
            completed_tasks = set()
        if partial_schedule is None:
            partial_schedule = {}
        if do_fast:
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                starts_dict, ends_dict, unfeasible = {}, {}, True
            else:
                starts_dict, ends_dict, unfeasible = self.problem.func_sgs_2(
                    current_time=current_t,
                    completed_task_indicator=np.array(
                        [
                            1 if self.problem.tasks_list[i] in completed_tasks else 0
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    partial_schedule_starts=np.array(
                        [
                            [
                                partial_schedule.get(
                                    self.problem.tasks_list[i], {}
                                ).get("starts", [])[k]
                                if k
                                < len(
                                    partial_schedule.get(
                                        self.problem.tasks_list[i], {}
                                    ).get("starts", [])
                                )
                                else -1
                                for k in range(10)
                            ]
                            for i in range(self.problem.n_jobs)
                        ],
                        np.int_,
                    ),
                    partial_schedule_ends=np.array(
                        [
                            [
                                partial_schedule.get(
                                    self.problem.tasks_list[i], {}
                                ).get("ends", [])[k]
                                if k
                                < len(
                                    partial_schedule.get(
                                        self.problem.tasks_list[i], {}
                                    ).get("ends", [])
                                )
                                else -1
                                for k in range(10)
                            ]
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    permutation_task=permutation_do_to_permutation_sgs_fast(
                        self.problem, self.rcpsp_permutation
                    ),
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,
                )
            self.rcpsp_schedule_feasible = not unfeasible
            self.rcpsp_schedule = {}
            for k in starts_dict:
                self.rcpsp_schedule[self.problem.tasks_list[k]] = {
                    "starts": list(starts_dict[k]),
                    "ends": list(ends_dict[k]),
                }
            self._schedule_to_recompute = False
        else:
            (
                schedule,
                feasible,
            ) = generate_schedule_from_permutation_serial_sgs_partial_schedule(
                solution=self,
                current_t=current_t,
                completed_tasks=completed_tasks,
                partial_schedule=partial_schedule,
                rcpsp_problem=self.problem,
            )
            self.rcpsp_schedule = schedule
            self.rcpsp_schedule_feasible = feasible
            self._schedule_to_recompute = False

    def __hash__(self):
        return hash((tuple(self.rcpsp_permutation), tuple(self.rcpsp_modes)))

    def __eq__(self, other):
        return (
            self.rcpsp_permutation == other.rcpsp_permutation
            and self.rcpsp_modes == other.rcpsp_modes
        )


class PartialPreemptiveRcpspSolution:
    def __init__(
        self,
        task_mode: dict[int, int] = None,
        start_times: dict[int, int] = None,
        end_times: dict[int, int] = None,
        partial_permutation: list[int] = None,
        list_partial_order: list[list[int]] = None,
        start_together: list[tuple[int, int]] = None,
        start_at_end: list[tuple[int, int]] = None,
        start_at_end_plus_offset: list[tuple[int, int, int]] = None,
        start_after_nunit: list[tuple[int, int, int]] = None,
        disjunctive_tasks: list[tuple[int, int]] = None,
        start_times_window: dict[Hashable, tuple[int, int]] = None,
        end_times_window: dict[Hashable, tuple[int, int]] = None,
    ):
        self.task_mode = task_mode
        self.start_times = start_times
        self.end_times = end_times
        self.partial_permutation = partial_permutation
        self.list_partial_order = list_partial_order
        self.start_together = start_together
        self.start_at_end = start_at_end
        self.start_after_nunit = start_after_nunit
        self.start_at_end_plus_offset = start_at_end_plus_offset
        self.disjunctive_tasks = disjunctive_tasks
        self.start_times_window = start_times_window
        self.end_times_window = end_times_window
        # one element in self.list_partial_order is a list [l1, l2, l3]
        # indicating that l1 should be started before l1, and  l2 before l3 for example


class PreemptiveRcpspProblem(Problem):
    sgs: ScheduleGenerationScheme
    resources: Union[
        dict[str, int], dict[str, list[int]]
    ]  # {resource_name: number_of_resource}
    non_renewable_resources: list[str]  # e.g. [resource_name3, resource_name4]
    n_jobs: int  # excluding dummy activities Start (0) and End (n)
    mode_details: dict[Hashable, dict[int, dict[str, int]]]
    # e.g. {job_id: {mode_id: {resource_name1: number_of_resources_needed, resource_name2: ...}}
    # one key being "duration"
    successors: dict[int, list[int]]  # {task_id: list of successor task ids}

    def __init__(
        self,
        resources: Union[dict[str, int], dict[str, list[int]]],
        non_renewable_resources: list[str],
        mode_details: dict[Hashable, dict[Union[str, int], dict[str, int]]],
        successors: dict[Union[int, str], list[Union[str, int]]],
        horizon,
        horizon_multiplier=1,
        tasks_list: list[Union[int, str]] = None,
        source_task=None,
        sink_task=None,
        preemptive_indicator: dict[Hashable, bool] = None,
        duration_subtask: dict[Hashable, tuple[bool, int]] = None,
        name_task: dict[int, str] = None,
    ):
        self.resources = resources
        self.resources_list = list(self.resources.keys())
        self.non_renewable_resources = non_renewable_resources
        self.mode_details = mode_details
        self.successors = successors
        self.horizon = horizon
        self.horizon_multiplier = horizon_multiplier
        self.name_task = name_task
        if name_task is None:
            self.name_task = {x: str(x) for x in self.mode_details}
        self.tasks_list = tasks_list
        if tasks_list is None:
            self.tasks_list = list(self.mode_details.keys())
        self.preemptive_indicator = preemptive_indicator
        if preemptive_indicator is None:
            self.preemptive_indicator = {t: True for t in self.tasks_list}
        self.duration_subtask = duration_subtask
        if self.duration_subtask is None:
            self.duration_subtask = {t: (False, 0) for t in self.tasks_list}
        self.any_duration_subtask_limited = any(
            self.duration_subtask[x][0] for x in self.duration_subtask
        )
        self.n_jobs = len(self.tasks_list)
        self.n_jobs_non_dummy = self.n_jobs - 2
        self.index_task = {self.tasks_list[i]: i for i in range(self.n_jobs)}
        self.source_task = source_task
        if source_task is None:
            self.source_task = min(self.tasks_list)
        self.sink_task = sink_task
        if sink_task is None:
            self.sink_task = max(self.tasks_list)
        self.tasks_list_non_dummy = [
            t for t in self.tasks_list if t not in {self.source_task, self.sink_task}
        ]
        self.index_task_non_dummy = {
            self.tasks_list_non_dummy[i]: i for i in range(self.n_jobs_non_dummy)
        }
        self.max_number_of_mode = max(
            [len(self.mode_details[key1].keys()) for key1 in self.mode_details.keys()]
        )
        self.is_multimode = self.max_number_of_mode > 1
        self.is_calendar = False
        if any(isinstance(self.resources[res], Iterable) for res in self.resources):
            self.is_calendar = (
                max(
                    [
                        len(
                            set(self.resources[res])
                            if isinstance(self.resources[res], Iterable)
                            else {self.resources[res]}
                        )
                        for res in self.resources
                    ]
                )
                > 1
            )
            if not self.is_calendar:
                self.resources = {r: int(self.resources[r][0]) for r in self.resources}
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(self)

    def get_resource_names(self):
        return self.resources_list

    def get_resource_availability_array(self, res):
        if self.is_varying_resource():
            return self.resources[res]
        else:
            return np.full(self.horizon, self.resources[res])

    def get_max_resource_capacity(self, res):
        if self.is_calendar:
            return max(self.resources.get(res, [0]))
        return self.resources.get(res, 0)

    def get_modes_dict(self, rcpsp_solution: PreemptiveRcpspSolution):
        return self.build_mode_dict(rcpsp_solution.rcpsp_modes)

    def update_function(self):
        (
            self.func_sgs,
            self.func_sgs_2,
            self.compute_mean_resource,
        ) = create_np_data_and_jit_functions(self)

    def is_rcpsp_multimode(self):
        return self.is_multimode

    def is_varying_resource(self):
        return self.is_calendar

    def is_duration_minimum_preemption(self):
        return self.any_duration_subtask_limited

    def is_preemptive(self):
        return True

    def is_multiskill(self):
        return False

    def can_be_preempted(self, task):
        return self.preemptive_indicator.get(task, False)

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

    def evaluate_function(self, rcpsp_sol: PreemptiveRcpspSolution):
        if rcpsp_sol._schedule_to_recompute:
            rcpsp_sol.generate_schedule_from_permutation_serial_sgs()
        makespan = rcpsp_sol.rcpsp_schedule[self.sink_task]["ends"][-1]
        return makespan, 0.0

    def evaluate(self, variable: PreemptiveRcpspSolution) -> dict[str, float]:
        obj_makespan, obj_mean_resource_reserve = self.evaluate_function(variable)
        return {
            "makespan": obj_makespan,
            "mean_resource_reserve": obj_mean_resource_reserve,
        }

    def evaluate_mobj(self, variable: PreemptiveRcpspSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(variable))

    def evaluate_mobj_from_dict(self, dict_values: dict[str, float]) -> TupleFitness:
        return TupleFitness(
            np.array([-dict_values["makespan"], dict_values["mean_resource_reserve"]]),
            2,
        )

    def build_mode_dict(self, rcpsp_modes_from_solution):
        modes_dict = {
            self.tasks_list_non_dummy[i]: rcpsp_modes_from_solution[i]
            for i in range(self.n_jobs_non_dummy)
        }
        modes_dict[self.source_task] = 1
        modes_dict[self.sink_task] = 1
        return modes_dict

    def build_mode_array(self, rcpsp_modes_from_solution):
        modes_dict = {
            self.tasks_list_non_dummy[i]: rcpsp_modes_from_solution[i]
            for i in range(self.n_jobs_non_dummy)
        }
        modes_dict[self.source_task] = 1
        modes_dict[self.sink_task] = 1
        return [modes_dict[t] for t in self.tasks_list]

    def return_index_task(self, task, offset=0):
        return self.index_task[task] + offset

    def satisfy(self, variable: PreemptiveRcpspSolution) -> bool:
        if variable.rcpsp_schedule_feasible is False:
            logger.debug("Schedule flagged as infeasible when generated")
            return False
        else:
            modes_dict = self.build_mode_dict(
                rcpsp_modes_from_solution=variable.rcpsp_modes
            )
            resource_avail_in_time = compute_resource(
                solution=variable, rcpsp_problem=self
            )
            for r in resource_avail_in_time:
                if np.any(resource_avail_in_time[r] < 0):
                    return False
            # Check for non-renewable resource violation
            for res in self.non_renewable_resources:
                usage = 0
                for act_id in variable.rcpsp_schedule:
                    mode = modes_dict[act_id]
                    usage += self.mode_details[act_id][mode][res]
                    if usage > self.get_max_resource_capacity(res):
                        logger.debug(
                            f"Non-renewable resource violation: act_id: {act_id}"
                            f"res {res}"
                            f"res_usage: {usage}"
                            f"res_avail: {self.resources[res]}"
                        )
                        return False
            # Check precedences / successors
            for act_id in self.successors:
                for succ_id in self.successors[act_id]:
                    start_succ = variable.rcpsp_schedule[succ_id]["starts"][0]
                    end_pred = variable.rcpsp_schedule[act_id]["ends"][-1]
                    if start_succ < end_pred:
                        logger.debug(
                            f"Precedence relationship broken: {act_id}"
                            f"end at {end_pred}"
                            f"while {succ_id} start at {start_succ}"
                        )
                        return False
            # Check sum of working time
            for t in self.tasks_list:
                dur = self.mode_details[t][modes_dict[t]]["duration"]
                sum_dur = sum(
                    e - s
                    for s, e in zip(
                        variable.get_start_times_list(t), variable.get_end_times_list(t)
                    )
                )
                if sum_dur < dur:
                    logger.info(
                        f"Task {t} is not executed long enough {sum_dur} vs {dur}"
                    )
                    return False
            return True

    def get_solution_type(self) -> type[Solution]:
        return PreemptiveRcpspSolution

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            {
                "rcpsp_permutation": PermutationRcpsp(
                    range=range(self.n_jobs_non_dummy)
                ),
                "rcpsp_modes": ListIntegerRcpsp(
                    length=self.n_jobs_non_dummy,
                    lows=1,
                    ups=[
                        len(self.mode_details[task])
                        for task in self.tasks_list_non_dummy
                    ],
                ),
            }
        )

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0),
            "mean_resource_reserve": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=1.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc=dict_objective,
        )

    def compute_resource_consumption(self, rcpsp_sol: PreemptiveRcpspSolution):
        modes_dict = self.build_mode_dict(rcpsp_sol.rcpsp_modes)
        last_activity = max(rcpsp_sol.rcpsp_schedule)
        makespan = rcpsp_sol.rcpsp_schedule[last_activity]["end_time"]
        consumptions = np.zeros((len(self.resources), makespan + 1))
        for act_id in rcpsp_sol.rcpsp_schedule:
            for ir in range(len(self.resources)):
                consumptions[
                    ir,
                    rcpsp_sol.rcpsp_schedule[act_id][
                        "start_time"
                    ] : rcpsp_sol.rcpsp_schedule[act_id]["end_time"] + 1,
                ] += self.mode_details[act_id][modes_dict[act_id]][
                    self.resources_list[ir]
                ]
        return consumptions

    def plot_ressource_view(self, rcpsp_sol: PreemptiveRcpspSolution):
        consumption = self.compute_resource_consumption(rcpsp_sol=rcpsp_sol)
        fig, ax = plt.subplots(nrows=len(self.resources_list), sharex=True)
        for i in range(len(self.resources_list)):
            ax[i].axhline(
                y=self.resources[self.resources_list[i]], label=self.resources_list[i]
            )
            ax[i].plot(consumption[i, :])
            ax[i].legend()

    def copy(self):
        return PreemptiveRcpspProblem(
            resources=self.resources,
            non_renewable_resources=self.non_renewable_resources,
            mode_details=deepcopy(self.mode_details),
            successors=deepcopy(self.successors),
            horizon=self.horizon,
            horizon_multiplier=self.horizon_multiplier,
        )

    def copy_with_multiplier(self, multiplier=0.5):
        mode_details = deepcopy(self.mode_details)
        n = int(1 / multiplier)
        for t in mode_details:
            for m in mode_details[t]:
                mode_details[t][m]["duration"] = int(
                    math.ceil(multiplier * mode_details[t][m]["duration"])
                )
        return PreemptiveRcpspProblem(
            resources={r: self.resources[r][::n] * n for r in self.resources},
            non_renewable_resources=self.non_renewable_resources,
            mode_details=mode_details,
            successors=deepcopy(self.successors),
            horizon=int(self.horizon / n),
            horizon_multiplier=self.horizon_multiplier,
        )

    def get_dummy_solution(self):
        sol = PreemptiveRcpspSolution(
            problem=self,
            rcpsp_permutation=list(range(self.n_jobs_non_dummy)),
            rcpsp_modes=[1 for i in range(self.n_jobs_non_dummy)],
        )
        return sol

    def get_resource_available(self, res, time):
        if self.is_calendar:
            return self.resources.get(res, [0])[time]
        return self.resources.get(res, 0)

    def __str__(self):
        val = (
            "I'm a RCPSP problem with "
            + str(self.n_jobs)
            + " tasks.."
            + " and ressources ="
            + str(self.resources_list)
        )
        return val


def generate_schedule_from_permutation_serial_sgs(
    solution: PreemptiveRcpspSolution, rcpsp_problem: PreemptiveRcpspProblem
):
    activity_end_times = {}

    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon

    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        minimum_starting_time[act] = 0
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)

    for k in modes_dict:
        if modes_dict[k] not in rcpsp_problem.mode_details[k]:
            modes_dict[k] = 1
    expected_durations_task = {
        k: rcpsp_problem.mode_details[k][modes_dict[k]]["duration"] for k in modes_dict
    }
    schedules = {}
    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        for id_successor in perm_extended:
            respected = True
            for pred in rcpsp_problem.successors:
                if (
                    id_successor in rcpsp_problem.successors[pred]
                    and pred in perm_extended
                ):
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        starts = []
        ends = []
        cur_duration = 0
        valid = False
        while not valid:
            reached_t = None
            if expected_durations_task[act_id] == 0:
                starts += [current_min_time]
                ends += [current_min_time]
                cur_duration += ends[-1] - starts[-1]
            else:
                reached_end = True
                for t in range(
                    current_min_time,
                    current_min_time + expected_durations_task[act_id] - cur_duration,
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    if any(
                        resource_avail_in_time[res][t]
                        < rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                            res, 0
                        )
                        for res in rcpsp_problem.resources_list
                    ):
                        reached_end = False
                        break
                    else:
                        reached_t = t
                if reached_t is not None and rcpsp_problem.can_be_preempted(act_id):
                    starts += [current_min_time]
                    ends += [reached_t + 1]
                    cur_duration += ends[-1] - starts[-1]
                if reached_end and not rcpsp_problem.can_be_preempted(act_id):
                    starts += [current_min_time]
                    ends += [reached_t + 1]
                    cur_duration += ends[-1] - starts[-1]
            valid = cur_duration == expected_durations_task[act_id]
            if not valid:
                current_min_time = next(
                    (
                        t
                        for t in range(
                            reached_t + 2
                            if reached_t is not None
                            else current_min_time + 1,
                            new_horizon,
                        )
                        if all(
                            resource_avail_in_time[res][t]
                            >= rcpsp_problem.mode_details[act_id][
                                modes_dict[act_id]
                            ].get(res, 0)
                            for res in rcpsp_problem.resources_list
                        )
                    ),
                    None,
                )
                if current_min_time is None:
                    unfeasible_non_renewable_resources = True
                    break
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for s, e in zip(starts, ends):
                for t in range(s, e):
                    for res in resource_avail_in_time:
                        if (
                            rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                                res, 0
                            )
                            == 0
                        ):
                            continue
                        resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]][res]
                        if (
                            res in rcpsp_problem.non_renewable_resources
                            and t == end_t - 1
                        ):
                            for tt in range(end_t, new_horizon):
                                resource_avail_in_time[res][tt] -= (
                                    rcpsp_problem.mode_details[act_id][
                                        modes_dict[act_id]
                                    ][res]
                                )
                                if resource_avail_in_time[res][tt] < 0:
                                    unfeasible_non_renewable_resources = True

            activity_end_times[act_id] = end_t
            schedules[act_id] = (starts, ends)
            perm_extended.remove(act_id)
            if unfeasible_non_renewable_resources:
                break
            for s in rcpsp_problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["starts"] = schedules[act_id][0]
        rcpsp_schedule[act_id]["ends"] = schedules[act_id][1]
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["starts"] = [9999999]
            rcpsp_schedule[last_act_id]["ends"] = [9999999]
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def generate_schedule_from_permutation_serial_sgs_partial_schedule(
    solution: PreemptiveRcpspSolution,
    rcpsp_problem: PreemptiveRcpspProblem,
    partial_schedule: dict[Hashable, dict[str, list[int]]],
    current_t: int,
    completed_tasks: set[Hashable],
) -> tuple[dict[int, dict[str, list[int]]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = list(
                rcpsp_problem.resources[res][: new_horizon + 1]
            )
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        minimum_starting_time[act] = current_t
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)

    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for k in modes_dict:
        if modes_dict[k] not in rcpsp_problem.mode_details[k]:
            modes_dict[k] = 1
    expected_durations_task = {
        k: rcpsp_problem.mode_details[k][modes_dict[k]]["duration"] for k in modes_dict
    }
    done_duration_task = {k: 0 for k in modes_dict}
    schedules = deepcopy(partial_schedule)
    # Update current resource usage by the scheduled task (ongoing task, in practice)
    for task in partial_schedule:
        starts = partial_schedule[task]["starts"]
        ends = partial_schedule[task]["ends"]
        done_duration_task[task] = sum(
            [ends[i] - starts[i] for i in range(len(starts))]
        )
        end_t = ends[-1]
        for s, e in zip(starts, ends):
            for t in range(s, e):
                for res in resource_avail_in_time:
                    resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[task][
                        modes_dict[task]
                    ].get(res, 0)
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t, new_horizon):
                            resource_avail_in_time[res][tt] -= (
                                rcpsp_problem.mode_details[task][modes_dict[task]].get(
                                    res, 0
                                )
                            )
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
        if done_duration_task[task] == expected_durations_task[task]:
            activity_end_times[task] = end_t
            perm_extended.remove(task)
            for s in rcpsp_problem.successors[task]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[task]
                )
        else:
            minimum_starting_time[task] = ends[-1]
    perm_extended = [x for x in perm_extended if x not in completed_tasks]
    # fix modes in case specified mode not in mode details for the activites
    for ac in modes_dict:
        if modes_dict[ac] not in rcpsp_problem.mode_details[ac]:
            modes_dict[ac] = 1
    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        # get first activity in perm with precedences respected
        for id_successor in perm_extended:
            respected = True
            for pred in rcpsp_problem.successors.keys():
                if (
                    id_successor in rcpsp_problem.successors[pred]
                    and pred in perm_extended
                ):
                    respected = False
                    break
            if respected:
                act_id = id_successor
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        starts = []
        ends = []
        while not valid:
            reached_t = None
            if expected_durations_task[act_id] == 0:
                starts += [current_min_time]
                ends += [current_min_time]
                done_duration_task[act_id] += ends[-1] - starts[-1]
            else:
                reached_end = True
                for t in range(
                    current_min_time,
                    current_min_time
                    + expected_durations_task[act_id]
                    - done_duration_task[act_id],
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                    if any(
                        resource_avail_in_time[res][t]
                        < rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                            res, 0
                        )
                        for res in rcpsp_problem.resources_list
                    ):
                        reached_end = False
                        break
                    else:
                        reached_t = t
                if reached_t is not None and rcpsp_problem.can_be_preempted(act_id):
                    starts += [current_min_time]
                    ends += [reached_t + 1]
                    done_duration_task[act_id] += ends[-1] - starts[-1]
                if reached_end and not rcpsp_problem.can_be_preempted(act_id):
                    starts += [current_min_time]
                    ends += [reached_t + 1]
                    done_duration_task[act_id] += ends[-1] - starts[-1]
            valid = done_duration_task[act_id] == expected_durations_task[act_id]
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
                        >= rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                            res, 0
                        )
                        for res in rcpsp_problem.resources_list
                    )
                )
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for s, e in zip(starts, ends):
                for t in range(s, e):
                    for res in resource_avail_in_time:
                        resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0)
                        if (
                            res in rcpsp_problem.non_renewable_resources
                            and t == end_t - 1
                        ):
                            for tt in range(end_t, new_horizon):
                                resource_avail_in_time[res][tt] -= (
                                    rcpsp_problem.mode_details[act_id][
                                        modes_dict[act_id]
                                    ].get(res, 0)
                                )
                                if resource_avail_in_time[res][tt] < 0:
                                    unfeasible_non_renewable_resources = True
            schedules[act_id] = {
                "starts": schedules.get(act_id, {}).get("starts", []) + starts,
                "ends": schedules.get(act_id, {}).get("ends", []) + ends,
            }
            activity_end_times[act_id] = end_t
            perm_extended.remove(act_id)
            for s in rcpsp_problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = schedules[act_id]
    for act_id in completed_tasks:
        rcpsp_schedule[act_id] = partial_schedule[act_id]
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["starts"] = [99999999]
            rcpsp_schedule[last_act_id]["ends"] = [9999999]
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def compute_mean_resource_reserve(
    solution: PreemptiveRcpspSolution, rcpsp_problem: PreemptiveRcpspProblem
):
    if not solution.rcpsp_schedule_feasible:
        return 0.0
    last_activity = rcpsp_problem.sink_task
    makespan = solution.rcpsp_schedule[last_activity]["ends"][-1]
    resource_avail_in_time = {}
    modes = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][: makespan + 1]
        else:
            resource_avail_in_time[res] = np.full(
                makespan, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    for act_id in rcpsp_problem.tasks_list:
        starts = solution.rcpsp_schedule[act_id]["starts"]
        ends = solution.rcpsp_schedule[act_id]["ends"]
        mode = modes[act_id]
        for s, e in zip(starts, ends):
            for t in range(s, e):
                for res in resource_avail_in_time:
                    if rcpsp_problem.mode_details[act_id][mode].get(res, 0) == 0:
                        continue
                    resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[
                        act_id
                    ][mode][res]
                    if res in rcpsp_problem.non_renewable_resources and t == e - 1:
                        for tt in range(e, makespan):
                            resource_avail_in_time[res][tt] -= (
                                rcpsp_problem.mode_details[act_id][mode][res]
                            )
    mean_avail = {}
    for res in resource_avail_in_time:
        mean_avail[res] = np.mean(resource_avail_in_time[res])
    mean_resource_reserve = np.mean(
        [
            mean_avail[res] / max(rcpsp_problem.resources[res])
            if rcpsp_problem.is_varying_resource()
            else mean_avail[res] / rcpsp_problem.resources[res]
            for res in rcpsp_problem.resources_list
        ]
    )
    return mean_resource_reserve


def compute_resource(
    solution: PreemptiveRcpspSolution, rcpsp_problem: PreemptiveRcpspProblem
):
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for r in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[r] = np.copy(
                rcpsp_problem.resources[r][:new_horizon]
            )
        else:
            resource_avail_in_time[r] = rcpsp_problem.resources[r] * np.ones(
                new_horizon
            )
    for t in solution.rcpsp_schedule:
        start_times = solution.rcpsp_schedule[t]["starts"]
        end_times = solution.rcpsp_schedule[t]["ends"]
        for s, e in zip(start_times, end_times):
            for r in rcpsp_problem.resources_list:
                resource_avail_in_time[r][s:e] -= rcpsp_problem.mode_details[t][
                    modes_dict[t]
                ].get(r, 0)
                if np.any(resource_avail_in_time[r][s:e] < 0):
                    logger.debug(f"Missing ressource {__file__}")
    return resource_avail_in_time


def permutation_do_to_permutation_sgs_fast(
    rcpsp_problem: PreemptiveRcpspProblem, permutation_do: Iterable[int]
) -> npt.NDArray[np.int_]:
    perm_extended = [
        rcpsp_problem.index_task[rcpsp_problem.tasks_list_non_dummy[x]]
        for x in permutation_do
    ]
    perm_extended.insert(0, rcpsp_problem.index_task[rcpsp_problem.source_task])
    perm_extended.append(rcpsp_problem.index_task[rcpsp_problem.sink_task])
    return np.array(perm_extended, dtype=np.int_)


def create_np_data_and_jit_functions(rcpsp_problem: Union[PreemptiveRcpspProblem]):
    consumption_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int_,
    )
    duration_array = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.max_number_of_mode), dtype=np.int_
    )
    predecessors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int_)
    successors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int_)
    preemptive_tag = np.zeros(rcpsp_problem.n_jobs, dtype=bool)
    horizon = rcpsp_problem.horizon
    ressource_available = np.zeros(
        (len(rcpsp_problem.resources_list), horizon), dtype=np.int_
    )
    ressource_renewable = np.ones((len(rcpsp_problem.resources_list)), dtype=bool)
    min_duration_preemptive_bool = np.zeros(rcpsp_problem.n_jobs, dtype=bool)
    min_duration_preemptive = np.zeros(rcpsp_problem.n_jobs, dtype=np.int_)
    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        min_duration_preemptive_bool[i] = rcpsp_problem.duration_subtask[task][0]
        min_duration_preemptive[i] = rcpsp_problem.duration_subtask[task][1]

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        preemptive_tag[i] = rcpsp_problem.can_be_preempted(task)
        index_mode = 0
        for mode in sorted(
            rcpsp_problem.mode_details[rcpsp_problem.tasks_list[i]].keys()
        ):
            for k in range(len(rcpsp_problem.resources_list)):
                consumption_array[i, index_mode, k] = rcpsp_problem.mode_details[task][
                    mode
                ].get(rcpsp_problem.resources_list[k], 0)
            duration_array[i, index_mode] = rcpsp_problem.mode_details[task][mode][
                "duration"
            ]
            index_mode += 1

    task_index = {rcpsp_problem.tasks_list[i]: i for i in range(rcpsp_problem.n_jobs)}
    for k in range(len(rcpsp_problem.resources_list)):
        if rcpsp_problem.is_varying_resource():
            ressource_available[k, :] = rcpsp_problem.resources[
                rcpsp_problem.resources_list[k]
            ][: ressource_available.shape[1]]
        else:
            ressource_available[k, :] = np.full(
                ressource_available.shape[1],
                rcpsp_problem.resources[rcpsp_problem.resources_list[k]],
                dtype=np.int_,
            )
        if rcpsp_problem.resources_list[k] in rcpsp_problem.non_renewable_resources:
            ressource_renewable[k] = False

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        for s in rcpsp_problem.successors[task]:
            index_s = task_index[s]
            predecessors[index_s, i] = 1
            successors[i, index_s] = 1
    minimum_starting_time_array = np.zeros(rcpsp_problem.n_jobs, dtype=np.int_)
    if not rcpsp_problem.is_duration_minimum_preemption():
        func_sgs = partial(
            sgs_fast_preemptive,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            minimum_starting_time_array=minimum_starting_time_array,
        )
        func_sgs_2 = partial(
            sgs_fast_partial_schedule_preemptive,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            minimum_starting_time_array=minimum_starting_time_array,
        )
    else:
        func_sgs = partial(
            sgs_fast_preemptive_minduration,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            min_duration_preemptive=min_duration_preemptive,
            min_duration_preemptive_bool=min_duration_preemptive_bool,
        )
        func_sgs_2 = partial(
            sgs_fast_partial_schedule_preemptive_minduration,
            consumption_array=consumption_array,
            preemptive_tag=preemptive_tag,
            duration_array=duration_array,
            predecessors=predecessors,
            successors=successors,
            horizon=horizon,
            ressource_available=ressource_available,
            ressource_renewable=ressource_renewable,
            min_duration_preemptive=min_duration_preemptive,
            min_duration_preemptive_bool=min_duration_preemptive_bool,
        )
    func_compute_mean_resource = partial(
        compute_mean_ressource,
        consumption_array=consumption_array,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
    )
    return func_sgs, func_sgs_2, func_compute_mean_resource


def get_rcpsp_problemp_preemptive(rcpsp_problem):
    return PreemptiveRcpspProblem(
        resources=rcpsp_problem.resources,
        non_renewable_resources=rcpsp_problem.non_renewable_resources,
        mode_details=rcpsp_problem.mode_details,
        successors=rcpsp_problem.successors,
        horizon=rcpsp_problem.horizon,
        horizon_multiplier=1,
        tasks_list=rcpsp_problem.tasks_list,
        source_task=rcpsp_problem.source_task,
        sink_task=rcpsp_problem.sink_task,
        preemptive_indicator={
            rcpsp_problem.tasks_list[k]: True for k in range(rcpsp_problem.n_jobs)
        },
        name_task=rcpsp_problem.name_task,
    )
