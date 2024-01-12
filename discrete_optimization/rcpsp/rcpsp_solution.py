#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import (
    annotations,  # make annotations be considered as string by default
)

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
from numpy import typing as npt

from discrete_optimization.generic_tools.do_problem import RobustProblem, Solution

if TYPE_CHECKING:  # avoid circular imports due to annotations
    from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel


class TaskDetails:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class RCPSPSolution(Solution):
    """Solution to RCPSPModel problems.

    Attributes:
        problem: RCPSP problem for which this is a solution
        rcpsp_permutation: Tasks permutation.
        rcpsp_schedule: Tasks schedule. ( task -> "start_time" or "end_time" -> time)
            Potentially only partial if not feasible with given permutation, or for aggregated models.
        rcpsp_modes: Mode used for each task. Same order as `problem.tasks_list_non_dummy`.
        rcpsp_schedule_feasible: False if schedule generation from permutation failed, True else.
        standardised_permutation: Permutation deduced uniquely from schedule.
            Can be different from `rcpsp_permutation` as different permutations can lead to same schedule.
        fast: boolean indicating if we us the fast functions to generate schedule from permutation.

    Args:
        problem: RCPSP problem for which this is a solution
        rcpsp_permutation: Tasks permutation.
            If given and schedule not given, used to reconstruct the schedule.
            if not given, deduced from schedule.
            If not given and schedule not given, it is set to `problem.fixed_permutation`
        rcpsp_schedule: Tasks schedule. ( task -> "start_time" or "end_time" -> time)
            If given used to construct `standardised_permutation`.
            If given and `rcpsp_permutation` not given, used to construct `rcpsp_permutation`.
            If given and `rcpsp_permutation` given, no consistency check.
            If not given, deduced from `rcpsp_permutation` if possible.
            If not possible, `rcpsp_schedule_feasible` set to False and
            `rcpsp_schedule` set to a partially filled schedule.
            In case of `Aggreg_RCPSPModel`, kept empty.
        rcpsp_modes: Mode used for each task. Same order as `problem.tasks_list_non_dummy`.
            If not given we use `problem.fixed_modes` if existing, else 1 for each task.
        rcpsp_schedule_feasible: True if a schedule can be deduced from permutation.
            False if it leads to incoherency preventing a schedule generation.
            Recomputed when schedule is (re)computed from permutation.
        standardised_permutation: Permutation deduced uniquely from schedule. If given, not recomputed.
            Can be different from `rcpsp_permutation` as different permutations can lead to same schedule.
        fast: boolean indicating if we us the fast functions to generate schedule from permutation.

    """

    def __init__(
        self,
        problem: RCPSPModel,
        rcpsp_permutation: Optional[List[int]] = None,
        rcpsp_schedule: Optional[Dict[Hashable, Dict[str, int]]] = None,
        rcpsp_modes: Optional[List[int]] = None,
        rcpsp_schedule_feasible: bool = True,
        standardised_permutation: Optional[List[int]] = None,
        fast: bool = True,
    ):
        self.problem = problem
        self.rcpsp_schedule_feasible = rcpsp_schedule_feasible
        self.fast = fast

        self.rcpsp_modes: List[int]
        self.rcpsp_schedule: Dict[Hashable, Dict[str, int]]
        self.rcpsp_permutation: List[int]
        self.standardised_permutation: List[int]

        # init rcpsp_modes
        if rcpsp_modes is None:
            if self.problem.fixed_modes is not None:
                self.rcpsp_modes = self.problem.fixed_modes
            else:
                self.rcpsp_modes = [1 for i in range(self.problem.n_jobs_non_dummy)]
        else:
            self.rcpsp_modes = rcpsp_modes

        # init rcpsp_permutation
        if rcpsp_permutation is None:
            if rcpsp_schedule is None:
                if self.problem.fixed_permutation is not None:
                    self.rcpsp_permutation = self.problem.fixed_permutation
                else:
                    raise ValueError(
                        "rcpsp_permutation and rcpsp_schedule can be None together "
                        "only if problem.fixed_permutation is not None"
                    )
            else:
                self.rcpsp_schedule = rcpsp_schedule
                standardised_permutation = self.generate_permutation_from_schedule()
                self.rcpsp_permutation = deepcopy(standardised_permutation)
        else:
            self.rcpsp_permutation = rcpsp_permutation

        # init rcpsp_schedule
        if rcpsp_schedule is None:
            self.generate_schedule_from_permutation_serial_sgs(do_fast=self.fast)
        else:
            self.rcpsp_schedule = rcpsp_schedule

        # init standardised_permutation
        if standardised_permutation is None:
            self.standardised_permutation = self.generate_permutation_from_schedule()
        else:
            self.standardised_permutation = standardised_permutation

        # schedule already computed (prevent issues with __setattr__ hack)
        self._schedule_to_recompute = False

    def change_problem(self, new_problem: RCPSPModel) -> None:  # type: ignore
        # set problem
        self.problem = new_problem
        # recompute schedule and standardised permutation with respect to the new problem
        self.generate_schedule_from_permutation_serial_sgs(do_fast=self.fast)
        self.standardised_permutation = self.generate_permutation_from_schedule()

    def __setattr__(self, key: str, value: Any) -> None:
        super.__setattr__(self, key, value)
        if key == "rcpsp_permutation":
            self._schedule_to_recompute = True

    def copy(self) -> "RCPSPSolution":
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_permutation=deepcopy(self.rcpsp_permutation),
            rcpsp_modes=deepcopy(self.rcpsp_modes),
            rcpsp_schedule=deepcopy(self.rcpsp_schedule),
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
            fast=self.fast,
        )

    def lazy_copy(self) -> "RCPSPSolution":
        return RCPSPSolution(
            problem=self.problem,
            rcpsp_permutation=self.rcpsp_permutation,
            rcpsp_modes=self.rcpsp_modes,
            rcpsp_schedule=self.rcpsp_schedule,
            rcpsp_schedule_feasible=self.rcpsp_schedule_feasible,
            standardised_permutation=self.standardised_permutation,
            fast=self.fast,
        )

    def __str__(self) -> str:
        if self.rcpsp_schedule is None:
            sched_str = "None"
        else:
            sched_str = str(self.rcpsp_schedule)
        val = "RCPSP solution (rcpsp_schedule): " + sched_str
        return val

    def generate_permutation_from_schedule(self) -> List[int]:
        sorted_task = [
            self.problem.index_task_non_dummy[i]
            for i in sorted(
                self.rcpsp_schedule, key=lambda x: self.rcpsp_schedule[x]["start_time"]
            )
            if i in self.problem.index_task_non_dummy
        ]
        return sorted_task

    def compute_mean_resource_reserve(self, fast: bool = True) -> float:
        if not fast:
            return compute_mean_resource_reserve(
                solution=self, rcpsp_problem=self.problem
            )
        else:
            if not self.rcpsp_schedule_feasible:
                return 0.0
            last_activity = self.problem.sink_task
            makespan = self.rcpsp_schedule[last_activity]["end_time"]
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                return 0.0
            else:
                return self.problem.compute_mean_resource(
                    horizon=makespan,
                    modes_array=np.array(
                        self.problem.build_mode_array(self.rcpsp_modes)
                    )
                    - 1,  # permutation_task=array(task)->task index
                    start_array=np.array(
                        [
                            self.rcpsp_schedule[t]["start_time"]
                            for t in self.problem.tasks_list
                        ]
                    ),
                    end_array=np.array(
                        [
                            self.rcpsp_schedule[t]["end_time"]
                            for t in self.problem.tasks_list
                        ]
                    ),
                )

    def generate_schedule_from_permutation_serial_sgs(
        self, do_fast: bool = True
    ) -> None:
        self._schedule_to_recompute = False
        if isinstance(self.problem, RobustProblem):
            self.rcpsp_schedule = {}
            self.rcpsp_schedule_feasible = True
        else:
            if do_fast:
                schedule: Dict[int, Tuple[int, int]]
                if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                    # non existing modes
                    schedule, unfeasible = {}, True
                else:
                    schedule, unfeasible = self.problem.func_sgs(
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
                for k in schedule:
                    self.rcpsp_schedule[self.problem.tasks_list[k]] = {
                        "start_time": schedule[k][0],
                        "end_time": schedule[k][1],
                    }
                if self.problem.sink_task not in self.rcpsp_schedule:
                    self.rcpsp_schedule[self.problem.sink_task] = {
                        "start_time": 99999999,
                        "end_time": 99999999,
                    }
            else:
                (
                    self.rcpsp_schedule,
                    self.rcpsp_schedule_feasible,
                ) = generate_schedule_from_permutation_serial_sgs(
                    solution=self, rcpsp_problem=self.problem
                )

    def generate_schedule_from_permutation_serial_sgs_2(
        self,
        current_t: int = 0,
        completed_tasks: Optional[Dict[Hashable, TaskDetails]] = None,
        scheduled_tasks_start_times: Optional[Dict[Hashable, int]] = None,
        do_fast: bool = True,
    ) -> None:
        if completed_tasks is None:
            completed_tasks = {}
        if scheduled_tasks_start_times is None:
            scheduled_tasks_start_times = {}
        if do_fast and not self.problem.do_special_constraints:
            schedule: Dict[int, Tuple[int, int]]
            if max(self.rcpsp_modes) > self.problem.max_number_of_mode:
                # non existing modes
                schedule, unfeasible = {}, True
            else:
                schedule, unfeasible = self.problem.func_sgs_2(
                    current_time=current_t,
                    completed_task_indicator=np.array(
                        [
                            1 if self.problem.tasks_list[i] in completed_tasks else 0
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    completed_task_times=np.array(
                        [
                            completed_tasks[self.problem.tasks_list[i]].end
                            if self.problem.tasks_list[i] in completed_tasks
                            else 0
                            for i in range(self.problem.n_jobs)
                        ]
                    ),
                    scheduled_task=np.array(
                        [
                            scheduled_tasks_start_times[self.problem.tasks_list[i]]
                            if self.problem.tasks_list[i] in scheduled_tasks_start_times
                            else -1
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
            self.rcpsp_schedule = {}
            for k in schedule:
                self.rcpsp_schedule[self.problem.tasks_list[k]] = {
                    "start_time": schedule[k][0],
                    "end_time": schedule[k][1],
                }
            if self.problem.sink_task not in self.rcpsp_schedule:
                self.rcpsp_schedule[self.problem.sink_task] = {
                    "start_time": 999999999,
                    "end_time": 999999999,
                }
            self.rcpsp_schedule_feasible = not unfeasible
            self._schedule_to_recompute = False
        else:
            if self.problem.do_special_constraints:
                (
                    self.rcpsp_schedule,
                    self.rcpsp_schedule_feasible,
                ) = generate_schedule_from_permutation_serial_sgs_partial_schedule_specialized_constraints(
                    solution=self,
                    current_t=current_t,
                    completed_tasks=completed_tasks,
                    scheduled_tasks_start_times=scheduled_tasks_start_times,
                    rcpsp_problem=self.problem,
                )
            else:
                (
                    self.rcpsp_schedule,
                    self.rcpsp_schedule_feasible,
                ) = generate_schedule_from_permutation_serial_sgs_partial_schedule(
                    solution=self,
                    current_t=current_t,
                    completed_tasks=completed_tasks,
                    scheduled_tasks_start_times=scheduled_tasks_start_times,
                    rcpsp_problem=self.problem,
                )
            self._schedule_to_recompute = False

    def get_max_end_time(self) -> int:
        return max([self.get_end_time(x) for x in self.rcpsp_schedule])

    def get_start_time(self, task: Hashable) -> int:
        return self.rcpsp_schedule[task]["start_time"]

    def get_end_time(self, task: Hashable) -> int:
        return self.rcpsp_schedule[task]["end_time"]

    def get_start_times_list(self, task: Hashable) -> List[int]:
        return [self.get_start_time(task)]

    def get_end_times_list(self, task: Hashable) -> List[int]:
        return [self.get_end_time(task)]

    def get_active_time(self, task: Hashable) -> List[int]:
        return list(range(self.get_start_time(task), self.get_end_time(task)))

    def get_mode(self, task: Hashable) -> int:
        return self.rcpsp_modes[self.problem.index_task_non_dummy[task]]

    def __hash__(self) -> int:
        return hash((tuple(self.rcpsp_permutation), tuple(self.rcpsp_modes)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RCPSPSolution)
            and self.rcpsp_permutation == other.rcpsp_permutation
            and self.rcpsp_modes == other.rcpsp_modes
        )


def permutation_do_to_permutation_sgs_fast(
    rcpsp_problem: RCPSPModel, permutation_do: Iterable[int]
) -> npt.NDArray[np.int_]:
    perm_extended = [
        rcpsp_problem.index_task[rcpsp_problem.tasks_list_non_dummy[x]]
        for x in permutation_do
    ]
    perm_extended.insert(0, rcpsp_problem.index_task[rcpsp_problem.source_task])
    perm_extended.append(rcpsp_problem.index_task[rcpsp_problem.sink_task])
    return np.array(perm_extended, dtype=np.int_)


def generate_schedule_from_permutation_serial_sgs(
    solution: RCPSPSolution, rcpsp_problem: RCPSPModel
) -> Tuple[Dict[Hashable, Dict[str, int]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon

    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][  # type: ignore
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
        # for act_id in perm_extended:
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            for t in range(
                current_min_time,
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"],
            ):
                for res in rcpsp_problem.resources_list:
                    if (
                        rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                            res, 0
                        )
                        == 0
                    ):
                        continue
                    if t < new_horizon:
                        if (
                            resource_avail_in_time[res][t]
                            < rcpsp_problem.mode_details[act_id][modes_dict[act_id]][
                                res
                            ]
                        ):
                            valid = False
                    else:
                        unfeasible_non_renewable_resources = True
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = (
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            for t in range(current_min_time, end_t):
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
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t, new_horizon):
                            resource_avail_in_time[res][
                                tt
                            ] -= rcpsp_problem.mode_details[act_id][modes_dict[act_id]][
                                res
                            ]
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            activity_end_times[act_id] = end_t
            perm_extended.remove(act_id)
            for s in rcpsp_problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["start_time"] = 99999999
            rcpsp_schedule[last_act_id]["end_time"] = 9999999
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def generate_schedule_from_permutation_serial_sgs_special_constraints(
    solution: RCPSPSolution, rcpsp_problem: RCPSPModel
) -> Tuple[Dict[Hashable, Dict[str, int]], bool]:
    activity_end_times = {}

    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][  # type: ignore
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        minimum_starting_time[act] = 0
        if rcpsp_problem.do_special_constraints:
            if act in rcpsp_problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    rcpsp_problem.special_constraints.start_times_window[act][0]  # type: ignore
                    if rcpsp_problem.special_constraints.start_times_window[act][0]
                    is not None
                    else 0
                )
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)

    def ressource_consumption(
        res: str, task: Hashable, duration: int, mode: int
    ) -> int:
        dur = rcpsp_problem.mode_details[task][mode]["duration"]
        if duration > dur:
            return 0
        return rcpsp_problem.mode_details[task][mode].get(res, 0)

    for k in modes_dict:
        if modes_dict[k] not in rcpsp_problem.mode_details[k]:
            modes_dict[k] = 1

    def look_for_task(perm: List[Hashable], ignore_sc: bool = False) -> List[Hashable]:
        act_ids = []
        for task_id in perm:
            respected = True
            # Check all kind of precedence constraints....
            for pred in rcpsp_problem.predecessors.get(task_id, {}):
                if pred in perm_extended:
                    respected = False
                    break
            if not ignore_sc:
                for (
                    pred
                ) in rcpsp_problem.special_constraints.dict_start_at_end_reverse.get(
                    task_id, {}
                ):
                    if pred in perm_extended:
                        respected = False
                        break
                for (
                    pred
                ) in rcpsp_problem.special_constraints.dict_start_at_end_offset_reverse.get(
                    task_id, {}
                ):
                    if pred in perm_extended:
                        respected = False
                        break
                for (
                    pred
                ) in rcpsp_problem.special_constraints.dict_start_after_nunit_reverse.get(
                    task_id, {}
                ):
                    if pred in perm_extended:
                        respected = False
                        break
            task_to_start_too = set()
            if respected:
                task_to_start_too = (
                    rcpsp_problem.special_constraints.dict_start_together.get(
                        task_id, set()
                    )
                )
                if not ignore_sc:
                    if len(task_to_start_too) > 0:
                        if not all(
                            s not in perm_extended
                            for t in task_to_start_too
                            for s in rcpsp_problem.predecessors[t]
                        ):
                            respected = False
            if respected:
                act_ids = [task_id] + list(task_to_start_too)
                break
        return act_ids

    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        act_ids = look_for_task(
            [
                k
                for k in rcpsp_problem.special_constraints.dict_start_at_end_reverse
                if k in perm_extended
            ]
        )
        act_ids = []
        if len(act_ids) == 0:
            act_ids = look_for_task(perm_extended)
        if len(act_ids) == 0:
            act_ids = look_for_task(perm_extended, ignore_sc=True)
        current_min_time = max([minimum_starting_time[act_id] for act_id in act_ids])
        max_duration = max(
            [
                rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                for act_id in act_ids
            ]
        )
        valid = False
        while not valid:
            valid = True
            for t in range(current_min_time, current_min_time + max_duration):
                for res in rcpsp_problem.resources_list:
                    r = sum(
                        [
                            ressource_consumption(
                                res=res,
                                task=task,
                                duration=t - current_min_time,
                                mode=modes_dict[task],
                            )
                            for task in act_ids
                        ]
                    )
                    if r == 0:
                        continue
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < r:
                            valid = False
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                if not valid:
                    break
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = current_min_time + max_duration
            for t in range(current_min_time, current_min_time + max_duration):
                for res in resource_avail_in_time:
                    r = sum(
                        [
                            ressource_consumption(
                                res=res,
                                task=task,
                                duration=t - current_min_time,
                                mode=modes_dict[task],
                            )
                            for task in act_ids
                        ]
                    )
                    resource_avail_in_time[res][t] -= r
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t, new_horizon):
                            resource_avail_in_time[res][tt] -= r
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            for act_id in act_ids:
                activity_end_times[act_id] = (
                    current_min_time
                    + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                )
                perm_extended.remove(act_id)
                for s in rcpsp_problem.successors[act_id]:
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[act_id]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end.get(
                    act_id, {}
                ):
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[act_id]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_after_nunit.get(
                    act_id, {}
                ):
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s],
                        current_min_time
                        + rcpsp_problem.special_constraints.dict_start_after_nunit[
                            act_id
                        ][s],
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end_offset.get(
                    act_id, {}
                ):
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s],
                        activity_end_times[act_id]
                        + rcpsp_problem.special_constraints.dict_start_at_end_offset[
                            act_id
                        ][s],
                    )
    rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["start_time"] = 99999999
            rcpsp_schedule[last_act_id]["end_time"] = 9999999
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def generate_schedule_from_permutation_serial_sgs_partial_schedule(
    solution: RCPSPSolution,
    rcpsp_problem: RCPSPModel,
    current_t: int,
    completed_tasks: Dict[Hashable, TaskDetails],
    scheduled_tasks_start_times: Dict[Hashable, int],
) -> Tuple[Dict[Hashable, Dict[str, int]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][  # type: ignore
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        if act in list(scheduled_tasks_start_times.keys()):
            minimum_starting_time[act] = scheduled_tasks_start_times[act]
        else:
            minimum_starting_time[act] = current_t
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)

    # Update current resource usage by the scheduled task (ongoing task, in practice)
    for act_id in scheduled_tasks_start_times:
        current_min_time = scheduled_tasks_start_times[act_id]
        end_t = (
            current_min_time
            + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        for t in range(current_min_time, end_t):
            for res in resource_avail_in_time:
                resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[act_id][
                    modes_dict[act_id]
                ].get(res, 0)
                if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                    for tt in range(end_t, new_horizon):
                        resource_avail_in_time[res][tt] -= rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0)
                        if resource_avail_in_time[res][tt] < 0:
                            unfeasible_non_renewable_resources = True
        activity_end_times[act_id] = end_t
        perm_extended.remove(act_id)
        for s in rcpsp_problem.successors[act_id]:
            minimum_starting_time[s] = max(
                minimum_starting_time[s], activity_end_times[act_id]
            )

    perm_extended = [x for x in perm_extended if x not in list(completed_tasks)]
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
        while not valid:
            valid = True
            for t in range(
                current_min_time,
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"],
            ):
                for res in resource_avail_in_time:
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0):
                            valid = False
                    else:
                        unfeasible_non_renewable_resources = True
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = (
                current_min_time
                + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
            )
            for t in range(current_min_time, end_t):
                for res in resource_avail_in_time:
                    resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[
                        act_id
                    ][modes_dict[act_id]].get(res, 0)
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t + 1, new_horizon):
                            resource_avail_in_time[res][
                                tt
                            ] -= rcpsp_problem.mode_details[act_id][
                                modes_dict[act_id]
                            ].get(
                                res, 0
                            )
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            activity_end_times[act_id] = end_t
            perm_extended.remove(act_id)
            for s in rcpsp_problem.successors[act_id]:
                minimum_starting_time[s] = max(
                    minimum_starting_time[s], activity_end_times[act_id]
                )
    rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    for act_id in completed_tasks:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = completed_tasks[act_id].start
        rcpsp_schedule[act_id]["end_time"] = completed_tasks[act_id].end
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["start_time"] = 99999999
            rcpsp_schedule[last_act_id]["end_time"] = 9999999
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def generate_schedule_from_permutation_serial_sgs_partial_schedule_specialized_constraints(
    solution: RCPSPSolution,
    rcpsp_problem: RCPSPModel,
    current_t: int,
    completed_tasks: Dict[Hashable, TaskDetails],
    scheduled_tasks_start_times: Dict[Hashable, int],
) -> Tuple[Dict[Hashable, Dict[str, int]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = rcpsp_problem.horizon
    resource_avail_in_time = {}
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][  # type: ignore
                : new_horizon + 1
            ]
        else:
            resource_avail_in_time[res] = np.full(
                new_horizon, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()

    def ressource_consumption(
        res: str, task: Hashable, duration: int, mode: int
    ) -> int:
        dur = rcpsp_problem.mode_details[task][mode]["duration"]
        if duration > dur:
            return 0
        return rcpsp_problem.mode_details[task][mode].get(res, 0)

    minimum_starting_time = {}
    for act in rcpsp_problem.tasks_list:
        if act in list(scheduled_tasks_start_times.keys()):
            minimum_starting_time[act] = scheduled_tasks_start_times[act]
        else:
            minimum_starting_time[act] = current_t
        if rcpsp_problem.do_special_constraints:
            if act in rcpsp_problem.special_constraints.start_times_window:
                minimum_starting_time[act] = (
                    max(  # type: ignore
                        rcpsp_problem.special_constraints.start_times_window[act][0],
                        minimum_starting_time[act],
                    )
                    if rcpsp_problem.special_constraints.start_times_window[act][0]
                    is not None
                    else minimum_starting_time[act]
                )
    perm_extended = [
        rcpsp_problem.tasks_list_non_dummy[x] for x in solution.rcpsp_permutation
    ]
    perm_extended.insert(0, rcpsp_problem.source_task)
    perm_extended.append(rcpsp_problem.sink_task)
    modes_dict = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    # Update current resource usage by the scheduled task (ongoing task, in practice)
    for act_id in scheduled_tasks_start_times:
        current_min_time = scheduled_tasks_start_times[act_id]
        end_t = (
            current_min_time
            + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        for t in range(current_min_time, end_t):
            for res in resource_avail_in_time:
                resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[act_id][
                    modes_dict[act_id]
                ].get(res, 0)
                if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                    for tt in range(end_t, new_horizon):
                        resource_avail_in_time[res][tt] -= rcpsp_problem.mode_details[
                            act_id
                        ][modes_dict[act_id]].get(res, 0)
                        if resource_avail_in_time[res][tt] < 0:
                            unfeasible_non_renewable_resources = True
        activity_end_times[act_id] = end_t
        perm_extended.remove(act_id)
        for s in rcpsp_problem.successors[act_id]:
            minimum_starting_time[s] = max(
                minimum_starting_time[s], activity_end_times[act_id]
            )

    perm_extended = [x for x in perm_extended if x not in list(completed_tasks)]
    for ac in modes_dict:
        if modes_dict[ac] not in rcpsp_problem.mode_details[ac]:
            modes_dict[ac] = 1

    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources:
        act_ids = []
        for task_id in perm_extended:
            respected = True
            # Check all kind of precedence constraints....
            for pred in rcpsp_problem.predecessors.get(task_id, {}):
                if pred in perm_extended:
                    respected = False
                    break
            for pred in rcpsp_problem.special_constraints.dict_start_at_end_reverse.get(
                task_id, {}
            ):
                if pred in perm_extended:
                    respected = False
                    break
            for (
                pred
            ) in rcpsp_problem.special_constraints.dict_start_at_end_offset_reverse.get(
                task_id, {}
            ):
                if pred in perm_extended:
                    respected = False
                    break
            for (
                pred
            ) in rcpsp_problem.special_constraints.dict_start_after_nunit_reverse.get(
                task_id, {}
            ):
                if pred in perm_extended:
                    respected = False
                    break
            task_to_start_too: List[Hashable] = []
            if respected:
                task_to_start_too = [
                    k
                    for k in rcpsp_problem.special_constraints.dict_start_together.get(
                        task_id, set()
                    )
                    if k in perm_extended
                ]
                if len(task_to_start_too) > 0:
                    if not all(
                        s not in perm_extended
                        for t in task_to_start_too
                        for s in rcpsp_problem.predecessors[t]
                    ):
                        respected = False
            if respected:
                act_ids = [task_id] + task_to_start_too
                break
        if len(act_ids) == 0:
            for task_id in perm_extended:
                respected = True
                # Check all kind of precedence constraints....
                for pred in rcpsp_problem.predecessors.get(task_id, {}):
                    if pred in perm_extended:
                        respected = False
                        break
                if respected:
                    act_ids = [task_id]
        current_min_time = max([minimum_starting_time[act_id] for act_id in act_ids])
        max_duration = max(
            [
                rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                for act_id in act_ids
            ]
        )
        valid = False
        while not valid:
            valid = True
            for t in range(current_min_time, current_min_time + max_duration):
                for res in rcpsp_problem.resources_list:
                    r = sum(
                        [
                            ressource_consumption(
                                res=res,
                                task=task,
                                duration=t - current_min_time,
                                mode=modes_dict[task],
                            )
                            for task in act_ids
                        ]
                    )
                    if r == 0:
                        continue
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < r:
                            valid = False
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                if not valid:
                    break
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = current_min_time + max_duration
            for t in range(current_min_time, current_min_time + max_duration):
                for res in resource_avail_in_time:
                    r = sum(
                        [
                            ressource_consumption(
                                res=res,
                                task=task,
                                duration=t - current_min_time,
                                mode=modes_dict[task],
                            )
                            for task in act_ids
                        ]
                    )
                    resource_avail_in_time[res][t] -= r
                    if res in rcpsp_problem.non_renewable_resources and t == end_t - 1:
                        for tt in range(end_t, new_horizon):
                            resource_avail_in_time[res][tt] -= r
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            for act_id in act_ids:
                activity_end_times[act_id] = (
                    current_min_time
                    + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                )
                perm_extended.remove(act_id)
                for s in rcpsp_problem.successors[act_id]:
                    minimum_starting_time[s] = max(
                        minimum_starting_time[s], activity_end_times[act_id]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end.get(
                    act_id, {}
                ):
                    minimum_starting_time[s] = activity_end_times[act_id]
                for s in rcpsp_problem.special_constraints.dict_start_after_nunit.get(
                    act_id, {}
                ):
                    minimum_starting_time[s] = (
                        current_min_time
                        + rcpsp_problem.special_constraints.dict_start_after_nunit[
                            act_id
                        ][s]
                    )
                for s in rcpsp_problem.special_constraints.dict_start_at_end_offset.get(
                    act_id, {}
                ):
                    minimum_starting_time[s] = (
                        activity_end_times[act_id]
                        + rcpsp_problem.special_constraints.dict_start_at_end_offset[
                            act_id
                        ][s]
                    )
    rcpsp_schedule: Dict[Hashable, Dict[str, int]] = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = (
            activity_end_times[act_id]
            - rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
        )
        rcpsp_schedule[act_id]["end_time"] = activity_end_times[act_id]
    for act_id in completed_tasks:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]["start_time"] = completed_tasks[act_id].start
        rcpsp_schedule[act_id]["end_time"] = completed_tasks[act_id].end
    if unfeasible_non_renewable_resources:
        rcpsp_schedule_feasible = False
        last_act_id = rcpsp_problem.sink_task
        if last_act_id not in rcpsp_schedule:
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]["start_time"] = 99999999
            rcpsp_schedule[last_act_id]["end_time"] = 9999999
    else:
        rcpsp_schedule_feasible = True
    return rcpsp_schedule, rcpsp_schedule_feasible


def compute_mean_resource_reserve(
    solution: RCPSPSolution, rcpsp_problem: RCPSPModel
) -> float:
    if not solution.rcpsp_schedule_feasible:
        return 0.0
    last_activity = rcpsp_problem.sink_task
    makespan = solution.rcpsp_schedule[last_activity]["end_time"]
    resource_avail_in_time = {}
    modes = rcpsp_problem.build_mode_dict(solution.rcpsp_modes)
    for res in rcpsp_problem.resources_list:
        if rcpsp_problem.is_varying_resource():
            resource_avail_in_time[res] = rcpsp_problem.resources[res][: makespan + 1]  # type: ignore
        else:
            resource_avail_in_time[res] = np.full(
                makespan, rcpsp_problem.resources[res], dtype=np.int_
            ).tolist()
    for act_id in rcpsp_problem.tasks_list:
        start_time = solution.rcpsp_schedule[act_id]["start_time"]
        end_time = solution.rcpsp_schedule[act_id]["end_time"]
        mode = modes[act_id]
        for t in range(start_time, end_time):
            for res in resource_avail_in_time:
                if rcpsp_problem.mode_details[act_id][mode].get(res, 0) == 0:
                    continue
                resource_avail_in_time[res][t] -= rcpsp_problem.mode_details[act_id][
                    mode
                ][res]
                if res in rcpsp_problem.non_renewable_resources and t == end_time:
                    for tt in range(end_time, makespan):
                        resource_avail_in_time[res][tt] -= rcpsp_problem.mode_details[
                            act_id
                        ][mode][res]
    mean_avail = {}
    for res in resource_avail_in_time:
        mean_avail[res] = np.mean(resource_avail_in_time[res])
    mean_resource_reserve = np.mean(
        [
            mean_avail[res] / rcpsp_problem.get_max_resource_capacity(res)
            for res in rcpsp_problem.resources_list
        ]
    )
    return float(mean_resource_reserve)


class PartialSolution:
    def __init__(
        self,
        task_mode: Optional[Dict[int, int]] = None,
        start_times: Optional[Dict[int, int]] = None,
        end_times: Optional[Dict[int, int]] = None,
        partial_permutation: Optional[List[int]] = None,
        list_partial_order: Optional[List[List[int]]] = None,
        start_together: Optional[List[Tuple[int, int]]] = None,
        start_at_end: Optional[List[Tuple[int, int]]] = None,
        start_at_end_plus_offset: Optional[List[Tuple[int, int, int]]] = None,
        start_after_nunit: Optional[List[Tuple[int, int, int]]] = None,
        disjunctive_tasks: Optional[List[Tuple[int, int]]] = None,
        start_times_window: Optional[Dict[Hashable, Tuple[int, int]]] = None,
        end_times_window: Optional[Dict[Hashable, Tuple[int, int]]] = None,
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
