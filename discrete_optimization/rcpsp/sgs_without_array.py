#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Dict, Hashable, Optional, Tuple

import numpy as np
from numpy import typing as npt
from sortedcontainers import SortedDict

from discrete_optimization.rcpsp import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution


class SGSWithoutArray:
    def __init__(self, rcpsp_model: RCPSPModel):
        self.rcpsp_model = rcpsp_model
        self.resource_avail_in_time: Dict[str, npt.NDArray[np.int_]] = {}
        for res in self.rcpsp_model.resources_list:
            if self.rcpsp_model.is_varying_resource():
                self.resource_avail_in_time[res] = np.array(
                    self.rcpsp_model.resources[res][: self.rcpsp_model.horizon + 1]  # type: ignore
                )
            else:
                self.resource_avail_in_time[res] = np.full(
                    self.rcpsp_model.horizon,
                    self.rcpsp_model.resources[res],
                    dtype=np.int_,
                )
        self.dict_step_ressource: Dict[str, SortedDict] = {}
        for res in self.resource_avail_in_time:
            self.dict_step_ressource[res] = SGSWithoutArray.create_absolute_dict(
                self.resource_avail_in_time[res]
            )

    @staticmethod
    def get_available_from_delta(sdict: SortedDict, time: int) -> int:
        index = sdict.bisect_left(time)
        if time in sdict:
            r = range(index + 1)
        else:
            r = range(index)
        s = sum(sdict.peekitem(j)[1] for j in r)
        return s

    @staticmethod
    def get_available_from_absolute(sdict: SortedDict, time: int) -> int:
        index = sdict.bisect_right(time)
        s = sdict.peekitem(index - 1)[1]
        return s

    @staticmethod
    def add_event_delta(
        sdict: SortedDict, time_start: int, delta: int, time_end: int
    ) -> None:
        if time_start in sdict:
            sdict[time_start] += delta
        else:
            sdict.update({time_start: delta})
        if time_end in sdict:
            sdict[time_end] -= delta
        else:
            sdict.update({time_end: -delta})

    @staticmethod
    def add_event_delta_in_absolute(
        sdict: SortedDict,
        time_start: int,
        delta: int,  # Negative usually
        time_end: int,
        liberate: bool = True,
    ) -> None:

        for t in [k for k in sdict if time_start <= k < time_end]:
            sdict[t] += delta
        i = sdict.bisect_right(time_start)
        if time_start not in sdict:
            sdict[time_start] = sdict.peekitem(i - 1)[1] + delta
        i = sdict.bisect_right(time_end)
        if time_end not in sdict:
            sdict[time_end] = sdict.peekitem(i - 1)[1] - delta

    @staticmethod
    def create_delta_dict(vector: npt.ArrayLike) -> SortedDict:
        v = np.array(vector)
        delta = v[:-1] - v[1:]
        index_non_zero = np.nonzero(delta)[0]
        l = {0: v[0]}
        for j in range(len(index_non_zero)):
            ind = index_non_zero[j]
            l[ind + 1] = -delta[ind]
        l = SortedDict(l)
        return l

    @staticmethod
    def create_absolute_dict(vector: npt.ArrayLike) -> SortedDict:
        v = np.array(vector)
        delta = v[:-1] - v[1:]
        index_non_zero = np.nonzero(delta)[0]
        l = {0: v[0]}
        for j in range(len(index_non_zero)):
            ind = index_non_zero[j]
            l[ind + 1] = v[ind + 1]
        l = SortedDict(l)
        return l

    def generate_schedule_from_permutation_serial_sgs(
        self, solution: RCPSPSolution, rcpsp_problem: RCPSPModel
    ) -> Tuple[Dict[Hashable, Dict[str, int]], bool, Dict[str, SortedDict]]:
        activity_end_times = {}
        unfeasible_non_renewable_resources = False
        new_horizon = rcpsp_problem.horizon
        resource_avail_in_time = deepcopy(self.dict_step_ressource)
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
            current_min_time: Optional[int] = minimum_starting_time[act_id]
            start_time: int
            end_time: int
            valid = False
            while not valid:
                valid = True
                if current_min_time is None:
                    unfeasible_non_renewable_resources = True
                    break
                start_time = current_min_time
                end_time = (
                    current_min_time
                    + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                )
                for res in rcpsp_problem.resources_list:
                    need = rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                        res, 0
                    )
                    if need == 0:
                        continue
                    else:
                        if (
                            self.get_available_from_absolute(
                                sdict=resource_avail_in_time[res], time=current_min_time  # type: ignore
                            )
                            < need
                        ):
                            current_min_time = next(
                                (
                                    k
                                    for k in resource_avail_in_time[res]
                                    if k > start_time
                                    and resource_avail_in_time[res][k] >= need
                                ),
                                None,
                            )
                            valid = False
                            break
                        keys = [
                            k
                            for k in resource_avail_in_time[res]
                            if start_time <= k < end_time
                        ]
                        for k in keys:
                            if resource_avail_in_time[res][k] < need:
                                current_min_time = next(
                                    (
                                        ki
                                        for ki in resource_avail_in_time[res]
                                        if ki > k
                                        and resource_avail_in_time[res][ki] >= need
                                    ),
                                    None,
                                )
                                valid = False
                                break
            if not unfeasible_non_renewable_resources and current_min_time is not None:
                end_t = (
                    current_min_time
                    + rcpsp_problem.mode_details[act_id][modes_dict[act_id]]["duration"]
                )
                for res in resource_avail_in_time:
                    need = rcpsp_problem.mode_details[act_id][modes_dict[act_id]].get(
                        res, 0
                    )
                    if need == 0:
                        continue
                    if res not in rcpsp_problem.non_renewable_resources:
                        self.add_event_delta_in_absolute(
                            resource_avail_in_time[res],
                            time_start=current_min_time,
                            delta=-need,
                            time_end=end_t,
                        )
                    else:
                        self.add_event_delta_in_absolute(
                            resource_avail_in_time[res],
                            time_start=new_horizon,
                            delta=-need,
                            time_end=new_horizon + 2,
                        )
                        if resource_avail_in_time[res][new_horizon] < 0:
                            unfeasible_non_renewable_resources = True
                if unfeasible_non_renewable_resources:
                    break
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
        return rcpsp_schedule, rcpsp_schedule_feasible, resource_avail_in_time
