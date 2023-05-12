#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Tuple

import numba.typed
import numba.types
import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore

logger = logging.getLogger(__name__)

int_array = numba.types.Array(numba.types.int_, 1, "C")


@njit
def sgs_fast(
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    minimum_starting_time_array: npt.NDArray[np.int_],
) -> Tuple[Dict[int, Tuple[int, int]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = minimum_starting_time_array[act]
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors[permutation_task, :], axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        index_id = 0
        found = False
        for i in range(nb_task):
            if pred_links[i] == 0 and done_np[i] == 0:
                act_id = permutation_task[i]
                index_id = i
                found = True
                break
        if not found:
            break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            end_time = current_min_time + duration_array[act_id, modes_array[act_id]]
            for t in range(current_min_time, end_time):
                for res in range(ressource_available.shape[0]):
                    if t < new_horizon:
                        if (
                            resource_avail_in_time[res][t]
                            < consumption_array[act_id, modes_array[act_id], res]
                        ):  # 11
                            valid = False
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        break
                if not valid:
                    break
            if not valid:
                current_min_time += 1
        if not unfeasible_non_renewable_resources:
            end_t = current_min_time + duration_array[act_id, modes_array[act_id]]
            for res in range(ressource_available.shape[0]):
                if ressource_renewable[res]:
                    resource_avail_in_time[res][
                        current_min_time:end_t
                    ] -= consumption_array[act_id, modes_array[act_id], res]
                else:
                    resource_avail_in_time[res][current_min_time:] -= consumption_array[
                        act_id, modes_array[act_id], res
                    ]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            done_np[index_id] = 1
            done += 1
            # for s in range(successors.shape[1]):
            for j in range(nb_task):
                if successors[act_id, permutation_task[j]] == 1:
                    minimum_starting_time[permutation_task[j]] = max(
                        int(minimum_starting_time[permutation_task[j]]),
                        int(activity_end_times[act_id]),
                    )
                    pred_links[j] -= 1
    rcpsp_schedule: Dict[int, Tuple[int, int]] = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = (
            activity_end_times[act_id] - duration_array[act_id, modes_array[act_id]],
            activity_end_times[act_id],
        )
    return rcpsp_schedule, unfeasible_non_renewable_resources


@njit
def sgs_fast_preemptive(
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    preemptive_tag: npt.NDArray[np.bool_],  # array(task)->bool
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    minimum_starting_time_array: npt.NDArray[np.int_],
) -> Tuple[Dict[int, npt.NDArray[np.int_]], Dict[int, npt.NDArray[np.int_]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    starts_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    ends_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = minimum_starting_time_array[act]
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    done_duration = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                break
        current_min_time = int(minimum_starting_time[act_id])
        valid = False
        starts = []
        ends = []
        while not valid:
            valid = True
            reached_t = None
            reached_end = True
            if duration_array[act_id, modes_array[act_id]] == 0:
                starts.append(current_min_time)
                ends.append(current_min_time)
                done_duration[act_id] = 0
            else:
                for t in range(
                    current_min_time,
                    current_min_time
                    + duration_array[act_id, modes_array[act_id]]
                    - done_duration[act_id],
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    for res in range(ressource_available.shape[0]):
                        if t < new_horizon:
                            if (
                                resource_avail_in_time[res][t]
                                < consumption_array[act_id, modes_array[act_id], res]
                            ):
                                reached_end = False
                                break
                        else:
                            unfeasible_non_renewable_resources = True
                            break
                    if reached_end:
                        reached_t = t
                    else:
                        break
                if (
                    reached_t is not None
                    and preemptive_tag[act_id]
                    and (reached_t + 1 - current_min_time >= 1 or reached_end)
                ):
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and not preemptive_tag[act_id] and reached_t is not None:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    current_min_time = (
                        reached_t + 2 if reached_t is not None else current_min_time + 1
                    )
                    if unfeasible_non_renewable_resources:
                        break
        if unfeasible_non_renewable_resources:
            break
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            starts[i] : ends[i]
                        ] -= consumption_array[act_id, modes_array[act_id], res]
                    else:
                        if i == 0:
                            resource_avail_in_time[res][
                                starts[i] :
                            ] -= consumption_array[act_id, modes_array[act_id], res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                            break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(starts, dtype=np.int_)
            ends_dict[act_id] = np.array(ends, dtype=np.int_)
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    return starts_dict, ends_dict, unfeasible_non_renewable_resources


@njit
def sgs_fast_preemptive_some_special_constraints(
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    preemptive_tag: npt.NDArray[np.bool_],  # array(task)->bool
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    start_at_end_plus_offset: npt.NDArray[
        np.int_
    ],  # array(N, 3) -> (task1, task2, offset)
    start_after_nunit: npt.NDArray[np.int_],  # array(N, 3) -> (task1, task2, offset)
    minimum_starting_time_array: npt.NDArray[np.int_],
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
) -> Tuple[Dict[int, npt.NDArray[np.int_]], Dict[int, npt.NDArray[np.int_]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    starts_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    ends_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    nb_task = permutation_task.shape[0]
    for act in range(nb_task):
        minimum_starting_time[act] = minimum_starting_time_array[act]
    start_after_nunit_links = np.zeros(nb_task)
    for task in range(nb_task):
        start_after_nunit_links[task] = np.sum(start_after_nunit[:, 1] == task)
    start_at_end_plus_offset_links = np.zeros(nb_task)
    for task in range(nb_task):
        start_at_end_plus_offset_links[task] = np.sum(
            start_at_end_plus_offset[:, 1] == task
        )

    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    done_duration = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        found = False
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
                and start_after_nunit_links[permutation_task[i]] == 0
                and start_at_end_plus_offset_links[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                found = True
                break
        if not found:
            for i in range(nb_task):
                if (
                    pred_links[permutation_task[i]] == 0
                    and done_np[permutation_task[i]] == 0
                ):
                    act_id = permutation_task[i]
                    break
        current_min_time = int(minimum_starting_time[act_id])
        valid = False
        starts = []
        ends = []
        while not valid:
            valid = True
            reached_t = None
            reached_end = True
            if duration_array[act_id, modes_array[act_id]] == 0:
                starts.append(current_min_time)
                ends.append(current_min_time)
                done_duration[act_id] = 0
            else:
                for t in range(
                    current_min_time,
                    current_min_time
                    + duration_array[act_id, modes_array[act_id]]
                    - done_duration[act_id],
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    for res in range(ressource_available.shape[0]):
                        if t < new_horizon:
                            if (
                                resource_avail_in_time[res][t]
                                < consumption_array[act_id, modes_array[act_id], res]
                            ):
                                reached_end = False
                                break
                        else:
                            unfeasible_non_renewable_resources = True
                            break
                    if reached_end:
                        reached_t = t
                    else:
                        break
                if (
                    reached_t is not None
                    and preemptive_tag[act_id]
                    and (
                        reached_t + 1 - current_min_time
                        >= duration_array[act_id, modes_array[act_id]] / 8
                        or reached_end
                    )
                ):
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and not preemptive_tag[act_id] and reached_t is not None:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    current_min_time = (
                        reached_t + 2 if reached_t is not None else current_min_time + 1
                    )
                    if unfeasible_non_renewable_resources:
                        break
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            starts[i] : ends[i]
                        ] -= consumption_array[act_id, modes_array[act_id], res]
                    else:
                        if i == 0:
                            resource_avail_in_time[res][
                                starts[i] :
                            ] -= consumption_array[act_id, modes_array[act_id], res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                            break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(starts, dtype=np.int_)
            ends_dict[act_id] = np.array(ends, dtype=np.int_)
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1

            for t in range(start_after_nunit.shape[0]):
                if start_after_nunit[t, 0] == act_id:
                    task = start_after_nunit[t, 1]
                    off = start_after_nunit[t, 2]
                    minimum_starting_time[task] = max(
                        int(minimum_starting_time[task]), starts_dict[act_id][0] + off
                    )
                    start_after_nunit_links[task] -= 1
            for t in range(start_at_end_plus_offset.shape[0]):
                if start_at_end_plus_offset[t, 0] == act_id:
                    task = start_at_end_plus_offset[t, 1]
                    off = start_at_end_plus_offset[t, 2]
                    minimum_starting_time[task] = max(
                        int(minimum_starting_time[task]),
                        activity_end_times[act_id] + off,
                    )
                    start_at_end_plus_offset_links[task] -= 1
    return starts_dict, ends_dict, unfeasible_non_renewable_resources


@njit
def sgs_fast_preemptive_minduration(
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    preemptive_tag: npt.NDArray[np.bool_],  # array(task)->bool
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    min_duration_preemptive_bool: npt.NDArray[np.bool_],
    min_duration_preemptive: npt.NDArray[np.int_],
) -> Tuple[Dict[int, npt.NDArray[np.int_]], Dict[int, npt.NDArray[np.int_]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    starts_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    ends_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = 0
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    done_duration = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                break
        current_min_time = int(minimum_starting_time[act_id])
        valid = False
        starts = []
        ends = []
        while not valid:
            valid = True
            reached_t = None
            reached_end = True
            if duration_array[act_id, modes_array[act_id]] == 0:
                starts.append(current_min_time)
                ends.append(current_min_time)
                done_duration[act_id] = 0
            else:
                for t in range(
                    current_min_time,
                    current_min_time
                    + duration_array[act_id, modes_array[act_id]]
                    - done_duration[act_id],
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    for res in range(ressource_available.shape[0]):
                        if t < new_horizon:
                            if (
                                resource_avail_in_time[res][t]
                                < consumption_array[act_id, modes_array[act_id], res]
                            ):
                                reached_end = False
                                break
                        else:
                            unfeasible_non_renewable_resources = True
                            break
                    if reached_end:
                        reached_t = t
                    else:
                        break
                if reached_t is not None and preemptive_tag[act_id]:
                    if (
                        not reached_end
                        and min_duration_preemptive_bool[act_id]
                        and (
                            min_duration_preemptive[act_id]
                            > (reached_t + 1 - current_min_time)
                            or (
                                duration_array[act_id, modes_array[act_id]]
                                - done_duration[act_id]
                                < min_duration_preemptive[act_id]
                            )
                        )
                    ):
                        pass
                    else:
                        starts.append(current_min_time)
                        ends.append(reached_t + 1)
                        done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and not preemptive_tag[act_id] and reached_t is not None:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    current_min_time = (
                        reached_t + 2 if reached_t is not None else current_min_time + 1
                    )
                    if unfeasible_non_renewable_resources:
                        break
        if unfeasible_non_renewable_resources:
            break
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            starts[i] : ends[i]
                        ] -= consumption_array[act_id, modes_array[act_id], res]
                    else:
                        if i == 0:
                            resource_avail_in_time[res][
                                starts[i] :
                            ] -= consumption_array[act_id, modes_array[act_id], res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                            break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(starts, dtype=np.int_)
            ends_dict[act_id] = np.array(ends, dtype=np.int_)
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    return starts_dict, ends_dict, unfeasible_non_renewable_resources


@njit
def sgs_fast_partial_schedule(
    current_time: int,
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    completed_task_indicator: npt.NDArray[np.int_],
    completed_task_times: npt.NDArray[np.int_],
    scheduled_task: npt.NDArray[np.int_],
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    minimum_starting_time_array: npt.NDArray[np.int_],
) -> Tuple[Dict[int, Tuple[int, int]], bool]:
    activity_end_times: Dict[int, int] = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = max(minimum_starting_time_array[act], current_time)
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros(nb_task, dtype=np.int_)
    for t in range(nb_task):
        if scheduled_task[t] != -1:
            activity_end_times[t] = (
                scheduled_task[t] + duration_array[t, modes_array[t]]
            )
            for res in range(ressource_available.shape[0]):
                if ressource_renewable[res]:
                    resource_avail_in_time[res][
                        scheduled_task[t] : activity_end_times[t]
                    ] -= consumption_array[t, modes_array[t], res]
                else:
                    resource_avail_in_time[res][
                        scheduled_task[t] :
                    ] -= consumption_array[t, modes_array[t], res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            if unfeasible_non_renewable_resources:
                break
            for s in range(successors.shape[1]):
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[t])
                    )
                    pred_links[s] -= 1
            done += 1
            done_np[t] = 1
        if completed_task_indicator[t] == 1:
            done += 1
            done_np[t] = 1
            activity_end_times[t] = completed_task_times[t]
            for s in range(successors.shape[1]):
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[t])
                    )
                    pred_links[s] -= 1

    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            end_time = current_min_time + duration_array[act_id, modes_array[act_id]]
            for t in range(current_min_time, end_time):
                for res in range(ressource_available.shape[0]):
                    if t < new_horizon:
                        if (
                            resource_avail_in_time[res][t]
                            < consumption_array[act_id, modes_array[act_id], res]
                        ):
                            valid = False
                    else:
                        unfeasible_non_renewable_resources = True
                        break
            if not valid:
                current_min_time += 1
        if unfeasible_non_renewable_resources:
            break
        if not unfeasible_non_renewable_resources:
            end_t = current_min_time + duration_array[act_id, modes_array[act_id]]
            for res in range(ressource_available.shape[0]):
                if ressource_renewable[res]:
                    resource_avail_in_time[res][
                        current_min_time:end_t
                    ] -= consumption_array[act_id, modes_array[act_id], res]
                else:
                    resource_avail_in_time[res][current_min_time:] -= consumption_array[
                        act_id, modes_array[act_id], res
                    ]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    rcpsp_schedule: Dict[int, Tuple[int, int]] = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = (
            activity_end_times[act_id] - duration_array[act_id, modes_array[act_id]],
            activity_end_times[act_id],
        )

    return rcpsp_schedule, unfeasible_non_renewable_resources


@njit
def sgs_fast_partial_schedule_incomplete_permutation_tasks(
    current_time: int,
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    completed_task_indicator: npt.NDArray[np.int_],
    completed_task_times: npt.NDArray[np.int_],
    scheduled_task: npt.NDArray[np.int_],
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    minimum_starting_time_array: npt.NDArray[np.int_],
) -> Tuple[Dict[int, Tuple[int, int]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[permutation_task[act]] = max(
            current_time, minimum_starting_time_array[act]
        )
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors[permutation_task, :], axis=1)
    done_np = np.zeros((predecessors.shape[0]), dtype=np.int_)
    for t in range(nb_task):
        activity_end_times[t] = 0
    for t in range(nb_task):
        if scheduled_task[t] != -1:
            activity_end_times[t] = (
                scheduled_task[t] + duration_array[t, modes_array[t]]
            )
            for res in range(ressource_available.shape[0]):
                if ressource_renewable[res]:
                    resource_avail_in_time[res][
                        scheduled_task[t] : activity_end_times[t]
                    ] -= consumption_array[t, modes_array[t], res]
                else:
                    resource_avail_in_time[res][
                        scheduled_task[t] :
                    ] -= consumption_array[t, modes_array[t], res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            if unfeasible_non_renewable_resources:
                break
            for j in range(nb_task):
                s = permutation_task[j]
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[t])
                    )
                    pred_links[j] -= 1
            done += 1
            done_np[t] = 1
        if completed_task_indicator[t] == 1:
            done += 1
            done_np[t] = 1
            activity_end_times[t] = completed_task_times[t]
            for j in range(nb_task):
                s = permutation_task[j]
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[t])
                    )
                    pred_links[j] -= 1

    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        found = False
        for i in range(nb_task):
            if pred_links[i] == 0 and done_np[permutation_task[i]] == 0:
                act_id = permutation_task[i]
                found = True
                break
        if not found:
            break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        while not valid:
            valid = True
            end_time = current_min_time + duration_array[act_id, modes_array[act_id]]
            for t in range(current_min_time, end_time):
                for res in range(ressource_available.shape[0]):
                    if t < new_horizon:
                        if (
                            resource_avail_in_time[res][t]
                            < consumption_array[act_id, modes_array[act_id], res]
                        ):
                            valid = False
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        break
            if not valid:
                current_min_time += 1
        if unfeasible_non_renewable_resources:
            break
        if not unfeasible_non_renewable_resources:
            end_t = current_min_time + duration_array[act_id, modes_array[act_id]]
            for res in range(ressource_available.shape[0]):
                if ressource_renewable[res]:
                    resource_avail_in_time[res][
                        current_min_time:end_t
                    ] -= consumption_array[act_id, modes_array[act_id], res]
                else:
                    resource_avail_in_time[res][current_min_time:] -= consumption_array[
                        act_id, modes_array[act_id], res
                    ]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            done_np[act_id] = 1
            done += 1
            for j in range(nb_task):
                if successors[act_id, permutation_task[j]] == 1:
                    minimum_starting_time[permutation_task[j]] = max(
                        int(minimum_starting_time[permutation_task[j]]),
                        int(activity_end_times[act_id]),
                    )
                    pred_links[j] -= 1
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = (
            activity_end_times[act_id] - duration_array[act_id, modes_array[act_id]],
            activity_end_times[act_id],
        )

    return rcpsp_schedule, unfeasible_non_renewable_resources


@njit
def sgs_fast_partial_schedule_preemptive(
    current_time: int,
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    completed_task_indicator: npt.NDArray[np.int_],
    partial_schedule_starts: npt.NDArray[np.int_],  # array(task, 5)
    partial_schedule_ends: npt.NDArray[np.int_],  # array(task, 5)
    preemptive_tag: npt.NDArray[np.bool_],  # array(task)->bool
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    minimum_starting_time_array: npt.NDArray[np.int_],
) -> Tuple[Dict[int, npt.NDArray[np.int_]], Dict[int, npt.NDArray[np.int_]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = max(minimum_starting_time_array[act], current_time)
    starts_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    ends_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros(nb_task, dtype=np.int_)
    done_duration = np.zeros(nb_task, dtype=np.int_)
    for t in range(nb_task):
        end = None
        for i in range(partial_schedule_starts.shape[1]):
            if partial_schedule_starts[t, i] != -1:
                end = partial_schedule_ends[t, i]
                done_duration[t] += (
                    partial_schedule_ends[t, i] - partial_schedule_starts[t, i]
                )
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            partial_schedule_starts[t, i] : partial_schedule_ends[t, i]
                        ] -= consumption_array[t, modes_array[t], res]
                    else:
                        if done_duration[t] == duration_array[t, modes_array[t]]:
                            resource_avail_in_time[res][
                                partial_schedule_starts[t, i] :
                            ] -= consumption_array[t, modes_array[t], res]
                            if resource_avail_in_time[res][-1] < 0:
                                unfeasible_non_renewable_resources = True
                                break
                if unfeasible_non_renewable_resources:
                    break
        if (
            done_duration[t] == duration_array[t, modes_array[t]]
            and duration_array[t, modes_array[t]] >= 1
        ):
            completed_task_indicator[t] = 1
        if end is not None:
            for s in range(successors.shape[1]):
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(int(minimum_starting_time[s]), end)
        if completed_task_indicator[t] == 1:
            done += 1
            done_np[t] = 1
            activity_end_times[t] = end
            starts_dict[t] = np.array(
                [k for k in partial_schedule_starts[t, :] if k != -1], dtype=np.int_
            )
            ends_dict[t] = np.array(
                [k for k in partial_schedule_ends[t, :] if k != -1], dtype=np.int_
            )
            for s in range(successors.shape[1]):
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[t])
                    )
                    pred_links[s] -= 1
    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        starts = []
        ends = []
        while not valid:
            valid = True
            reached_t = None
            reached_end = True
            if duration_array[act_id, modes_array[act_id]] == 0:
                starts.append(current_min_time)
                ends.append(current_min_time)
                done_duration[act_id] = 0
            else:
                for t in range(
                    current_min_time,
                    current_min_time
                    + duration_array[act_id, modes_array[act_id]]
                    - done_duration[act_id],
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    for res in range(ressource_available.shape[0]):
                        if t < new_horizon:
                            if (
                                resource_avail_in_time[res][t]
                                < consumption_array[act_id, modes_array[act_id], res]
                            ):
                                reached_end = False
                                break
                        else:
                            unfeasible_non_renewable_resources = True
                            break
                    if reached_end:
                        reached_t = t
                    else:
                        break
                if reached_t is not None and preemptive_tag[act_id]:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and not preemptive_tag[act_id] and reached_t is not None:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    current_min_time = (
                        reached_t + 2 if reached_t is not None else current_min_time + 1
                    )
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            starts[i] : ends[i]
                        ] -= consumption_array[act_id, modes_array[act_id], res]
                    else:
                        if i == 0:
                            resource_avail_in_time[res][
                                starts[i] :
                            ] -= consumption_array[act_id, modes_array[act_id], res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                            break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(
                [k for k in partial_schedule_starts[act_id, :] if k != -1] + starts,
                dtype=np.int_,
            )
            ends_dict[act_id] = np.array(
                [k for k in partial_schedule_ends[act_id, :] if k != -1] + ends,
                dtype=np.int_,
            )
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    return starts_dict, ends_dict, unfeasible_non_renewable_resources


@njit
def sgs_fast_partial_schedule_preemptive_minduration(
    current_time: int,
    permutation_task: npt.NDArray[np.int_],  # permutation_task=array(task)->task index
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    completed_task_indicator: npt.NDArray[np.int_],
    partial_schedule_starts: npt.NDArray[np.int_],  # array(task, 5)
    partial_schedule_ends: npt.NDArray[np.int_],  # array(task, 5)
    preemptive_tag: npt.NDArray[np.bool_],  # array(task)->bool
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    duration_array: npt.NDArray[np.int_],
    predecessors: npt.NDArray[np.int_],  # array(task, task) -> bool
    successors: npt.NDArray[np.int_],  # array(task, task)->bool
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
    min_duration_preemptive_bool: npt.NDArray[np.bool_],
    min_duration_preemptive: npt.NDArray[np.int_],
) -> Tuple[Dict[int, npt.NDArray[np.int_]], Dict[int, npt.NDArray[np.int_]], bool]:
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = current_time
    starts_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    ends_dict = numba.typed.Dict.empty(
        key_type=numba.types.intp,
        value_type=int_array,
    )
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros(nb_task, dtype=np.int_)
    done_duration = np.zeros(nb_task, dtype=np.int_)
    for t in range(nb_task):
        end = None
        for i in range(partial_schedule_starts.shape[1]):
            if partial_schedule_starts[t, i] != -1:
                end = partial_schedule_ends[t, i]
                done_duration[t] += (
                    partial_schedule_ends[t, i] - partial_schedule_starts[t, i]
                )
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            partial_schedule_starts[t, i] : partial_schedule_ends[t, i]
                        ] -= consumption_array[t, modes_array[t], res]
                    else:
                        if done_duration[t] == duration_array[t, modes_array[t]]:
                            resource_avail_in_time[res][
                                partial_schedule_starts[t, i] :
                            ] -= consumption_array[t, modes_array[t], res]
                            if resource_avail_in_time[res][-1] < 0:
                                unfeasible_non_renewable_resources = True
                                break
                if unfeasible_non_renewable_resources:
                    break
        if (
            done_duration[t] == duration_array[t, modes_array[t]]
            and duration_array[t, modes_array[t]] >= 1
        ):
            completed_task_indicator[t] = 1
        if end is not None:
            for s in range(successors.shape[1]):
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(int(minimum_starting_time[s]), end)
        if completed_task_indicator[t] == 1:
            done += 1
            done_np[t] = 1
            activity_end_times[t] = end
            starts_dict[t] = np.array(
                [k for k in partial_schedule_starts[t, :] if k != -1], dtype=np.int_
            )
            ends_dict[t] = np.array(
                [k for k in partial_schedule_ends[t, :] if k != -1], dtype=np.int_
            )
            for s in range(successors.shape[1]):
                if successors[t, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[t])
                    )
                    pred_links[s] -= 1

    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                break
        current_min_time = minimum_starting_time[act_id]
        valid = False
        starts = []
        ends = []
        while not valid:
            valid = True
            reached_t = None
            reached_end = True
            if duration_array[act_id, modes_array[act_id]] == 0:
                starts.append(current_min_time)
                ends.append(current_min_time)
                done_duration[act_id] = 0
            else:
                for t in range(
                    current_min_time,
                    current_min_time
                    + duration_array[act_id, modes_array[act_id]]
                    - done_duration[act_id],
                ):
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        break
                    for res in range(ressource_available.shape[0]):
                        if t < new_horizon:
                            if (
                                resource_avail_in_time[res][t]
                                < consumption_array[act_id, modes_array[act_id], res]
                            ):
                                reached_end = False
                                break
                        else:
                            unfeasible_non_renewable_resources = True
                            break
                    if reached_end:
                        reached_t = t
                    else:
                        break
                if reached_t is not None and preemptive_tag[act_id]:
                    if (
                        not reached_end
                        and min_duration_preemptive_bool[act_id]
                        and (
                            min_duration_preemptive[act_id]
                            > (reached_t + 1 - current_min_time)
                            or (
                                duration_array[act_id, modes_array[act_id]]
                                - done_duration[act_id]
                                < min_duration_preemptive[act_id]
                            )
                        )
                    ):
                        logger.debug("passed")
                        pass
                    else:
                        starts.append(current_min_time)
                        ends.append(reached_t + 1)
                        done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and not preemptive_tag[act_id] and reached_t is not None:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    current_min_time = (
                        reached_t + 2 if reached_t is not None else current_min_time + 1
                    )
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        resource_avail_in_time[res][
                            starts[i] : ends[i]
                        ] -= consumption_array[act_id, modes_array[act_id], res]
                    else:
                        if i == 0:
                            resource_avail_in_time[res][
                                starts[i] :
                            ] -= consumption_array[act_id, modes_array[act_id], res]
                        if resource_avail_in_time[res][-1] < 0:
                            unfeasible_non_renewable_resources = True
                            break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t

            starts_dict[act_id] = np.array(
                [k for k in partial_schedule_starts[act_id, :] if k != -1] + starts,
                dtype=np.int_,
            )
            ends_dict[act_id] = np.array(
                [k for k in partial_schedule_ends[act_id, :] if k != -1] + ends,
                dtype=np.int_,
            )
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    return starts_dict, ends_dict, unfeasible_non_renewable_resources


@njit
def compute_mean_ressource(
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    start_array: npt.NDArray[np.int_],
    end_array: npt.NDArray[np.int_],
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
) -> float:
    new_horizon = horizon
    resource_avail_in_time = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index, : new_horizon + 1]
        )
    nb_task = start_array.shape[0]
    for t in range(nb_task):
        start_time = start_array[t]
        end_time = end_array[t]
        for res_index in resource_avail_in_time:
            if ressource_renewable[res_index]:
                resource_avail_in_time[res_index][
                    start_time:end_time
                ] -= consumption_array[t, modes_array[t], res_index]
            else:
                resource_avail_in_time[res_index][start_time:] -= consumption_array[
                    t, modes_array[t], res_index
                ]
    mean_avail = {}
    for res in resource_avail_in_time:
        mean_avail[res] = np.mean(resource_avail_in_time[res])
    mean_resource_reserve = np.mean(
        np.array(
            [
                mean_avail[res_index] / np.max(ressource_available[res_index, :])
                for res_index in range(ressource_available.shape[0])
            ]
        )
    )
    return float(mean_resource_reserve)


@njit
def compute_ressource_consumption(
    modes_array: npt.NDArray[np.int_],  # modes=array(task)->0, 1...
    consumption_array: npt.NDArray[
        np.int_
    ],  # consumption_array=array3D(task, mode, res),
    start_array: npt.NDArray[np.int_],
    end_array: npt.NDArray[np.int_],
    horizon: int,
    ressource_available: npt.NDArray[np.int_],
    ressource_renewable: npt.NDArray[np.bool_],
) -> Dict[int, npt.NDArray[np.int_]]:
    new_horizon = horizon
    resource_avail_in_time: Dict[int, npt.NDArray[np.int_]] = {}
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.zeros(new_horizon + 1, dtype=np.int_)
    nb_task = start_array.shape[0]
    for t in range(nb_task):
        start_time = start_array[t]
        end_time = end_array[t]
        for res_index in resource_avail_in_time:
            if ressource_renewable[res_index]:
                resource_avail_in_time[res_index][
                    start_time:end_time
                ] += consumption_array[t, modes_array[t], res_index]
            else:
                resource_avail_in_time[res_index][start_time:] += consumption_array[
                    t, modes_array[t], res_index
                ]
    return resource_avail_in_time
