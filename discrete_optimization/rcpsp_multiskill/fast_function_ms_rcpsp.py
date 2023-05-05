#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numba.typed
import numba.types
import numpy as np
from numba import njit

int_array = numba.types.Array(numba.types.int_, 1, "C")


@njit
def sgs_fast_ms(
    permutation_task,  # permutation_task=array(task)->task index
    priority_worker_per_task,  # array(task, worker)
    modes_array,  # modes=array(task)->0, 1...
    consumption_array,  # consumption_array=array3D(task, mode, res),
    skills_needs,  # array(task, mode, skill)
    duration_array,  # array(task, mode) -> d
    predecessors,  # array(task, task) -> bool
    successors,  # array(task, task)->bool
    horizon,  # int
    ressource_available,  # array(res, times)->int
    ressource_renewable,  # array(res)->bool
    worker_available,  # array(workers, int)->bool
    worker_skills,  # array(workers, skills)->int
    minimum_starting_time_array,
    one_unit_per_task: bool = True,
):
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    worker_avail_in_time = np.copy(worker_available)
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = minimum_starting_time_array[act]
    skills_usage = {}
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
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
                            current_min_time = t
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        break
                if not valid:
                    break
            if valid:
                skills = skills_needs[act_id, modes_array[act_id]]
                if end_time - current_min_time > 0 and np.max(skills) > 0:
                    indexes_present_worker = np.array(
                        [
                            priority_worker_per_task[act_id, i]
                            for i in range(worker_avail_in_time.shape[0])
                            if np.min(
                                worker_avail_in_time[
                                    priority_worker_per_task[act_id, i],
                                    current_min_time:end_time,
                                ]
                            )
                            > 0
                        ]
                    )
                    if one_unit_per_task:
                        indexes_present_worker = np.array(
                            [
                                i
                                for i in indexes_present_worker
                                if np.all(worker_skills[i, :] >= skills)
                            ]
                        )
                    if len(indexes_present_worker) > 0:
                        available_skills_t = np.sum(
                            worker_skills[indexes_present_worker, :], axis=0
                        )
                        if (
                            np.min(
                                available_skills_t
                                - skills_needs[act_id, modes_array[act_id]]
                            )
                            < 0
                        ):
                            valid = False
                    else:
                        valid = False
            if not valid:
                current_min_time += 1
                if current_min_time > horizon:
                    unfeasible_non_renewable_resources = True
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
                    if resource_avail_in_time[res][horizon] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            skills = skills_needs[act_id, modes_array[act_id]]
            skills_done = np.zeros((skills.shape[0]))
            skills_usage_i = np.zeros(
                (worker_avail_in_time.shape[0], skills_needs.shape[2])
            )
            used = [0]
            if np.max(skills) > 0:
                while True:
                    score = [
                        np.sum(worker_skills[p, :] * ((skills - skills_done) > 0))
                        for p in indexes_present_worker
                    ]
                    sort = [
                        indexes_present_worker[p]
                        for p in np.argsort(-np.array(score))
                        if indexes_present_worker[p] not in used[1:]
                    ]
                    j = sort[0]
                    nz = np.nonzero(worker_skills[j, :] * skills > 0)[0]
                    if len(nz) > 0:
                        for nnz in nz:
                            skills_usage_i[j, nnz] = 1
                            skills_done[nnz] += worker_skills[j, nnz]
                        worker_avail_in_time[j, current_min_time:end_t] = 0
                        used += [j]
                    if np.all(skills_done >= skills):
                        break

            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            done_np[act_id] = 1
            skills_usage[act_id] = skills_usage_i
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = (
            activity_end_times[act_id] - duration_array[act_id, modes_array[act_id]],
            activity_end_times[act_id],
        )
    return rcpsp_schedule, skills_usage, unfeasible_non_renewable_resources


@njit
def sgs_fast_ms_partial_schedule(
    permutation_task,  # permutation_task=array(task)->task index
    priority_worker_per_task,  # array(task, worker)
    modes_array,  # modes=array(task)->0, 1...
    scheduled_task_indicator,  # array(task)->bool
    scheduled_start_task_times,  # array(task)->int
    scheduled_end_task_times,  # array(task)->int
    worker_used,  # array(task, worker)->bool
    current_time,  # int
    consumption_array,  # consumption_array=array3D(task, mode, res),
    skills_needs,  # array(task, mode, skill)
    duration_array,  # array(task, mode) -> d
    predecessors,  # array(task, task) -> bool
    successors,  # array(task, task)->bool
    horizon,  # int
    ressource_available,  # array(res, times)->int
    ressource_renewable,  # array(res)->bool
    worker_available,  # array(workers, int)->bool
    worker_skills,  # array(workers, skills)->int
    minimum_starting_time_array,
    one_unit_per_task: bool = True,
):
    activity_end_times = {}
    unfeasible_non_renewable_resources = False
    new_horizon = horizon
    resource_avail_in_time = {}
    worker_avail_in_time = np.copy(worker_available)
    for index in range(ressource_available.shape[0]):
        resource_avail_in_time[index] = np.copy(
            ressource_available[index][: new_horizon + 1]
        )
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = max(current_time, minimum_starting_time_array[act])
    skills_usage = {}
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    rcpsp_schedule = {}
    for t in range(nb_task):
        if scheduled_task_indicator[t] != 0:
            rcpsp_schedule[t] = (
                scheduled_start_task_times[t],
                scheduled_end_task_times[t],
            )
            for res in range(ressource_available.shape[0]):
                if ressource_renewable[res]:
                    resource_avail_in_time[res][
                        int(scheduled_start_task_times[t]) : int(
                            scheduled_end_task_times[t]
                        )
                    ] -= consumption_array[t, modes_array[t], res]
                else:
                    resource_avail_in_time[res][
                        scheduled_end_task_times[t] :
                    ] -= consumption_array[t, modes_array[t], res]
                    if resource_avail_in_time[res][-1] < 0:
                        unfeasible_non_renewable_resources = True
                        break
            activity_end_times[t] = scheduled_end_task_times[t]
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
            skills_act_id = np.zeros(
                (worker_avail_in_time.shape[0], skills_needs.shape[2])
            )
            # nz[0] : nb_preemptive, nz[1] workers
            # Assign workers :
            skills = skills_needs[t, modes_array[t]]
            if np.max(skills) > 0:
                wused = worker_used[t, :]
                nz = np.nonzero(wused)
                skills_done = np.zeros((skills.shape[0]))
                for w in nz[0]:
                    skills_worker = np.nonzero(worker_skills[w, :] * skills > 0)[0]
                    if len(skills_worker) > 0:
                        for s in skills_worker:
                            skills_act_id[w, s] = 1
                            skills_done[s] += worker_skills[w, s]
                        worker_avail_in_time[
                            w,
                            int(scheduled_start_task_times[t]) : int(
                                scheduled_end_task_times[t]
                            ),
                        ] = 0
                    if np.all(skills_done > skills):
                        break
            skills_usage[t] = skills_act_id
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
                            current_min_time = t
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        break
                if not valid:
                    break
            if valid:
                skills = skills_needs[act_id, modes_array[act_id]]
                if end_time - current_min_time > 0 and np.max(skills) > 0:
                    indexes_present_worker = np.array(
                        [
                            priority_worker_per_task[act_id, i]
                            for i in range(worker_avail_in_time.shape[0])
                            if np.min(
                                worker_avail_in_time[
                                    priority_worker_per_task[act_id, i],
                                    current_min_time:end_time,
                                ]
                            )
                            > 0
                        ]
                    )
                    if one_unit_per_task:
                        indexes_present_worker = np.array(
                            [
                                i
                                for i in indexes_present_worker
                                if np.all(worker_skills[i, :] >= skills)
                            ]
                        )
                    if len(indexes_present_worker) > 0:
                        available_skills_t = np.sum(
                            worker_skills[indexes_present_worker, :], axis=0
                        )
                        if (
                            np.min(
                                available_skills_t
                                - skills_needs[act_id, modes_array[act_id]]
                            )
                            < 0
                        ):
                            valid = False
                    else:
                        valid = False
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
            skills = skills_needs[act_id, modes_array[act_id]]
            skills_done = np.zeros((skills.shape[0]))
            skills_usage_i = np.zeros(
                (worker_avail_in_time.shape[0], skills_needs.shape[2])
            )
            if np.max(skills) > 0:
                for employee in indexes_present_worker:
                    nz = np.nonzero(worker_skills[employee, :] * skills > 0)[0]
                    if len(nz) > 0:
                        for nnz in nz:
                            skills_usage_i[employee, nnz] = 1
                            skills_done[nnz] += worker_skills[employee, nnz]
                        worker_avail_in_time[employee, current_min_time:end_t] = 0
                    if np.all(skills_done >= skills):
                        break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            done_np[act_id] = 1
            skills_usage[act_id] = skills_usage_i
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = (
            activity_end_times[act_id] - duration_array[act_id, modes_array[act_id]],
            activity_end_times[act_id],
        )
    return rcpsp_schedule, skills_usage, unfeasible_non_renewable_resources


@njit
def sgs_fast_ms_preemptive(
    permutation_task,  # permutation_task=array(task)->task index
    priority_worker_per_task,  # array(task, worker)
    modes_array,  # permutation_task=array(task)->task index
    consumption_array,  # modes = array(task) -> 0, 1...
    # consumption_array=array3D(task, mode, res),
    skills_needs,
    duration_array,
    preemptive_tag,  # array(task)->bool
    predecessors,  # array(task, task) -> bool
    successors,  # array(task, task)->bool
    horizon,
    ressource_available,
    ressource_renewable,
    worker_available,  # array(workers, int)->bool
    worker_skills,  # array(workers, skills)->int
    minimum_starting_time_array,
    is_releasable,  # [task, mode, res]-> bool
    one_unit_per_task: bool = True,
    consider_partial_preemptive: bool = False,
    strictly_disjunctive_subtasks: bool = False,
):  # array(res)->bool
    activity_end_times = {}
    unfeasible_sched = False
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
    worker_avail_in_time = np.copy(worker_available)
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = minimum_starting_time_array[act]
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    done_duration = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    skills_usage = {}
    worker_usage = np.zeros((worker_skills.shape[0]), dtype=np.int_)
    while done < nb_task and not unfeasible_sched:
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
        workers_av = []
        skills = skills_needs[act_id, modes_array[act_id]]
        need_worker = np.max(skills) > 0
        unfeasible_non_renewable_resources = False
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
                    if need_worker:
                        if t == current_min_time:
                            indexes_present_worker_t = np.array(
                                [
                                    priority_worker_per_task[act_id, i]
                                    for i in range(priority_worker_per_task.shape[1])
                                    if worker_avail_in_time[
                                        priority_worker_per_task[act_id, i],
                                        current_min_time,
                                    ]
                                    > 0
                                ]
                            )
                            if one_unit_per_task:
                                indexes_present_worker_t = np.array(
                                    [
                                        i
                                        for i in indexes_present_worker_t
                                        if np.all(
                                            worker_skills[i, :]
                                            >= skills_needs[act_id, modes_array[act_id]]
                                        )
                                    ]
                                )
                            if len(indexes_present_worker_t) > 0:
                                available_skills_t = np.sum(
                                    worker_skills[indexes_present_worker_t, :], axis=0
                                )
                                if (
                                    np.min(
                                        available_skills_t
                                        - skills_needs[act_id, modes_array[act_id]]
                                    )
                                    < 0
                                ):
                                    reached_end = False
                                    break
                            else:
                                reached_end = False
                                break
                        else:
                            indexes_present_worker_next_t = np.array(
                                [
                                    i
                                    for i in indexes_present_worker_t
                                    if worker_avail_in_time[i, t] > 0
                                ]
                            )
                            if one_unit_per_task:
                                indexes_present_worker_next_t = np.array(
                                    [
                                        i
                                        for i in indexes_present_worker_next_t
                                        if np.all(
                                            worker_skills[i, :]
                                            >= skills_needs[act_id, modes_array[act_id]]
                                        )
                                    ]
                                )
                            if len(indexes_present_worker_next_t) > 0:
                                available_skills_tt = np.sum(
                                    worker_skills[indexes_present_worker_next_t, :],
                                    axis=0,
                                )
                                if (
                                    np.min(
                                        available_skills_tt
                                        - skills_needs[act_id, modes_array[act_id]]
                                    )
                                    < 0
                                ):
                                    reached_end = False
                                    break
                                else:
                                    indexes_present_worker_t = (
                                        indexes_present_worker_next_t
                                    )
                            else:
                                reached_end = False
                                break
                    if t >= new_horizon:
                        reached_end = False
                        unfeasible_non_renewable_resources = True
                        unfeasible_sched = True
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
                            unfeasible_sched = True
                            break
                    if reached_end:
                        reached_t = t
                    else:
                        break
                if (
                    reached_t is not None
                    and preemptive_tag[act_id] == 1
                    and (
                        True
                        or reached_t + 1 - current_min_time
                        >= duration_array[act_id, modes_array[act_id]] / 8
                        or reached_end
                    )
                ):
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    workers_av.append(indexes_present_worker_t)
                    done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and preemptive_tag[act_id] == 0:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    workers_av.append(indexes_present_worker_t)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )

                if not valid:
                    if strictly_disjunctive_subtasks:
                        current_min_time = (
                            reached_t + 2
                            if reached_t is not None
                            else current_min_time + 1
                        )
                    else:
                        current_min_time = (
                            reached_t + 1
                            if reached_t is not None
                            else current_min_time + 1
                        )
                    if current_min_time >= new_horizon:
                        return None, None, None, True

                if consider_partial_preemptive and len(starts) > 0:
                    to_break = False
                    for res in range(ressource_available.shape[0]):
                        if consumption_array[act_id, modes_array[act_id], res] > 0:
                            if is_releasable[act_id, modes_array[act_id], res] == 0:
                                if (
                                    min(
                                        [
                                            resource_avail_in_time[res][p]
                                            for p in range(starts[0], ends[-1])
                                            if ressource_available[res][p] > 0
                                        ]
                                    )
                                    < consumption_array[
                                        act_id, modes_array[act_id], res
                                    ]
                                ):
                                    indexes = [
                                        p
                                        for p in range(starts[0], ends[-1])
                                        if ressource_available[res][p] > 0
                                        and resource_avail_in_time[res][p]
                                        < consumption_array[
                                            act_id, modes_array[act_id], res
                                        ]
                                    ]
                                    minimum_starting_time[act_id] = indexes[-1] + 1
                                    done_duration[
                                        act_id
                                    ] = 0  # a bit conservative here.
                                    to_break = True
                                    unfeasible_non_renewable_resources = True
                                    break
                    if to_break:
                        break
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            skills_act_id = np.zeros(
                (len(starts), worker_avail_in_time.shape[0], skills_needs.shape[2])
            )
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        if (
                            consider_partial_preemptive
                            and is_releasable[act_id, modes_array[act_id], res] == 0
                        ):
                            if i == 0:
                                for p in range(starts[i], ends[-1]):
                                    if ressource_available[res][p] > 0:
                                        resource_avail_in_time[res][
                                            p
                                        ] -= consumption_array[
                                            act_id, modes_array[act_id], res
                                        ]
                                if (
                                    np.min(
                                        resource_avail_in_time[res][
                                            starts[i] : ends[-1]
                                        ]
                                    )
                                    < 0
                                ):
                                    unfeasible_non_renewable_resources = True
                                    unfeasible_sched = True
                                    break
                        else:
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
                skills = skills_needs[act_id, modes_array[act_id]]
                skills_done = np.zeros((skills.shape[0]))
                if need_worker and duration_array[act_id, modes_array[act_id]] > 0:
                    used = [0]
                    while True:
                        score = [
                            np.sum(worker_skills[p, :] * ((skills - skills_done) > 0))
                            for p in workers_av[i]
                        ]
                        sort = [
                            workers_av[i][p]
                            for p in np.argsort(-np.array(score))
                            if workers_av[i][p] not in used[1:]
                        ]
                        j = sort[0]
                        nz = np.nonzero(worker_skills[j, :] * skills > 0)[0]
                        if len(nz) > 0:
                            for nnz in nz:
                                skills_act_id[i, j, nnz] = 1
                                skills_done[nnz] += worker_skills[j, nnz]
                            worker_avail_in_time[j, starts[i] : ends[i]] = 0
                            used += [j]
                        if np.all(skills_done >= skills):
                            break
                    for w in used[1:]:
                        worker_usage[w] += ends[i] - starts[i]
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(starts, dtype=np.int_)
            ends_dict[act_id] = np.array(ends, dtype=np.int_)
            skills_usage[act_id] = skills_act_id
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    return starts_dict, ends_dict, skills_usage, unfeasible_sched


@njit
def sgs_fast_ms_preemptive_some_special_constraints(
    permutation_task,  # permutation_task=array(task)->task index
    priority_worker_per_task,  # array(task, worker)
    modes_array,  # modes = array(task) -> 0, 1...
    consumption_array,  # consumption_array=array3D(task, mode, res),
    skills_needs,
    duration_array,
    preemptive_tag,  # array(task)->bool
    predecessors,  # array(task, task) -> bool
    successors,  # array(task, task)->bool
    start_at_end_plus_offset,  # array(N, 3) -> (task1, task2, offset)
    start_after_nunit,  # array(N, 3) -> (task1, task2, offset)
    horizon,
    ressource_available,
    ressource_renewable,
    worker_available,  # array(workers, int)->bool
    worker_skills,  # array(workers, skills)->int
    minimum_starting_time_array,
    is_releasable,  # [task, mode, res]-> bool
    one_unit_per_task: bool = True,
    consider_partial_preemptive: bool = False,
    strictly_disjunctive_subtasks: bool = False,
):  # array(res)->bool
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
    worker_avail_in_time = np.copy(worker_available)
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = minimum_starting_time_array[act]
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)
    done_duration = np.zeros((permutation_task.shape[0]), dtype=np.int_)

    start_after_nunit_links = np.zeros(nb_task)
    for task in range(nb_task):
        start_after_nunit_links[task] = np.sum(start_after_nunit[:, 1] == task)

    start_at_end_plus_offset_links = np.zeros(nb_task)
    for task in range(nb_task):
        start_at_end_plus_offset_links[task] = np.sum(
            start_at_end_plus_offset[:, 1] == task
        )

    skills_usage = {}
    worker_usage = np.zeros((worker_skills.shape[0]), dtype=np.int_)
    while done < nb_task and not unfeasible_non_renewable_resources:
        act_id = 0
        for i in range(nb_task):
            if (
                pred_links[permutation_task[i]] == 0
                and done_np[permutation_task[i]] == 0
                and start_after_nunit_links[permutation_task[i]] == 0
                and start_at_end_plus_offset_links[permutation_task[i]] == 0
            ):
                act_id = permutation_task[i]
                break
        current_min_time = int(minimum_starting_time[act_id])
        valid = False
        starts = []
        ends = []
        workers_av = []
        skills = skills_needs[act_id, modes_array[act_id]]
        need_worker = np.max(skills) > 0
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
                    if need_worker:
                        if t == current_min_time:
                            indexes_present_worker_t = np.array(
                                [
                                    priority_worker_per_task[act_id, i]
                                    for i in range(priority_worker_per_task.shape[1])
                                    if worker_avail_in_time[
                                        priority_worker_per_task[act_id, i],
                                        current_min_time,
                                    ]
                                    > 0
                                ]
                            )
                            if one_unit_per_task:
                                indexes_present_worker_t = np.array(
                                    [
                                        i
                                        for i in indexes_present_worker_t
                                        if np.all(
                                            worker_skills[i, :]
                                            >= skills_needs[act_id, modes_array[act_id]]
                                        )
                                    ]
                                )
                            if len(indexes_present_worker_t) > 0:
                                available_skills_t = np.sum(
                                    worker_skills[indexes_present_worker_t, :], axis=0
                                )
                                if (
                                    np.min(
                                        available_skills_t
                                        - skills_needs[act_id, modes_array[act_id]]
                                    )
                                    < 0
                                ):
                                    reached_end = False
                                    break
                            else:
                                reached_end = False
                                break
                        else:
                            indexes_present_worker_next_t = np.array(
                                [
                                    i
                                    for i in indexes_present_worker_t
                                    if worker_avail_in_time[i, t] > 0
                                ]
                            )
                            if one_unit_per_task:
                                indexes_present_worker_next_t = np.array(
                                    [
                                        i
                                        for i in indexes_present_worker_next_t
                                        if np.all(
                                            worker_skills[i, :]
                                            >= skills_needs[act_id, modes_array[act_id]]
                                        )
                                    ]
                                )
                            if len(indexes_present_worker_next_t) > 0:
                                available_skills_tt = np.sum(
                                    worker_skills[indexes_present_worker_next_t, :],
                                    axis=0,
                                )
                                if (
                                    np.min(
                                        available_skills_tt
                                        - skills_needs[act_id, modes_array[act_id]]
                                    )
                                    < 0
                                ):
                                    reached_end = False
                                    break
                                else:
                                    indexes_present_worker_t = (
                                        indexes_present_worker_next_t
                                    )
                            else:
                                reached_end = False
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
                    workers_av.append(indexes_present_worker_t)
                    done_duration[act_id] += ends[-1] - starts[-1]

                if reached_end and not preemptive_tag[act_id]:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    workers_av.append(indexes_present_worker_t)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    if strictly_disjunctive_subtasks:
                        current_min_time = (
                            reached_t + 2
                            if reached_t is not None
                            else current_min_time + 1
                        )
                    else:
                        current_min_time = (
                            reached_t + 1
                            if reached_t is not None
                            else current_min_time + 1
                        )
                    if current_min_time >= new_horizon:
                        return None, None, None, True

                if valid and consider_partial_preemptive:
                    for res in range(ressource_available.shape[0]):
                        if consumption_array[act_id, modes_array[act_id], res] > 0:
                            if is_releasable[act_id, modes_array[act_id], res] == 0:
                                if (
                                    np.min(
                                        ressource_available[res][starts[0] : ends[-1]]
                                    )
                                    < consumption_array[
                                        act_id, modes_array[act_id], res
                                    ]
                                ):
                                    unfeasible_non_renewable_resources = True
                                    minimum_starting_time[act_id] = ends[
                                        -1
                                    ]  # a bit conservative here.
                if unfeasible_non_renewable_resources:
                    break
        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            skills_act_id = np.zeros(
                (len(starts), worker_avail_in_time.shape[0], skills_needs.shape[2])
            )
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        if (
                            consider_partial_preemptive
                            and not is_releasable[act_id, modes_array[act_id], res]
                        ):
                            if i == 0:
                                resource_avail_in_time[res][
                                    starts[i] : ends[-1]
                                ] -= consumption_array[act_id, modes_array[act_id], res]
                        else:
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
                skills = skills_needs[act_id, modes_array[act_id]]
                skills_done = np.zeros((skills.shape[0]))
                if need_worker and duration_array[act_id, modes_array[act_id]] > 0:
                    used = [0]
                    sort = [
                        workers_av[i][p]
                        for p in np.argsort(worker_usage[workers_av[i]])
                    ]
                    for j in sort:
                        nz = np.nonzero(worker_skills[j, :] * skills > 0)[0]
                        if len(nz) > 0:
                            for nnz in nz:
                                skills_act_id[i, j, nnz] = 1
                                skills_done[nnz] += worker_skills[j, nnz]
                            worker_avail_in_time[j, starts[i] : ends[i]] = 0
                            used += [j]
                        if np.all(skills_done >= skills):
                            break
                    for w in used[1:]:
                        worker_usage[w] += ends[i] - starts[i]
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(starts, dtype=np.int_)
            ends_dict[act_id] = np.array(ends, dtype=np.int_)
            skills_usage[act_id] = skills_act_id
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

    return starts_dict, ends_dict, skills_usage, unfeasible_non_renewable_resources


@njit
def sgs_fast_ms_preemptive_partial_schedule(
    permutation_task,  # permutation_task=array(task)->task index
    priority_worker_per_task,  # array(task, worker)
    modes_array,  # modes=array(task)->0, 1...
    scheduled_task_indicator,  # array(task)->bool
    scheduled_start_task_times,  # array(task, nb_preemptive)->int
    scheduled_end_task_times,  # array(task, nb_preemptive)->int
    nb_subparts,  # array(task) -> int
    worker_used,  # array(task, nb_preemptive, worker)->bool
    current_time,  # int
    consumption_array,
    skills_needs,
    duration_array,
    preemptive_tag,  # array(task)->bool
    predecessors,  # array(task, task) -> bool
    successors,  # array(task, task)->bool
    horizon,
    ressource_available,
    ressource_renewable,
    worker_available,  # array(workers, int)->bool
    worker_skills,  # array(workers, skills)->int
    minimum_starting_time_array,
    is_releasable,  # [task, mode, res]-> bool
    one_unit_per_task: bool = True,
    consider_partial_preemptive: bool = False,
):  # array(res)->bool
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
    worker_avail_in_time = np.copy(worker_available)
    minimum_starting_time = {}
    for act in range(permutation_task.shape[0]):
        minimum_starting_time[act] = max(current_time, minimum_starting_time_array[act])
    skills_usage = {}
    done = 0
    nb_task = permutation_task.shape[0]
    pred_links = np.sum(predecessors, axis=1)
    done_np = np.zeros((permutation_task.shape[0]), dtype=np.int_)

    for t in range(nb_task):
        if scheduled_task_indicator[t] != 0:
            starts_dict[t] = scheduled_start_task_times[t, : nb_subparts[t]]
            ends_dict[t] = scheduled_end_task_times[t, : nb_subparts[t]]
            for j in range(starts_dict[t].shape[0]):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        if (
                            consider_partial_preemptive
                            and not is_releasable[t, modes_array[t], res]
                        ):
                            if j == 0:
                                resource_avail_in_time[res][
                                    starts_dict[t][j] : ends_dict[t][-1]
                                ] -= consumption_array[t, modes_array[t], res]
                        else:
                            resource_avail_in_time[res][
                                starts_dict[t][j] : ends_dict[t][j]
                            ] -= consumption_array[t, modes_array[t], res]
                    else:
                        if j == starts_dict[t].shape[0] - 1:
                            resource_avail_in_time[res][
                                ends_dict[t][j] :
                            ] -= consumption_array[t, modes_array[t], res]
                            if resource_avail_in_time[res][-1] < 0:
                                unfeasible_non_renewable_resources = True
                                break
            activity_end_times[t] = ends_dict[t][-1]
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
            skills_act_id = np.zeros(
                (
                    starts_dict[t].shape[0],
                    worker_avail_in_time.shape[0],
                    skills_needs.shape[2],
                )
            )
            # nz[0] : nb_preemptive, nz[1] workers
            # Assign workers :
            skills = skills_needs[t, modes_array[t]]
            if np.max(skills) > 0:
                wused = worker_used[t, :, :]
                nz = np.nonzero(wused)
                skills_done = np.zeros((skills.shape[0]))
                for i, j in zip(nz[0], nz[1]):
                    skills_worker = np.nonzero(worker_skills[j, :] * skills > 0)[0]
                    if len(skills_worker) > 0:
                        for s in skills_worker:
                            skills_act_id[i, j, s] = 1
                            skills_done[s] += worker_skills[j, s]
                        worker_avail_in_time[j, starts_dict[t][i] : ends_dict[t][i]] = 0
                    if np.all(skills_done > skills):
                        break

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
        workers_av = []
        skills = skills_needs[act_id, modes_array[act_id]]
        need_worker = np.max(skills) > 0
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
                    if need_worker:
                        if t == current_min_time:
                            indexes_present_worker_t = np.array(
                                [
                                    priority_worker_per_task[act_id, i]
                                    for i in range(priority_worker_per_task.shape[1])
                                    if worker_avail_in_time[
                                        priority_worker_per_task[act_id, i],
                                        current_min_time,
                                    ]
                                    > 0
                                ]
                            )
                            if one_unit_per_task:
                                indexes_present_worker_t = np.array(
                                    [
                                        i
                                        for i in indexes_present_worker_t
                                        if np.all(
                                            worker_skills[i, :]
                                            >= skills_needs[act_id, modes_array[act_id]]
                                        )
                                    ]
                                )
                            if len(indexes_present_worker_t) > 0:
                                available_skills_t = np.sum(
                                    worker_skills[indexes_present_worker_t, :], axis=0
                                )
                                if (
                                    np.min(
                                        available_skills_t
                                        - skills_needs[act_id, modes_array[act_id]]
                                    )
                                    < 0
                                ):
                                    reached_end = False
                                    break
                            else:
                                reached_end = False
                                break
                        else:
                            indexes_present_worker_next_t = np.array(
                                [
                                    i
                                    for i in indexes_present_worker_t
                                    if worker_avail_in_time[i, t] > 0
                                ]
                            )
                            if one_unit_per_task:
                                indexes_present_worker_next_t = np.array(
                                    [
                                        i
                                        for i in indexes_present_worker_next_t
                                        if np.all(
                                            worker_skills[i, :]
                                            >= skills_needs[act_id, modes_array[act_id]]
                                        )
                                    ]
                                )
                            if len(indexes_present_worker_next_t) > 0:
                                available_skills_tt = np.sum(
                                    worker_skills[indexes_present_worker_next_t, :],
                                    axis=0,
                                )
                                if (
                                    np.min(
                                        available_skills_tt
                                        - skills_needs[act_id, modes_array[act_id]]
                                    )
                                    < 0
                                ):
                                    reached_end = False
                                    break
                                else:
                                    indexes_present_worker_t = (
                                        indexes_present_worker_next_t
                                    )
                            else:
                                reached_end = False
                                break
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
                    workers_av.append(indexes_present_worker_t)
                    done_duration[act_id] += ends[-1] - starts[-1]
                if reached_end and not preemptive_tag[act_id]:
                    starts.append(current_min_time)
                    ends.append(reached_t + 1)
                    workers_av.append(indexes_present_worker_t)
                    done_duration[act_id] += ends[-1] - starts[-1]
                valid = (
                    done_duration[act_id] == duration_array[act_id, modes_array[act_id]]
                )
                if not valid:
                    current_min_time = (
                        reached_t + 2 if reached_t is not None else current_min_time + 1
                    )
                if valid and consider_partial_preemptive:
                    for res in range(ressource_available.shape[0]):
                        if consumption_array[act_id, modes_array[act_id], res] > 0:
                            if not is_releasable[act_id, modes_array[act_id], res]:
                                if (
                                    np.min(
                                        ressource_available[res][starts[0] : ends[-1]]
                                    )
                                    < consumption_array[
                                        act_id, modes_array[act_id], res
                                    ]
                                ):
                                    unfeasible_non_renewable_resources = True
                                    minimum_starting_time[act_id] = ends[
                                        -1
                                    ]  # a bit conservative here.

        if not unfeasible_non_renewable_resources:
            end_t = ends[-1]
            skills_act_id = np.zeros(
                (len(starts), worker_avail_in_time.shape[0], skills_needs.shape[2])
            )
            for i in range(len(starts)):
                for res in range(ressource_available.shape[0]):
                    if ressource_renewable[res]:
                        if (
                            consider_partial_preemptive
                            and not is_releasable[act_id, modes_array[act_id], res]
                        ):
                            if i == 0:
                                resource_avail_in_time[res][
                                    starts[i] : ends[-1]
                                ] -= consumption_array[act_id, modes_array[act_id], res]
                        else:
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
                skills = skills_needs[act_id, modes_array[act_id]]
                skills_done = np.zeros((skills.shape[0]))
                if need_worker and duration_array[act_id, modes_array[act_id]] > 0:
                    for j in workers_av[i]:
                        nz = np.nonzero(worker_skills[j, :] * skills > 0)[0]
                        if len(nz) > 0:
                            for nnz in nz:
                                skills_act_id[i, j, nnz] = 1
                                skills_done[nnz] += worker_skills[j, nnz]
                            worker_avail_in_time[j, starts[i] : ends[i]] = 0
                        if np.all(skills_done >= skills):
                            break
            if unfeasible_non_renewable_resources:
                break
            activity_end_times[act_id] = end_t
            starts_dict[act_id] = np.array(starts, dtype=np.int_)
            ends_dict[act_id] = np.array(ends, dtype=np.int_)
            skills_usage[act_id] = skills_act_id
            done_np[act_id] = 1
            done += 1
            for s in range(successors.shape[1]):
                if successors[act_id, s] == 1:
                    minimum_starting_time[s] = max(
                        int(minimum_starting_time[s]), int(activity_end_times[act_id])
                    )
                    pred_links[s] -= 1
    return starts_dict, ends_dict, skills_usage, unfeasible_non_renewable_resources
