#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from discrete_optimization.flex_scheduling.problem import (
    RESOURCE_KEY,
    ConstraintsTask,
    FlexProblem,
    GroupType,
    ObjectiveParams,
    ObjectivesEnum,
    ResourceData,
    ScheduleSolution,
    ScheduleSolutionPreemptive,
    TaskData,
    TaskObject,
)
from discrete_optimization.rcpsp.problem import (
    RcpspProblem,
)


def from_rcpsp_to_fsp(problem: RcpspProblem) -> FlexProblem:
    resources = [
        ResourceData(
            id=r,
            name=r,
            calendar_availability=np.array(
                problem.get_resource_availability_array(r)
                + [problem.get_max_resource_capacity(r)]
            ),
            renewable=not (r in problem.non_renewable_resources),
            max_capacity=problem.get_max_resource_capacity(r),
            is_disjunctive=problem.get_max_resource_capacity(r) == 1,
            is_station=False,
            is_operator=False,
        )  # we don't know this
        for r in problem.resources_list
    ]
    tasks = [
        TaskObject(
            id=t,
            name=t,
            modes={
                i: TaskData(
                    duration=problem.mode_details[t][i]["duration"],
                    resource_consumption={
                        r: problem.mode_details[t][i][r]
                        for r in problem.resources_list
                        if r in problem.mode_details[t][i]
                    },
                    preemptive_on_resource_break=True,
                )
                for i in problem.mode_details[t]
            },
            min_starting_date=None,
            max_starting_date=None,
            min_ending_date=None,
            max_ending_date=None,
        )
        for t in problem.mode_details
    ]
    tasks_group = []
    constraints_data = ConstraintsTask(
        successors={t: set(problem.successors[t]) for t in problem.successors}
    )
    return FlexProblem(
        resources=resources,
        tasks=tasks,
        tasks_group=tasks_group,
        constraints=constraints_data,
        objective_params=ObjectiveParams(params_obj={ObjectivesEnum.MAKESPAN: 1}),
        horizon=problem.horizon,
    )


def from_fsp_to_rcpsp(problem: FlexProblem) -> RcpspProblem:
    mode_details = {}
    for t in problem.tasks:
        id_ = t.id
        mode_details[id_] = {}
        for mode in t.modes:
            mode_details[id_][mode] = {}
            mode_details[id_][mode]["duration"] = t.modes[mode].duration
            mode_details[id_][mode].update(t.modes[mode].resource_consumption)

    return RcpspProblem(
        resources={r.id: r.calendar_availability for r in problem.resources},
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=problem.constraints.successors,
        horizon=problem.horizon,
        horizon_multiplier=1,
    )


def get_lb_ub_start_end_date(problem: FlexProblem):
    min_start_time = {i: 0 for i in range(problem.nb_tasks)}
    max_start_time = {i: problem.horizon for i in range(problem.nb_tasks)}
    min_end_time = {i: 0 for i in range(problem.nb_tasks)}
    max_end_time = {i: problem.horizon for i in range(problem.nb_tasks)}
    for i in range(problem.nb_tasks):
        task = problem.tasks[i]
        if task.min_starting_date is not None:
            min_start_time[i] = task.min_starting_date
        if task.max_starting_date is not None:
            max_start_time[i] = task.max_starting_date
        if task.min_ending_date is not None:
            min_end_time[i] = task.min_ending_date
        if task.max_ending_date is not None:
            max_end_time[i] = task.max_ending_date
    return min_start_time, max_start_time, min_end_time, max_end_time


def get_lb_ub_start_end_date_group_of_task(problem: FlexProblem):
    nb_group = len(problem.tasks_group)
    ids = {g.id for g in problem.tasks_group}
    min_start_time = {i: 0 for i in ids}
    max_start_time = {i: problem.horizon for i in ids}
    min_end_time = {i: 0 for i in ids}
    max_end_time = {i: problem.horizon for i in ids}
    for i in range(nb_group):
        group = problem.tasks_group[i]
        group_id = group.id
        if group.min_starting_date is not None:
            min_start_time[group_id] = group.min_starting_date
        if group.max_starting_date is not None:
            max_start_time[group_id] = group.max_starting_date
        if group.min_ending_date is not None:
            min_end_time[group_id] = group.min_ending_date
        if group.max_ending_date is not None:
            max_end_time[group_id] = group.max_ending_date
    return min_start_time, max_start_time, min_end_time, max_end_time


def create_resource_consumption_from_calendar(
    calendar_availability: np.ndarray,
) -> List[Dict[str, int]]:
    max_capacity = np.max(calendar_availability)
    fake_tasks: List[Dict[str, int]] = []
    delta = calendar_availability[:-1] - calendar_availability[1:]
    index_non_zero = np.nonzero(delta)[0]
    if calendar_availability[0] < max_capacity:
        consume = {
            "value": int(max_capacity - calendar_availability[0]),
            "duration": int(index_non_zero[0] + 1),
            "start": 0,
        }
        fake_tasks += [consume]
    for j in range(len(index_non_zero) - 1):
        ind = index_non_zero[j]
        value = calendar_availability[ind + 1]
        if value != max_capacity:
            consume = {
                "value": int(max_capacity - value),
                "duration": int(index_non_zero[j + 1] - ind),
                "start": int(ind + 1),
            }
            fake_tasks += [consume]
    # Last part could also be added..
    if len(index_non_zero) > 0:
        value = calendar_availability[index_non_zero[-1] + 1]
        if value != max_capacity:
            fake_tasks.append(
                {
                    "value": int(max_capacity - value),
                    "duration": int(
                        calendar_availability.shape[0] + 1 - index_non_zero[-1]
                    ),
                    "start": index_non_zero[-1] + 1,
                }
            )
    return fake_tasks


def build_pattern_resource(resource_data: ResourceData):
    calendar_availability = resource_data.calendar_availability
    values = calendar_availability[calendar_availability != 0]
    values = sorted(set(values))
    resource_maps = {}
    for j in range(len(values)):
        res = ResourceData(
            id=str(resource_data.id) + f"-{j}",
            name=resource_data.name + f"-{j}",
            calendar_availability=values[j] * (calendar_availability == values[j]),
            renewable=resource_data.renewable,
            max_capacity=values[j],
            is_disjunctive=resource_data.is_disjunctive,
            is_station=resource_data.is_station,
            is_operator=resource_data.is_operator,
        )
        resource_maps[resource_data.name + f"-{j}"] = res
        if j >= 1:
            res_slide = ResourceData(
                id=str(resource_data.id) + f"-{(0, j)}",
                name=resource_data.name + f"-{(0, j)}",
                calendar_availability=calendar_availability
                * (calendar_availability <= values[j]),
                renewable=resource_data.renewable,
                max_capacity=int(values[j]),
                is_disjunctive=resource_data.is_disjunctive,
                is_station=resource_data.is_station,
                is_operator=resource_data.is_operator,
                child_resource=set(
                    [str(resource_data.id) + f"-{jj}" for jj in range(j + 1)]
                ),
            )
            resource_maps[resource_data.name + f"-{(0, j)}"] = res_slide
    return resource_maps


def transform_problem_into_multimode(fsp: FlexProblem) -> FlexProblem:
    new_resource = []
    resource_split = {}
    for resource in fsp.resources:
        if resource.renewable and resource.max_capacity != 1:
            calendar_availability = resource.calendar_availability
            values = calendar_availability[calendar_availability != 0]
            values = sorted(set(values))
            if len(values) >= 2:
                new_res = build_pattern_resource(resource)
                resource_split[resource.id] = new_res
                for r in new_res:
                    new_resource.append(new_res[r])
            else:
                new_resource.append(resource)
        else:
            new_resource.append(resource)

    for task in fsp.tasks:
        new_modes = {}
        modes = task.modes
        sorted_modes = sorted(modes.keys())
        init_mode_index = sorted_modes[0]
        for i_mode in sorted_modes:
            mode = modes[i_mode]
            res_conso = mode.resource_consumption
            non_zeros_resource = [r for r in res_conso if res_conso[r] > 0]
            resource_to_split = [r for r in non_zeros_resource if r in resource_split]
            if len(resource_to_split) == 0:
                new_modes[init_mode_index] = mode
                init_mode_index += 1
            else:
                # TODO do product of modes..
                for sub_res in resource_split[resource_to_split[0]]:
                    new_res_conso = {
                        r: res_conso[r] for r in res_conso if r != resource_to_split[0]
                    }
                    new_res_conso[resource_split[resource_to_split[0]][sub_res].id] = (
                        res_conso[resource_to_split[0]]
                    )
                    new_modes[init_mode_index] = TaskData(
                        duration=mode.duration,
                        resource_consumption=new_res_conso,
                        preemptive_on_resource_break=mode.preemptive_on_resource_break,
                    )
                    init_mode_index += 1
        task.modes = new_modes

    fsp.resources = new_resource
    fsp.nb_resources = len(new_resource)
    fsp.resource_dict = {r.id: r for r in new_resource}
    fsp.resource_id_to_index = {fsp.resources[i].id: i for i in range(fsp.nb_resources)}
    if ObjectivesEnum.RESOURCE_COST in fsp.objective_params.params_obj:
        obj_params_resource = fsp.objective_params.params_obj[
            ObjectivesEnum.RESOURCE_COST
        ]
        new_weight = {}
        for r in obj_params_resource.weight_per_resource_unit:
            if r in resource_split:
                for rr in resource_split[r]:
                    new_weight[rr] = obj_params_resource.weight_per_resource_unit[r]
            else:
                new_weight[r] = obj_params_resource.weight_per_resource_unit[r]
        obj_params_resource.weight_per_resource_unit = new_weight
    return fsp


def compute_duration_function_time_cluster(
    orig_duration: int,
    resource_calendar: np.ndarray,
    cumulative_resource_calendar: np.ndarray,
):
    duration = -np.ones((cumulative_resource_calendar.shape[0]))
    dict_of_interval_per_duration = {}
    current_interval = [0, 0]
    cur_duration = -1
    for i in range(cumulative_resource_calendar.shape[0]):
        if resource_calendar[i] == 0:
            if duration[i] == duration[i - 1]:
                current_interval[1] = i
            else:
                prev_d = duration[i - 1]
                if prev_d not in dict_of_interval_per_duration:
                    dict_of_interval_per_duration[prev_d] = []
                dict_of_interval_per_duration[prev_d] += [
                    [current_interval[0], current_interval[1]]
                ]
                current_interval = [i, i]
            continue
        x = cumulative_resource_calendar[i]
        if x == 0:
            continue
        index = next(
            (
                j
                for j in range(i, cumulative_resource_calendar.shape[0])
                if cumulative_resource_calendar[j] == x + orig_duration - 1
            ),
            None,
        )
        if index is not None:
            duration[i] = index - i + 1
            cur_duration = duration[i]
            if i >= 1:
                if duration[i] == duration[i - 1]:
                    current_interval[1] = i
                else:
                    prev_d = duration[i - 1]
                    if prev_d not in dict_of_interval_per_duration:
                        dict_of_interval_per_duration[prev_d] = []
                    dict_of_interval_per_duration[prev_d] += [
                        [current_interval[0], current_interval[1]]
                    ]
                    current_interval = [i, i]
        else:
            break
    if current_interval[0] != current_interval[1]:
        d = cur_duration
        if d not in dict_of_interval_per_duration:
            dict_of_interval_per_duration[d] = []
        dict_of_interval_per_duration[d] += [[current_interval[0], current_interval[1]]]
    if len(dict_of_interval_per_duration) == 0:
        dict_of_interval_per_duration[orig_duration] = current_interval
    return duration, dict_of_interval_per_duration


def compute_duration_tasks_function_time(problem: FlexProblem):
    method = compute_duration_function_time_cluster
    resource_calendar_dict = {
        problem.resources[i].id: problem.resources[i].calendar_availability > 0
        for i in range(len(problem.resources))
    }
    cumulative_calendar_dict = {
        r: np.cumsum(resource_calendar_dict[r]) for r in resource_calendar_dict
    }
    durations = {
        (i, m): None for i in range(problem.nb_tasks) for m in problem.tasks[i].modes
    }
    res_arrays = {}
    for i in range(problem.nb_tasks):
        for m in problem.tasks[i].modes:
            task_data: TaskData = problem.tasks[i].modes[m]
            resource_non_zeros = [
                r
                for r in task_data.resource_consumption
                if task_data.resource_consumption[r] > 0
            ]
            if len(resource_non_zeros) == 0:
                durations[i, m] = ([], {task_data.duration: [[0, problem.horizon]]})
            elif len(resource_non_zeros) == 1:
                # One resource pool is used.
                orig_duration = task_data.duration
                res_consumption = task_data.resource_consumption[resource_non_zeros[0]]
                c = (
                    problem.resources[
                        problem.resource_id_to_index[resource_non_zeros[0]]
                    ].calendar_availability
                    >= res_consumption
                )
                durations[i, m] = method(
                    orig_duration=orig_duration,
                    resource_calendar=c,  # resource_calendar_dict[resource_non_zeros[0]],
                    cumulative_resource_calendar=np.cumsum(c),
                    # cumulative_calendar_dict[
                    #     resource_non_zeros[0]
                    # ],
                )
                res_arrays[i, m] = c
            else:
                orig_duration = task_data.duration
                tuple_res = tuple(
                    [(r, task_data.resource_consumption[r]) for r in resource_non_zeros]
                )
                if tuple_res not in resource_calendar_dict:
                    # For the first resource in the tuple, b  "availability >= consumption"
                    first_res_id, first_consumption = tuple_res[0]
                    b = (
                        problem.resources[
                            problem.resource_id_to_index[first_res_id]
                        ].calendar_availability
                        >= first_consumption
                    )

                    for res_id, cons in tuple_res[1:]:
                        b &= (
                            problem.resources[
                                problem.resource_id_to_index[res_id]
                            ].calendar_availability
                            >= cons
                        )
                    resource_calendar_dict[tuple_res] = b
                    cumulative_calendar_dict[tuple_res] = np.cumsum(
                        resource_calendar_dict[tuple_res]
                    )
                durations[i, m] = method(
                    orig_duration=orig_duration,
                    resource_calendar=resource_calendar_dict[tuple_res],
                    cumulative_resource_calendar=cumulative_calendar_dict[tuple_res],
                )
                res_arrays[i, m] = resource_calendar_dict[tuple_res]

    return durations, res_arrays


def resource_consumption_modes(flex_problem: FlexProblem):
    # Build an array resource_consumptions[i, r]
    max_nb_modes = max(
        len(flex_problem.tasks[i].modes) for i in range(flex_problem.nb_tasks)
    )
    resource_consumptions = np.zeros(
        (flex_problem.nb_tasks, max_nb_modes, flex_problem.nb_resources), dtype=int
    )
    for i in range(flex_problem.nb_tasks):
        modes = sorted(list(flex_problem.tasks[i].modes.keys()))
        for j, mode_j in enumerate(modes):
            consumption_dict = flex_problem.tasks[i].modes[mode_j].resource_consumption
            for r_idx, r_name in enumerate(flex_problem.resources):
                resource_consumptions[i, j, r_idx] = consumption_dict.get(r_name.id, 0)
    return resource_consumptions


debug_log = logging.getLogger("debug")


class SolutionDetails:
    def __init__(
        self,
        problem: FlexProblem,
        solution: ScheduleSolution,
        durations_data: dict = None,
        res_arrays_data: dict = None,
    ):
        self.problem = problem
        self.solution = solution
        if durations_data is None:
            durations_data, res_arrays_data = compute_duration_tasks_function_time(
                problem
            )
        self.durations = durations_data
        self.res_arrays = res_arrays_data
        self.resource_consumption_mode = resource_consumption_modes(self.problem)
        self.resource_availability = {
            r.id: np.copy(self.problem.resource_dict[r.id].calendar_availability)
            for r in self.problem.resources
        }
        self.resource_usage = {
            r: np.zeros(self.resource_availability[r].shape)
            for r in self.resource_availability
        }
        self.schedule_det = {}

    def compute_details(self):
        schedule_det = {}
        for index_task in range(self.problem.nb_tasks):
            start = int(self.solution.schedule[index_task, 0])
            end = int(self.solution.schedule[index_task, 1])
            mode = int(self.solution.modes[index_task])
            original_duration = self.problem.tasks[index_task].modes[mode].duration
            index_mode = self.problem.tasks[index_task].modes_id_to_index[mode]
            if (
                original_duration == 0
                or len(self.problem.tasks[index_task].modes[mode].resource_consumption)
                == 0
            ):
                continue
            duration_data = self.durations[index_task, mode]
            binary_res_array = self.res_arrays[index_task, mode]
            dur_for_this_start = duration_data[0][start]
            if dur_for_this_start != end - start:
                logging.error(
                    f"dur for this start, {dur_for_this_start, end - start, start, end}"
                )
            assert end - start == dur_for_this_start
            times = []
            for t in range(int(start), int(end)):
                if binary_res_array[t] > 0:
                    times.append(t)
                    for i_r in range(self.resource_consumption_mode.shape[2]):
                        resource_ = self.problem.resources[i_r].id
                        self.resource_availability[resource_][t] -= (
                            self.resource_consumption_mode[index_task, index_mode, i_r]
                        )
                        self.resource_usage[resource_][t] += (
                            self.resource_consumption_mode[index_task, index_mode, i_r]
                        )
                        if self.resource_availability[resource_][t] < 0:
                            logging.info(
                                f"Problem at time {t} for resource {self.problem.resources[i_r].id}"
                            )
            assert len(times) == original_duration
            schedule_det[index_task] = times
        self.schedule_det = schedule_det
        self.group_consumption()
        self.resource_blocked_gen()
        self.resource_blocked_1()

    def group_consumption(self):
        for group in self.problem.tasks_group:
            if group.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE:
                res_not_release: dict[RESOURCE_KEY, int] = group.res_not_released
                tasks = group.tasks_group
                indexes = [self.problem.task_id_to_index[t] for t in tasks]
                start_group = int(
                    min(self.solution.schedule[ind, 0] for ind in indexes)
                )
                end_group = int(max(self.solution.schedule[ind, 1] for ind in indexes))
                for r in res_not_release:
                    index_r = self.problem.resource_id_to_index[r]
                    cal = self.problem.resource_dict[r].calendar_availability
                    for t in range(start_group, end_group):
                        if cal[t] == 0:
                            continue
                        tasks_active_there = [
                            ind
                            for ind in indexes
                            if self.solution.schedule[ind, 0]
                            <= t
                            < self.solution.schedule[ind, 1]
                        ]
                        res_already_counted = False
                        for task in tasks_active_there:
                            mode = self.solution.modes[task]
                            index_mode = self.problem.tasks[task].modes_id_to_index[
                                mode
                            ]
                            if (
                                self.resource_consumption_mode[
                                    task, index_mode, index_r
                                ]
                                > 0
                            ):
                                res_already_counted = True
                                break
                        if res_already_counted:
                            logging.debug("res already counted")
                            continue
                        logging.debug("res not already counted")
                        self.resource_usage[r][t] += res_not_release[r]
                        self.resource_availability[r][t] -= res_not_release[r]
                        if self.resource_availability[r][t] < 0:
                            logging.info(
                                f"Group consumption : Problem at time {t}, for resource {r}"
                            )

    def resource_blocked_gen(self):
        if not self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic:
            return
        for (
            gr1,
            gr2,
            res_blocked,
        ) in self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic:
            end_gr1 = None
            if gr1.is_a_task:
                task_id = gr1.task_id
                index = self.problem.task_id_to_index[task_id]
                end_gr1 = int(self.solution.schedule[index, 1])
            else:
                group = self.problem.tasks_group[
                    self.problem.group_id_to_index[gr1.group_id]
                ]
                indexes = [self.problem.task_id_to_index[t] for t in group.tasks_group]
                end_gr1 = int(max(self.solution.schedule[ind, 1] for ind in indexes))
            start_gr2 = None
            if gr2.is_a_task:
                task_id = gr2.task_id
                index = self.problem.task_id_to_index[task_id]
                start_gr2 = int(self.solution.schedule[index, 0])
            else:
                group = self.problem.tasks_group[
                    self.problem.group_id_to_index[gr2.group_id]
                ]
                indexes = [self.problem.task_id_to_index[t] for t in group.tasks_group]
                start_gr2 = int(min(self.solution.schedule[ind, 0] for ind in indexes))
            logging.debug(f"Blocked during {end_gr1, start_gr2}")
            for r in res_blocked:
                cal = self.problem.resource_dict[r].calendar_availability
                for t in range(end_gr1, start_gr2):
                    if cal[t] > 0:
                        self.resource_availability[r][t] -= res_blocked[r]
                        self.resource_usage[r][t] += res_blocked[r]
                        if self.resource_availability[r][t] < 0:
                            logging.info(
                                f"Resource blocked Gen : Problem at time {t} for resource {r},"
                                f"{self.resource_availability[r][t]}"
                            )

    def resource_blocked_1(self):
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor
            is not None
        ):
            for (
                t1,
                t2,
                res_blocked,
            ) in self.problem.constraints.successor_with_res_release_at_start_of_successor:
                index1 = self.problem.task_id_to_index[t1]
                end_t1 = int(self.solution.schedule[index1, 1])
                index2 = self.problem.task_id_to_index[t2]
                start_t2 = int(self.solution.schedule[index2, 0])
                logging.debug(f"Blocked during {end_t1, start_t2}")
                for r in res_blocked:
                    cal = self.problem.resource_dict[r].calendar_availability
                    for t in range(end_t1, start_t2):
                        if cal[t] > 0:
                            self.resource_availability[r][t] -= res_blocked[r]
                            self.resource_usage[r][t] += res_blocked[r]
                            if self.resource_availability[r][t] < 0:
                                logging.info(
                                    f"Res block 1 : Problem at time {t} for resource {r},"
                                    f"{self.resource_availability[r][t]}"
                                )

    def satisfy(self):
        for index_task in range(self.problem.nb_tasks):
            mode = self.solution.modes[index_task]
            original_duration = self.problem.tasks[index_task].modes[mode].duration
            if (
                original_duration == 0
                or len(self.problem.tasks[index_task].modes[mode].resource_consumption)
                == 0
            ):
                continue
            if len(self.schedule_det[index_task]) != original_duration:
                return False
        for r in self.resource_availability:
            if np.min(self.resource_availability[r]) < 0:
                return False
        return True

    def build_scheduling_preemptive(self) -> "ScheduleSolutionPreemptive":
        sched_list = []
        for index_task in range(self.problem.nb_tasks):
            sched = []
            if index_task in self.schedule_det:
                times = self.schedule_det[index_task]
                left_side = times[0]
                right_side = left_side + 1
                for i in range(1, len(times)):
                    cur = times[i]
                    if cur > right_side:
                        sched.append((int(left_side), int(right_side)))
                        left_side = cur
                        right_side = cur + 1
                    else:
                        right_side = cur + 1
                if len(sched) == 0:
                    sched.append((int(left_side), int(right_side)))
                else:
                    if sched[-1][1] != right_side:
                        # Last part.
                        sched.append((int(left_side), int(right_side)))
                sched_list.append(sched)
            else:
                sched_list.append(
                    [
                        (
                            int(self.solution.schedule[index_task, 0]),
                            int(self.solution.schedule[index_task, 1]),
                        )
                    ]
                )
            mode = self.solution.modes[index_task]
            original_duration = self.problem.tasks[index_task].modes[mode].duration
            assert sum([x[1] - x[0] for x in sched_list[-1]]) == original_duration
        return ScheduleSolutionPreemptive(
            problem=self, schedule=sched_list, modes=self.solution.modes
        )
