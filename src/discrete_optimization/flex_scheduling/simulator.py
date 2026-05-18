#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import dataclasses
from typing import Hashable

import numpy as np

from discrete_optimization.flex_scheduling.problem import (
    RESOURCE_KEY,
    FlexProblem,
    GroupType,
    ScheduleSolution,
    ScheduleSolutionPreemptive,
    TaskGroupAbstraction,
    TasksGroups,
)


def init_capacity_resource(flex_problem: FlexProblem):
    nb_resources = flex_problem.nb_resources
    max_len_calendar = max(
        len(res.calendar_availability) for res in flex_problem.resources
    )
    capacity = np.zeros((nb_resources, max_len_calendar), dtype=int)
    for r_idx, res in enumerate(flex_problem.resources):
        arr = np.array(res.calendar_availability, dtype=int)
        capacity[r_idx, : len(arr)] = arr
    return capacity


def resource_consumption(flex_problem: FlexProblem, solution: ScheduleSolution):
    # Build an array resource_consumptions[i, r]
    resource_consumptions = np.zeros(
        (flex_problem.nb_tasks, flex_problem.nb_resources), dtype=int
    )
    for i in range(flex_problem.nb_tasks):
        mode_i = solution.modes[i]
        consumption_dict = flex_problem.tasks[i].modes[mode_i].resource_consumption
        for r_idx, r_name in enumerate(flex_problem.resources):
            resource_consumptions[i, r_idx] = consumption_dict.get(r_name.id, 0)
    return resource_consumptions


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
                resource_consumptions[i, mode_j, r_idx] = consumption_dict.get(
                    r_name.id, 0
                )
    return resource_consumptions


def build_task_to_set_of_groups(flex_problem: FlexProblem):
    task_to_group_set = {}
    for group in flex_problem.tasks_group:
        group_id = group.id
        for task in group.tasks_group:
            if task not in task_to_group_set:
                task_to_group_set[task] = set()
            task_to_group_set[task].add(group_id)
    return task_to_group_set


def post_process_schedule(
    flex_problem: FlexProblem,
    solution: ScheduleSolution,
    keep_min_time: bool = False,
    keep_strict_order_task: bool = False,
):
    # resource_to_index = {r: i for i, r in enumerate(flex_problem.resources)}
    capacity = init_capacity_resource(flex_problem)
    resource_consumptions = resource_consumption(
        flex_problem=flex_problem, solution=solution
    )
    task_to_group_set = build_task_to_set_of_groups(flex_problem)
    tasks_order = np.argsort(solution.schedule[:, 0])
    new_schedule = {}
    current_group = {}
    successors_task = flex_problem.constraints.successors
    predecessors_task = {}
    if successors_task is not None:
        for i in successors_task:
            for t in successors_task[i]:
                if t not in predecessors_task:
                    predecessors_task[t] = set()
                predecessors_task[t].add(i)
    successors_group = flex_problem.constraints.successors_group_tasks
    predecessors_group = {}
    if successors_group is not None:
        for i in successors_group:
            for t in successors_group[i]:
                if t not in predecessors_group:
                    predecessors_group[t] = set()
                predecessors_group[t].add(i)
    successor_with_res_release_at_start_of_successor = (
        flex_problem.constraints.successor_with_res_release_at_start_of_successor
    )
    successor_generic_with_res_release_at_start_of_successor_generic = flex_problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
    scheduled_task = set()
    scheduled_group = set()
    cur_end_time = 0
    index_task = 0
    for i_task in tasks_order:
        id_task = flex_problem.index_to_task_id[i_task]
        print("Scheduling...", id_task)
        min_time = flex_problem.tasks[i_task].min_starting_date
        if min_time is None:
            min_time = 0
        cons = resource_consumptions[i_task, :]
        if id_task in predecessors_task:
            min_time = max(
                min_time,
                max([new_schedule[j][-1][1] for j in predecessors_task[id_task]]),
            )  # Max time of predecessors
        if id_task in task_to_group_set:
            for g in task_to_group_set[id_task]:
                if g in predecessors_group:
                    for pred_g in predecessors_group[g]:
                        min_time = max(
                            min_time,
                            max(
                                [
                                    new_schedule[jj][-1][1]
                                    for jj in flex_problem.tasks_group[
                                        flex_problem.group_id_to_index[pred_g]
                                    ].tasks_group
                                ]
                            ),
                        )
        duration = flex_problem.tasks[i_task].modes[solution.modes[i_task]].duration
        duration_done = 0
        part = 0
        cur_time = min_time
        if keep_min_time:
            cur_time = max(min_time, int(solution.schedule[i_task, 0]))
        if keep_strict_order_task:
            if index_task > 0:
                st_prev = int(solution.schedule[tasks_order[index_task - 1], 0])
                id_task_prev = flex_problem.index_to_task_id[
                    tasks_order[index_task - 1]
                ]
                actual_st = int(new_schedule[id_task_prev][0][0])
                cur_time = max(
                    min_time, actual_st + (int(solution.schedule[i_task, 0]) - st_prev)
                )

        new_schedule[id_task] = []
        if duration_done == duration:
            new_schedule[id_task].append((min_time, min_time))
        while duration_done < duration:
            next_start = next(
                i
                for i in range(cur_time, capacity.shape[1])
                if np.min(capacity[:, i] - cons) >= 0
            )
            next_end = next(
                (
                    j
                    for j in range(next_start, next_start + duration - duration_done)
                    if np.min(capacity[:, j] - cons) < 0
                ),
                None,
            )
            if next_end is None:
                next_end = next_start + duration - duration_done
            new_schedule[id_task].append((next_start, next_end))
            for k in range(next_start, next_end):
                capacity[:, k] -= cons
            duration_done += next_end - next_start
            cur_time = next_end + 1
        cur_end_time = new_schedule[id_task][0][1]
        scheduled_task.add(id_task)
        if id_task in task_to_group_set:
            for g in task_to_group_set[id_task]:
                group = flex_problem.tasks_group[flex_problem.group_id_to_index[g]]
                if all(t_id in scheduled_task for t_id in group.tasks_group):
                    scheduled_group.add(g)
        index_task += 1
    return new_schedule


def build_graph_dependencies(flex_problem: FlexProblem):
    import networkx as nx

    fsp = flex_problem
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [flex_problem.tasks[i].id for i in range(flex_problem.nb_tasks)]
    )
    successors_task = fsp.constraints.successors
    if successors_task is not None:
        for i in successors_task:
            for t in successors_task[i]:
                graph.add_edge(i, t, label="classic")
    successors_group_tasks = fsp.constraints.successors_group_tasks
    if successors_group_tasks is not None:
        for g in successors_group_tasks:
            gr = flex_problem.tasks_group[flex_problem.group_id_to_index[g]]
            for oth_g in successors_group_tasks[g]:
                ogr = flex_problem.tasks_group[flex_problem.group_id_to_index[oth_g]]
                for i_gr in gr.tasks_group:
                    for i_ogr in ogr.tasks_group:
                        graph.add_edge(i_gr, i_ogr, label="via_group")
    for (
        t1,
        t2,
        res,
    ) in flex_problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic:
        if t1.is_a_task:
            id_1 = [t1.task_id]
        else:
            group = flex_problem.tasks_group[
                flex_problem.group_id_to_index[t1.group_id]
            ]
            id_1 = list(group.tasks_group)
        if t2.is_a_task:
            id_2 = [t2.task_id]
        else:
            group = flex_problem.tasks_group[
                flex_problem.group_id_to_index[t2.group_id]
            ]
            id_2 = list(group.tasks_group)
        for id1 in id_1:
            for id2 in id_2:
                graph.add_edge(id1, id2, label="constr")
    return graph


def get_tasks_in_events(
    problem: FlexProblem,
    event: tuple[TaskGroupAbstraction, TaskGroupAbstraction, dict[RESOURCE_KEY, int]],
):
    source = set()
    sink = set()
    if event[0].is_a_task:
        source.add(event[0].task_id)
    if event[1].is_a_task:
        sink.add(event[1].task_id)
    if not event[0].is_a_task:
        source.update(
            problem.tasks_group[
                problem.group_id_to_index[event[0].group_id]
            ].tasks_group
        )
    if not event[1].is_a_task:
        sink.update(
            problem.tasks_group[
                problem.group_id_to_index[event[1].group_id]
            ].tasks_group
        )
    return source, sink


@dataclasses.dataclass
class StateOfEvent:
    event: tuple[TaskGroupAbstraction, TaskGroupAbstraction, dict[RESOURCE_KEY, int]]
    source: set[Hashable]
    sink: set[Hashable]
    active: bool = False
    start_event_triggered: int = None
    end_event_triggered: int = None

    def update_state_of_event(
        self, full_schedule: dict[Hashable, list[tuple[int, int]]]
    ):
        if all(t in full_schedule for t in self.source):
            self.start_event_triggered = max(
                [full_schedule[t][-1][1] for t in self.source]
            )
            self.active = True
        if any(t in full_schedule for t in self.sink):
            self.end_event_triggered = min(
                [full_schedule[t][0][0] for t in self.sink if t in full_schedule]
            )


@dataclasses.dataclass
class StateOfGroup:
    event: TasksGroups
    active: bool = False
    start_event_triggered: int = None
    end_event_triggered: int = None

    def update_state_of_event(
        self, full_schedule: dict[Hashable, list[tuple[int, int]]]
    ):
        if all(t in full_schedule for t in self.event.tasks_group):
            self.start_event_triggered = min(
                [full_schedule[t][0][0] for t in self.event.tasks_group]
            )
            self.end_event_triggered = max(
                [full_schedule[t][-1][1] for t in self.event.tasks_group]
            )
            self.active = True
        if any(t in full_schedule for t in self.event.tasks_group):
            self.start_event_triggered = min(
                [
                    full_schedule[t][0][0]
                    for t in self.event.tasks_group
                    if t in full_schedule
                ]
            )
            self.active = True


class PostprocessTool:
    def __init__(self, flex_problem: FlexProblem, solution: ScheduleSolution):
        self.problem = flex_problem
        self.solution = solution
        self.capacity = init_capacity_resource(flex_problem)  # array (i_resource, time)
        self.resource_consumption = resource_consumption(
            flex_problem=flex_problem, solution=solution
        )  # array (i_task, i_resource)
        self.graph = build_graph_dependencies(flex_problem=flex_problem)
        self.tasks_id_to_group = build_task_to_set_of_groups(self.problem)
        self.predecessors = {
            t: list(self.graph.predecessors(t)) for t in self.graph.nodes
        }
        self.nr_events = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
        self.states_events = []
        for i in range(len(self.nr_events)):
            source, sink = get_tasks_in_events(
                problem=self.problem, event=self.nr_events[i]
            )
            self.states_events.append(
                StateOfEvent(event=self.nr_events[i], source=source, sink=sink)
            )
        self.state_groups = []
        for g in self.problem.tasks_group:
            if g.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE:
                self.state_groups.append(StateOfGroup(g))

    def post_process_left(
        self,
        flex_problem: FlexProblem,
        solution: ScheduleSolution,
        keep_min_time: bool = False,
        keep_strict_order_task: bool = False,
    ):
        # resource_to_index = {r: i for i, r in enumerate(flex_problem.resources)}
        capacity = self.capacity
        resource_consumptions = self.resource_consumption
        tasks_order = list(np.argsort(solution.schedule[:, 0]))
        tasks_order = sorted(
            range(solution.schedule.shape[0]),
            key=lambda x: (
                solution.schedule[x, 0],
                -1
                if any(
                    self.problem.index_to_task_id[x] in event.sink
                    for event in self.states_events
                )
                else 0,
            ),
        )

        new_schedule = {}
        scheduled_task = set()
        prev_id_task = None
        currently_failed = set()
        while len(new_schedule) < self.problem.nb_tasks:
            i_task = next(
                (
                    j
                    for j in tasks_order
                    if self.problem.index_to_task_id[j] not in new_schedule
                    and j not in currently_failed
                    and all(
                        t in new_schedule
                        for t in self.predecessors[self.problem.index_to_task_id[j]]
                    )
                    # and
                )
            )
            id_task = flex_problem.index_to_task_id[i_task]
            print("Scheduling...", id_task)
            min_time = flex_problem.tasks[i_task].min_starting_date
            if min_time is None:
                min_time = 0
            cons = resource_consumptions[i_task, :]
            if id_task in self.predecessors and len(self.predecessors[id_task]) > 0:
                min_time = max(
                    min_time,
                    max([new_schedule[j][-1][1] for j in self.predecessors[id_task]]),
                )  # Max time of predecessors
            if id_task in self.tasks_id_to_group:
                for g in self.tasks_id_to_group[id_task]:
                    group = self.problem.tasks_group[self.problem.group_id_to_index[g]]
                    if group.no_overlap:
                        if any(jj in new_schedule for jj in group.tasks_group):
                            # avoid overlap..
                            print("prev", min_time)
                            min_time = max(
                                min_time,
                                max(
                                    [
                                        new_schedule[jj][-1][1]
                                        for jj in group.tasks_group
                                        if jj in new_schedule
                                    ]
                                ),
                            )
                            print("new", min_time)
            duration = flex_problem.tasks[i_task].modes[solution.modes[i_task]].duration
            duration_done = 0
            cur_time = min_time
            if keep_min_time:
                cur_time = max(min_time, int(solution.schedule[i_task, 0]))
            if keep_strict_order_task:
                if prev_id_task is not None:
                    cur_time = max(min_time, new_schedule[prev_id_task][0][0])
            new_schedule[id_task] = []
            if duration_done == duration:
                new_schedule[id_task].append((min_time, min_time))

            def get_capacity_index_task(index_task: int, time: int):
                conso = self.resource_consumption[index_task, :]
                nz = np.nonzero(conso)[0]
                capa = np.copy(self.capacity[:, time])
                id_task_ = self.problem.index_to_task_id[index_task]
                for event in self.states_events:
                    if id_task_ not in event.sink:
                        if event.active:
                            if (
                                event.end_event_triggered is None
                                or event.end_event_triggered > time
                            ):
                                # print("End event trigerred Release", event.end_event_triggered)
                                for r in event.event[2]:
                                    id_r = self.problem.resource_id_to_index[r]
                                    if id_r in nz:
                                        capa[id_r] -= event.event[2][r]
                                        if capa[id_r] < cons[id_r]:
                                            pass
                                            # print("Current task", id_task_)
                                            # print("sink", event.sink)
                                            # print("Current task schedule",
                                            #       solution.schedule[index_task])
                                            # for s in event.sink:
                                            #     print(solution.schedule[self.problem.task_id_to_index[s]])
                for event in self.state_groups:
                    if id_task_ not in event.event.tasks_group:
                        if event.active:
                            if (
                                event.end_event_triggered is None
                                or event.end_event_triggered > time
                            ):
                                # print("End event trigerred Group", event.end_event_triggered)
                                if event.event.res_not_released is not None:
                                    for r in event.event.res_not_released:
                                        id_r = self.problem.resource_id_to_index[r]
                                        if id_r in nz:
                                            capa[id_r] -= event.event.res_not_released[
                                                r
                                            ]
                    if id_task_ in event.event.tasks_group:
                        if event.event.no_overlap:
                            pass  # Do something here.
                return capa

            failed = False
            while duration_done < duration:
                next_start = next(
                    (
                        i
                        for i in range(cur_time, capacity.shape[1])
                        if np.min(
                            get_capacity_index_task(index_task=i_task, time=i) - cons
                        )
                        >= 0
                    ),
                    None,
                )
                if next_start is None:
                    tasks_order = tasks_order
                    currently_failed.add(i_task)
                    if id_task in new_schedule:
                        for ns, ne in new_schedule[id_task]:
                            for k in range(ns, ne):
                                capacity[:, k] += cons
                        new_schedule.pop(id_task)
                        failed = True
                    break
                next_end = next(
                    (
                        j
                        for j in range(
                            next_start, next_start + duration - duration_done
                        )
                        if np.min(
                            get_capacity_index_task(index_task=i_task, time=j) - cons
                        )
                        < 0
                    ),
                    None,
                )
                if next_end is None:
                    next_end = next_start + duration - duration_done
                new_schedule[id_task].append((next_start, next_end))
                for k in range(next_start, next_end):
                    capacity[:, k] -= cons
                duration_done += next_end - next_start
                cur_time = next_end + 1
            if not failed:
                currently_failed = set()
                scheduled_task.add(id_task)
                for e in self.state_groups:
                    e.update_state_of_event(new_schedule)
                for e in self.states_events:
                    e.update_state_of_event(new_schedule)
                prev_id_task = id_task
        return new_schedule


def compute_equivalent_preemptive_solution(
    flex_problem: FlexProblem, solution: ScheduleSolution
):
    """Dont change the actual schedule but make the solution preemptive"""
    schedule = []
    capacity = init_capacity_resource(flex_problem)
    resource_consumptions = resource_consumption(
        flex_problem=flex_problem, solution=solution
    )
    for i in range(flex_problem.nb_tasks):
        st, end = solution.schedule[i, :]
        st = int(st)
        end = int(end)
        if st == end:
            schedule.append([(st, end)])
            continue
        active_time = []
        active_slots = []
        start_slot = None
        for time in range(st, end):
            if np.min(capacity[:, time] - resource_consumptions[i, :]) >= 0:
                # All resource here.
                active_time.append(time)
                if start_slot is None:
                    start_slot = time
            else:
                if start_slot is not None:
                    active_slots.append((start_slot, active_time[-1] + 1))
                start_slot = None
        if start_slot is not None:
            active_slots.append((start_slot, end))
        sum_ = sum([x[1] - x[0] for x in active_slots])
        print(sum_, flex_problem.tasks[i].modes[1].duration)
        schedule.append(active_slots)
    return ScheduleSolutionPreemptive(
        problem=flex_problem, schedule=schedule, modes=solution.modes
    )


def check_solution(solution: ScheduleSolutionPreemptive, problem: FlexProblem):
    """To rework, do some basic check (was used to debug the postpro methods.."""
    resource_consumption = np.zeros((problem.nb_resources, problem.horizon))
    for i in range(len(solution.schedule)):
        mode = solution.modes[i]
        data = problem.tasks[i].modes[mode]
        duration = data.duration
        sum_ = sum([x[1] - x[0] for x in solution.schedule[i]])
        if sum_ != duration:
            print("Problem duration...", sum_, duration)
        res_conso = data.resource_consumption
        for ns, ne in solution.schedule[i]:
            if ne > ns:
                for r in res_conso:
                    id_r = problem.resource_id_to_index[r]
                    resource_consumption[id_r, ns:ne] += res_conso[r]
                    if (
                        np.min(
                            problem.resources[id_r].calendar_availability[ns:ne]
                            - resource_consumption[id_r, ns:ne]
                        )
                        < 0
                    ):
                        print(
                            resource_consumption[id_r, ns:ne],
                            problem.resources[id_r].calendar_availability[ns:ne],
                        )
                        print("PROBLEM !")

    for group in problem.tasks_group:
        if group.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE:
            tasks = [problem.task_id_to_index[t] for t in group.tasks_group]
            min_st = int(min([solution.schedule[i][0][0] for i in tasks]))
            max_st = int(max([solution.schedule[i][-1][1] for i in tasks]))
            for r in group.res_not_released:
                id_r = problem.resource_id_to_index[r]
                ind = [
                    i
                    for i in range(min_st, max_st)
                    if problem.resources[id_r].calendar_availability[i] > 0
                ]
                for j in ind:
                    resource_consumption[id_r, j] += group.res_not_released[r]
                    if (
                        resource_consumption[id_r, j]
                        - problem.resources[id_r].calendar_availability[j]
                        > 0
                    ):
                        print("PROBLEM ")
                        print(
                            f"PROBLEM with group consuming at time {j},",
                            resource_consumption[id_r, j],
                            problem.resources[id_r].calendar_availability[j],
                        )
