#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import dataclasses
import logging
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import ortools.sat.python.cp_model
from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    Domain,
    IntervalVar,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.flex_scheduling.fsp_utils import (
    create_resource_consumption_from_calendar,
    get_lb_ub_start_end_date,
    get_lb_ub_start_end_date_group_of_task,
)
from discrete_optimization.flex_scheduling.problem import (
    RESOURCE_KEY,
    CumulativeResource,
    FlexProblem,
    GroupType,
    NonRenewableResource,
    NoUnaryResource,
    ObjectiveParamEarliness,
    ObjectiveParamResource,
    ObjectiveParamTardiness,
    ObjectivesEnum,
    ResourceData,
    ScheduleSolution,
    Task,
    TaskData,
    TasksGroups,
)
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.cumulative_resource import (
    CumulativeResourceSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.non_renewable_resource import (
    NonRenewableCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.precedence_scheduling import (
    PrecedenceSchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConstraintIncluding:
    include_non_released_resource: bool = dataclasses.field(default=True)
    include_group_variables: bool = dataclasses.field(default=True)
    include_constraint_precedence_on_groups: bool = dataclasses.field(default=True)
    include_constraints_on_groups: bool = dataclasses.field(default=True)
    include_generalized_time_constraints: bool = dataclasses.field(default=True)
    include_variable_resource: bool = dataclasses.field(default=True)


class DurationEncodingEnum(Enum):
    INDICATOR = 0
    ELEMENT = 1


class CpSatFlexSolver(
    PrecedenceSchedulingCpSatSolver[Task],
    CumulativeResourceSchedulingCpSatSolver[Task, CumulativeResource, NoUnaryResource],
    NonRenewableCpSatSolver[Task, NonRenewableResource],
    WarmstartMixin,
):
    hyperparameters = [
        EnumHyperparameter(
            name="duration_encoding",
            enum=DurationEncodingEnum,
            default=DurationEncodingEnum.INDICATOR,
        )
    ]
    problem: FlexProblem

    def __init__(self, problem: FlexProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        (
            self.min_start_time,
            self.max_start_time,
            self.min_end_time,
            self.max_end_time,
        ) = get_lb_ub_start_end_date(problem=self.problem)
        self.durs: Dict[
            Tuple[int, int], Tuple[List[int], Dict[int, List[Tuple[int, int]]]]
        ] = self.problem.durations_data
        self.duration_encoding: DurationEncodingEnum = None

    def set_warm_start(self, solution: Solution) -> None:
        if self.solver is not None:
            self.cp_model.ClearHints()
            response = self.solver.ResponseProto()  # Get the raw response
            for i in range(len(response.solution)):
                var = self.cp_model.GetIntVarFromProtoIndex(i)
                # print(f"Variable {var} = {response.solution[i]}")
                self.cp_model.AddHint(var, response.solution[i])
        else:
            self.set_warm_start_from_sol(solution)

    def set_warm_start_from_sol(self, solution: Solution) -> None:
        l = self.from_solution_to_hint(solution)
        self.cp_model.ClearHints()
        done = set()
        for var, val in l:
            if str(var) not in done and isinstance(
                var, ortools.sat.python.cp_model.IntVar
            ):
                self.cp_model.AddHint(var, int(val))
                done.add(str(var))
            else:
                pass
                # print(var, " already")

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        index = self.problem.task_id_to_index[task]
        if start_or_end == StartOrEnd.START:
            return self.variables["starts"][index]
        elif start_or_end == StartOrEnd.END:
            return self.variables["ends"][index]
        return None

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        index = self.problem.task_id_to_index[task]
        return self.variables["is_present"][index][mode]

    def from_solution_to_hint(
        self, solution: ScheduleSolution
    ) -> list[tuple[ortools.sat.python.cp_model.VariableT, int]]:
        list_variables_value = []
        # Main variables
        for i in range(self.problem.nb_tasks):
            list_variables_value.append(
                (self.variables["starts"][i], solution.schedule[i, 0])
            )
            list_variables_value.append(
                (self.variables["ends"][i], solution.schedule[i, 1])
            )
            list_variables_value.append(
                (
                    self.variables["durations"][i],
                    solution.schedule[i, 1] - solution.schedule[i, 0],
                )
            )
        # Modes variables
        for i in range(self.problem.nb_tasks):
            mode_chosen = solution.modes[i]
            for mode in self.problem.tasks[i].modes:
                if mode == mode_chosen:
                    if len(self.problem.tasks[i].modes) > 1:
                        list_variables_value.append(
                            (self.variables["is_present"][i][mode], 1)
                        )
                    list_variables_value.append(
                        (
                            self.variables["opt_durations"][i][mode],
                            solution.schedule[i, 1] - solution.schedule[i, 0],
                        )
                    )
                else:
                    if len(self.problem.tasks[i].modes) > 1:
                        list_variables_value.append(
                            (self.variables["is_present"][i][mode], 0)
                        )
                    list_variables_value.append(
                        (self.variables["opt_durations"][i][mode], 0)
                    )

        # Group variables
        for group in self.problem.tasks_group:
            group_id = group.id
            ft = group.first_task_if_any
            lt = group.last_task_if_any
            if ft is not None:
                i = self.problem.task_id_to_index[ft]
                st_grp = solution.schedule[i, 0]
            else:
                st_grp = min(
                    [
                        solution.schedule[self.problem.task_id_to_index[id_task], 0]
                        for id_task in group.tasks_group
                    ]
                )
            if lt is not None:
                i = self.problem.task_id_to_index[lt]
                end_grp = solution.schedule[i, 1]
            else:
                end_grp = max(
                    [
                        solution.schedule[self.problem.task_id_to_index[id_task], 1]
                        for id_task in group.tasks_group
                    ]
                )

            list_variables_value.append(
                (self.variables["start_span_variables"][group_id], st_grp)
            )
            list_variables_value.append(
                (self.variables["end_span_variables"][group_id], end_grp)
            )
            list_variables_value.append(
                (self.variables["duration_span_variables"][group_id], end_grp - st_grp)
            )

        list_variables_value += self.from_solution_to_hint_non_released_delta(
            solution=solution
        )

        # Resource
        for resource in self.problem.resources:
            if "resource_capacity_variables" in self.variables:
                if resource.id in self.variables["resource_capacity_variables"]:
                    list_variables_value.append(
                        (
                            self.variables["resource_capacity_variables"][resource.id],
                            int(resource.max_capacity),
                        )
                    )
        # Indicator for duration..
        if "dictionary_indicators" in self.variables:
            for (index, mode), dur in self.variables["dictionary_indicators"]:
                if (
                    solution.schedule[index, 1] - solution.schedule[index, 0] == dur
                    and solution.modes[index] == mode
                ):
                    list_variables_value.append(
                        (
                            self.variables["dictionary_indicators"][
                                ((index, mode), dur)
                            ],
                            1,
                        )
                    )
                else:
                    list_variables_value.append(
                        (
                            self.variables["dictionary_indicators"][
                                ((index, mode), dur)
                            ],
                            0,
                        )
                    )

        groups = [
            g
            for g in self.problem.tasks_group
            if g.type_of_group == GroupType.SUBGROUP_TASK_FOR_OBJECTIVE
        ]
        nb_groups = len(groups)
        if "capacity_group_execution" in self.variables:
            list_variables_value.append(
                (self.variables["capacity_group_execution"], nb_groups)
            )  # could be reduced
        if "earliness_task" in self.variables:
            for t in self.variables["earliness_task"]:
                it = self.problem.task_id_to_index[t]
                end = solution.schedule[it, 1]
                list_variables_value.append(
                    (
                        self.variables["earliness_task"][t],
                        max([0, self.problem.tasks[it].max_ending_date - end]),
                    )
                )
        list_variables_value.append(
            (self.variables["makespan"], max(solution.schedule[:, 1]))
        )
        list_variables_value += self.from_solution_to_hint_earliness(solution)
        return list_variables_value

    def from_solution_to_hint_earliness(self, solution: ScheduleSolution):
        list_variables_value = []
        for id_task in self.variables["tardiness"]["tasks"]:
            index = self.problem.task_id_to_index[id_task]
            deadline = self.problem.task_id_dict[id_task].max_ending_date
            end = solution.schedule[index, 1]
            if end > deadline:
                list_variables_value.append(
                    (
                        self.variables["tardiness"]["tasks"][id_task]["tardiness"],
                        end - deadline,
                    )
                )
                if "earliness" in self.variables["tardiness"]["tasks"][id_task]:
                    list_variables_value.append(
                        (self.variables["tardiness"]["tasks"][id_task]["earliness"], 0)
                    )
            else:
                list_variables_value.append(
                    (self.variables["tardiness"]["tasks"][id_task]["tardiness"], 0)
                )
                if "earliness" in self.variables["tardiness"]["tasks"][id_task]:
                    list_variables_value.append(
                        (
                            self.variables["tardiness"]["tasks"][id_task]["earliness"],
                            deadline - end,
                        )
                    )
        for id_group in self.variables["tardiness"]["groups"]:
            group = self.problem.tasks_group[self.problem.group_id_to_index[id_group]]
            deadline = group.max_ending_date
            end = max(
                [
                    solution.schedule[self.problem.task_id_to_index[id_task], 1]
                    for id_task in group.tasks_group
                ]
            )
            if end > deadline:
                list_variables_value.append(
                    (
                        self.variables["tardiness"]["groups"][id_group]["tardiness"],
                        end - deadline,
                    )
                )
                if "earliness" in self.variables["tardiness"]["groups"][id_group]:
                    list_variables_value.append(
                        (
                            self.variables["tardiness"]["groups"][id_group][
                                "earliness"
                            ],
                            0,
                        )
                    )
            else:
                list_variables_value.append(
                    (self.variables["tardiness"]["groups"][id_group]["tardiness"], 0)
                )
                if "earliness" in self.variables["tardiness"]["groups"][id_group]:
                    list_variables_value.append(
                        (
                            self.variables["tardiness"]["groups"][id_group][
                                "earliness"
                            ],
                            deadline - end,
                        )
                    )
        return list_variables_value

    def from_solution_to_hint_non_released_delta(self, solution: ScheduleSolution):
        list_variables_value = []
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor
            is not None
        ):
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor
            for t1, t2, d_res in data:
                # This is the simple case where, the first element of the tuple is the task id, without any mode.
                if t1 in self.problem.task_id_to_index:
                    i1 = self.problem.task_id_to_index[t1]
                    i2 = self.problem.task_id_to_index[t2]
                    list_variables_value.append(
                        (
                            self.variables["durations_non_release_1"][(i1, i2)],
                            solution.schedule[i2, 0] - solution.schedule[i1, 1],
                        )
                    )
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            is not None
        ):
            durations_non_release = {}
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            for (t1, mode), t2, d_res in data:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                assert mode in self.variables["is_present"][i1]
                if (i1, mode, i2) not in durations_non_release:
                    if solution.modes[i1] == mode:
                        list_variables_value.append(
                            (
                                self.variables["durations_non_release_2"][
                                    (i1, mode, i2)
                                ],
                                solution.schedule[i2, 0] - solution.schedule[i1, 1],
                            )
                        )
                    else:
                        list_variables_value.append(
                            (
                                self.variables["durations_non_release_2"][
                                    (i1, mode, i2)
                                ],
                                0,
                            )
                        )
        if (
            self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            is not None
        ):
            data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            for t1, t2, d_res in data:
                tag = []
                st_ = None
                end_ = None
                if t1.is_a_task:
                    i1 = self.problem.task_id_to_index[t1.task_id]
                    st_ = solution.schedule[i1, 1]
                    tag.append(("task", t1.task_id))
                else:
                    group = t1.group_id
                    gr: TasksGroups = [
                        g for g in self.problem.tasks_group if g.id == group
                    ][0]
                    st_ = max(
                        [
                            solution.schedule[self.problem.task_id_to_index[id_task], 1]
                            for id_task in gr.tasks_group
                        ]
                    )
                    tag.append(("group", group))

                if t2.is_a_task:
                    i2 = self.problem.task_id_to_index[t2.task_id]
                    end_ = solution.schedule[i2, 0]
                    tag.append(("task", t2.task_id))

                else:
                    group = t2.group_id
                    gr: TasksGroups = [
                        g for g in self.problem.tasks_group if g.id == group
                    ][0]
                    end_ = min(
                        [
                            solution.schedule[self.problem.task_id_to_index[id_task], 0]
                            for id_task in gr.tasks_group
                        ]
                    )
                    tag.append(("group", group))
                tag = tuple(tag)
                if self.variables["durations_non_release_3"][tag] not in [
                    x[0] for x in list_variables_value
                ]:
                    list_variables_value.append(
                        (self.variables["durations_non_release_3"][tag], end_ - st_)
                    )
        return list_variables_value

    def init_main_variables(self):
        starts_variable = {}
        ends_variable = {}
        durations_variable = {}
        intervals_variable = {}
        for i in range(self.problem.nb_tasks):
            domain_intervals = []
            max_duration = 0
            for key in self.durs:
                if key[0] == i:
                    for val in self.durs[key][1]:
                        if val >= 0:
                            domain_intervals.extend(self.durs[key][1][val])
                            max_duration = max(max_duration, int(val))
            domain = Domain.FromIntervals(domain_intervals)
            starts_variable[i] = self.cp_model.NewIntVarFromDomain(
                domain=domain, name=f"start_{i}"
            )
            #     lb=self.min_start_time[i], ub=self.max_start_time[i], name=f"start_{i}"
            # )
            self.cp_model.add(starts_variable[i] >= int(self.min_start_time[i]))
            self.cp_model.add(starts_variable[i] <= int(self.max_start_time[i]))

            ends_variable[i] = self.cp_model.NewIntVar(
                lb=int(self.min_end_time[i]),
                ub=int(self.max_end_time[i])
                if not self.problem.tasks[i].soft_max_end_date
                else int(self.problem.horizon),
                # deadline is actually soft
                name=f"end_{i}",
            )
            min_duration = min(
                [
                    self.problem.tasks[i].modes[m].duration
                    for m in self.problem.tasks[i].modes
                ]
            )
            durations_variable[i] = self.cp_model.NewIntVar(
                lb=min_duration, ub=max_duration, name=f"duration_{i}"
            )
            intervals_variable[i] = self.cp_model.NewIntervalVar(
                start=starts_variable[i],
                size=durations_variable[i],
                end=ends_variable[i],
                name=f"interval_{i}",
            )
            allowed = []
            for val in self.durs[(i, 1)][1]:
                if val >= 0:
                    for lb, ub in self.durs[(i, 1)][1][val]:
                        for j in range(lb, ub + 1):
                            allowed.append((j, int(val)))
            # self.cp_model.add_allowed_assignments([starts_variable[i], durations_variable[i]],
            #                                      allowed)
        self.variables["starts"] = starts_variable
        self.variables["ends"] = ends_variable
        self.variables["durations"] = durations_variable
        self.variables["intervals"] = intervals_variable

    def init_optional_interval_variables(self):
        # This is for multimode tasks !
        is_present_variable = {}
        # opt_starts_variable = {} # might be useful one day.
        # opt_end_variable = {}
        opt_durations_variable = {}
        opt_intervals_variable = {}
        for i in range(self.problem.nb_tasks):
            nb_modes = len(self.problem.tasks[i].modes)
            is_present_variable[i] = {}
            opt_durations_variable[i] = {}
            opt_intervals_variable[i] = {}
            for mode in self.problem.tasks[i].modes:
                if nb_modes > 1:
                    is_present_variable[i][mode] = self.cp_model.NewBoolVar(
                        name=f"task_{i}_m_{mode}"
                    )
                else:
                    is_present_variable[i][mode] = (
                        True  # Maybe reconsider this when the task is totally.
                    )
                dur = self.problem.tasks[i].modes[mode].duration
                lb = dur
                ub = (
                    dur
                    if not self.problem.tasks[i]
                    .modes[mode]
                    .preemptive_on_resource_break
                    else 100 * dur
                )
                opt_durations_variable[i][mode] = self.cp_model.NewIntVar(
                    lb=lb, ub=ub, name=f"dur_task_{i}_m_{mode}"
                )
                if nb_modes == 1:
                    self.cp_model.Add(
                        opt_durations_variable[i][mode]
                        == self.variables["durations"][i]
                    )

                opt_intervals_variable[i][mode] = self.cp_model.NewOptionalIntervalVar(
                    start=self.variables["starts"][i],
                    size=opt_durations_variable[i][mode],
                    end=self.variables["ends"][i],
                    is_present=is_present_variable[i][mode],
                    name=f"interval_task_{i}_m_{mode}",
                )
            self.cp_model.AddExactlyOne(
                [is_present_variable[i][mode] for mode in is_present_variable[i]]
            )  # I choose 1 and only 1 mode.
        self.variables["opt_intervals"] = opt_intervals_variable
        self.variables["opt_durations"] = opt_durations_variable
        self.variables["is_present"] = is_present_variable

    def init_group_variables(self):
        if len(self.problem.tasks_group) == 0:
            return
        (
            gmin_start_time,
            gmax_start_time,
            gmin_end_time,
            gmax_end_time,
        ) = get_lb_ub_start_end_date_group_of_task(self.problem)
        span_interval_variables = {}
        start_span_variables = {}
        duration_span_variables = {}
        end_span_variables = {}
        for group in self.problem.tasks_group:
            group_id = group.id
            ft = group.first_task_if_any
            lt = group.last_task_if_any
            index_group = {self.problem.task_id_to_index[t] for t in group.tasks_group}
            if ft is not None:
                start_span_variables[group_id] = self.variables["starts"][
                    self.problem.task_id_to_index[ft]
                ]
            else:
                start_span_variables[group_id] = self.cp_model.NewIntVar(
                    lb=int(gmin_start_time[group_id]),
                    ub=int(gmax_start_time[group_id]),
                    name=f"start_span_gr_{group_id}",
                )
                self.cp_model.AddMinEquality(
                    start_span_variables[group_id],
                    [self.variables["starts"][i_x] for i_x in index_group],
                )
            if lt is not None:
                end_span_variables[group_id] = self.variables["ends"][
                    self.problem.task_id_to_index[lt]
                ]
            else:
                end_span_variables[group_id] = self.cp_model.NewIntVar(
                    lb=int(gmin_end_time[group_id]),
                    ub=int(gmax_end_time[group_id])
                    if not group.soft_max_end_date
                    else int(self.problem.horizon),
                    # deadline is actually soft,
                    name=f"end_span_gr_{group_id}",
                )
                self.cp_model.AddMaxEquality(
                    end_span_variables[group_id],
                    [self.variables["ends"][i_x] for i_x in index_group],
                )
            duration_span_variables[group_id] = self.cp_model.NewIntVar(
                lb=0, ub=int(self.problem.horizon), name=f"duration_span_gr_{group_id}"
            )
            # self.cp_model.Add(
            #     end_span_variables[group_id]
            #     == start_span_variables[group_id] + duration_span_variables[group_id]
            # )

            span_interval_variables[group_id] = self.cp_model.NewIntervalVar(
                start=start_span_variables[group_id],
                size=duration_span_variables[group_id],
                end=end_span_variables[group_id],
                name=f"interval_span_gr_{group_id}",
            )

        self.variables["start_span_variables"] = start_span_variables
        self.variables["duration_span_variables"] = duration_span_variables
        self.variables["end_span_variables"] = end_span_variables
        self.variables["span_interval_variables"] = span_interval_variables

    def constraint_reservoir(self, constraint_including: ConstraintIncluding = None):
        for res in self.problem.resources:
            if res.max_capacity == 1:
                task_mode_consume = [
                    (
                        index_task,
                        mode,
                        self.problem.tasks[index_task]
                        .modes[mode]
                        .get_res_consumption(res.id),
                    )
                    for index_task in range(self.problem.nb_tasks)
                    for mode in self.problem.tasks[index_task].modes
                    if self.problem.tasks[index_task]
                    .modes[mode]
                    .get_res_consumption(res.id)
                    > 0
                ]
                active = [
                    self.variables["is_present"][x[0]][x[1]] for x in task_mode_consume
                ]
                consos = [x[2] for x in task_mode_consume]
                times = [0]
                vals = [1]
                actives = [1]
                for i in range(len(consos)):
                    times.append(self.variables["starts"][task_mode_consume[i][0]])
                    vals.append(-consos[i])
                    actives.append(active[i])
                    times.append(self.variables["ends"][task_mode_consume[i][0]])
                    vals.append(consos[i])
                    actives.append(active[i])
                if (
                    self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
                    is not None
                ):
                    data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
                    for t1, t2, d_res in data:
                        if not t1.is_a_task or not t2.is_a_task:
                            continue
                        i1 = self.problem.task_id_to_index[t1.task_id]
                        i2 = self.problem.task_id_to_index[t2.task_id]
                        if res.id in d_res:
                            times.append(self.variables["ends"][i1])
                            vals.append(-d_res[res.id])
                            actives.append(1)
                            times.append(self.variables["starts"][i2])
                            vals.append(d_res[res.id])
                            actives.append(1)
                self.cp_model.AddReservoirConstraint(times, vals, 0, 1)

    def init_intervals_of_non_released_resource(
        self, constraint_including: ConstraintIncluding = None
    ):
        if constraint_including is None:
            constraint_including = ConstraintIncluding()
        intervals_non_release = {}
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor
            is not None
        ):
            durations_non_release = {}
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor
            for t1, t2, d_res in data:
                # This is the simple case where, the first element of the tuple is the task id, without any mode.
                if t1 in self.problem.task_id_to_index:
                    i1 = self.problem.task_id_to_index[t1]
                    i2 = self.problem.task_id_to_index[t2]
                    if (i1, i2) not in durations_non_release:
                        delta = self.cp_model.NewIntVar(
                            lb=0,
                            ub=self.max_start_time[i2] - self.min_end_time[i1],
                            name=f"delta_end_{i1}_start_{i2}",
                        )
                        durations_non_release[(i1, i2)] = delta
                    itv = self.cp_model.NewIntervalVar(
                        start=self.variables["ends"][i1],
                        size=durations_non_release[(i1, i2)],
                        end=self.variables["starts"][i2],
                        name=f"interval_end_{i1}_start_{i2}",
                    )
                    for res in d_res:
                        if res not in intervals_non_release:
                            intervals_non_release[res] = []
                        intervals_non_release[res].append((itv, d_res[res]))
            self.variables["durations_non_release_1"] = durations_non_release
        if (
            self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            is not None
        ):
            durations_non_release = {}
            data = self.problem.constraints.successor_with_res_release_at_start_of_successor_mode
            for (t1, mode), t2, d_res in data:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                assert mode in self.variables["is_present"][i1]
                if (i1, mode, i2) not in durations_non_release:
                    delta = self.cp_model.NewIntVar(
                        lb=0,
                        ub=self.max_start_time[i2] - self.min_end_time[i1],
                        name=f"delta_end_{i1}_start_{i2}",
                    )
                    durations_non_release[(i1, mode, i2)] = delta
                # This
                itv = self.cp_model.NewOptionalIntervalVar(
                    start=self.variables["ends"][i1],
                    size=durations_non_release[(i1, mode, i2)],
                    end=self.variables["starts"][i2],
                    is_present=self.variables["is_present"][i1][mode],
                    name=f"interval_end_{i1}_start_{i2}",
                )
                for res in d_res:
                    if res not in intervals_non_release:
                        intervals_non_release[res] = []
                    intervals_non_release[res].append((itv, d_res[res]))
                self.variables["durations_non_release_2"] = durations_non_release
        if (
            self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            is not None
        ):
            durations_non_release = {}
            data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
            for t1, t2, d_res in data:
                tag = []
                if (
                    not t1.is_a_task or not t2.is_a_task
                ) and not constraint_including.include_group_variables:
                    continue
                if t1.is_a_task:
                    i1 = self.problem.task_id_to_index[t1.task_id]
                    min_end_time = self.min_end_time[i1]
                    end_t1 = self.variables["ends"][i1]
                    tag.append(("task", t1.task_id))
                else:
                    group = t1.group_id
                    gr: TasksGroups = [
                        g for g in self.problem.tasks_group if g.id == group
                    ][0]
                    min_end_time = max(
                        [
                            self.min_end_time[self.problem.task_id_to_index[i]]
                            for i in gr.tasks_group
                        ]
                    )
                    end_t1 = self.variables["end_span_variables"][group]
                    tag.append(("group", group))

                if t2.is_a_task:
                    i2 = self.problem.task_id_to_index[t2.task_id]
                    max_start_time = self.max_start_time[i2]
                    start_t2 = self.variables["starts"][i2]
                    tag.append(("task", t2.task_id))
                else:
                    group = t2.group_id
                    gr: TasksGroups = [
                        g for g in self.problem.tasks_group if g.id == group
                    ][0]
                    max_start_time = min(
                        [
                            self.max_start_time[self.problem.task_id_to_index[i]]
                            for i in gr.tasks_group
                        ]
                    )
                    start_t2 = self.variables["start_span_variables"][group]
                    tag.append(("group", group))
                tag = tuple(tag)
                if tag not in durations_non_release:
                    delta = self.cp_model.NewIntVar(
                        lb=0,
                        ub=int(max_start_time - min_end_time),
                        name=f"delta_end_{tag[0]}_start_{tag[1]}",
                    )
                    durations_non_release[tag] = delta
                self.cp_model.add(start_t2 >= end_t1)
                itv = self.cp_model.NewIntervalVar(
                    start=end_t1,
                    size=durations_non_release[tag],
                    end=start_t2,
                    name=f"interval_end_{tag[0]}_start_{tag[1]}",
                )
                for res in d_res:
                    if res not in intervals_non_release:
                        intervals_non_release[res] = []
                    intervals_non_release[res].append((itv, d_res[res]))
                self.variables["durations_non_release_3"] = durations_non_release

        self.variables["intervals_non_release"]: Dict[
            RESOURCE_KEY, list[tuple[IntervalVar, int]]
        ] = intervals_non_release

    def init_resource_variables(self):
        resource_capacity_var = {}
        obj_param = self.problem.objective_params.params_obj.get(
            ObjectivesEnum.RESOURCE_COST, None
        )
        if obj_param is not None:
            for resource in self.problem.resources:
                if (
                    obj_param.weight_per_resource_unit.get(resource.id, 0) > 0
                    or obj_param.consider_in_objectives[resource.id]
                ):
                    resource_capacity_var[resource.id] = self.cp_model.NewIntVar(
                        lb=0,
                        ub=int(resource.max_capacity),
                        name=f"res_capacity_{resource.id}",
                    )
        self.variables["resource_capacity_variables"] = resource_capacity_var

    def init_and_constraint_wip_variables(self):
        # This is a variable to store some "capacity" variable on the number of work in progress.
        if self.problem.tasks_group is None or len(self.problem.tasks_group) == 0:
            return
        groups = [
            g
            for g in self.problem.tasks_group
            if g.type_of_group == GroupType.SUBGROUP_TASK_FOR_OBJECTIVE
        ]
        nb_groups = len(groups)
        if nb_groups == 0:
            return
        capacity_group_execution = self.cp_model.NewIntVar(
            lb=1, ub=nb_groups, name="capacity_nb_group_in_progress"
        )
        self.cp_model.AddCumulative(
            [
                self.variables["span_interval_variables"][
                    g.id
                ]  # this spans over the entire group of task.
                for g in groups
            ],
            [1] * nb_groups,
            capacity_group_execution,
        )
        self.variables["capacity_group_execution"] = capacity_group_execution

    def constraint_precedence(self):
        """
        Basic precedence constraint
        """
        for t_id in self.problem.constraints.successors:
            index = self.problem.task_id_to_index[t_id]
            for succ in self.problem.constraints.successors[t_id]:
                succ_index = self.problem.task_id_to_index[succ]
                self.cp_model.Add(
                    self.variables["starts"][succ_index]
                    >= self.variables["ends"][index]
                )

    def constraint_precedence_on_groups(self):
        """
        Basic precedence constraint between group of tasks
        """
        if self.problem.constraints.successors_group_tasks is None:
            return
        for g_id in self.problem.constraints.successors_group_tasks:
            for g_succ_id in self.problem.constraints.successors_group_tasks[g_id]:
                self.cp_model.Add(
                    self.variables["start_span_variables"][g_succ_id]
                    >= self.variables["end_span_variables"][g_id]
                )

    def constraint_duration_of_tasks(self):
        """
        Tricky constraint : should take into account the partial preemption possibility,
        which makes duration variable based on calendars
        """
        durs = self.durs
        dictionary_indicators = {}
        if self.duration_encoding == DurationEncodingEnum.INDICATOR:
            for task_index, mode in durs:
                d = self.constraint_duration_of_task(
                    task_index=task_index,
                    mode=mode,
                    duration_per_interval=durs[(task_index, mode)][1],
                )
                dictionary_indicators.update(d)
            self.variables["dictionary_indicators"] = dictionary_indicators
            for index in self.variables["is_present"]:
                all_key = [
                    x
                    for x in self.variables["dictionary_indicators"]
                    if x[0][0] == index
                ]
                self.cp_model.AddExactlyOne(
                    [self.variables["dictionary_indicators"][x] for x in all_key]
                )
        elif self.duration_encoding == DurationEncodingEnum.ELEMENT:
            for task_index, mode in durs:
                self.constraint_duration_of_task_element(task_index, mode)

    def constraint_duration_of_task_element(
        self,
        task_index: int,
        mode: int,
    ):
        positive_durations = [d for d in self.durs[task_index, mode][1] if d >= 0]
        duration_per_interval = self.durs[task_index, mode][1]
        duration_list = self.durs[task_index, mode][0]
        if len(positive_durations) == 1:
            dur = int(positive_durations[0])
            interval = Domain.FromIntervals(duration_per_interval[dur])
            self.cp_model.AddLinearExpressionInDomain(
                self.variables["starts"][task_index], interval
            )
            self.cp_model.Add(self.variables["opt_durations"][task_index][mode] == dur)
        else:
            duration_list = [int(x) for x in duration_list]
            self.cp_model.add_element(
                self.variables["starts"][task_index],
                expressions=duration_list,
                target=self.variables["opt_durations"][task_index][mode],
            )
        return None

    def constraint_duration_of_task(
        self,
        task_index: int,
        mode: int,
        duration_per_interval: Dict[int, List[Tuple[int, int]]],
    ):
        dictionary_indicators = {}
        positive_durations = [d for d in duration_per_interval if d >= 0]
        if len(positive_durations) == 1:
            dur = int(positive_durations[0])
            interval = Domain.FromIntervals(duration_per_interval[dur])
            self.cp_model.AddLinearExpressionInDomain(
                self.variables["starts"][task_index], interval
            )
            self.cp_model.Add(self.variables["opt_durations"][task_index][mode] == dur)
            dictionary_indicators[((task_index, mode), dur)] = self.variables[
                "is_present"
            ][task_index][mode]
        else:
            for possible_duration in duration_per_interval:
                if possible_duration < 0:
                    continue
                interval = Domain.FromIntervals(
                    duration_per_interval[possible_duration]
                )
                dictionary_indicators[((task_index, mode), possible_duration)] = (
                    self.cp_model.NewBoolVar(
                        f"d_{(task_index, mode), possible_duration}"
                    )
                )
                self.cp_model.AddLinearExpressionInDomain(
                    self.variables["starts"][task_index], interval
                ).OnlyEnforceIf(
                    dictionary_indicators[((task_index, mode), possible_duration)]
                )
                self.cp_model.Add(
                    self.variables["opt_durations"][task_index][mode]
                    == int(possible_duration)
                ).OnlyEnforceIf(
                    dictionary_indicators[((task_index, mode), possible_duration)]
                )

            # corrected version (to be confirmed)
            self.cp_model.Add(
                sum([dictionary_indicators[k] for k in dictionary_indicators])
                == self.variables["is_present"][task_index][mode]
            )

            # Seems incorrect with non duplicated starts variables, even though it works most of the time
            # self.cp_model.Add(
            #    sum([dictionary_indicators[k] for k in dictionary_indicators]) == 1
            # )  # Only one indicator activated.
            # self.cp_model.AddExactlyOne([dictionary_indicators[k] for k in dictionary_indicators])
        return dictionary_indicators

    def constraint_cumulative(self, constraint_including: ConstraintIncluding):
        for r in self.problem.resources:
            if r.renewable:
                if constraint_including.include_variable_resource:
                    if r.id in self.variables["resource_capacity_variables"]:
                        self.constraint_cumulative_resource(
                            resource=r,
                            variable_max_capacity=True,
                            constraint_including=constraint_including,
                        )
                self.constraint_cumulative_resource(
                    resource=r,
                    variable_max_capacity=False,
                    constraint_including=constraint_including,
                )
            else:
                self.constraint_non_renewable_resource(
                    resource=r, variable_max_capacity=True
                )
                self.constraint_non_renewable_resource(
                    resource=r, variable_max_capacity=False
                )

    def constraint_non_renewable_resource(
        self, resource: ResourceData, variable_max_capacity: bool = False
    ):
        capa = resource.max_capacity
        id_resource = resource.id
        task_mode_consume = [
            (
                self.variables["is_present"][i][mode],
                int(self.problem.tasks[i].modes[mode].get_res_consumption(id_resource)),
            )
            for i in self.variables["opt_intervals"]
            for mode in self.variables["opt_intervals"][i]
            if self.problem.tasks[i].modes[mode].get_res_consumption(id_resource) > 0
        ]
        if not variable_max_capacity:
            self.cp_model.Add(sum([x[0] * x[1] for x in task_mode_consume]) <= capa)
        else:
            self.cp_model.Add(
                sum([x[0] * x[1] for x in task_mode_consume])
                == self.variables["resource_capacity_variables"][resource.id]
            )

    def constraint_cumulative_resource(
        self,
        resource: ResourceData,
        variable_max_capacity: bool = False,
        constraint_including: ConstraintIncluding = None,
    ):
        post_cumulative_constraints(
            problem=self.problem,
            resource=resource,
            solver=self,
            variable_max_capacity=variable_max_capacity,
            include_intervals_non_release=constraint_including.include_non_released_resource,
        )

    def constraint_cumulative_resource_depr(
        self, resource: ResourceData, variable_max_capacity: bool = False
    ):
        """
        Simple and tested version
        """
        res_comp: List[Dict[str, int]] = create_resource_consumption_from_calendar(
            calendar_availability=resource.calendar_availability
        )
        id_resource = resource.id
        task_mode_consume = [
            (
                self.variables["opt_intervals"][i][mode],
                int(self.problem.tasks[i].modes[mode].get_res_consumption(id_resource)),
            )
            for i in self.variables["opt_intervals"]
            for mode in self.variables["opt_intervals"][i]
            if self.problem.tasks[i].modes[mode].get_res_consumption(id_resource) > 0
        ]
        if "intervals_non_release" in self.variables:
            if resource.id in self.variables["intervals_non_release"]:
                task_mode_consume += self.variables["intervals_non_release"][
                    resource.id
                ]
        fake_task_res = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name=f"res_"
                ),
                f["value"],
            )
            for f in res_comp
            if f["value"] > 0 and f["value"] != resource.max_capacity
        ]
        if not variable_max_capacity:
            self.cp_model.AddCumulative(
                [x[0] for x in task_mode_consume] + [x[0] for x in fake_task_res],
                [x[1] for x in task_mode_consume] + [x[1] for x in fake_task_res],
                resource.max_capacity,
            )
        else:
            self.cp_model.AddCumulative(
                [x[0] for x in task_mode_consume] + [x[0] for x in fake_task_res],
                [x[1] for x in task_mode_consume] + [x[1] for x in fake_task_res],
                self.variables["resource_capacity_variables"][resource.id],
            )

    def constraint_on_groups_of_task(self):
        # self.constraint_group_non_release_resource_reservoir()
        self.constraint_group_non_release_resource()
        self.constraint_non_overlap_group()

    def constraint_sequencing_on_disjunctive_resource(self):
        # Sequence var like idea.
        for res in self.problem.resources:
            if res.max_capacity == 1:
                task_mode_consume = [
                    (
                        index_task,
                        mode,
                        self.problem.tasks[index_task]
                        .modes[mode]
                        .get_res_consumption(res.id),
                    )
                    for index_task in range(self.problem.nb_tasks)
                    for mode in self.problem.tasks[index_task].modes
                    if self.problem.tasks[index_task]
                    .modes[mode]
                    .get_res_consumption(res.id)
                    > 0
                ]
                tasks = []
                for i in range(len(task_mode_consume)):
                    tasks.append(task_mode_consume[i][0])
                list_next = []
                if (
                    self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
                    is not None
                ):
                    data = self.problem.constraints.successor_generic_with_res_release_at_start_of_successor_generic
                    for t1, t2, d_res in data:
                        if not t1.is_a_task or not t2.is_a_task:
                            continue
                        i1 = self.problem.task_id_to_index[t1.task_id]
                        i2 = self.problem.task_id_to_index[t2.task_id]
                        list_next += [(i1, i2)]
                        if i1 not in tasks:
                            tasks.append(i1)
                        if i2 not in tasks:
                            tasks.append(i2)
                if len(list_next) > 0:
                    tasks.append(-1)
                    vars = {
                        (i, j): self.cp_model.NewBoolVar(name=f"{j}_after_{i}")
                        for i in tasks
                        for j in tasks
                        if i != j
                    }
                    for i in tasks:
                        if i != -1:
                            self.cp_model.add(
                                sum(vars[i, j] for j in tasks if i != j) == 1
                            )
                        else:
                            self.cp_model.add(
                                sum(vars[j, i] for j in tasks if i != j) == 1
                            )
                    for i, j in vars:
                        if i != -1 and j != -1:
                            self.cp_model.add(
                                self.variables["starts"][j] >= self.variables["ends"][i]
                            ).only_enforce_if(vars[i, j])
                        if (i, j) in list_next:
                            self.cp_model.add(vars[i, j] == 1)

    def constraint_group_non_release_resource(self):
        groups_non_release_resource = [
            g
            for g in self.problem.tasks_group
            if g.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE
        ]
        all_resource_concerned = set()
        for g in groups_non_release_resource:
            all_resource_concerned.update(set([r for r in g.res_not_released]))
        for resource in all_resource_concerned:
            intervals_and_consumption = []
            tasks_covered_in_group = set()
            for g in groups_non_release_resource:
                if "span_interval_variables" in self.variables:
                    if (
                        resource in g.res_not_released
                        and g.res_not_released[resource] > 0
                    ):
                        intervals_and_consumption.append(
                            (
                                self.variables["span_interval_variables"][g.id],
                                g.res_not_released[resource],
                            )
                        )
                        tasks_covered_in_group.update(g.tasks_group)
            task_mode_consume = [
                (
                    self.variables["opt_intervals"][i][mode],
                    int(
                        self.problem.tasks[i].modes[mode].get_res_consumption(resource)
                    ),
                )
                for i in self.variables["opt_intervals"]
                for mode in self.variables["opt_intervals"][i]
                if self.problem.tasks[i].modes[mode].get_res_consumption(resource) > 0
                and self.problem.index_to_task_id[i] not in tasks_covered_in_group
            ]
            task_non_release = self.variables["intervals_non_release"].get(resource, [])
            self.cp_model.AddCumulative(
                [
                    x[0]
                    for x in task_mode_consume
                    + intervals_and_consumption
                    + task_non_release
                ],
                [
                    x[1]
                    for x in task_mode_consume
                    + intervals_and_consumption
                    + task_non_release
                ],
                self.problem.resource_dict[resource].max_capacity,
            )

    def constraint_group_non_release_resource_reservoir(self):
        groups_non_release_resource = [
            g
            for g in self.problem.tasks_group
            if g.type_of_group == GroupType.GROUP_TASK_NON_RELEASED_RESOURCE
        ]
        all_resource_concerned = set()
        for g in groups_non_release_resource:
            all_resource_concerned.update(set([r for r in g.res_not_released]))
        for resource in all_resource_concerned:
            intervals_and_consumption = []
            tasks_covered_in_group = set()
            for g in groups_non_release_resource:
                if resource in g.res_not_released and g.res_not_released[resource] > 0:
                    intervals_and_consumption.append(
                        (
                            self.variables["start_span_variables"][g.id],
                            self.variables["end_span_variables"][g.id],
                            g.res_not_released[resource],
                        )
                    )
                    tasks_covered_in_group.update(g.tasks_group)
            task_mode_consume = [
                (
                    (i, mode),
                    int(
                        self.problem.tasks[i].modes[mode].get_res_consumption(resource)
                    ),
                )
                for i in self.variables["opt_intervals"]
                for mode in self.variables["opt_intervals"][i]
                if self.problem.tasks[i].modes[mode].get_res_consumption(resource) > 0
                and self.problem.index_to_task_id[i] not in tasks_covered_in_group
            ]
            times = [0]
            active = [1]
            vals = [self.problem.resource_dict[resource].max_capacity]
            for i in range(len(task_mode_consume)):
                (index, mode), val = task_mode_consume[i]
                times.append(self.variables["starts"][index])
                vals.append(-val)
                active.append(self.variables["is_present"][index][mode])
                times.append(self.variables["ends"][index])
                vals.append(val)
                active.append(self.variables["is_present"][index][mode])
            for i in range(len(intervals_and_consumption)):
                times.append(intervals_and_consumption[i][0])
                vals.append(-intervals_and_consumption[i][2])
                active.append(1)
                times.append(intervals_and_consumption[i][1])
                vals.append(intervals_and_consumption[i][2])
                active.append(1)
            self.cp_model.AddReservoirConstraint(
                times=times,
                level_changes=vals,
                min_level=0,
                max_level=self.problem.resource_dict[resource].max_capacity,
            )
            # self.cp_model.AddReservoirConstraintWithActive(times=times,
            #                                               level_changes=vals,
            #                                               actives=active,
            #                                               min_level=0,
            #                                               max_level=self.problem.resource_dict[resource].max_capacity)

    def constraint_non_overlap_group(self):
        for group in self.problem.tasks_group:
            if group.no_overlap:
                intervals = [
                    self.variables["intervals"][self.problem.task_id_to_index[i_t]]
                    for i_t in group.tasks_group
                ]
                self.cp_model.AddNoOverlap(intervals)
                # TODO Be able to remove dummy tasks from the group,
                # self.cp_model.AddCumulative(intervals, [1] * len(intervals), 1)

    def constraint_generalized_time_constraint(self):
        if self.problem.constraints.start_at_start is not None:
            for t1, t2 in self.problem.constraints.start_at_start:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.Add(
                    self.variables["starts"][i1] == self.variables["starts"][i2]
                )
        if self.problem.constraints.start_at_end is not None:
            for t1, t2 in self.problem.constraints.start_at_end:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.Add(
                    self.variables["starts"][i1] == self.variables["ends"][i2]
                )
        if self.problem.constraints.start_at_end_plus_offset is not None:
            for t1, t2, offset in self.problem.constraints.start_at_end_plus_offset:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.Add(
                    self.variables["starts"][i1] == self.variables["ends"][i2] + offset
                )
        if self.problem.constraints.start_after_end_plus_offset is not None:
            for t1, t2, offset in self.problem.constraints.start_after_end_plus_offset:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.Add(
                    self.variables["starts"][i1] >= self.variables["ends"][i2] + offset
                )
        if self.problem.constraints.start_at_start_plus_offset is not None:
            for t1, t2, offset in self.problem.constraints.start_at_start_plus_offset:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.Add(
                    self.variables["starts"][i1]
                    == self.variables["starts"][i2] + offset
                )
        if self.problem.constraints.start_after_start_plus_offset is not None:
            for (
                t1,
                t2,
                offset,
            ) in self.problem.constraints.start_after_start_plus_offset:
                i1 = self.problem.task_id_to_index[t1]
                i2 = self.problem.task_id_to_index[t2]
                self.cp_model.Add(
                    self.variables["starts"][i1]
                    >= self.variables["starts"][i2] + offset
                )

    def create_objectives(self, constraint_including: ConstraintIncluding = None):
        if constraint_including is None:
            constraint_including = ConstraintIncluding()
        objs = []
        weights = []
        names = []
        for obj_enum in self.problem.objective_params.params_obj:
            if obj_enum == ObjectivesEnum.MAKESPAN:
                var, name = self.create_makespan()
                objs.append(var)
                weights.append(self.problem.objective_params.params_obj[obj_enum])
                names.append(name)
            if (
                obj_enum == ObjectivesEnum.RESOURCE_COST
                and constraint_including.include_variable_resource
            ):
                var, name = self.create_resource_objective(
                    obj_params_resource=self.problem.objective_params.params_obj[
                        obj_enum
                    ]
                )
                objs.append(var)
                weights.append(
                    self.problem.objective_params.params_obj[obj_enum].weight
                )
                names.append(name)
            if obj_enum == ObjectivesEnum.WORK_IN_PROGRESS:
                obj_params_wip = self.problem.objective_params.params_obj[obj_enum]
                if obj_params_wip.count_nb_group_in_progress:
                    self.init_and_constraint_wip_variables()
                    objs.append(self.variables["capacity_group_execution"])
                    weights.append(obj_params_wip.coefficient_on_nb_group_in_progress)
                    names.append("wip_cost")
            if obj_enum == ObjectivesEnum.TARDINESS:
                var, name = self.create_tardiness_objective(
                    obj_tardiness=self.problem.objective_params.params_obj[obj_enum]
                )
                objs.append(var)
                weights.append(1)
                names.append(name)
            if obj_enum == ObjectivesEnum.EARLINESS:
                var, name = self.create_earliness_objective(
                    obj_earliness=self.problem.objective_params.params_obj[obj_enum]
                )
                objs.append(var)
                weights.append(1)
                names.append(name)
            if obj_enum == ObjectivesEnum.NON_RELEASE_DURATION:
                expr = sum(
                    [
                        self.variables["durations_non_release_1"][t]
                        for t in self.variables.get("durations_non_release_1", [])
                    ]
                    + [
                        self.variables["durations_non_release_2"][t]
                        for t in self.variables.get("durations_non_release_2", [])
                    ]
                    + [
                        self.variables["durations_non_release_3"][t]
                        for t in self.variables.get("durations_non_release_3", [])
                    ]
                )
                objs.append(expr)
                weights.append(1)
                names.append("non_release_duration")
        self.variables["obj_data"] = (objs, weights, names)
        self.cp_model.Minimize(sum([objs[i] * weights[i] for i in range(len(objs))]))

    def create_makespan(self):
        makespan = self.cp_model.NewIntVar(
            lb=0, ub=int(self.problem.horizon), name="makespan"
        )
        self.cp_model.AddMaxEquality(
            makespan, [self.variables["ends"][x] for x in self.variables["ends"]]
        )
        self.variables["makespan"] = makespan
        return makespan, "makespan"

    def create_resource_objective(self, obj_params_resource: ObjectiveParamResource):
        resource_cost = [
            self.variables["resource_capacity_variables"][r]
            * obj_params_resource.weight_per_resource_unit[r]
            for r in obj_params_resource.weight_per_resource_unit
            if obj_params_resource.weight_per_resource_unit[r] != 0
            and (
                (r not in obj_params_resource.consider_in_objectives)
                or (obj_params_resource.consider_in_objectives[r])
            )
        ]
        return sum(resource_cost), "resource_cost"

    def create_earliness_objective(self, obj_earliness: ObjectiveParamEarliness):
        self.variables["earliness"] = {"tasks": {}, "groups": {}}
        cost_list: List[Tuple[LinearExprT, float]] = []
        for id_task in obj_earliness.weight_per_task:
            if obj_earliness.weight_per_task[id_task] > 0:
                index = self.problem.task_id_to_index[id_task]
                deadline = self.problem.task_id_dict[id_task].max_ending_date
                if deadline is not None:
                    end = self.variables["ends"][index]
                    # Create lateness/earliness variables
                    earliness = self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.horizon, name=f"earliness_task_{id_task}"
                    )
                    self.cp_model.Add(earliness >= deadline - end)
                    # cost_expr = penalty * lateness + earliness
                    cost_expr = earliness
                    cost_list.append(
                        (cost_expr, obj_earliness.weight_per_task[id_task])
                    )
                    self.variables["earliness"]["tasks"][id_task] = {
                        "earliness": earliness,
                    }
        for id_group in obj_earliness.weight_per_groups:
            if obj_earliness.weight_per_groups[id_group] > 0:
                index = self.problem.group_id_to_index[id_group]
                deadline = self.problem.tasks_group[index].max_ending_date
                soft = self.problem.tasks_group[index].soft_max_end_date
                if deadline is not None:
                    end = self.variables["end_span_variables"][id_group]
                    # Create lateness/earliness variables
                    earliness = self.cp_model.NewIntVar(
                        lb=0,
                        ub=self.problem.horizon,
                        name=f"earliness_group_{id_group}",
                    )
                    self.cp_model.Add(earliness >= deadline - end)
                    # cost_expr = penalty * lateness + earliness
                    cost_expr = earliness
                    cost_list.append(
                        (cost_expr, obj_earliness.weight_per_groups[id_group])
                    )
                    self.variables["earliness"]["groups"][id_group] = {
                        "earliness": earliness,
                    }
        return (
            LinearExpr.weighted_sum(
                [x[0] for x in cost_list], [x[1] for x in cost_list]
            ),
            "earliness",
        )

    def create_tardiness_objective(self, obj_tardiness: ObjectiveParamTardiness):
        self.variables["tardiness"] = {"tasks": {}, "groups": {}}
        cost_list: List[Tuple[LinearExprT, float]] = []
        for id_task in obj_tardiness.weight_per_task:
            if obj_tardiness.weight_per_task[id_task] > 0:
                index = self.problem.task_id_to_index[id_task]
                deadline = self.problem.task_id_dict[id_task].max_ending_date
                if deadline is not None:
                    end = self.variables["ends"][index]

                    # Create lateness/earliness variables
                    lateness = self.cp_model.NewIntVar(
                        lb=0,
                        ub=int(self.problem.horizon),
                        name=f"lateness_task_{id_task}",
                    )

                    self.cp_model.Add(lateness >= end - int(deadline))

                    cost_list.append((lateness, obj_tardiness.weight_per_task[id_task]))

                    self.variables["tardiness"]["tasks"][id_task] = {
                        "tardiness": lateness,
                    }
        for id_group in obj_tardiness.weight_per_groups:
            if obj_tardiness.weight_per_groups[id_group] > 0:
                index = self.problem.group_id_to_index[id_group]
                deadline = self.problem.tasks_group[index].max_ending_date
                soft = self.problem.tasks_group[index].soft_max_end_date
                if deadline is not None:
                    end = self.variables["end_span_variables"][id_group]

                    # Create lateness/earliness variables
                    lateness = self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.horizon, name=f"lateness_group_{id_group}"
                    )

                    self.cp_model.Add(lateness >= end - int(deadline))

                    cost_list.append(
                        (lateness, obj_tardiness.weight_per_groups[id_group])
                    )

                    self.variables["tardiness"]["groups"][id_group] = {
                        "tardiness": lateness,
                    }
        return (
            LinearExpr.weighted_sum(
                [x[0] for x in cost_list], [x[1] for x in cost_list]
            ),
            "tardiness",
        )

    def init_model(
        self, constraint_including: Optional[ConstraintIncluding] = None, **args: Any
    ) -> None:
        if constraint_including is None:
            constraint_including = ConstraintIncluding()
        args = self.complete_with_default_hyperparameters(args)
        super().init_model(**args)
        self.duration_encoding = args["duration_encoding"]
        self.init_main_variables()  # main interval variables
        self.init_optional_interval_variables()  # for multimode
        if constraint_including.include_group_variables:
            self.init_group_variables()  # create span interval variable for relevant group of tasks.
        # for cases where some resource are not released at the end of a task, but rather on the starting of one another
        if constraint_including.include_non_released_resource:
            self.init_intervals_of_non_released_resource(
                constraint_including=constraint_including
            )
            # self.constraint_reservoir(constraint_including=constraint_including)
            # self.constraint_sequencing_on_disjunctive_resource()
        if constraint_including.include_variable_resource:
            self.init_resource_variables()  #
        self.constraint_precedence()
        if constraint_including.include_constraint_precedence_on_groups:
            self.constraint_precedence_on_groups()
        self.constraint_duration_of_tasks()
        self.constraint_cumulative(constraint_including=constraint_including)
        if constraint_including.include_constraints_on_groups:
            self.constraint_on_groups_of_task()
        if constraint_including.include_generalized_time_constraints:
            self.constraint_generalized_time_constraint()
        self.create_objectives(constraint_including)

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return self.variables["obj_data"][2]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        if isinstance(obj, tuple):
            return sum(
                [
                    obj[2 * i + 1] * res[-1][0]._intern_obj[obj[2 * i]]
                    for i in range(len(obj) // 2)
                ]
            )
        sol = res[-1][0]
        return sol._intern_obj[obj]

    def get_objr_expr(self, obj: Union[str, tuple]):
        if isinstance(obj, tuple):
            nb_objective = len(obj) // 2
            objs = [obj[2 * i] for i in range(nb_objective)]
            weights = [obj[2 * i + 1] for i in range(nb_objective)]
            objs_expr = [self.get_objr_expr(ob) for ob in objs]
            return LinearExpr.weighted_sum(objs_expr, weights)
        ind_obj = next(
            (
                i
                for i in range(len(self.variables["obj_data"][2]))
                if self.variables["obj_data"][2][i] == obj
            ),
            None,
        )
        return self.variables["obj_data"][0][ind_obj]

    def set_lexico_objective(self, obj: str) -> None:
        expr = self.get_objr_expr(obj)
        if expr is not None:
            self.cp_model.Minimize(expr)
        else:
            logger.warning(f"{obj} objective is absent it seems")

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        expr = self.get_objr_expr(obj)
        if expr is not None:
            self.cp_model.Add(expr <= value)
        else:
            logger.warning(f"{obj} objective is absent it seems")

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> ScheduleSolution:
        logger.info(
            f"cur obj value : {cpsolvercb.ObjectiveValue()}, bound={cpsolvercb.BestObjectiveBound()}"
        )
        logger.info("Sub-objectives :")
        details_subobj = {}
        for i in range(len(self.variables["obj_data"][0])):
            try:
                logger.info(
                    f"{self.variables['obj_data'][2][i]} : {cpsolvercb.Value(self.variables['obj_data'][0][i])}"
                )
                details_subobj[self.variables["obj_data"][2][i]] = cpsolvercb.Value(
                    self.variables["obj_data"][0][i]
                )
            except Exception as e:
                print(e)
        if "resource_capacity_variables" in self.variables:
            logger.info("Resource capacity data : ")
            logger.info(
                f"{[cpsolvercb.Value(self.variables['resource_capacity_variables'][x]) for x in self.variables['resource_capacity_variables']]}"
            )
        if "tardiness" in self.variables:
            logger.info("Tardiness data")
            for group in self.variables["tardiness"]["groups"]:
                logger.info(f"Group {group}")
                logger.info(
                    f"tardiness : {cpsolvercb.Value(self.variables['tardiness']['groups'][group]['tardiness'])}"
                )
                logger.info(
                    f"earliness : {cpsolvercb.Value(self.variables['tardiness']['groups'][group]['earliness'])}"
                )
        schedule = np.zeros((self.problem.nb_tasks, 2))
        allocation = np.zeros(self.problem.nb_tasks)
        for i in range(self.problem.nb_tasks):
            schedule[i, 0] = cpsolvercb.Value(self.variables["starts"][i])
            schedule[i, 1] = cpsolvercb.Value(self.variables["ends"][i])
            for j in self.variables["is_present"][i]:
                if cpsolvercb.Value(self.variables["is_present"][i][j]):
                    allocation[i] = j
        sol = ScheduleSolution(
            problem=self.problem, schedule=schedule, modes=allocation
        )
        sol._intern_obj = details_subobj
        return sol

    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        index = self.problem.task_id_to_index[task]
        if mode in self.variables["opt_intervals"][index]:
            return self.variables["opt_intervals"][index][mode]
        else:
            return None

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        # No Unary Resource modeled
        return None

    def get_task_unary_resource_interval(
        self, task: Task, unary_resource: UnaryResource
    ) -> IntervalVar:
        # No Unary Resource modeled
        return None


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
    # print(dict_of_interval_per_duration)
    return duration, dict_of_interval_per_duration


def compute_duration_tasks_function_time(problem: FlexProblem, method=None):
    if method is None:
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
    return durations


def build_multiple_cumulative_constraints_inputs(
    problem: FlexProblem, resource: ResourceData
):
    """
    This is a utility function to define the set of interval variable to include in separated cumulative constraints.

    this handle corner case where : non-zero resource availability happens in some ressource, and this non-zero level is
    lower than the resource need of the task.
    -When this happen, it shouldn't prevent the task to overlap with this period of time -> it just means that the task
    has a longer duration, as computed in compute_duration_tasks_function_time function.
    -With previous formulation of cumulative constraint, this corner case is not well taken into account,
    We put in a cumulative constraint all task consuming some resource and all interval corresponding to (partial)
    resource unavaibility, and this was leading to unsat problem
    """
    res_comp: List[Dict[str, int]] = create_resource_consumption_from_calendar(
        calendar_availability=resource.calendar_availability
    )
    max_capacity = resource.max_capacity
    resource_calendar = [
        f for f in res_comp if f["value"] > 0 and f["value"] != resource.max_capacity
    ]
    values = sorted(set([f["value"] for f in resource_calendar]))
    task_mode_consume = [
        (
            index_task,
            mode,
            problem.tasks[index_task].modes[mode].get_res_consumption(resource.id),
        )
        for index_task in range(problem.nb_tasks)
        for mode in problem.tasks[index_task].modes
        if problem.tasks[index_task].modes[mode].get_res_consumption(resource.id) > 0
    ]
    inputs_for_cumulative_constraint = [
        {
            "val": 0,
            "task_mode_conso": task_mode_consume,
            "set_task_mode_conso": set(task_mode_consume),
            "calendar_tasks": [],
        }
    ]
    for i in range(len(values)):
        val = values[i]
        task_mode_c = [x for x in task_mode_consume if val + x[2] <= max_capacity]
        ftask_mode_c = [x for x in resource_calendar if x["value"] <= val]
        inputs_for_cumulative_constraint.append(
            (
                {
                    "val": val,
                    "task_mode_conso": task_mode_c,
                    "set_task_mode_conso": set(task_mode_c),
                    "calendar_tasks": ftask_mode_c,
                }
            )
        )
    if len(inputs_for_cumulative_constraint) == 0:
        return []
    if len(inputs_for_cumulative_constraint) == 1:
        return inputs_for_cumulative_constraint
    if len(inputs_for_cumulative_constraint) > 1:
        index_to_keep = []
        # current_set = inputs_for_cumulative_constraint[0]["set_task_mode_conso"].copy()
        cur_index = 0
        inputs_for_cumulative_constraint = sorted(
            inputs_for_cumulative_constraint, key=lambda x: x["val"], reverse=True
        )
        for j in range(1, len(inputs_for_cumulative_constraint)):
            if (
                inputs_for_cumulative_constraint[j]["set_task_mode_conso"]
                != inputs_for_cumulative_constraint[cur_index]["set_task_mode_conso"]
            ):
                index_to_keep.append(j - 1)
                cur_index = j
        index_to_keep.append(cur_index)
        filtered_index_to_keep = [
            inputs_for_cumulative_constraint[ind] for ind in index_to_keep
        ]
        return inputs_for_cumulative_constraint  # filtered_index_to_keep


def post_cumulative_constraints(
    problem: FlexProblem,
    resource: ResourceData,
    solver: CpSatFlexSolver,
    variable_max_capacity: bool,
    include_intervals_non_release: bool = True,
):
    inputs_constraint = build_multiple_cumulative_constraints_inputs(
        problem=problem, resource=resource
    )
    task_non_release = []
    if (
        "intervals_non_release" in solver.variables
        and include_intervals_non_release
        and not variable_max_capacity
    ):
        if resource.id in solver.variables["intervals_non_release"]:
            task_non_release = solver.variables["intervals_non_release"][resource.id]

    for input_data in inputs_constraint:
        val = input_data["val"]
        set_task_mode_conso = list(input_data["set_task_mode_conso"])
        intervals_ = [
            solver.variables["opt_intervals"][x[0]][x[1]] for x in set_task_mode_conso
        ]
        consos = [x[2] for x in set_task_mode_conso]
        other_intervals_c = [
            x for x in task_non_release if x[1] + val <= resource.max_capacity
        ]
        calendar_intervals = [
            (
                solver.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name=f"res_"
                ),
                f["value"],
            )
            for f in input_data["calendar_tasks"]
        ]
        if len(intervals_) + len(other_intervals_c) == 0:
            # Useless
            continue

        if not variable_max_capacity:
            solver.cp_model.AddCumulative(
                intervals_
                + [x[0] for x in other_intervals_c]
                + [x[0] for x in calendar_intervals],
                consos
                + [x[1] for x in other_intervals_c]
                + [x[1] for x in calendar_intervals],
                resource.max_capacity,
            )
        else:
            if "resource_capacity_variables" in solver.variables:
                solver.cp_model.AddCumulative(
                    intervals_
                    + [x[0] for x in other_intervals_c]
                    + [x[0] for x in calendar_intervals],
                    consos
                    + [x[1] for x in other_intervals_c]
                    + [x[1] for x in calendar_intervals],
                    solver.variables["resource_capacity_variables"][resource.id],
                )
