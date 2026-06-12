#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, List

import numpy as np
import ortools
from ortools.sat.python.cp_model import (
    CpModel,
    CpSolverSolutionCallback,
    Domain,
    LinearExpr,
    LinearExprT,
)

from discrete_optimization.flex_scheduling.fsp_utils import (
    create_resource_consumption_from_calendar,
)
from discrete_optimization.flex_scheduling.problem import (
    FlexProblem,
    GroupType,
    ObjectivesEnum,
    ResourceData,
    ScheduleSolution,
    ScheduleSolutionPreemptive,
    TasksGroups,
)
from discrete_optimization.flex_scheduling.solvers.cpsat import CpSatFlexSolver
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.skill import Skill

logger = logging.getLogger(__name__)


def resource_consumption(flex_problem: FlexProblem, solution: ScheduleSolution):
    # Build an array resource_consumptions[i, r]
    max_nb_modes = max([len(t.modes) for t in flex_problem.tasks])
    resource_consumptions = np.zeros(
        (flex_problem.nb_tasks, flex_problem.nb_resources), dtype=int
    )
    for i in range(flex_problem.nb_tasks):
        mode_i = solution.modes[i]
        consumption_dict = flex_problem.tasks[i].modes[mode_i].resource_consumption
        for r_idx, r_name in enumerate(flex_problem.resources):
            resource_consumptions[i, r_idx] = consumption_dict.get(r_name.id, 0)
    return resource_consumptions


class CPSatFlexSPPreempt(CpSatFlexSolver):
    def get_skill_variable(
        self, task: Task, unary_resource: UnaryResource, skill: Skill
    ) -> LinearExprT:
        return 0

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> ScheduleSolution:
        logger.info(
            f"cur obj value : {cpsolvercb.ObjectiveValue()}, bound={cpsolvercb.BestObjectiveBound()}"
        )
        logger.info("Sub-objectives :")
        details_subobj = {}
        if "obj_data" in self.variables:
            for i in range(len(self.variables["obj_data"][0])):
                logger.info(
                    f"{self.variables['obj_data'][2][i]} : {cpsolvercb.Value(self.variables['obj_data'][0][i])}"
                )
                details_subobj[self.variables["obj_data"][2][i]] = cpsolvercb.Value(
                    self.variables["obj_data"][0][i]
                )
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
        schedule_l = {}
        for i in range(self.problem.nb_tasks):
            schedule_l[i] = []
            schedule[i, 0] = cpsolvercb.Value(self.variables["starts"][i])
            schedule[i, 1] = cpsolvercb.Value(self.variables["ends"][i])
            for j in range(len(self.variables["starts_preempt"][i])):
                st = cpsolvercb.Value(self.variables["starts_preempt"][i][j])
                end = cpsolvercb.Value(self.variables["ends_preempt"][i][j])
                dur = cpsolvercb.Value(self.variables["durs_preempt"][i][j])
                present = cpsolvercb.Value(self.variables["is_present_preempt"][i][j])
                if present:
                    schedule_l[i].append((st, end))

            for j in self.variables["modes_variable"][i]:
                if cpsolvercb.Value(self.variables["modes_variable"][i][j]):
                    allocation[i] = j
        sol = ScheduleSolutionPreemptive(
            problem=self.problem,
            schedule=[schedule_l[i] for i in range(self.problem.nb_tasks)],
            modes=allocation,
        )
        sol._intern_obj = details_subobj
        return sol

    def from_solution_to_hint(
        self, solution: ScheduleSolutionPreemptive
    ) -> list[tuple[ortools.sat.python.cp_model.VariableT, int]]:
        list_variables_value = []
        for i in range(self.problem.nb_tasks):
            for j in range(len(self.variables["starts_preempt"][i])):
                if j + 1 <= len(solution.schedule[i]):
                    list_variables_value.append(
                        (
                            self.variables["starts_preempt"][i][j],
                            solution.schedule[i][j][0],
                        )
                    )
                    list_variables_value.append(
                        (
                            self.variables["ends_preempt"][i][j],
                            solution.schedule[i][j][1],
                        )
                    )
                    list_variables_value.append(
                        (
                            self.variables["durs_preempt"][i][j],
                            solution.schedule[i][j][1] - solution.schedule[i][j][0],
                        )
                    )
                    list_variables_value.append(
                        (self.variables["is_present_preempt"][i][j], 1)
                    )
                else:
                    list_variables_value.append(
                        (
                            self.variables["starts_preempt"][i][j],
                            solution.schedule[i][-1][1],
                        )
                    )
                    list_variables_value.append(
                        (
                            self.variables["ends_preempt"][i][j],
                            solution.schedule[i][-1][1],
                        )
                    )
                    list_variables_value.append(
                        (self.variables["durs_preempt"][i][j], 0)
                    )
                    list_variables_value.append(
                        (self.variables["is_present_preempt"][i][j], 0)
                    )

        # Main variables
        if "starts" in self.variables:
            for i in range(self.problem.nb_tasks):
                list_variables_value.append(
                    (self.variables["starts"][i], solution.schedule[i][0][0])
                )
                list_variables_value.append(
                    (self.variables["ends"][i], solution.schedule[i][-1][1])
                )
                list_variables_value.append(
                    (
                        self.variables["durations"][i],
                        solution.schedule[i][-1][1] - solution.schedule[i][0][0],
                    )
                )
        # Modes variables
        for i in range(self.problem.nb_tasks):
            mode_chosen = solution.modes[i]
            for mode in self.problem.tasks[i].modes:
                if mode == mode_chosen:
                    if len(self.problem.tasks[i].modes) > 1:
                        list_variables_value.append(
                            (self.variables["modes_variable"][i][mode], 1)
                        )
                    list_variables_value.append(
                        (
                            self.variables["durations_executed"][i],
                            self.problem.tasks[i].modes[mode].duration,
                        )
                    )
                else:
                    if len(self.problem.tasks[i].modes) > 1:
                        list_variables_value.append(
                            (self.variables["modes_variable"][i][mode], 0)
                        )

        # Group variables
        if "resource_capacity_variables" in self.variables:
            for group in self.problem.tasks_group:
                group_id = group.id
                ft = group.first_task_if_any
                lt = group.last_task_if_any
                if ft is not None:
                    i = self.problem.task_id_to_index[ft]
                    st_grp = solution.schedule[i][0][0]
                else:
                    st_grp = min(
                        [
                            solution.schedule[self.problem.task_id_to_index[id_task]][
                                0
                            ][0]
                            for id_task in group.tasks_group
                        ]
                    )
                if lt is not None:
                    i = self.problem.task_id_to_index[lt]
                    end_grp = solution.schedule[i, 1]
                else:
                    end_grp = max(
                        [
                            solution.schedule[self.problem.task_id_to_index[id_task]][
                                -1
                            ][1]
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
                    (
                        self.variables["duration_span_variables"][group_id],
                        end_grp - st_grp,
                    )
                )

        list_variables_value += self.from_solution_to_hint_non_released_delta(
            solution=solution
        )

        # Resource
        if "resource_capacity_variables" in self.variables:
            for resource in self.problem.resources:
                if resource.id in self.variables["resource_capacity_variables"]:
                    list_variables_value.append(
                        (
                            self.variables["resource_capacity_variables"][resource.id],
                            int(resource.max_capacity),
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
        if "makespan" in self.variables:
            list_variables_value.append(
                (
                    self.variables["makespan"],
                    max(
                        [
                            solution.schedule[i][-1][1]
                            for i in range(self.problem.nb_tasks)
                        ]
                    ),
                )
            )
        if "tardiness" in self.variables:
            list_variables_value += self.from_solution_to_hint_earliness(solution)
        return list_variables_value

    def from_solution_to_hint_non_released_delta(
        self, solution: ScheduleSolutionPreemptive
    ):
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
                            solution.schedule[i2][0][0] - solution.schedule[i1][-1][1],
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
                                solution.schedule[i2][0][0]
                                - solution.schedule[i1][-1][1],
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
                    st_ = solution.schedule[i1][-1][1]
                    tag.append(("task", t1.task_id))
                else:
                    group = t1.group_id
                    gr: TasksGroups = [
                        g for g in self.problem.tasks_group if g.id == group
                    ][0]
                    st_ = max(
                        [
                            solution.schedule[self.problem.task_id_to_index[id_task]][
                                -1
                            ][1]
                            for id_task in gr.tasks_group
                        ]
                    )
                    tag.append(("group", group))

                if t2.is_a_task:
                    i2 = self.problem.task_id_to_index[t2.task_id]
                    end_ = solution.schedule[i2][0][0]
                    tag.append(("task", t2.task_id))

                else:
                    group = t2.group_id
                    gr: TasksGroups = [
                        g for g in self.problem.tasks_group if g.id == group
                    ][0]
                    end_ = min(
                        [
                            solution.schedule[self.problem.task_id_to_index[id_task]][
                                0
                            ][0]
                            for id_task in gr.tasks_group
                        ]
                    )
                    tag.append(("group", group))
                tag = tuple(tag)
                if "durations_non_release_3" in self.variables:
                    if self.variables["durations_non_release_3"][tag] not in [
                        x[0] for x in list_variables_value
                    ]:
                        list_variables_value.append(
                            (self.variables["durations_non_release_3"][tag], end_ - st_)
                        )
        return list_variables_value

    def from_solution_to_hint_earliness(self, solution: ScheduleSolutionPreemptive):
        list_variables_value = []
        for id_task in self.variables["tardiness"]["tasks"]:
            index = self.problem.task_id_to_index[id_task]
            deadline = self.problem.task_id_dict[id_task].max_ending_date
            end = solution.schedule[index][-1][1]
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
                    solution.schedule[self.problem.task_id_to_index[id_task]][-1][1]
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
                list_variables_value.append(
                    (self.variables["tardiness"]["groups"][id_group]["earliness"], 0)
                )
            else:
                list_variables_value.append(
                    (self.variables["tardiness"]["groups"][id_group]["tardiness"], 0)
                )
                list_variables_value.append(
                    (
                        self.variables["tardiness"]["groups"][id_group]["earliness"],
                        deadline - end,
                    )
                )
        return list_variables_value

    def init_model(self, **args: Any) -> None:
        model = CpModel()

        self.cp_model = model
        self.init_main_variables_preempt(
            nb_max_preemption=45
        )  # main interval variables
        self.init_span_task_variables()
        self.init_modes_variables()
        # Create some ghost intervals
        self.init_variable_preempt_mode()
        # Constraint preemption variables
        self.constraint_convention_preemption()
        self.constraint_sum_duration()

        self.init_group_variables()  # create span interval variable for relevant group of tasks.
        # for cases where some resource are not released at the end of a task, but rather on the starting of one another
        self.init_intervals_of_non_released_resource()
        self.init_resource_variables()  #
        self.constraint_precedence()
        self.constraint_precedence_on_groups()
        self.constraint_cumulative()
        self.constraint_on_groups_of_task()
        self.constraint_generalized_time_constraint()
        self.create_objectives()

    def create_objectives(self):
        super().create_objectives()
        init = self.variables["obj_data"]
        nb_preemption = sum(
            [
                self.variables["is_present_preempt"][i][j]
                for i in self.variables["is_present_preempt"]
                for j in range(len(self.variables["is_present_preempt"][i]))
            ]
        )
        objs = init[0] + [nb_preemption]
        weights = init[1] + [10000]
        names = init[2] + ["nb_preemption"]
        self.variables["obj_data"] = (objs, weights, names)
        self.cp_model.Minimize(sum([objs[i] * weights[i] for i in range(len(objs))]))

    def compute_possible_start_end_duration_values(self, index_task: int):
        conso = self.problem.tasks[index_task].modes[1]

    def init_main_variables_preempt(self, nb_max_preemption: int = None):
        starts_variable = {}
        ends_variable = {}
        durations_variable = {}
        is_present_variable = {}
        intervals_variable = {}
        for i in range(self.problem.nb_tasks):
            max_duration = max(
                [
                    self.problem.tasks[i].modes[m].duration
                    for m in self.problem.tasks[i].modes
                ]
            )
            res = [
                r
                for m in self.problem.tasks[i].modes
                for r in self.problem.tasks[i].modes[m].resource_consumption
                if self.problem.tasks[i].modes[m].resource_consumption[r] > 0
                and len(set(self.problem.resource_dict[r].calendar_availability > 0))
                >= 2
            ]
            nb_preemption = max(1, max_duration)
            if len(res) == 0 and False:
                # print('No preemption')
                nb_preemption = 1
            else:
                pass
                # print('Preemption')
            if nb_max_preemption is not None:
                nb_preemption = min(nb_preemption, nb_max_preemption)
            starts_variable[i] = [
                self.cp_model.NewIntVar(
                    lb=0, ub=self.problem.horizon, name=f"start_{i}_{j}"
                )
                for j in range(nb_preemption)
            ]
            ends_variable[i] = [
                self.cp_model.NewIntVar(
                    lb=0, ub=self.problem.horizon, name=f"end_{i}_{j}"
                )
                for j in range(nb_preemption)
            ]
            durations_variable[i] = [
                self.cp_model.NewIntVar(lb=0, ub=max_duration, name=f"duration_{i}_{j}")
                for j in range(nb_preemption)
            ]
            is_present_variable[i] = [
                self.cp_model.NewBoolVar(name=f"is_present_{i}_{j}")
                for j in range(nb_preemption)
            ]
            intervals_variable[i] = [
                self.cp_model.NewOptionalIntervalVar(
                    start=starts_variable[i][j],
                    end=ends_variable[i][j],
                    size=durations_variable[i][j],
                    is_present=is_present_variable[i][j],
                    name=f"interval_{i}_{j}",
                )
                for j in range(nb_preemption)
            ]
        self.variables["starts_preempt"] = starts_variable
        self.variables["ends_preempt"] = ends_variable
        self.variables["durs_preempt"] = durations_variable
        self.variables["is_present_preempt"] = is_present_variable
        self.variables["interval_preempt"] = intervals_variable

    def init_variable_preempt_mode(self):
        opt_variables = {}
        for i in range(self.problem.nb_tasks):
            opt_variables[i] = {}
            for mode in self.variables["modes_variable"][i]:
                # pre = [self.cp_model.NewBoolVar(name=f"present_{j}_{mode}")
                #        for j in range(len(self.variables["starts_preempt"][i]))]
                # for j in range(len(pre)):
                # self.cp_model.AddBoolAnd([self.variables["modes_variable"][i][mode],
                #                          self.variables["is_present_preempt"][i][j]],
                #                         pre[j])
                #    self.cp_model.add(pre[j] == 1).OnlyEnforceIf(self.variables["modes_variable"][i][mode],
                #                                                 self.variables["is_present_preempt"][i][j])
                opt_variables[i][mode] = [
                    self.cp_model.NewOptionalIntervalVar(
                        start=self.variables["starts_preempt"][i][j],
                        end=self.variables["ends_preempt"][i][j],
                        size=self.variables["durs_preempt"][i][j],
                        is_present=self.variables["modes_variable"][i][mode],
                        # pre[j],
                        name=f"int_{i}_{j}_{mode}",
                    )
                    for j in range(len(self.variables["starts_preempt"][i]))
                ]
        self.variables["interval_preempt_mode"] = opt_variables

    def init_span_task_variables(self):
        span_interval_variables = {}
        start_span_variables = {}
        duration_span_variables = {}
        end_span_variables = {}
        for i in range(self.problem.nb_tasks):
            start_span_variables[i] = self.variables["starts_preempt"][i][0]
            end_span_variables[i] = self.variables["ends_preempt"][i][-1]
            duration_span_variables[i] = self.cp_model.NewIntVar(
                # lb=0, ub=self.max_end_time[i]-self.min_end_time[i],
                lb=0,
                ub=self.problem.horizon,
                name=f"duration_span_{i}",
            )
            self.cp_model.Add(
                end_span_variables[i]
                == start_span_variables[i] + duration_span_variables[i]
            )
            span_interval_variables[i] = self.cp_model.NewIntervalVar(
                start=start_span_variables[i],
                size=duration_span_variables[i],
                end=end_span_variables[i],
                name=f"span_{i}",
            )
        self.variables["starts"] = start_span_variables
        self.variables["ends"] = end_span_variables
        self.variables["durations"] = duration_span_variables
        self.variables["intervals"] = span_interval_variables

    def constraint_convention_preemption(self):
        for i in self.variables["starts_preempt"]:
            min_duration = min(
                [
                    self.problem.tasks[i].modes[m].duration
                    for m in self.problem.tasks[i].modes
                ]
            )
            starts = self.variables["starts_preempt"][i]
            ends = self.variables["ends_preempt"][i]
            durs = self.variables["durs_preempt"][i]
            presents = self.variables["is_present_preempt"][i]
            for j in range(1, len(starts) - 1):
                self.cp_model.AddImplication(presents[j].Not(), presents[j + 1].Not())
            for j in range(1, len(starts)):
                self.cp_model.Add(starts[j] >= ends[j - 1])
                self.cp_model.Add(starts[j] > ends[j - 1]).OnlyEnforceIf(
                    presents[j]
                )  # disjoint..
                # self.cp_model.Add(starts[j] < ends[j-1]+5).OnlyEnforceIf(presents[j])  # disjoint..
                # if min_duration >= 2:
                #    self.cp_model.Add(durs[j] >= 2).OnlyEnforceIf(presents[j])  # disjoint..

                self.cp_model.Add(presents[j] == 0).OnlyEnforceIf(presents[j - 1].Not())
                self.cp_model.Add(starts[j] == ends[j - 1]).OnlyEnforceIf(
                    presents[j].Not()
                )
                self.cp_model.Add(ends[j] == ends[j - 1]).OnlyEnforceIf(
                    presents[j].Not()
                )
            self.cp_model.Add(presents[0] == 1)
            if min_duration > 0:
                # First subtask is not dummy
                self.cp_model.Add(durs[0] > 0)
                self.cp_model.Add(presents[0] == 1)
                for j in range(len(starts)):
                    self.cp_model.Add(durs[j] > 0).OnlyEnforceIf(presents[j])
                    self.cp_model.Add(durs[j] == 0).OnlyEnforceIf(presents[j].Not())
            self.cp_model.AddNoOverlap(self.variables["interval_preempt"][i])

    def constraint_sum_duration(self):
        for i in range(self.problem.nb_tasks):
            durs = self.variables["durs_preempt"][i]
            self.cp_model.Add(
                LinearExpr.sum(durs) == self.variables["durations_executed"][i]
            )

    def init_modes_variables(self):
        modes_variable = {}
        duration_variable = {}
        for i in range(self.problem.nb_tasks):
            nb_modes = len(self.problem.tasks[i].modes)
            modes_variable[i] = {}
            possible_durs = list(
                set(
                    [
                        self.problem.tasks[i].modes[mode].duration
                        for mode in self.problem.tasks[i].modes
                    ]
                )
            )
            if len(possible_durs) == 1:
                duration_variable[i] = possible_durs[0]
            else:
                duration_variable[i] = self.cp_model.NewIntVarFromDomain(
                    Domain.from_values(possible_durs), name=f"duration_execution_{i}"
                )
            for mode in self.problem.tasks[i].modes:
                if nb_modes > 1:
                    modes_variable[i][mode] = self.cp_model.NewBoolVar(
                        name=f"task_{i}_m_{mode}"
                    )
                    (
                        self.cp_model.Add(
                            duration_variable[i]
                            == self.problem.tasks[i].modes[mode].duration
                        ).OnlyEnforceIf(modes_variable[i][mode])
                    )
                else:
                    modes_variable[i][mode] = True
            if nb_modes > 1:
                self.cp_model.AddExactlyOne(
                    [modes_variable[i][mode] for mode in modes_variable[i]]
                )
        self.variables["modes_variable"] = modes_variable
        self.variables["durations_executed"] = duration_variable

    def init_main_variables(self):
        starts_variable = {}
        ends_variable = {}
        durations_variable = {}
        intervals_variable = {}
        for i in range(self.problem.nb_tasks):
            starts_variable[i] = self.cp_model.NewIntVar(
                lb=self.min_start_time[i], ub=self.max_start_time[i], name=f"start_{i}"
            )
            ends_variable[i] = self.cp_model.NewIntVar(
                lb=self.min_end_time[i],
                ub=self.max_end_time[i]
                if not self.problem.tasks[i].soft_max_end_date
                else self.problem.horizon,
                # deadline is actually soft
                name=f"end_{i}",
            )
            max_duration = max(
                [
                    self.problem.tasks[i].modes[m].duration
                    for m in self.problem.tasks[i].modes
                ]
            )
            min_duration = min(
                [
                    self.problem.tasks[i].modes[m].duration
                    for m in self.problem.tasks[i].modes
                ]
            )
            durations_variable[i] = self.cp_model.NewIntVar(
                lb=min_duration, ub=100 * max_duration, name=f"duration_{i}"
            )
            intervals_variable[i] = self.cp_model.NewIntervalVar(
                start=starts_variable[i],
                size=durations_variable[i],
                end=ends_variable[i],
                name=f"interval_{i}",
            )
        self.variables["starts"] = starts_variable
        self.variables["ends"] = ends_variable
        self.variables["durations"] = durations_variable
        self.variables["intervals"] = intervals_variable

    def constraint_cumulative(self):
        for r in self.problem.resources:
            if r.renewable:
                self.constraint_cumulative_resource(
                    resource=r, variable_max_capacity=False
                )
                if (
                    ObjectivesEnum.RESOURCE_COST
                    in self.problem.objective_params.params_obj
                ):
                    if (
                        r.id
                        in self.problem.objective_params.params_obj[
                            ObjectivesEnum.RESOURCE_COST
                        ].weight_per_resource_unit
                    ):
                        self.constraint_cumulative_resource(
                            resource=r, variable_max_capacity=True
                        )

    def constraint_cumulative_resource(
        self, resource: ResourceData, variable_max_capacity: bool = False
    ):
        res_comp: List[Dict[str, int]] = create_resource_consumption_from_calendar(
            calendar_availability=resource.calendar_availability
        )
        id_resource = resource.id
        task_mode_consume = [
            (
                self.variables["interval_preempt_mode"][i][mode][j],
                int(self.problem.tasks[i].modes[mode].get_res_consumption(id_resource)),
            )
            for i in self.variables["interval_preempt_mode"]
            for mode in self.variables["interval_preempt_mode"][i]
            for j in range(len(self.variables["interval_preempt_mode"][i][mode]))
            if self.problem.tasks[i].modes[mode].get_res_consumption(id_resource) > 0
        ]
        task_mode_consume_with_non_release = None
        if "intervals_non_release" in self.variables:
            if resource.id in self.variables["intervals_non_release"]:
                task_mode_consume_with_non_release = (
                    task_mode_consume
                    + self.variables["intervals_non_release"][resource.id]
                )
        fake_task_res = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name=f"res_"
                ),
                f["value"],
            )
            for f in res_comp
            if f["value"] > 0  # and f["value"] != resource.max_capacity
        ]
        fake_task_res_bis = [
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
            if task_mode_consume_with_non_release:
                self.cp_model.AddCumulative(
                    [x[0] for x in task_mode_consume_with_non_release]
                    + [x[0] for x in fake_task_res_bis],
                    [x[1] for x in task_mode_consume_with_non_release]
                    + [x[0] for x in fake_task_res_bis],
                    resource.max_capacity,
                )
            self.cp_model.AddCumulative(
                [x[0] for x in task_mode_consume] + [x[0] for x in fake_task_res],
                [x[1] for x in task_mode_consume] + [x[1] for x in fake_task_res],
                resource.max_capacity,
            )
        else:
            if task_mode_consume_with_non_release:
                self.cp_model.AddCumulative(
                    [x[0] for x in task_mode_consume_with_non_release]
                    + [x[0] for x in fake_task_res_bis],
                    [x[1] for x in task_mode_consume_with_non_release]
                    + [x[0] for x in fake_task_res_bis],
                    self.variables["resource_capacity_variables"][resource.id],
                )
            self.cp_model.AddCumulative(
                [
                    x[0] for x in task_mode_consume
                ],  # + [x[0] for x in fake_task_res_bis],
                [
                    x[1] for x in task_mode_consume
                ],  # + [x[1] for x in fake_task_res_bis],
                self.variables["resource_capacity_variables"][resource.id],
            )

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
                if resource in g.res_not_released and g.res_not_released[resource] > 0:
                    intervals_and_consumption.append(
                        (
                            self.variables["span_interval_variables"][g.id],
                            g.res_not_released[resource],
                        )
                    )
                    tasks_covered_in_group.update(g.tasks_group)

            task_mode_consume = [
                (
                    self.variables["interval_preempt_mode"][i][mode][j],
                    int(
                        self.problem.tasks[i].modes[mode].get_res_consumption(resource)
                    ),
                )
                for i in self.variables["interval_preempt_mode"]
                for mode in self.variables["interval_preempt_mode"][i]
                for j in range(len(self.variables["interval_preempt_mode"][i][mode]))
                if self.problem.tasks[i].modes[mode].get_res_consumption(resource) > 0
                and self.problem.index_to_task_id[i] not in tasks_covered_in_group
            ]
            if "resource_capacity_variables" in self.variables and False:
                self.cp_model.AddCumulative(
                    [x[0] for x in task_mode_consume]
                    + [x[0] for x in intervals_and_consumption],
                    [x[1] for x in task_mode_consume]
                    + [x[1] for x in intervals_and_consumption],
                    self.variables["resource_capacity_variables"][resource],
                )
            self.cp_model.AddCumulative(
                [x[0] for x in task_mode_consume]
                + [x[0] for x in intervals_and_consumption],
                [x[1] for x in task_mode_consume]
                + [x[1] for x in intervals_and_consumption],
                self.problem.resource_dict[resource].max_capacity,
            )
