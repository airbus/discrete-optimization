#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
import random
from abc import abstractmethod
from enum import Enum
from typing import Any, Hashable, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from minizinc import Instance

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    GraphRCPSP,
    GraphRCPSPSpecialConstraints,
    build_graph_rcpsp_object,
)
from discrete_optimization.generic_rcpsp_tools.typing import (
    ANY_CP_SOLVER,
    ANY_MSRCPSP,
    ANY_RCPSP,
    ANY_SOLUTION,
    ANY_SOLUTION_PREEMPTIVE,
)
from discrete_optimization.generic_tools.cp_tools import CPSolver, SignEnum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.lns_cp import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPSolutionPreemptive
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.solver.cp_solvers import CP_MRCPSP_MZN_PREEMPTIVE
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraintsPreemptive,
    compute_constraints_details,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive,
    compute_overskill,
    employee_usage,
    start_together_problem_description,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solver_mspsp_instlib import (
    CP_MSPSP_MZN,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
)

logger = logging.getLogger(__name__)


def get_max_time_solution(solution: ANY_SOLUTION):
    return solution.get_max_end_time()


class ParamsConstraintBuilder(Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="minus_delta_primary", low=0, default=100),
        IntegerHyperparameter(name="plus_delta_primary", low=0, default=100),
        IntegerHyperparameter(name="minus_delta_secondary", low=0, default=0),
        IntegerHyperparameter(name="plus_delta_secondary", low=0, default=0),
        IntegerHyperparameter(name="minus_delta_primary_duration", default=5, low=0),
        IntegerHyperparameter(name="plus_delta_primary_duration", default=5, low=0),
        IntegerHyperparameter(name="minus_delta_secondary_duration", default=5, low=0),
        IntegerHyperparameter(name="plus_delta_secondary_duration", default=5, low=0),
        CategoricalHyperparameter(
            name="constraint_max_time_to_current_solution",
            choices=[True, False],
            default=False,
        ),
        FloatHyperparameter(
            name="fraction_of_task_assigned_multiskill", default=0.6, low=0.0, high=1.0
        ),
        CategoricalHyperparameter(
            name="except_assigned_multiskill_primary_set",
            choices=[True, False],
            default=False,
        ),
        CategoricalHyperparameter(
            name="first_method_multiskill", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="second_method_multiskill", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="additional_methods", choices=[True, False], default=False
        ),
    ]

    def __init__(
        self,
        minus_delta_primary: int,
        plus_delta_primary: int,
        minus_delta_secondary: int,
        plus_delta_secondary: int,
        minus_delta_primary_duration: int = 5,
        plus_delta_primary_duration: int = 5,
        minus_delta_secondary_duration: int = 5,
        plus_delta_secondary_duration: int = 5,
        constraint_max_time_to_current_solution: bool = True,
        fraction_of_task_assigned_multiskill: float = 0.6,
        except_assigned_multiskill_primary_set: bool = False,
        first_method_multiskill: bool = True,
        second_method_multiskill: bool = False,
        additional_methods: bool = False,
    ):
        self.minus_delta_primary = minus_delta_primary
        self.plus_delta_primary = plus_delta_primary
        self.minus_delta_secondary = minus_delta_secondary
        self.plus_delta_secondary = plus_delta_secondary
        self.minus_delta_primary_duration = minus_delta_primary_duration
        self.plus_delta_primary_duration = plus_delta_primary_duration
        self.minus_delta_secondary_duration = minus_delta_secondary_duration
        self.plus_delta_secondary_duration = plus_delta_secondary_duration
        self.constraint_max_time_to_current_solution = (
            constraint_max_time_to_current_solution
        )
        self.fraction_of_task_assigned_multiskill = fraction_of_task_assigned_multiskill
        self.except_assigned_multiskill_primary_set = (
            except_assigned_multiskill_primary_set
        )
        self.first_method_multiskill = first_method_multiskill
        self.second_method_multiskill = second_method_multiskill
        self.additional_methods = additional_methods

    @staticmethod
    def default():
        return ParamsConstraintBuilder(
            minus_delta_primary=100,
            plus_delta_primary=100,
            minus_delta_secondary=0,
            plus_delta_secondary=0,
            constraint_max_time_to_current_solution=False,
            fraction_of_task_assigned_multiskill=0.6,
        )


def constraints_strings(
    current_solution: Union[RCPSPSolution, MS_RCPSPSolution],
    cp_solver: ANY_CP_SOLVER,
    tasks_primary: Set[Hashable],
    tasks_secondary: Set[Hashable],
    params_constraints: ParamsConstraintBuilder,
):
    max_time = get_max_time_solution(solution=current_solution)
    list_strings = []
    for job in tasks_primary:
        start_time_j = current_solution.get_start_time(job)
        string_start_1 = cp_solver.constraint_start_time_string(
            task=job,
            start_time=max(0, start_time_j - params_constraints.minus_delta_primary),
            sign=SignEnum.UEQ,
        )
        string_start_2 = cp_solver.constraint_start_time_string(
            task=job,
            start_time=min(
                max_time, start_time_j + params_constraints.plus_delta_primary
            )
            if params_constraints.constraint_max_time_to_current_solution
            else start_time_j + params_constraints.plus_delta_primary,
            sign=SignEnum.LEQ,
        )
        list_strings += [string_start_1, string_start_2]
    for job in tasks_secondary:
        if job in tasks_primary:
            continue
        start_time_j = current_solution.get_start_time(job)
        if (
            params_constraints.minus_delta_secondary == 0
            and params_constraints.plus_delta_secondary == 0
        ):
            string_start = cp_solver.constraint_start_time_string(
                task=job, start_time=start_time_j, sign=SignEnum.EQUAL
            )
            list_strings += [string_start]
        else:
            string_start_1 = cp_solver.constraint_start_time_string(
                task=job,
                start_time=max(
                    0, start_time_j - params_constraints.minus_delta_secondary
                ),
                sign=SignEnum.UEQ,
            )
            string_start_2 = cp_solver.constraint_start_time_string(
                task=job,
                start_time=min(
                    max_time, start_time_j + params_constraints.plus_delta_secondary
                )
                if params_constraints.constraint_max_time_to_current_solution
                else start_time_j + params_constraints.plus_delta_secondary,
                sign=SignEnum.LEQ,
            )
            list_strings += [string_start_1, string_start_2]
    if params_constraints.constraint_max_time_to_current_solution:
        string = cp_solver.constraint_start_time_string(
            current_solution.problem.sink_task, start_time=max_time, sign=SignEnum.LEQ
        )
        list_strings += [string]
    return list_strings


def constraints_strings_preemptive(
    current_solution: ANY_SOLUTION_PREEMPTIVE,
    cp_solver: ANY_CP_SOLVER,
    tasks_primary: Set[Hashable],
    tasks_secondary: Set[Hashable],
    params_constraints: ParamsConstraintBuilder,
):
    max_time = get_max_time_solution(solution=current_solution)
    multimode = isinstance(cp_solver, (CP_MRCPSP_MZN_PREEMPTIVE, CP_MS_MRCPSP_MZN))
    modes_dict = None
    if multimode:
        modes_dict = current_solution.problem.get_modes_dict(current_solution)
    list_strings = []
    for job in tasks_primary:
        list_starts = current_solution.get_start_times_list(job)
        list_ends = current_solution.get_end_times_list(job)
        for j in range(len(list_starts)):
            start_time_j = list_starts[j]
            end_time_j = list_ends[j]
            duration_j = end_time_j - start_time_j
            string1_start = cp_solver.constraint_start_time_string_preemptive_i(
                task=job,
                start_time=max(
                    0, start_time_j - params_constraints.minus_delta_primary
                ),
                sign=SignEnum.UEQ,
                part_id=j + 1,
            )
            string2_start = cp_solver.constraint_start_time_string_preemptive_i(
                task=job,
                start_time=min(
                    max_time, start_time_j + params_constraints.plus_delta_primary
                )
                if params_constraints.constraint_max_time_to_current_solution
                else start_time_j + params_constraints.plus_delta_primary,
                sign=SignEnum.LEQ,
                part_id=j + 1,
            )
            string1_dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job,
                duration=max(
                    duration_j - params_constraints.minus_delta_primary_duration, 0
                ),
                sign=SignEnum.UEQ,
                part_id=j + 1,
            )
            string2_dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job,
                duration=duration_j + params_constraints.plus_delta_primary_duration,
                sign=SignEnum.LEQ,
                part_id=j + 1,
            )
            list_strings += [string1_dur, string1_start, string2_dur, string2_start]
        for k in range(len(list_starts), cp_solver.nb_preemptive):
            if isinstance(
                cp_solver,
                (
                    CP_MS_MRCPSP_MZN,
                    CP_MS_MRCPSP_MZN_PREEMPTIVE,
                    CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE,
                ),
            ):
                # WARNING, for multiskill, possible to put 10 for example
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job,
                    duration=params_constraints.plus_delta_primary_duration,
                    sign=SignEnum.LEQ,
                    part_id=k + 1,
                )
            else:
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job, duration=0, sign=SignEnum.LEQ, part_id=k + 1
                )
            list_strings += [string1_dur]
    for job in tasks_secondary:
        is_paused = len(current_solution.get_start_times_list(job)) > 1
        is_paused_str = "true" if is_paused else "false"
        list_strings += [
            "constraint is_paused["
            + str(cp_solver.index_in_minizinc[job])
            + "]=="
            + is_paused_str
            + ";\n"
        ]
    for job in tasks_secondary:
        if multimode:
            list_strings += cp_solver.constraint_task_to_mode(
                task_id=job, mode=modes_dict[job]
            )
        if job in tasks_primary:
            continue
        list_starts = current_solution.get_start_times_list(job)
        list_ends = current_solution.get_end_times_list(job)
        for j in range(len(list_starts)):
            start_time_j = list_starts[j]
            end_time_j = list_ends[j]
            duration_j = end_time_j - start_time_j
            if (
                params_constraints.minus_delta_secondary == 0
                and params_constraints.plus_delta_secondary == 0
            ):
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=start_time_j,
                    sign=SignEnum.EQUAL,
                    part_id=j + 1,
                )
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job, duration=duration_j, sign=SignEnum.EQUAL, part_id=j + 1
                )
                list_strings += [string1_dur, string1_start]
            else:
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=max(
                        0, start_time_j - params_constraints.minus_delta_secondary
                    ),
                    sign=SignEnum.UEQ,
                    part_id=j + 1,
                )
                string2_start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=min(
                        max_time, start_time_j + params_constraints.plus_delta_secondary
                    )
                    if params_constraints.constraint_max_time_to_current_solution
                    else start_time_j + params_constraints.plus_delta_secondary,
                    sign=SignEnum.LEQ,
                    part_id=j + 1,
                )
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job,
                    duration=max(
                        duration_j - params_constraints.minus_delta_secondary_duration,
                        0,
                    ),
                    sign=SignEnum.UEQ,
                    part_id=j + 1,
                )
                string2_dur = cp_solver.constraint_duration_string_preemptive_i(
                    task=job,
                    duration=duration_j
                    + params_constraints.plus_delta_secondary_duration,
                    sign=SignEnum.LEQ,
                    part_id=j + 1,
                )
                list_strings += [string1_dur, string1_start, string2_dur, string2_start]
        for k in range(len(list_starts), cp_solver.nb_preemptive):
            if (
                params_constraints.minus_delta_secondary == 0
                and params_constraints.plus_delta_secondary == 0
            ):
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=current_solution.get_end_time(job),
                    sign=SignEnum.EQUAL,
                    part_id=k + 1,
                )
                list_strings += [string1_start]
            else:
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=current_solution.get_end_time(job)
                    - params_constraints.minus_delta_secondary,
                    sign=SignEnum.UEQ,
                    part_id=k + 1,
                )
                string2_start = cp_solver.constraint_start_time_string_preemptive_i(
                    task=job,
                    start_time=min(
                        max_time,
                        current_solution.get_end_time(job)
                        + params_constraints.plus_delta_secondary,
                    )
                    if params_constraints.constraint_max_time_to_current_solution
                    else current_solution.get_end_time(job)
                    + params_constraints.plus_delta_secondary,
                    sign=SignEnum.LEQ,
                    part_id=k + 1,
                )
                list_strings += [string2_start, string1_start]
            string1_dur = cp_solver.constraint_duration_string_preemptive_i(
                task=job, duration=0, sign=SignEnum.EQUAL, part_id=k + 1
            )
            list_strings += [string1_dur]
    return list_strings


def constraints_strings_multiskill(
    current_solution: MS_RCPSPSolution,
    cp_solver: ANY_CP_SOLVER,
    tasks_primary: Set[Hashable],
    tasks_secondary: Set[Hashable],
    params_constraints: ParamsConstraintBuilder,
):
    list_strings = constraints_strings(
        current_solution=current_solution,
        tasks_primary=tasks_primary,
        tasks_secondary=tasks_secondary,
        cp_solver=cp_solver,
        params_constraints=params_constraints,
    )
    if params_constraints.first_method_multiskill:
        if not params_constraints.except_assigned_multiskill_primary_set:
            list_strings += constraint_unit_used_to_tasks(
                tasks_set=set(
                    random.sample(
                        current_solution.problem.tasks_list,
                        min(
                            int(
                                params_constraints.fraction_of_task_assigned_multiskill
                                * current_solution.problem.n_jobs
                            ),
                            current_solution.problem.n_jobs,
                        ),
                    )
                ),
                current_solution=current_solution,
                cp_solver=cp_solver,
            )
        else:
            list_strings += constraint_unit_used_to_tasks(
                tasks_set=tasks_secondary,
                current_solution=current_solution,
                cp_solver=cp_solver,
            )
        if params_constraints.additional_methods:
            list_strings += constraints_start_on_end(
                current_solution, cp_solver=cp_solver, frac=0.25
            )
            nb_usage = sum(
                [
                    len(current_solution.employee_usage[t])
                    for t in current_solution.employee_usage
                ]
            )
            nb_usage_per_employee = [
                sum(
                    [
                        e in current_solution.employee_usage[t]
                        for t in current_solution.employee_usage
                    ]
                )
                for e in current_solution.problem.employees_list
            ]
            list_strings += ["constraint sum(unit_used)<=" + str(nb_usage + 5) + ";\n"]
            for i in range(len(nb_usage_per_employee)):
                list_strings += [
                    "constraint sum(i in Act)(unit_used["
                    + str(i + 1)
                    + ",i])<="
                    + str(nb_usage_per_employee[i] + 3)
                    + ";\n"
                ]
                list_strings += [
                    "constraint sum(i in Act)(unit_used["
                    + str(i + 1)
                    + ",i])>="
                    + str(nb_usage_per_employee[i] - 3)
                    + ";\n"
                ]
    elif params_constraints.second_method_multiskill:
        employee_usage_matrix, sum_usage, employees_usage_dict = employee_usage(
            solution=current_solution, problem=current_solution.problem
        )
        sorted_employee = np.argsort(sum_usage)
        set_employees_to_fix = set(
            [
                current_solution.problem.employees_list[i]
                for i in sorted_employee[int(len(sorted_employee) / 2) :]
            ]
        )
        set_employees_to_fix = set(
            random.sample(
                current_solution.problem.employees_list,
                max(0, int(len(sorted_employee) / 2)),
            )
        )
        list_strings += constraint_unit_used_subset_employees(
            employees_set=set_employees_to_fix,
            current_solution=current_solution,
            cp_solver=cp_solver,
            employees_usage_dict=employees_usage_dict,
        )
        if params_constraints.additional_methods:
            list_strings += constraints_start_on_end(
                current_solution, cp_solver=cp_solver, frac=0.25
            )
            list_strings += [
                constraint_number_of_change_in_worker_allocation(
                    current_solution=current_solution, nb_moves=10
                )
            ]
    else:
        list_strings += [
            constraint_number_of_change_in_worker_allocation(
                current_solution=current_solution, nb_moves=20
            )
        ]
        list_strings += constraints_start_on_end(
            current_solution, cp_solver=cp_solver, frac=0.25
        )
    return list_strings


def constraint_number_of_change_in_worker_allocation(
    current_solution: MS_RCPSPSolution, nb_moves: int
):
    ref_unit_used = [
        [False for i in range(current_solution.problem.nb_tasks)]
        for j in range(current_solution.problem.nb_employees)
    ]
    for i in range(len(current_solution.problem.tasks_list)):
        t = current_solution.problem.tasks_list[i]
        for j in range(current_solution.problem.nb_employees):
            if current_solution.problem.employees_list[
                j
            ] in current_solution.employee_usage.get(t, {}):
                ref_unit_used[j][i] = True
    stringit = (
        "["
        + ",".join(
            [
                "false" if ref_unit_used[j][i] else "true"
                for j in range(len(ref_unit_used))
                for i in range(len(ref_unit_used[j]))
            ]
        )
        + "]"
    )
    s = (
        """
        int: nb_move="""
        + str(nb_moves)
        + """;\n
        set of int: MOVES=0..nb_move;\n
        array[Units, Act] of var bool: changed;\n
        constraint sum(w in Units, a in Act)(bool2int(changed[w, a]))<=nb_move;\n
        constraint forall(w in Units, a in Act)(changed[w, a]->unit_used[w, a]==ref_unit_used_not[w, a]);\n
        constraint forall(w in Units, a in Act)(not changed[w, a]->unit_used[w, a]==not ref_unit_used_not[w, a]);\n
        array[Units, Act] of bool: ref_unit_used_not=array2d(Units, Act, """
        + stringit
        + """);\n
        array[Units] of int: unit_usage=[sum(i in Act)(bool2int(not ref_unit_used_not[w, i])) | w in Units];\n
        %constraint forall(w in Units)(sum(row(unit_used, w))<=unit_usage[w]+nb_move/\sum(row(unit_used, w))
        %                              >=unit_usage[w]-nb_move);\n
        %constraint forall(w in Units)(sum(i in Act where not ref_unit_used_not[w, i])(bool2int(unit_used[w, i]))>=unit_usage[w]-nb_move);\n
        constraint sum(w in Units, a in Act)(unit_used[w, a]==ref_unit_used_not[w, a])<=nb_move;\n
        """
    )
    return s


def constraint_number_of_change_in_worker_allocation_preemptive(
    current_solution: MS_RCPSPSolution_Preemptive, nb_preemptive: int, nb_moves: int
):
    ref_unit_used = [
        [
            [False for k in range(nb_preemptive)]
            for i in range(current_solution.problem.nb_tasks)
        ]
        for j in range(current_solution.problem.nb_employees)
    ]
    for i in range(len(current_solution.problem.tasks_list)):
        t = current_solution.problem.tasks_list[i]
        employee_used = current_solution.employee_used(t)
        l = len(employee_used)
        for m in range(l):
            for j in range(current_solution.problem.nb_employees):
                if current_solution.problem.employees_list[j] in employee_used[m]:
                    ref_unit_used[j][i][m] = True
    stringit = (
        "["
        + ",".join(
            [
                "false" if ref_unit_used[j][i][l] else "true"
                for j in range(len(ref_unit_used))
                for i in range(len(ref_unit_used[j]))
                for l in range(len(ref_unit_used[j][i]))
            ]
        )
        + "]"
    )
    s = (
        """
        int: nb_move="""
        + str(nb_moves)
        + """;\n
        set of int: MOVES=0..nb_move;\n
        array[Units, Tasks, PREEMPTIVE] of var bool: changed;\n
        constraint sum(changed)<=nb_move;\n
        constraint forall(w in Units, a in Tasks, j in PREEMPTIVE)(changed[w, a, j]->unit_used_preemptive[w, a, j]==ref_unit_used_not[w, a, j]);\n
        constraint forall(w in Units, a in Tasks, j in PREEMPTIVE)(not changed[w, a, j]->unit_used_preemptive[w, a, j]==not ref_unit_used_not[w, a, j]);\n
        array[Units, Tasks, PREEMPTIVE] of bool: ref_unit_used_not=array3d(Units, Tasks, PREEMPTIVE, """
        + stringit
        + """);\n
        array[Units] of int: unit_usage=[sum(i in Tasks, j in PREEMPTIVE)(bool2int(not ref_unit_used_not[w, i, j])) | w in Units];\n
        constraint sum(w in Units, a in Tasks, j in PREEMPTIVE)(unit_used_preemptive[w, a, j]==ref_unit_used_not[w, a, j])<=nb_move;\n
        """
    )
    return s


def constraints_exchange_worker(
    current_solution: MS_RCPSPSolution, nb_moves: int, cp_solver: ANY_CP_SOLVER
):
    ref_unit_used = [
        [False for i in range(current_solution.problem.nb_tasks)]
        for j in range(current_solution.problem.nb_employees)
    ]
    for i in range(len(current_solution.problem.tasks_list)):
        t = current_solution.problem.tasks_list[i]
        for j in range(current_solution.problem.nb_employees):
            if current_solution.problem.employees_list[
                j
            ] in current_solution.employee_usage.get(t, {}):
                ref_unit_used[j][i] = True
    stringit = (
        "["
        + ",".join(
            [
                "true" if ref_unit_used[j][i] else "false"
                for j in range(len(ref_unit_used))
                for i in range(len(ref_unit_used[j]))
            ]
        )
        + "]"
    )
    s = (
        """
        int: nb_move="""
        + str(nb_moves)
        + """;\n
        set of int: UnitsWithZero = 0..nb_units;\n
        set of int: ActsWithZero = 0..nb_act;\n
        array[Units, Act] of bool: ref_unit_used==array2d(Units, Act, """
        + stringit
        + """)\n;
        array[1..nb_move, 1..2] of var UnitsWithZero: moves_unit;\n
        array[1..nb_move, 1..2] of var ActsWithZero: moves_act;\n
        constraint forall(i in 1..nb_move)(moves_unit[i, 1]==0<->moves_act[i, 1]==0 /\ moves_act[i, 2]==0);\n
        constraint forall(i in 1..nb_move)(moves_unit[i, 1]!=0->(moves_act[i,1]!=0
                                           /\ ref_unit_used[max([1, moves_unit[i,1]]), max([1, moves_act[i,1]])]
                                           /\ moves_act[i,2] != 0 /\ ref_unit_used[max([1, moves_unit[i,2]]),
                                                                                   max([1, moves_act[i,2]])] /\
                                           unit_used[max([1, moves_unit[i, 1]]), max([1,moves_act[i,2]])] /\
                                           unit_used[max([1, moves_unit[i, 2]]), max([1, moves_act[i,1]])]));\n
        constraint forall(i in 1..nb_move)(moves_unit[i, 1]<=moves_unit[i,2]);\n
        constraint forall(i in 1..nb_move)(moves_unit[i, 1]!=0->moves_unit[i, 2]>moves_unit[i, 1]);\n
        include "decreasing.mzn";
        constraint decreasing([moves_unit[j, 1]| j in 1..nb_move]);\n
        constraint forall(w in Units)(if not exists(j in 1..nb_move, i in 1..2)(moves_unit[j, i]==w) then
                                         forall(a in Act)(unit_used[w, a]==ref_unit_used[w,a])
                                      else
                                        true
                                      endif);\n
        constraint forall(a in Act)(if not exists(j in 1..nb_move, i in 1..2)(moves_act[j, i]==a) then
                                       forall(w in Units)(unit_used[w, a]==ref_unit_used[w,a])
                                    else
                                       true
                                    endif);\n
        constraint forall(w in Units, a in Act)(bool2int(unit_used[w,a])!=ref_unit_used[w, a])!=2*nb_move;\n
        """
    )
    return s


def constraints_start_on_end(
    current_solution: MS_RCPSPSolution, cp_solver: ANY_CP_SOLVER, frac=0.5
):
    end_times = {
        t: current_solution.get_end_time(t) for t in current_solution.problem.tasks_list
    }
    data = {}
    listed = []
    for t in current_solution.schedule:
        st = current_solution.get_start_time(t)
        tasks_ending = [tt for tt in end_times if end_times[tt] == st]
        listed += [(t, tt) for tt in tasks_ending]
        data[st] = tasks_ending
    selected = listed
    selected_task = random.sample(
        cp_solver.problem.tasks_list, k=int(frac * cp_solver.problem.nb_tasks)
    )
    s = []
    cnt = 0
    for i1, i2 in selected:
        if i1 not in selected_task or i2 not in selected_task:
            continue
        if isinstance(cp_solver, CP_MSPSP_MZN):
            st = (
                "constraint start["
                + str(cp_solver.index_in_minizinc[i1])
                + "]==start["
                + str(cp_solver.index_in_minizinc[i2])
                + "]+dur["
                + str(cp_solver.index_in_minizinc[i2])
                + "];\n"
            )
        else:
            st = (
                "constraint start["
                + str(cp_solver.index_in_minizinc[i1])
                + "]==start["
                + str(cp_solver.index_in_minizinc[i2])
                + "]+adur["
                + str(cp_solver.index_in_minizinc[i2])
                + "];\n"
            )
        s += [st]
        cnt += 1
    return s


def constraints_start_on_end_preemptive(
    current_solution: ANY_SOLUTION_PREEMPTIVE, cp_solver: ANY_CP_SOLVER, frac=0.5
):
    end_times = {
        t: current_solution.get_end_time(t) for t in current_solution.problem.tasks_list
    }
    data = {}
    listed = []
    for t in current_solution.schedule:
        st = current_solution.get_start_time(t)
        tasks_ending = [tt for tt in end_times if end_times[tt] == st]
        listed += [(t, tt) for tt in tasks_ending]
        data[st] = tasks_ending
    selected = listed
    selected_task = random.sample(
        cp_solver.problem.tasks_list, k=int(frac * cp_solver.problem.nb_tasks)
    )
    s = []
    cnt = 0
    for i1, i2 in selected:
        if i1 not in selected_task or i2 not in selected_task:
            continue
        st = (
            "constraint s_preemptive["
            + str(cp_solver.index_in_minizinc[i1])
            + ", 1]==s_preemptive["
            + str(cp_solver.index_in_minizinc[i2])
            + ",nb_preemptive];\n"
        )
        s += [st]
        cnt += 1
    return s


def constraints_strings_multiskill_preemptive(
    current_solution: ANY_SOLUTION,
    cp_solver: ANY_CP_SOLVER,
    tasks_primary: Set[Hashable],
    tasks_secondary: Set[Hashable],
    params_constraints: ParamsConstraintBuilder,
):
    list_strings = constraints_strings_preemptive(
        current_solution=current_solution,
        tasks_primary=tasks_primary,
        tasks_secondary=tasks_secondary,
        cp_solver=cp_solver,
        params_constraints=params_constraints,
    )
    if random.random() < 0.99 and params_constraints.first_method_multiskill:

        constraint_description = start_together_problem_description(
            solution=current_solution,
            constraints=current_solution.problem.special_constraints,
        )
        exceptions = []
        exceptions += [(x[1], 0) for x in constraint_description]
        exceptions += [(x[2], 0) for x in constraint_description]
        list_strings += constraint_unit_used_to_tasks_preemptive(
            tasks_set=set(
                random.sample(
                    current_solution.problem.tasks_list,
                    min(
                        int(
                            params_constraints.fraction_of_task_assigned_multiskill
                            * current_solution.problem.n_jobs
                        ),
                        current_solution.problem.n_jobs,
                    ),
                )
            ),
            current_solution=current_solution,
            cp_solver=cp_solver,
            exceptions=exceptions,
        )
        if params_constraints.additional_methods:
            list_strings += constraints_start_on_end_preemptive(
                current_solution, cp_solver=cp_solver, frac=0.3
            )
            list_strings += [
                constraint_number_of_change_in_worker_allocation_preemptive(
                    current_solution=current_solution,
                    nb_moves=4,
                    nb_preemptive=cp_solver.nb_preemptive,
                )
            ]
    elif params_constraints.second_method_multiskill:
        employee_usage_matrix, sum_usage, employees_usage_dict = employee_usage(
            solution=current_solution, problem=current_solution.problem
        )
        set_employees_to_fix = set(
            random.sample(
                current_solution.problem.employees_list,
                int(len(current_solution.problem.employees_list) - 2),
            )
        )
        list_strings += constraint_unit_used_subset_employees_preemptive(
            employees_set=set_employees_to_fix,
            current_solution=current_solution,
            cp_solver=cp_solver,
            employees_usage_dict=employees_usage_dict,
            exceptions=None,
        )
    return list_strings


def constraint_unit_used_to_tasks(
    tasks_set: Set[Hashable],
    current_solution: MS_RCPSPSolution,
    cp_solver: Union[CP_MS_MRCPSP_MZN],
):
    all_employees = set(current_solution.problem.employees_list)
    strings = []
    for task in tasks_set:
        if task in current_solution.employee_usage:
            employees = set(
                [
                    e
                    for e in current_solution.employee_usage[task]
                    if len(current_solution.employee_usage[task][e]) > 0
                ]
            )
            for e in all_employees:
                strings += cp_solver.constraint_used_employee(
                    task=task, employee=e, indicator=e in employees
                )
        else:
            for e in all_employees:
                strings += cp_solver.constraint_used_employee(
                    task=task, employee=e, indicator=False
                )
    return strings


def constraint_unit_used_subset_employees(
    employees_set: Set[Hashable],
    current_solution: MS_RCPSPSolution,
    cp_solver: Union[CP_MS_MRCPSP_MZN],
    employees_usage_dict=None,
):
    if employees_usage_dict is None:
        employee_usage_matrix, sum_usage, employees_usage_dict = employee_usage(
            current_solution, current_solution.problem
        )
    strings = []
    for e in employees_set:
        for (task, j) in employees_usage_dict[e]:
            strings += cp_solver.constraint_used_employee(
                task=task, employee=e, indicator=True
            )
        for t in current_solution.problem.tasks_list:
            if (t, 0) not in employees_usage_dict[e]:
                strings += cp_solver.constraint_used_employee(
                    task=t, employee=e, indicator=False
                )
    return strings


def constraint_unit_used_to_tasks_preemptive(
    tasks_set: Set[Hashable],
    current_solution: MS_RCPSPSolution_Preemptive,
    cp_solver: Union[CP_MS_MRCPSP_MZN_PREEMPTIVE],
    exceptions=None,
):
    if exceptions is None:
        exceptions = []
    all_employees = set(current_solution.problem.employees_list)
    strings = []
    for task in tasks_set:
        if task in current_solution.employee_usage:
            len_emp_usage = len(current_solution.employee_usage[task])
            for i in range(len_emp_usage):
                if (task, i) in exceptions:
                    continue
                employees = set(
                    [
                        e
                        for e in current_solution.employee_usage[task][i]
                        if len(current_solution.employee_usage[task][i][e]) > 0
                    ]
                )
                for e in all_employees:
                    strings += cp_solver.constraint_used_employee(
                        task=task, employee=e, part_id=i + 1, indicator=e in employees
                    )
            for i in range(len_emp_usage, cp_solver.nb_preemptive):
                for e in all_employees:
                    strings += cp_solver.constraint_used_employee(
                        task=task, employee=e, part_id=i + 1, indicator=False
                    )
        else:
            for i in range(cp_solver.nb_preemptive):
                for e in all_employees:
                    strings += cp_solver.constraint_used_employee(
                        task=task, employee=e, part_id=i + 1, indicator=False
                    )
    return strings


def constraint_unit_used_subset_employees_preemptive(
    employees_set: Set[Hashable],
    current_solution: MS_RCPSPSolution_Preemptive,
    cp_solver: Union[CP_MS_MRCPSP_MZN_PREEMPTIVE],
    employees_usage_dict=None,
    exceptions=None,
):
    if exceptions is None:
        exceptions = []
    if employees_usage_dict is None:
        employee_usage_matrix, sum_usage, employees_usage_dict = employee_usage(
            current_solution, current_solution.problem
        )
    strings = []
    for e in employees_set:
        for (task, j) in employees_usage_dict[e]:
            if (task, j) in exceptions:
                continue
            strings += cp_solver.constraint_used_employee(
                task=task, employee=e, part_id=j + 1, indicator=True
            )
        for t in current_solution.problem.tasks_list:
            for j in range(cp_solver.nb_preemptive):
                if (t, j) in exceptions:
                    continue
                if (t, j) not in employees_usage_dict[e]:
                    strings += cp_solver.constraint_used_employee(
                        task=t, employee=e, part_id=j + 1, indicator=False
                    )
    return strings


class NeighborBuilder:
    @abstractmethod
    def find_subtasks(
        self, current_solution: ANY_SOLUTION, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        """
        Split the scheduling task set in 2 part, it can then be used by constraint handler to introduce different
        constraints in those two subsets. Usually the first returned set will be considered
        like the subproblem in LNS
        Args:
            current_solution: current solution to consider
            subtasks: possibly existing subset of tasks that are in the neighborhood

        Returns:

        """
        ...


def intersect(i1, i2):
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


class NeighborBuilderSubPart(NeighborBuilder):
    """
    Cut the schedule in different subpart in the increasing order of the schedule.
    """

    def __init__(
        self, problem: RCPSPModel, graph: GraphRCPSP = None, nb_cut_part: int = 10
    ):
        self.problem = problem
        self.graph = graph
        self.nb_cut_part = nb_cut_part
        self.current_sub_part = 0
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self, current_solution: RCPSPSolution, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        nb_job_sub = math.ceil(self.problem.n_jobs / self.nb_cut_part)
        task_of_interest = sorted(
            self.problem.tasks_list, key=lambda x: current_solution.get_end_time(x)
        )
        task_of_interest = task_of_interest[
            self.current_sub_part
            * nb_job_sub : (self.current_sub_part + 1)
            * nb_job_sub
        ]
        if subtasks is None:
            subtasks = task_of_interest
        else:
            subtasks.update(task_of_interest)
        if len(subtasks) == 0:
            subtasks = [self.problem.sink_task]
        self.current_sub_part = (self.current_sub_part + 1) % self.nb_cut_part
        return subtasks, self.set_tasks.difference(subtasks)


class NeighborRandom(NeighborBuilder):
    def __init__(
        self,
        problem: ANY_RCPSP,
        graph: GraphRCPSP = None,
        fraction_subproblem: float = 0.9,
        delta_abs_time_from_makespan_to_not_fix: int = 5,
        delta_rel_time_from_makespan_to_not_fix: float = 0.0,
    ):
        self.problem = problem
        self.graph = graph
        if self.graph is None:
            self.graph = build_graph_rcpsp_object(rcpsp_problem=problem)
        self.fraction_subproblem = fraction_subproblem
        self.delta_abs_time_from_makespan_to_not_fix = (
            delta_abs_time_from_makespan_to_not_fix
        )
        self.delta_rel_time_from_makespan_to_not_fix = (
            delta_rel_time_from_makespan_to_not_fix
        )
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self, current_solution: ANY_SOLUTION, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        if subtasks is None:
            subtasks = set()
        max_time = current_solution.get_end_time(self.problem.sink_task)
        last_jobs = [
            x
            for x in self.problem.tasks_list
            if max_time - self.delta_abs_time_from_makespan_to_not_fix
            <= current_solution.get_end_time(x)
            <= max_time
            or (1 - self.delta_rel_time_from_makespan_to_not_fix) * max_time
            <= current_solution.get_end_time(x)
            <= max_time
        ]
        nb_jobs = self.problem.n_jobs
        tasks_subproblem = set(
            random.sample(
                self.problem.tasks_list, int(self.fraction_subproblem * nb_jobs)
            )
        )
        for lj in last_jobs:
            if lj not in tasks_subproblem:
                tasks_subproblem.add(lj)
        subtasks.update(tasks_subproblem)
        return subtasks, self.set_tasks.difference(subtasks)


class NeighborRandomAndNeighborGraph(NeighborBuilder):
    def __init__(
        self,
        problem: ANY_RCPSP,
        graph: GraphRCPSP = None,
        fraction_subproblem: float = 0.1,
    ):
        self.problem = problem
        self.graph = graph
        if self.graph is None:
            self.graph = build_graph_rcpsp_object(rcpsp_problem=self.problem)
        self.fraction_subproblem = fraction_subproblem
        self.nb_jobs_subproblem = math.ceil(
            self.problem.n_jobs * self.fraction_subproblem
        )
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self, current_solution: RCPSPSolution, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        if subtasks is None:
            subtasks = set()
            len_subtask = 0
        else:
            len_subtask = len(subtasks)
        while len_subtask < self.nb_jobs_subproblem:
            random_pick = random.choice(self.problem.tasks_list)
            interval = (
                current_solution.get_start_time(random_pick),
                current_solution.get_end_time(random_pick),
            )
            task_intersect = [
                t
                for t in self.problem.tasks_list
                if intersect(
                    interval,
                    (
                        current_solution.get_start_time(t),
                        current_solution.get_end_time(t),
                    ),
                )
                is not None
            ]
            for k in set(task_intersect):
                task_intersect += list(self.graph.get_pred_activities(k)) + list(
                    self.graph.get_next_activities(k)
                )
                if isinstance(self.graph, GraphRCPSPSpecialConstraints):
                    task_intersect += self.graph.get_neighbors_constraints(k)
                    task_intersect += self.graph.get_neighbors_constraints(k)
            subtasks.update(task_intersect)
            len_subtask = len(subtasks)
        if len(subtasks) >= self.nb_jobs_subproblem:
            subtasks = set(random.sample(list(subtasks), self.nb_jobs_subproblem))
        return subtasks, self.set_tasks.difference(subtasks)


class NeighborConstraintBreaks(NeighborBuilder):
    def __init__(
        self,
        problem: Union[RCPSPModelSpecialConstraintsPreemptive, RCPSPModel],
        graph: GraphRCPSP = None,
        fraction_subproblem: float = 0.1,
        other_constraint_handler: NeighborBuilder = None,
    ):
        if (
            not hasattr(problem, "do_special_constraints")
            or not problem.includes_special_constraint()
        ):
            raise ValueError(
                "NeighborConstraintBreaks is meant for problems with special constraints"
            )
        self.problem = problem
        self.graph = graph
        if self.graph is None:
            self.graph = build_graph_rcpsp_object(rcpsp_problem=problem)
        self.fraction_subproblem = fraction_subproblem
        self.nb_jobs_subproblem = math.ceil(
            self.problem.n_jobs * self.fraction_subproblem
        )
        self.other = other_constraint_handler
        if self.other is None:
            self.other = NeighborBuilderSubPart(
                problem=self.problem,
                graph=self.graph,
                nb_cut_part=int(math.ceil(1 / self.fraction_subproblem)),
            )
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self, current_solution: RCPSPSolution, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        if "special_constraints" in self.problem.__dict__.keys():
            details_constraints = compute_constraints_details(
                solution=current_solution, constraints=self.problem.special_constraints
            )
            sorted_constraints = sorted(details_constraints, key=lambda x: -x[-1])

            if len(sorted_constraints) == 1:
                t1, t2 = sorted_constraints[0][1], sorted_constraints[0][2]
                st1, end1 = current_solution.get_start_time(
                    t1
                ), current_solution.get_end_time(t1)
                st2, end2 = current_solution.get_start_time(
                    t2
                ), current_solution.get_end_time(t2)
                st_min = min(st1, st2)
                st_max = max(end1, end2)
                tasks = [
                    t
                    for t in self.problem.tasks_list
                    if st_min - 5 <= current_solution.get_start_time(t) <= st_max + 5
                ]
                subtasks = set(tasks)
                return subtasks, self.set_tasks.difference(subtasks)
            random.shuffle(sorted_constraints)
            subtasks = set()
            len_subtasks = 0
            j = 0
            while (
                j <= len(sorted_constraints) - 1
                and len_subtasks < 2 * self.nb_jobs_subproblem
            ):
                t1, t2 = sorted_constraints[j][1], sorted_constraints[j][2]
                subtasks.add(t1)
                subtasks.add(t2)
                if t1 in self.graph.index_components:
                    subtasks.update(
                        self.graph.components_graph_constraints[
                            self.graph.index_components[t1]
                        ]
                    )
                    for c in set(
                        self.graph.components_graph_constraints[
                            self.graph.index_components[t1]
                        ]
                    ):
                        subtasks.update(self.graph.get_next_activities(c))
                        subtasks.update(self.graph.get_descendants_activities(c))
                        subtasks.update(self.graph.get_ancestors_activities(c))
                        subtasks.update(self.graph.get_pred_activities(c))
                else:
                    subtasks.update(list(self.graph.get_next_activities(t1)))
                    subtasks.update(list(self.graph.get_pred_activities(t1)))
                len_subtasks = len(subtasks)
                j += 1
            if len_subtasks < self.nb_jobs_subproblem:
                subtasks, _ = self.other.find_subtasks(
                    current_solution=current_solution, subtasks=subtasks
                )
            len_subtasks = len(subtasks)
            if len_subtasks > self.nb_jobs_subproblem:
                subtasks = set(random.sample(list(subtasks), self.nb_jobs_subproblem))
            return subtasks, self.set_tasks.difference(subtasks)
        else:
            return self.other.find_subtasks(
                current_solution=current_solution, subtasks=subtasks
            )


class NeighborBuilderMix(NeighborBuilder):
    def __init__(
        self,
        list_neighbor: List[NeighborBuilder],
        weight_neighbor: Union[List[float], np.array],
        verbose: bool = False,
    ):
        self.list_neighbor = list_neighbor
        self.weight_neighbor = weight_neighbor
        if isinstance(self.weight_neighbor, list):
            self.weight_neighbor = np.array(self.weight_neighbor)
        self.weight_neighbor = self.weight_neighbor / np.sum(self.weight_neighbor)
        self.index_np = np.array(range(len(self.list_neighbor)), dtype=np.int_)
        self.verbose = verbose

    def find_subtasks(
        self, current_solution: RCPSPSolution, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        choice = np.random.choice(self.index_np, size=1, p=self.weight_neighbor)[0]
        return self.list_neighbor[choice].find_subtasks(
            current_solution=current_solution, subtasks=subtasks
        )


class NeighborBuilderTimeWindow(NeighborBuilder):
    def find_subtasks(
        self, current_solution: ANY_SOLUTION, subtasks: Optional[Set[Hashable]] = None
    ) -> Tuple[Set[Hashable], Set[Hashable]]:
        last_time = current_solution.get_end_time(self.problem.sink_task)
        if self.current_time_window[0] >= last_time:
            self.current_time_window = [0, self.time_window_length]
        tasks_of_interest = [
            t
            for t in self.problem.tasks_list
            if any(
                current_solution.get_start_time(t)
                <= x
                <= current_solution.get_end_time(t)
                for x in range(self.current_time_window[0], self.current_time_window[1])
            )
        ]
        other_tasks = [t for t in self.problem.tasks_list if t not in tasks_of_interest]
        self.current_time_window = [
            self.current_time_window[0] + self.time_window_length,
            self.current_time_window[1] + self.time_window_length,
        ]
        return set(tasks_of_interest), set(other_tasks)

    def __init__(
        self, problem: ANY_RCPSP, graph: GraphRCPSP = None, time_window_length: int = 10
    ):
        self.problem = problem
        self.graph = graph
        self.time_window_length = time_window_length
        self.current_time_window = [0, self.time_window_length]


class BasicConstraintBuilder:
    def __init__(
        self,
        neighbor_builder: NeighborBuilder,
        params_constraint_builder: ParamsConstraintBuilder = None,
        preemptive: bool = False,
        multiskill: bool = False,
        verbose: bool = False,
    ):
        self.params_constraint_builder = params_constraint_builder
        if self.params_constraint_builder is None:
            self.params_constraint_builder = ParamsConstraintBuilder.default()
        self.preemptive = preemptive
        self.neighbor_builder = neighbor_builder
        self.matrix_func = {
            True: {
                True: constraints_strings_multiskill_preemptive,
                False: constraints_strings_multiskill,
            },
            False: {True: constraints_strings_preemptive, False: constraints_strings},
        }
        self.func = self.matrix_func[multiskill][preemptive]
        self.verbose = verbose

    def return_constraints(
        self,
        current_solution: ANY_SOLUTION,
        cp_solver: ANY_CP_SOLVER,
        params_constraint_builder: ParamsConstraintBuilder = None,
    ):
        p = (
            params_constraint_builder
            if params_constraint_builder is not None
            else self.params_constraint_builder
        )
        subtasks_1, subtasks_2 = self.neighbor_builder.find_subtasks(
            current_solution=current_solution
        )
        logger.debug(self.__class__.__name__)
        logger.debug(f"{len(subtasks_1)} in first set, {len(subtasks_2)} in second set")
        list_strings = self.func(
            current_solution=current_solution,
            cp_solver=cp_solver,
            tasks_primary=subtasks_1,
            tasks_secondary=subtasks_2,
            params_constraints=p,
        )
        return list_strings, subtasks_1, subtasks_2


class ObjectiveSubproblem(Enum):
    MAKESPAN_SUBTASKS = 0
    SUM_START_SUBTASKS = 1
    SUM_END_SUBTASKS = 2
    GLOBAL_MAKESPAN = 3


class ConstraintHandlerScheduling(ConstraintHandler):
    def __init__(
        self,
        problem: ANY_RCPSP,
        basic_constraint_builder: BasicConstraintBuilder,
        params_list: List[ParamsConstraintBuilder] = None,
        use_makespan_of_subtasks: bool = True,
        objective_subproblem: ObjectiveSubproblem = ObjectiveSubproblem.GLOBAL_MAKESPAN,
        verbose: bool = True,
    ):
        self.problem = problem
        self.basic_constraint_builder = basic_constraint_builder

        if isinstance(self.problem, RCPSPModelSpecialConstraintsPreemptive,) or (
            isinstance(self.problem, RCPSPModel) and self.problem.do_special_constraints
        ):
            self.graph_rcpsp = GraphRCPSPSpecialConstraints(problem=self.problem)
            self.special_constraints = True
        else:
            self.graph_rcpsp = GraphRCPSP(problem=self.problem)
            self.special_constraints = False
        if params_list is None:
            self.params_list = [
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=1,
                    plus_delta_secondary=1,
                    constraint_max_time_to_current_solution=True,
                ),
                ParamsConstraintBuilder(
                    minus_delta_primary=6000,
                    plus_delta_primary=6000,
                    minus_delta_secondary=300,
                    plus_delta_secondary=300,
                    constraint_max_time_to_current_solution=True,
                ),
            ]
        else:
            self.params_list = params_list
        self.use_makespan_of_subtasks = use_makespan_of_subtasks
        self.objective_subproblem = objective_subproblem
        if self.use_makespan_of_subtasks:
            self.objective_subproblem = ObjectiveSubproblem.MAKESPAN_SUBTASKS
        self.verbose = verbose

    def adding_constraint_from_results_store(
        self,
        cp_solver: ANY_CP_SOLVER,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution: RCPSPSolutionPreemptive = current_solution
        evaluation = self.problem.evaluate(current_solution)
        logger.debug(self.__class__.__name__)
        logger.debug(f"Current Eval : {evaluation}")
        if evaluation.get("constraint_penalty", 0) == 0:
            p = self.params_list[0]
        else:
            # Allow for some tricks. multistage optim
            p = self.params_list[min(len(self.params_list) - 1, 1)]
        (
            list_strings,
            subtasks_1,
            subtasks_2,
        ) = self.basic_constraint_builder.return_constraints(
            current_solution=current_solution,
            cp_solver=cp_solver,
            params_constraint_builder=p,
        )
        for s in list_strings:
            child_instance.add_string(s)
        if evaluation.get("constraint_penalty", 0) > 0:
            child_instance.add_string(
                "constraint objective=" + str(evaluation["makespan"]) + ";\n"
            )
            # We ignore this part of objective.
        else:
            string = cp_solver.constraint_start_time_string(
                task=self.problem.sink_task,
                start_time=current_solution.get_start_time(self.problem.sink_task) + 20,
                sign=SignEnum.LEQ,
            )
            child_instance.add_string(string)
        if evaluation.get("constraint_penalty", 0) > 0:
            child_instance.add_string(
                "constraint sec_objective<="
                + str(int(1.01 * 100 * evaluation.get("constraint_penalty", 0)) + 1000)
                + ";\n"
            )
        child_instance.add_string("constraint sec_objective>=0;\n")
        if "constraint_penalty" not in evaluation:
            child_instance.add_string("constraint sec_objective==0;\n")
        if evaluation.get("constraint_penalty", 0) > 0:
            strings = []
        else:
            try:
                strings = cp_solver.add_hard_special_constraints(
                    self.problem.special_constraints
                )
                for s in strings:
                    child_instance.add_string(s)
                child_instance.add_string("constraint sec_objective==0;\n")
            except Exception as e:
                logger.warning(
                    "Hard constraint failed no method add_hard_special_constraints inside your cpsolver class,"
                    " but the code should work fine anyway !"
                )
            strings = []
            if self.objective_subproblem == ObjectiveSubproblem.MAKESPAN_SUBTASKS:
                strings = cp_solver.constraint_objective_max_time_set_of_jobs(
                    subtasks_1
                )
                current_max = max(
                    [current_solution.get_end_time(t) for t in subtasks_1]
                )
                strings += ["constraint objective<=" + str(current_max) + ";\n"]
            elif self.objective_subproblem == ObjectiveSubproblem.GLOBAL_MAKESPAN:
                strings = cp_solver.constraint_objective_equal_makespan(
                    self.problem.sink_task
                )
            elif self.objective_subproblem == ObjectiveSubproblem.SUM_START_SUBTASKS:
                strings = cp_solver.constraint_sum_of_starting_time(subtasks_1)
                sum_end = sum(
                    [
                        (10 if t == self.problem.sink_task else 1)
                        * current_solution.get_start_time(t)
                        for t in subtasks_1
                    ]
                )
                strings += ["constraint objective<=" + str(sum_end) + ";\n"]
            elif self.objective_subproblem == ObjectiveSubproblem.SUM_END_SUBTASKS:
                strings = cp_solver.constraint_sum_of_ending_time(subtasks_1)
                sum_end = sum(
                    [
                        (10 if t == self.problem.sink_task else 1)
                        * current_solution.get_end_time(t)
                        for t in subtasks_1
                    ]
                )
                strings += ["constraint objective<=" + str(sum_end) + ";\n"]
        for s in strings:
            child_instance.add_string(s)
            list_strings += [s]
        return list_strings

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: ANY_CP_SOLVER,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass


class ConstraintHandlerMultiskillAllocation(ConstraintHandler):
    def __init__(
        self,
        problem: ANY_MSRCPSP,
        params_list: List[ParamsConstraintBuilder] = None,
        verbose: bool = False,
    ):
        self.problem = problem

    def adding_constraint_from_results_store(
        self,
        cp_solver: ANY_CP_SOLVER,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution = current_solution
        evaluation = self.problem.evaluate(current_solution)
        list_strings = constraints_strings(
            current_solution=current_solution,
            cp_solver=cp_solver,
            tasks_primary=self.problem.tasks_list,
            tasks_secondary=set(),
            params_constraints=ParamsConstraintBuilder(
                minus_delta_primary=0,
                plus_delta_primary=0,
                minus_delta_secondary=0,
                plus_delta_secondary=0,
                constraint_max_time_to_current_solution=True,
            ),
        )
        list_strings += constraint_unit_used_to_tasks(
            tasks_set=set(
                random.sample(
                    current_solution.problem.get_tasks_list(),
                    min(
                        int(0.8 * current_solution.problem.n_jobs),
                        current_solution.problem.n_jobs - 1,
                    ),
                )
            ),
            current_solution=current_solution,
            cp_solver=cp_solver,
        )
        list_strings += cp_solver.constraint_objective_makespan()
        for s in list_strings:
            child_instance.add_string(s)
        child_instance.add_string("constraint sec_objective==max(res_load);\n")
        return list_strings

    def remove_constraints_from_previous_iteration(
        self, cp_solver: CPSolver, child_instance, previous_constraints: Iterable[Any]
    ):
        pass


class EquilibrateMultiskillAllocationNonPreemptive(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self, cp_solver: CPSolver, child_instance, previous_constraints: Iterable[Any]
    ):
        pass

    def __init__(
        self,
        problem: ANY_MSRCPSP,
        params_list: List[ParamsConstraintBuilder] = None,
        verbose: bool = False,
    ):
        self.problem = problem

    def adding_constraint_from_results_store(
        self,
        cp_solver: ANY_CP_SOLVER,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution = current_solution
        evaluation = current_solution.problem.evaluate(current_solution)
        s = """array[Units] of var 0..max_time: res_load = [sum(a in Tasks)( adur[a] * unit_used[w,a] )| w in Units ];\n"""
        ss = """array[Tasks] of var int: overskill = [sum(sk in Skill where array_skills_required[sk, i]>0)(sum(w in Units)(unit_used[w, i]*skillunits[w, sk])-array_skills_required[sk, i])|i in Tasks];\n"""
        child_instance.add_string(s)
        child_instance.add_string(ss)
        list_strings = constraints_strings_multiskill(
            current_solution=current_solution,
            cp_solver=cp_solver,
            tasks_primary=set(),
            tasks_secondary=self.problem.tasks_list,
            params_constraints=ParamsConstraintBuilder(
                minus_delta_primary=0,
                plus_delta_primary=0,
                minus_delta_secondary=0,
                plus_delta_secondary=0,
                constraint_max_time_to_current_solution=True,
                except_assigned_multiskill_primary_set=False,
                fraction_of_task_assigned_multiskill=0.85,
                first_method_multiskill=True,
                second_method_multiskill=False,
                additional_methods=False,
            ),
        )
        list_strings += cp_solver.constraint_objective_max_time_set_of_jobs(
            [self.problem.sink_task]
        )
        list_strings += [
            "constraint objective="
            + str(current_solution.get_end_time(self.problem.sink_task))
            + ";\n"
        ]
        for s in list_strings:
            child_instance.add_string(s)
        if random.random() < 0.999:
            if evaluation.get("constraint_penalty", 0) == 0:
                child_instance.add_string("constraint sec_objective>=sum(res_load);\n")
            else:
                child_instance.add_string(
                    "constraint sec_objective>=100*sum(overskill)+10*sum(res_load)+max(res_load)-min(res_load);\n"
                )
        else:
            child_instance.add_string(
                "constraint sec_objective>=min(res_load)-max(res_load);\n"
            )
        return []


class EquilibrateMultiskillAllocation(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self, cp_solver: CPSolver, child_instance, previous_constraints: Iterable[Any]
    ):
        pass

    def __init__(
        self,
        problem: ANY_MSRCPSP,
        params_list: List[ParamsConstraintBuilder] = None,
        verbose: bool = False,
    ):
        self.problem = problem

    def adding_constraint_from_results_store(
        self,
        cp_solver: ANY_CP_SOLVER,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next(
                (
                    last_result_store.list_solution_fits[j]
                    for j in range(len(last_result_store.list_solution_fits))
                    if "opti_from_cp"
                    in last_result_store.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        else:
            current_solution, fit = next(
                (
                    result_storage.list_solution_fits[j]
                    for j in range(len(result_storage.list_solution_fits))
                    if "opti_from_cp"
                    in result_storage.list_solution_fits[j][0].__dict__.keys()
                ),
                (None, None),
            )
        if current_solution is None or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution = current_solution
        evaluation = current_solution.problem.evaluate(current_solution)
        overskill = compute_overskill(
            problem=current_solution.problem, solution=current_solution
        )
        cnt = 0
        for t in overskill:
            for s in overskill[t]:
                if s != "W":
                    continue
                for i in overskill[t][s]:
                    cnt += 1
        s = """array[Units] of var 0..max_time: res_load = [sum(a in Tasks, j in PREEMPTIVE)( d_preemptive[a, j] * unit_used_preemptive[w,a,j] )| w in Units ];\n"""
        ss = """array[Tasks] of var int: overskill = [sum(j in PREEMPTIVE, sk in [nb_skill] where array_skills_required[sk, i]>0)((d_preemptive[i, j]>0)*(sum(w in Units)(unit_used_preemptive[w, i, j]*skillunits[w, sk])-array_skills_required[sk, i]))|i in Tasks];\n"""
        child_instance.add_string(s)
        child_instance.add_string(ss)
        list_strings = constraints_strings_multiskill_preemptive(
            current_solution=current_solution,
            cp_solver=cp_solver,
            tasks_primary=set(),
            tasks_secondary=self.problem.tasks_list,
            params_constraints=ParamsConstraintBuilder(
                minus_delta_primary=0,
                plus_delta_primary=0,
                minus_delta_secondary=0,
                plus_delta_secondary=0,
                constraint_max_time_to_current_solution=True,
                fraction_of_task_assigned_multiskill=0.65,
            ),
        )
        list_strings += cp_solver.constraint_objective_max_time_set_of_jobs(
            [self.problem.sink_task]
        )
        list_strings += [
            "constraint objective="
            + str(current_solution.get_end_time(self.problem.sink_task))
            + ";\n"
        ]
        for s in list_strings:
            child_instance.add_string(s)
        if random.random() < 0.999:
            if evaluation.get("constraint_penalty", 0) == 0:
                child_instance.add_string(
                    "constraint sec_objective==100*sum(overskill)+10*sum(res_load)+max(res_load)-min(res_load);\n"
                )
            else:
                child_instance.add_string(
                    "constraint sec_objective>=100*sum(overskill)+10*sum(res_load)+max(res_load);\n"
                )
        else:
            child_instance.add_string(
                "constraint sec_objective>=min(res_load)-max(res_load);\n"
            )
        return []
