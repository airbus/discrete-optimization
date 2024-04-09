#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    MS_RCPSPModel,
    PrecomputeEmployeesForTasks,
)


class MultiSkillToRCPSP:
    def __init__(self, multiskill_model: MS_RCPSPModel):
        self.multiskill_model = multiskill_model
        self.worker_type_to_worker = None

    def is_compatible(
        self,
        task_requirements: Dict[str, int],
        ressource_availability: Dict[str, np.array],
        duration_task,
        horizon,
    ):
        non_zeros_res = [r for r in task_requirements if task_requirements[r] >= 0]
        p = np.multiply.reduce(
            [
                ressource_availability[non_zeros_res[j]][:horizon]
                >= task_requirements[non_zeros_res[j]]
                for j in range(len(non_zeros_res))
            ]
        )
        s = np.sum(p)
        return s >= duration_task

    def construct_rcpsp_by_worker_type(
        self,
        limit_number_of_mode_per_task: bool = True,
        max_number_of_mode: int = None,
        check_resource_compliance: bool = True,
        one_worker_type_per_task: bool = False,
    ):
        params_cp = ParametersCP(
            time_limit=30,
            intermediate_solution=True,
            all_solutions=False,
            nr_solutions=100,
        )
        solver = PrecomputeEmployeesForTasks(
            ms_rcpsp_model=self.multiskill_model, cp_solver_name=CPSolverName.CHUFFED
        )
        worker_type_name = sorted(solver.skills_dict)
        worker_type_container = solver.skills_representation_str
        self.skills_dict = solver.skills_dict
        self.skills_representation_str = solver.skills_representation_str
        calendar_worker_type = {}
        map_names_to_understandable = {
            worker_type_name[i]: "WorkerType-" + str(i)
            for i in range(len(worker_type_name))
        }
        self.map_names_to_understandable = map_names_to_understandable
        self.worker_type_to_worker = {
            self.map_names_to_understandable[k]: solver.skills_representation_str[k]
            for k in self.map_names_to_understandable
        }
        for worker_type in worker_type_name:
            employees = list(worker_type_container[worker_type])
            calend = np.array(
                self.multiskill_model.employees[employees[0]].calendar_employee,
                dtype=np.int_,
            )
            for j in range(1, len(employees)):
                calend += np.array(
                    self.multiskill_model.employees[employees[j]].calendar_employee,
                    dtype=np.int_,
                )
            calendar_worker_type[map_names_to_understandable[worker_type]] = calend
        resources_dict = self.multiskill_model.resources_availability
        usage_worker_in_chosen_modes = {k: 0 for k in calendar_worker_type}
        for k in calendar_worker_type:
            resources_dict[k] = calendar_worker_type[k]
        initial_mode_details = self.multiskill_model.mode_details
        mode_details_post_compute = {}
        dictionnary_precompute = {}
        for task in self.multiskill_model.tasks_list:
            solver.init_model(
                tasks_of_interest=[task],
                consider_units=False,
                consider_worker_type=True,
                one_ressource_per_task=self.multiskill_model.one_unit_per_task_max,
                one_worker_type_per_task=one_worker_type_per_task,
            )
            results = solver.solve(parameters_cp=params_cp)
            best_overskill_results = min(
                results, key=lambda x: x.overskill_type
            ).overskill_type
            pruned_results = [
                r for r in results if r.overskill_type == best_overskill_results
            ]
            dictionnary_precompute[task] = pruned_results
            mode_details_post_compute[task] = {}
            task_requirement_list = []
            for i in range(len(pruned_results)):
                ddd = initial_mode_details[task][pruned_results[i].mode_dict[task]]
                ddd = {
                    key: ddd[key]
                    for key in ddd
                    if key not in self.multiskill_model.skills_set
                }  # remove the skills
                wtype_used = [
                    (j, pruned_results[i].worker_type_used[j][0])
                    for j in range(len(pruned_results[i].worker_type_used))
                ]
                index_non_zeros = [k for k in wtype_used if k[1] > 0]

                task_requirement = ddd
                for k, c in index_non_zeros:
                    task_requirement[
                        map_names_to_understandable[worker_type_name[k]]
                    ] = c
                task_requirement_list += [task_requirement]
            if check_resource_compliance:
                tt = []
                for t in task_requirement_list:
                    b = self.is_compatible(
                        task_requirements={r: t[r] for r in t if r != "duration"},
                        duration_task=t["duration"],
                        horizon=self.multiskill_model.horizon,
                        ressource_availability=resources_dict,
                    )
                    if b:
                        tt += [t]
            else:
                tt = task_requirement_list
            tt = sorted(
                tt,
                key=lambda x: min(
                    [
                        usage_worker_in_chosen_modes[y]
                        for y in x
                        if y in usage_worker_in_chosen_modes
                    ],
                    default=0,
                ),
            )
            if limit_number_of_mode_per_task:
                number_of_modes = min(max_number_of_mode, len(tt))
                tt = tt[:number_of_modes]
            else:
                number_of_modes = len(tt)
            for i in range(number_of_modes):
                mode_details_post_compute[task][i + 1] = tt[i]
                for yy in tt[i]:
                    if yy in usage_worker_in_chosen_modes:
                        usage_worker_in_chosen_modes[yy] += 1
        rcpsp_model = RCPSPModel(
            resources=resources_dict,
            non_renewable_resources=list(self.multiskill_model.non_renewable_resources),
            mode_details=mode_details_post_compute,
            successors=self.multiskill_model.successors,
            horizon=self.multiskill_model.horizon,
            horizon_multiplier=self.multiskill_model.horizon_multiplier,
            tasks_list=self.multiskill_model.tasks_list,
            source_task=self.multiskill_model.source_task,
            sink_task=self.multiskill_model.sink_task,
        )
        return rcpsp_model
