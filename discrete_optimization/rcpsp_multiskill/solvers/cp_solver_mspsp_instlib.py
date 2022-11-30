#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

# Binding to use the mzn model provided by
# https://github.com/youngkd/MSPSP-InstLib

import logging
import os
from datetime import timedelta
from typing import Hashable, Optional, Set

from minizinc import Instance, Model, Solver

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
    build_unrelated_task,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    ParametersCP,
    SignEnum,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModelCalendar
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    SkillDetail,
)

logger = logging.getLogger(__name__)
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {
    "mspsp": os.path.join(this_path, "../minizinc/mspsp.mzn"),
    "mspsp_compatible": os.path.join(
        this_path, "../minizinc/mspsp_compatible_all_solvers.mzn"
    ),
}


class MS_RCPSPSolCP:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        logger.debug(f"One solution {self.objective}")
        if "nb_preemption_subtasks" in self.dict:
            logger.debug(
                ("nb_preemption_subtasks", self.dict["nb_preemption_subtasks"])
            )
        if "nb_small_tasks" in self.dict:
            logger.debug(("nb_small_tasks", self.dict["nb_small_tasks"]))
        keys = [k for k in self.dict if "penalty" in k]
        logger.debug("".join([str(k) + " : " + str(self.dict[k]) + "\n" for k in keys]))
        logger.debug(_output_item)

    def check(self) -> bool:
        return True


def create_usefull_res_data(rcpsp_model: MS_RCPSPModel):
    # USEFUL_RES  : array[ACT] of set of RESOURCE
    # POTENTIAL_ACT : array[RESOURCE] of set of ACT
    employees_position = {
        rcpsp_model.employees_list[i]: i + 1 for i in range(rcpsp_model.nb_employees)
    }
    non_zero_skills = {
        rcpsp_model.employees_list[i]: set(
            rcpsp_model.employees[rcpsp_model.employees_list[i]].get_non_zero_skills()
        )
        for i in range(rcpsp_model.nb_employees)
    }
    useful_res = [set() for i in range(rcpsp_model.n_jobs)]
    potential_act = [set() for j in range(rcpsp_model.nb_employees)]
    for task_number in range(rcpsp_model.n_jobs):
        task = rcpsp_model.tasks_list[task_number]
        skills_required = [
            sk
            for m in rcpsp_model.mode_details[task]
            for sk in rcpsp_model.skills_set
            if rcpsp_model.mode_details[task][m].get(sk, 0) > 0
        ]
        possibly_interested_employee = [
            emp
            for emp in non_zero_skills
            if any(sk in non_zero_skills[emp] for sk in skills_required)
        ]
        useful_res[task_number].update(
            set([employees_position[emp] for emp in possibly_interested_employee])
        )
        for emp in possibly_interested_employee:
            potential_act[employees_position[emp] - 1].add(task_number + 1)
    return useful_res, potential_act


class CP_MSPSP_MZN(MinizincCPSolver):
    def __init__(
        self,
        rcpsp_model: MS_RCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        silent_solve_error: bool = False,
        **kwargs,
    ):
        self.silent_solve_error = silent_solve_error
        self.rcpsp_model = rcpsp_model
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = [
            "start",
            "mrun",
        ]  # For now, I've put the var names of the CP model (not the rcpsp_model)
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.rcpsp_model, params_objective_function=params_objective_function
        )
        self.calendar = True
        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            self.calendar = True
        self.one_ressource_per_task = kwargs.get("one_ressource_per_task", False)
        self.resources_index = None

    def init_model(self, **args):
        model_type = args.get("model_type", "mspsp")
        model = Model(files_mzn[model_type])

        custom_output_type = args.get("output_type", False)
        if custom_output_type:
            model.output_type = MS_RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        dzn_file = args.get("dzn_file", None)
        instance = Instance(solver, model)
        if dzn_file is not None:
            model.add_file(args.get("dzn_file", None))
        else:
            dict_data = self.init_from_model(**args)
            for k in dict_data:
                instance[k] = dict_data[k]

        add_objective_makespan = args.get("add_objective_makespan", True)
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = args.get("ignore_sec_objective", True)

        resources_list = self.rcpsp_model.resources_list
        self.resources_index = resources_list
        sorted_tasks = self.rcpsp_model.tasks_list
        all_modes = [
            (act, mode, self.rcpsp_model.mode_details[act][mode])
            for act in sorted_tasks
            for mode in sorted(self.rcpsp_model.mode_details[act])
        ]
        self.modeindex_map = {
            i + 1: {"task": all_modes[i][0], "original_mode_index": all_modes[i][1]}
            for i in range(len(all_modes))
        }
        modes = [set() for t in sorted_tasks]
        for j in self.modeindex_map:
            modes[self.rcpsp_model.index_task[self.modeindex_map[j]["task"]]].add(j)
        self.mode_dict_task_mode_to_index_minizinc = {}
        for ind in self.modeindex_map:
            task = self.modeindex_map[ind]["task"]
            mode = self.modeindex_map[ind]["original_mode_index"]
            self.mode_dict_task_mode_to_index_minizinc[(task, mode)] = ind
        self.employees_position = self.rcpsp_model.employees_list
        self.index_employees_in_minizinc = {
            self.rcpsp_model.employees_list[i]: i + 1
            for i in range(len(self.rcpsp_model.employees_list))
        }
        self.instance = instance
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }

    def init_from_model(self, **args):
        dict_data = {}
        list_skills = sorted(list(self.rcpsp_model.skills_set))
        dict_data[
            "mint"
        ] = 10  # here put a better number (from critical path method for example)
        dict_data["nActs"] = self.rcpsp_model.n_jobs
        dict_data["dur"] = [
            self.rcpsp_model.mode_details[t][1]["duration"]
            for t in self.rcpsp_model.tasks_list
        ]
        dict_data["nSkills"] = len(self.rcpsp_model.skills_set)
        dict_data["sreq"] = [
            [self.rcpsp_model.mode_details[t][1].get(sk, 0) for sk in list_skills]
            for t in self.rcpsp_model.tasks_list
        ]
        dict_data["nResources"] = self.rcpsp_model.nb_employees
        dict_data["mastery"] = [
            [
                self.rcpsp_model.employees[self.rcpsp_model.employees_list[i]]
                .dict_skill.get(s, SkillDetail(0, 0, 0))
                .skill_value
                > 0
                for s in list_skills
            ]
            for i in range(self.rcpsp_model.nb_employees)
        ]
        dict_data["nPrecs"] = sum(
            [len(self.rcpsp_model.successors[n]) for n in self.rcpsp_model.successors]
        )
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        pred = []
        succ = []
        for task in self.rcpsp_model.tasks_list:
            pred += [self.index_in_minizinc[task]] * len(
                self.rcpsp_model.successors[task]
            )
            succ += [
                self.index_in_minizinc[successor]
                for successor in self.rcpsp_model.successors[task]
            ]
        dict_data["pred"] = pred
        dict_data["succ"] = succ
        self.graph = build_graph_rcpsp_object(rcpsp_problem=self.rcpsp_model)
        _, unrelated = build_unrelated_task(self.graph)
        dict_data["nUnrels"] = len(unrelated)
        dict_data["unpred"] = [self.index_in_minizinc[x[0]] for x in unrelated]
        dict_data["unsucc"] = [self.index_in_minizinc[x[1]] for x in unrelated]
        useful_res, potential_act = create_usefull_res_data(self.rcpsp_model)
        dict_data["USEFUL_RES"] = useful_res
        dict_data["POTENTIAL_ACT"] = potential_act
        return dict_data

    def constraint_task_to_mode(self, task_id, mode):
        return []

    def constraint_start_time_string(
        self, task, start_time, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint start["
            + str(self.index_in_minizinc[task])
            + "]"
            + str(sign.value)
            + str(start_time)
            + ";\n"
        )

    def constraint_end_time_string(
        self, task, end_time, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint start["
            + str(self.index_in_minizinc[task])
            + "]+dur["
            + str(self.index_in_minizinc[task])
            + "]"
            + str(sign.value)
            + str(end_time)
            + ";\n"
        )

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = "constraint start[" + str(ind) + "]==objective;\n"
        return [s]

    def constraint_objective_makespan(self):
        return self.constraint_objective_equal_makespan(self.rcpsp_model.sink_task)

    def constraint_used_employee(self, task, employee, indicator: bool = False):
        id_task = self.index_in_minizinc[task]
        id_employee = self.index_employees_in_minizinc[employee]
        tag = "true" if indicator else "false"
        return [
            "constraint assign["
            + str(id_task)
            + ","
            + str(id_employee)
            + "]=="
            + str(tag)
            + ";\n"
        ]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint start["
                + str(self.index_in_minizinc[j])
                + "]+dur["
                + str(self.index_in_minizinc[j])
                + "]<=objective;\n"
            ]
        return s

    def constraint_sum_of_ending_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        weights = [10 if s == self.rcpsp_model.sink_task else 1 for s in set_subtasks]
        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of int: weights = """
            + str(weights)
            + """;\n
               array[1..nb_indexes] of ACT: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in 1..nb_indexes)(weights[j]*(start[index_tasks[j]]+dur[index_tasks[j]]));\n"""
        )
        return [s]

    def constraint_sum_of_starting_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        weights = [10 if s == self.rcpsp_model.sink_task else 1 for s in set_subtasks]
        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of int: weights = """
            + str(weights)
            + """;\n
               array[1..nb_indexes] of ACT: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in 1..nb_indexes)(weights[j]*start[index_tasks[j]]);\n"""
        )
        return [s]

    def retrieve_solutions(self, result, parameters_cp: ParametersCP):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        best_objectives_cp = float("inf")
        list_solutions_fit = []
        starts = []
        mruns = []
        units_used = []
        objectives_cp = []
        index_best = 0
        # array[ACT,RESOURCE] of var bool: assign; % assignment of resources to activities
        # array[ACT,RESOURCE,SKILL] of var bool: contrib; % skill contribution assignment
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], MS_RCPSPSolCP):
                    starts += [result[i].dict["start"]]
                    mruns += [[1] * len(self.rcpsp_model.tasks_list)]
                    units_used += [result[i].dict["assign"]]
                    objectives_cp += [result[i].objective]
                else:
                    starts += [result[i, "start"]]
                    mruns += [[1] * len(self.rcpsp_model.tasks_list)]
                    units_used += [result[i, "assign"]]
                    objectives_cp += [result[i, "objective"]]
        else:
            if isinstance(result, MS_RCPSPSolCP):
                starts += [result.dict["start"]]
                mruns += [[1] * len(self.rcpsp_model.tasks_list)]
                units_used += [result.dict["unit_used"]]
                objectives_cp += [result.objective]
            else:
                starts += [result["start"]]
                mruns += [[1] * len(self.rcpsp_model.tasks_list)]
                units_used += [result["assign"]]
                objectives_cp += [result["objective"]]
        index_solution = 0
        skills_list = sorted(list(self.rcpsp_model.skills_set))
        for start_times, mrun, unit_used in zip(starts, mruns, units_used):
            usage = {}
            modes_dict = {}
            for i in range(len(mrun)):
                if mrun[i]:
                    modes_dict[self.modeindex_map[i + 1]["task"]] = self.modeindex_map[
                        i + 1
                    ]["original_mode_index"]
            for w in range(len(unit_used[0])):
                for task in range(len(unit_used)):
                    task_id = self.rcpsp_model.tasks_list[task]
                    if unit_used[task][w] == 1:
                        if isinstance(result[index_solution], MS_RCPSPSolCP):
                            if "contrib" in result[index_solution].dict:
                                intersection = [
                                    skills_list[i]
                                    for i in range(
                                        len(
                                            result[index_solution].dict["contrib"][
                                                task
                                            ][w]
                                        )
                                    )
                                    if result[index_solution].dict["contrib"][task][w][
                                        i
                                    ]
                                    == 1
                                ]
                        else:
                            mode = modes_dict[task_id]
                            skills_needed = set(
                                [
                                    s
                                    for s in self.rcpsp_model.skills_set
                                    if s in self.rcpsp_model.mode_details[task_id][mode]
                                    and self.rcpsp_model.mode_details[task_id][mode][s]
                                    > 0
                                ]
                            )
                            skills_worker = set(
                                [
                                    s
                                    for s in self.rcpsp_model.employees[
                                        self.employees_position[w]
                                    ].dict_skill
                                    if self.rcpsp_model.employees[
                                        self.employees_position[w]
                                    ]
                                    .dict_skill[s]
                                    .skill_value
                                    > 0
                                ]
                            )
                            intersection = skills_needed.intersection(skills_worker)
                        if len(intersection) > 0:
                            if task_id not in usage:
                                usage[task_id] = {}
                            usage[task_id][self.employees_position[w]] = intersection
            rcpsp_schedule = {}
            for i in range(len(start_times)):
                task_id = self.rcpsp_model.tasks_list[i]
                rcpsp_schedule[task_id] = {
                    "start_time": start_times[i],
                    "end_time": start_times[i]
                    + self.rcpsp_model.mode_details[task_id][modes_dict[task_id]][
                        "duration"
                    ],
                }
            sol = MS_RCPSPSolution(
                problem=self.rcpsp_model,
                modes=modes_dict,
                schedule=rcpsp_schedule,
                employee_usage=usage,
            )

            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            if objectives_cp[index_solution] < best_objectives_cp:
                index_best = index_solution
                best_objectives_cp = objectives_cp[index_solution]
            list_solutions_fit += [(sol, objective)]
            index_solution += 1
        if len(list_solutions_fit) > 0:
            list_solutions_fit[index_best][
                0
            ].opti_from_cp = True  # Flag the solution obtained for the optimum of cp
        result_storage = ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=best_solution,
            mode_optim=self.params_objective_function.sense_function,
            limit_store=False,
        )
        return result_storage


def chuffed_specific_code():
    s1 = """include "chuffed.mzn";\n"""
    s = """% Priority Searches\n
          ann: priority_search;

          ann: priority_input_order =
            priority_search(start,
                            [seq_search([
                                int_search([start[a]], input_order, indomain_min, complete),
                                bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                            input_order, indomain_max, complete)])
                             | a in ACT ],
                             input_order, complete);\n
          ann: priority_smallest =
            priority_search(start,
                              [seq_search([
                                  int_search([start[a]], input_order, indomain_min, complete),
                                  bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                              input_order, indomain_max, complete)])
                               | a in ACT ],
                               smallest, complete);\n"""
    s2 = """ann: priority_smallest_load =
            priority_search(start,
                [
                    seq_search([
                        int_search([start[a]], input_order, indomain_min, complete),
                        priority_search(res_load,
                            [
                                seq_search([
                                    bool_search([assign[a,r]], input_order, indomain_max, complete),
                                    bool_search([contrib[a,r,s] | s in SKILL],  input_order, indomain_max, complete)
                                ])
                            | r in RESOURCE],
                            smallest, complete)
                    ])
                | a in ACT ],
                smallest, complete);\n

        ann: priority_smallest_largest =
          priority_search(start,
                          [seq_search([
                              int_search([start[a]], input_order, indomain_min, complete),
                              bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                          input_order, indomain_max, complete)])
                           | a in ACT ],
                           smallest_largest, complete);\n

        ann: priority_first_fail =
          priority_search(start,
                          [seq_search([
                              int_search([start[a]], input_order, indomain_min, complete),
                              bool_search([contrib[a,r,s] | r in RESOURCE, s in SKILL],
                                          input_order, indomain_max, complete)])
                           | a in ACT ],
                           first_fail, complete);\n"""
