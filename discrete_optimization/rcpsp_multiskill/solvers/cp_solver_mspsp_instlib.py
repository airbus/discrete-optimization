#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

# Binding to use the mzn model provided by
# https://github.com/youngkd/MSPSP-InstLib

import logging
import os
from typing import Any, Dict, Hashable, Optional, Set

from minizinc import Instance, Model, Solver

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
    build_unrelated_task,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    SignEnum,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
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


def _log_minzinc_result(_output_item: Optional[str] = None, **kwargs: Any) -> None:
    logger.debug(f"One solution {kwargs['objective']}")
    if "nb_preemption_subtasks" in kwargs:
        logger.debug(("nb_preemption_subtasks", kwargs["nb_preemption_subtasks"]))
    if "nb_small_tasks" in kwargs:
        logger.debug(("nb_small_tasks", kwargs["nb_small_tasks"]))
    keys = [k for k in kwargs if "penalty" in k]
    logger.debug("".join([str(k) + " : " + str(kwargs[k]) + "\n" for k in keys]))
    logger.debug(_output_item)


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
    problem: MS_RCPSPModel

    def __init__(
        self,
        problem: MS_RCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        silent_solve_error: bool = False,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = [
            "start",
            "mrun",
        ]  # For now, I've put the var names of the CP model (not the rcpsp_model)
        self.one_ressource_per_task = kwargs.get("one_ressource_per_task", False)
        self.resources_index = None

    def init_model(self, **args):
        model_type = args.get("model_type", "mspsp")
        model = Model(files_mzn[model_type])

        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
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

        resources_list = self.problem.resources_list
        self.resources_index = resources_list
        sorted_tasks = self.problem.tasks_list
        all_modes = [
            (act, mode, self.problem.mode_details[act][mode])
            for act in sorted_tasks
            for mode in sorted(self.problem.mode_details[act])
        ]
        self.modeindex_map = {
            i + 1: {"task": all_modes[i][0], "original_mode_index": all_modes[i][1]}
            for i in range(len(all_modes))
        }
        modes = [set() for t in sorted_tasks]
        for j in self.modeindex_map:
            modes[self.problem.index_task[self.modeindex_map[j]["task"]]].add(j)
        self.mode_dict_task_mode_to_index_minizinc = {}
        for ind in self.modeindex_map:
            task = self.modeindex_map[ind]["task"]
            mode = self.modeindex_map[ind]["original_mode_index"]
            self.mode_dict_task_mode_to_index_minizinc[(task, mode)] = ind
        self.employees_position = self.problem.employees_list
        self.index_employees_in_minizinc = {
            self.problem.employees_list[i]: i + 1
            for i in range(len(self.problem.employees_list))
        }
        self.instance = instance
        self.index_in_minizinc = {
            task: self.problem.return_index_task(task, offset=1)
            for task in self.problem.tasks_list
        }

    def init_from_model(self, **args):
        dict_data = {}
        list_skills = sorted(list(self.problem.skills_set))
        dict_data[
            "mint"
        ] = 10  # here put a better number (from critical path method for example)
        dict_data["nActs"] = self.problem.n_jobs
        dict_data["dur"] = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        dict_data["nSkills"] = len(self.problem.skills_set)
        dict_data["sreq"] = [
            [self.problem.mode_details[t][1].get(sk, 0) for sk in list_skills]
            for t in self.problem.tasks_list
        ]
        dict_data["nResources"] = self.problem.nb_employees
        dict_data["mastery"] = [
            [
                self.problem.employees[self.problem.employees_list[i]]
                .dict_skill.get(s, SkillDetail(0, 0, 0))
                .skill_value
                > 0
                for s in list_skills
            ]
            for i in range(self.problem.nb_employees)
        ]
        dict_data["nPrecs"] = sum(
            [len(self.problem.successors[n]) for n in self.problem.successors]
        )
        self.index_in_minizinc = {
            task: self.problem.return_index_task(task, offset=1)
            for task in self.problem.tasks_list
        }
        pred = []
        succ = []
        for task in self.problem.tasks_list:
            pred += [self.index_in_minizinc[task]] * len(self.problem.successors[task])
            succ += [
                self.index_in_minizinc[successor]
                for successor in self.problem.successors[task]
            ]
        dict_data["pred"] = pred
        dict_data["succ"] = succ
        self.graph = build_graph_rcpsp_object(rcpsp_problem=self.problem)
        _, unrelated = build_unrelated_task(self.graph)
        dict_data["nUnrels"] = len(unrelated)
        dict_data["unpred"] = [self.index_in_minizinc[x[0]] for x in unrelated]
        dict_data["unsucc"] = [self.index_in_minizinc[x[1]] for x in unrelated]
        useful_res, potential_act = create_usefull_res_data(self.problem)
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
        return self.constraint_objective_equal_makespan(self.problem.sink_task)

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
        weights = [10 if s == self.problem.sink_task else 1 for s in set_subtasks]
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
        weights = [10 if s == self.problem.sink_task else 1 for s in set_subtasks]
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

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> MS_RCPSPSolution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        _log_minzinc_result(_output_item=_output_item, **kwargs)

        # array[ACT,RESOURCE] of var bool: assign; % assignment of resources to activities
        # array[ACT,RESOURCE,SKILL] of var bool: contrib; % skill contribution assignment
        start_times = kwargs["start"]
        mrun = [1] * len(self.problem.tasks_list)
        unit_used = kwargs["assign"]
        skills_list = sorted(list(self.problem.skills_set))
        usage = {}
        modes_dict = {}
        for i in range(len(mrun)):
            if mrun[i]:
                modes_dict[self.modeindex_map[i + 1]["task"]] = self.modeindex_map[
                    i + 1
                ]["original_mode_index"]
        for w in range(len(unit_used[0])):
            for task in range(len(unit_used)):
                task_id = self.problem.tasks_list[task]
                if unit_used[task][w] == 1:
                    if "contrib" in kwargs:
                        intersection = [
                            skills_list[i]
                            for i in range(len(kwargs["contrib"][task][w]))
                            if kwargs["contrib"][task][w][i] == 1
                        ]
                    else:
                        mode = modes_dict[task_id]
                        skills_needed = set(
                            [
                                s
                                for s in self.problem.skills_set
                                if s in self.problem.mode_details[task_id][mode]
                                and self.problem.mode_details[task_id][mode][s] > 0
                            ]
                        )
                        skills_worker = set(
                            [
                                s
                                for s in self.problem.employees[
                                    self.employees_position[w]
                                ].dict_skill
                                if self.problem.employees[self.employees_position[w]]
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
            task_id = self.problem.tasks_list[i]
            rcpsp_schedule[task_id] = {
                "start_time": start_times[i],
                "end_time": start_times[i]
                + self.problem.mode_details[task_id][modes_dict[task_id]]["duration"],
            }
        return MS_RCPSPSolution(
            problem=self.problem,
            modes=modes_dict,
            schedule=rcpsp_schedule,
            employee_usage=usage,
        )


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
