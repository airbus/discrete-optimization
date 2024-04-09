#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
import os
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple, Union

from minizinc import Instance, Model, Solver, Status

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
    build_unrelated_task,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    ParametersCP,
    SignEnum,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPSolutionPreemptive
from discrete_optimization.rcpsp.rcpsp_solution import PartialSolution, RCPSPSolution
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive,
    cluster_employees_to_resource_types,
    create_fake_tasks_multiskills,
)

logger = logging.getLogger(__name__)
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {
    "multi-calendar": os.path.join(
        this_path, "../minizinc/ms_rcpsp_multi_mode_mzn_calendar.mzn"
    ),
    "multi-calendar-overlap": os.path.join(
        this_path, "../minizinc/ms_rcpsp_with_overlap_variable.mzn"
    ),
    "multi-calendar-no-ressource": os.path.join(
        this_path, "../minizinc/ms_rcpsp_multi_mode_mzn_calendar_no_ressource.mzn"
    ),
    "ms_no_multitasking": os.path.join(
        this_path,
        "../minizinc/ms_rcpsp_multi_mode_mzn_calendar_no_ressource_nomultitasking.mzn",
    ),
    "ms_rcpsp_preemptive": os.path.join(
        this_path, "../minizinc/ms_rcpsp_preemptive.mzn"
    ),
    "ms_rcpsp_partial_preemptive": os.path.join(
        this_path, "../minizinc/ms_rcpsp_preemptive_partially_preemptive.mzn"
    ),
    "compute_worker_for_tasks": os.path.join(
        this_path, "../minizinc/ms_rcpsp_compute_workers_for_tasks.mzn"
    ),
}


def add_fake_task_cp_data(
    rcpsp_model: Union[MS_RCPSPModel],
    ignore_fake_task: bool = True,
    max_time_to_consider: int = None,
):
    if not ignore_fake_task:
        fake_tasks_res, fake_tasks_unit = create_fake_tasks_multiskills(
            rcpsp_problem=rcpsp_model
        )
        if len(fake_tasks_res) > 0:
            max_time_to_consider = (
                rcpsp_model.horizon
                if max_time_to_consider is None
                else max_time_to_consider
            )
            fake_tasks_res = [
                f for f in fake_tasks_res if f["start"] <= max_time_to_consider
            ]
            n_fake_tasks_res = len(fake_tasks_res)
            fakestart_res = [
                fake_tasks_res[i]["start"] for i in range(n_fake_tasks_res)
            ]
            fake_dur_res = [
                fake_tasks_res[i]["duration"] for i in range(n_fake_tasks_res)
            ]
            max_duration_fake_task_res = max(fake_dur_res)
            fake_req_res = [
                [fake_tasks_res[i].get(res, 0) for i in range(n_fake_tasks_res)]
                for res in rcpsp_model.resources_list
            ]
            dict_to_add_in_instance = {
                "max_duration_fake_task_resource": max_duration_fake_task_res,
                "n_fake_task_resource": n_fake_tasks_res,
                "fakestart_resource": fakestart_res,
                "fakedur_resource": fake_dur_res,
                "fakereq_resource": fake_req_res,
                "include_fake_tasks_resource": True,
            }
        else:
            dict_to_add_in_instance = {
                "max_duration_fake_task_resource": 0,
                "n_fake_task_resource": 0,
                "fakestart_resource": [],
                "fakedur_resource": [],
                "fakereq_resource": [[] for req in rcpsp_model.resources_list],
                "include_fake_tasks_resource": False,
            }
        if len(fake_tasks_unit) > 0:
            fake_tasks_unit = [
                f for f in fake_tasks_unit if f["start"] <= max_time_to_consider
            ]
            n_fake_tasks_unit = len(fake_tasks_unit)
            fakestart_unit = [
                fake_tasks_unit[i]["start"] for i in range(n_fake_tasks_unit)
            ]
            fake_dur_unit = [
                fake_tasks_unit[i]["duration"] for i in range(n_fake_tasks_unit)
            ]
            max_duration_fake_task_unit = max(fake_dur_unit)
            fake_req_unit = [
                [fake_tasks_unit[i].get(res, 0) for i in range(n_fake_tasks_unit)]
                for res in rcpsp_model.employees_list
            ]
            dict_to_add_in_instance[
                "max_duration_fake_task_unit"
            ] = max_duration_fake_task_unit
            dict_to_add_in_instance["n_fake_task_unit"] = n_fake_tasks_unit
            dict_to_add_in_instance["fakestart_unit"] = fakestart_unit
            dict_to_add_in_instance["fakedur_unit"] = fake_dur_unit
            dict_to_add_in_instance["fakereq_unit"] = fake_req_unit
            dict_to_add_in_instance["include_fake_tasks_unit"] = True
        else:
            dict_to_add_in_instance["max_duration_fake_task_unit"] = 0
            dict_to_add_in_instance["n_fake_task_unit"] = 0
            dict_to_add_in_instance["fakestart_unit"] = []
            dict_to_add_in_instance["fakedur_unit"] = []
            dict_to_add_in_instance["fakereq_unit"] = [
                [] for e in rcpsp_model.employees_list
            ]
            dict_to_add_in_instance["include_fake_tasks_unit"] = False
        return dict_to_add_in_instance
    else:
        dict_to_add_in_instance = {
            "max_duration_fake_task_resource": 0,
            "n_fake_task_resource": 0,
            "fakestart_resource": [],
            "fakedur_resource": [],
            "fakereq_resource": [[] for r in rcpsp_model.resources_list],
            "include_fake_tasks_resource": False,
            "max_duration_fake_task_unit": 0,
            "n_fake_task_unit": 0,
            "fakestart_unit": [],
            "fakedur_unit": [],
            "fakereq_unit": [[] for e in rcpsp_model.employees_list],
            "include_fake_tasks_unit": False,
        }
        return dict_to_add_in_instance


def _log_minzinc_result(_output_item: Optional[str] = None, **kwargs: Any) -> None:
    logger.debug(f"One solution {kwargs['objective']}")
    if "nb_preemption_subtasks" in kwargs:
        logger.debug(("nb_preemption_subtasks", kwargs["nb_preemption_subtasks"]))
    if "nb_small_tasks" in kwargs:
        logger.debug(("nb_small_tasks", kwargs["nb_small_tasks"]))
    if "res_load" in kwargs:
        logger.debug(("res_load ", kwargs["res_load"]))
    keys = [k for k in kwargs if "penalty" in k]
    logger.debug("".join([str(k) + " : " + str(kwargs[k]) + "\n" for k in keys]))
    logger.debug(_output_item)


class SearchStrategyMS_MRCPSP(Enum):
    START_THEN_USED_UNIT = "durThenStartThenMode"
    PRIORITY_SEARCH_START_UNIT_USED = "priority_smallest"
    NONE = "none"


class CP_MS_MRCPSP_MZN(MinizincCPSolver):
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

    def manual_cumulative_resource_constraints(self, instance):
        blocking_per_resource = {
            r: {"blocking": [], "merged": set(), "not_blocked": set()}
            for r in self.problem.resources_list
        }
        array = [
            [0 for i in range(self.problem.n_jobs)] for r in self.problem.resources_list
        ]
        for l in self.problem.resource_blocking_data:
            set_res = l[-1]
            list_task = l[0]
            for res in set_res:
                blocking_per_resource[res]["blocking"] += list_task
                blocking_per_resource[res]["merged"].update(list_task)
                index_res = self.problem.resources_list.index(res)
                array[index_res][self.problem.index_task[list_task[0]]] = (
                    self.problem.index_task[list_task[-1]] + 1
                )
                for i in list_task[1:]:
                    array[index_res][self.problem.index_task[i]] = -1
        for r in blocking_per_resource:
            blocking_per_resource[r]["not_blocked"] = [
                t
                for t in self.problem.tasks_list
                if t not in blocking_per_resource[r]["merged"]
            ]
        s = """array[Res, Act] of int: resource_blocking_array;"""
        instance.add_string(s)
        instance["resource_blocking_array"] = array

        s = """constraint forall(k in RRes)(
                 if include_fake_tasks_resource then
                    let{set of Tasks: TasksR = {i| i in Act where mask_res_task[k, i]=false /\
                                                resource_blocking_array[k, i]==0},
                        set of Tasks: TaskBlocker = {j| j in Act where resource_blocking_array[k,j]>0},
                        set of FakeActRes: FTasks = {j | j in FakeActRes where fakereq_resource[k, j]>0}}
                          in(
                             cumulative([start[i]| i in TasksR]++[start[i]| i in TaskBlocker]++[fakestart_resource[p] | p in FTasks],
                                        [adur[i]| i in TasksR]++
                                        [max([0, start[resource_blocking_array[k,i]]+adur[resource_blocking_array[k,i]]-start[i]])|
                                         i in TaskBlocker]++[fakedur_resource[p] | p in FTasks],
                                        [arreq[k,i] | i in TasksR]++[arreq[k,i] | i in TaskBlocker]++[fakereq_resource[k, p] | p in FTasks],
                                        rcap[k])
                          )
                 else
                    cumulative(start, adur, [arreq[k,i] | i in Act], rcap[k])
                 endif);
            """
        instance.add_string(s)

    def manual_starting_time(self, instance):
        slist = []
        for i in range(self.problem.nb_tasks):
            t = self.problem.tasks_list[i]
            if t in self.problem.special_constraints.start_times_window:
                st = []
                if (
                    self.problem.special_constraints.start_times_window[t][0]
                    is not None
                ):
                    st += [self.problem.special_constraints.start_times_window[t][0]]
                if (
                    self.problem.special_constraints.start_times_window[t][1]
                    is not None
                ):
                    st += [self.problem.special_constraints.start_times_window[t][1]]
                s = (
                    """constraint (member([start[j]+adur[j]|j in Act]++[fakestart_resource[j]+fakedur_resource[j]|
                                                                        j in FakeActRes]++
                                          [fakestart_unit[j]+fakedur_unit[j]|j in FakeActUnit]++"""
                    + str(st)
                    + """,
                                           start["""
                    + str(i + 1)
                    + """]));\n"""
                )
            else:
                s = (
                    """constraint (member([start[j]+adur[j]|j in Act]++[fakestart_resource[j]+fakedur_resource[j]|
                                                                        j in FakeActRes]++
                                          [fakestart_unit[j]+fakedur_unit[j]|j in FakeActUnit],
                                           start["""
                    + str(i + 1)
                    + """]));\n"""
                )
            slist += [s]
        for s in slist:
            instance.add_string(s)

    def write_search_strategy_chuffed(self, instance):
        s = """ann: priority_smallest = priority_search(start,
                                                 [seq_search([int_search([start[a]], input_order, indomain_min),
                                                              priority_search([sum(sk in Skill)(
                                                                  skillunits[w, sk] * array_skills_required[
                                                                      sk, a]) | w in Units],
                                                                              [bool_search([unit_used[w, a]],
                                                                                           input_order,
                                                                                           indomain_min) | w in Units],
                                                                              smallest, complete)
                                                              ])
                                                  | a in Act],
                                                 smallest, complete);\n"""
        instance.add_string(s)
        if self.cp_solver_name == CPSolverName.CHUFFED:
            instance.add_string("""include "chuffed.mzn";\n""")

    def constraint_task_to_mode(self, task_id, mode):
        modes = self.problem.mode_details[task_id].keys()
        list_strings = []
        for m in modes:
            if m == mode:
                bool_str = "true"
            else:
                bool_str = "false"
            s = (
                "constraint mrun["
                + str(self.mode_dict_task_mode_to_index_minizinc[(task_id, m)])
                + "]="
                + bool_str
                + ";\n"
            )
            list_strings += [s]
        return list_strings

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
            + "]+adur["
            + str(self.index_in_minizinc[task])
            + "]"
            + str(sign.value)
            + str(end_time)
            + ";\n"
        )

    def constraint_used_employee(self, task, employee, indicator: bool = False):
        id_task = self.index_in_minizinc[task]
        id_employee = self.index_employees_in_minizinc[employee]
        tag = "true" if indicator else "false"
        return [
            "constraint unit_used["
            + str(id_employee)
            + ","
            + str(id_task)
            + "]=="
            + str(tag)
            + ";\n"
        ]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint s["
                + str(self.index_in_minizinc[j])
                + "]+adur["
                + str(self.index_in_minizinc[j])
                + "]<=objective;\n"
            ]
        return s

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = "constraint s[" + str(ind) + "]==objective;\n"
        return [s]

    def constraint_sum_of_ending_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of Tasks: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in index_tasks)(start[j]+adur[j]);\n"""
        )
        return [s]

    def constraint_sum_of_starting_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of Tasks: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in index_tasks)(start[j]);\n"""
        )
        return [s]

    def add_hard_special_constraints(self, partial_solution):
        return add_hard_special_constraints(partial_solution, self)

    def init_model(self, **args):
        no_ressource = False
        model_type = args.get(
            "model_type",
            "multi-calendar" if not no_ressource else "multi-calendar-no-ressource",
        )
        model = Model(files_mzn[model_type])
        exact_skills_need = args.get("exact_skills_need", False)
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        include_cumulative_resource = args.get("include_cumulative_resource", True)
        include_constraint_on_start_value = args.get(
            "include_constraint_on_start_value", False
        )
        include_constraint_on_start_value_manual = args.get(
            "include_constraint_on_start_value_manual", False
        )
        search_strategy = args.get(
            "search_strategy", SearchStrategyMS_MRCPSP.START_THEN_USED_UNIT
        )
        if include_constraint_on_start_value_manual:
            include_constraint_on_start_value = False
        fake_tasks = args.get(
            "fake_tasks", True
        )  # to modelize varying quantity of resource.
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        resources_list = self.problem.resources_list
        self.resources_index = resources_list
        instance = Instance(solver, model)
        if self.cp_solver_name.value == CPSolverName.CHUFFED.value:
            self.write_search_strategy_chuffed(instance)
        if include_constraint_on_start_value_manual:
            self.manual_starting_time(instance)
        if search_strategy.value != "none":
            instance.add_string("my_search=" + str(search_strategy.value) + ";\n")
        else:
            instance.add_string(
                "my_search="
                + str(SearchStrategyMS_MRCPSP.START_THEN_USED_UNIT.value)
                + ";\n"
            )  # we have to put one by default.
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        instance["include_cumulative_resource"] = include_cumulative_resource
        instance[
            "include_constraint_on_start_value"
        ] = include_constraint_on_start_value
        n_res = len(resources_list)
        keys = []
        if not no_ressource:
            instance["n_res"] = n_res
            keys += ["n_res"]
        if model_type in {"multi-calendar", "multi-calendar-overlap"}:
            instance["exact_skills_need"] = exact_skills_need
            keys += ["exact_skills_need"]
        if model_type == "multi-calendar-overlap":
            self.graph = build_graph_rcpsp_object(rcpsp_problem=self.problem)
            _, unrelated = build_unrelated_task(self.graph)
            instance["nUnrel"] = len(unrelated)
            instance["unpred"] = [self.problem.index_task[x[0]] + 1 for x in unrelated]
            instance["unsucc"] = [self.problem.index_task[x[1]] + 1 for x in unrelated]
        if (
            not include_cumulative_resource
            and len(self.problem.resource_blocking_data) > 0
        ):
            self.manual_cumulative_resource_constraints(instance)

        instance["one_ressource_per_task"] = self.one_ressource_per_task
        keys += ["one_ressource_per_task"]
        n_tasks = self.problem.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = self.problem.tasks_list
        n_opt = sum(
            [len(list(self.problem.mode_details[key].keys())) for key in sorted_tasks]
        )
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
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

        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["dur"] = dur
        keys += ["dur"]

        skills_set = sorted(list(self.problem.skills_set))
        nb_units = len(self.problem.employees)
        skill_required = [
            [int(all_modes[i][2].get(s, 0)) for i in range(len(all_modes))]
            for s in skills_set
        ]
        if "max_time" in args:
            instance["max_time"] = args["max_time"]
        else:
            instance["max_time"] = self.problem.horizon
        keys += ["max_time"]
        dict_to_add = add_fake_task_cp_data(
            rcpsp_model=self.problem,
            ignore_fake_task=not fake_tasks,
            max_time_to_consider=instance["max_time"],
        )
        for key in dict_to_add:
            instance[key] = dict_to_add[key]
            keys += [key]
        instance["nb_skill"] = len(self.problem.skills_set)
        instance["skillreq"] = skill_required
        instance["nb_units"] = nb_units
        keys += ["nb_skill", "skillreq", "nb_units"]
        instance["source"] = self.problem.index_task[self.problem.source_task] + 1
        skillunits = [
            [
                int(math.floor(self.problem.employees[j].dict_skill[s].skill_value))
                if s in self.problem.employees[j].dict_skill
                else 0
                for s in skills_set
            ]
            for j in self.problem.employees_list
        ]
        self.employees_position = self.problem.employees_list
        self.index_employees_in_minizinc = {
            self.problem.employees_list[i]: i + 1
            for i in range(len(self.problem.employees_list))
        }
        instance["skillunits"] = skillunits
        keys += ["skillunits"]
        logger.debug(f"Employee position CP {self.employees_position}")
        if not no_ressource:
            rreq = [
                [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
                for res in resources_list
            ]
            instance["rreq"] = rreq
            keys += ["rreq"]
            instance["rreq"] = rreq
            keys += ["rreq"]

            rcap = [
                int(max(self.problem.resources_availability[x])) for x in resources_list
            ]
            instance["rcap"] = rcap
            keys += ["rcap"]
            rtype = [
                2 if res in self.problem.non_renewable_resources else 1
                for res in resources_list
            ]
            instance["rtype"] = rtype
            keys += ["rtype"]

        succ = [
            set(
                [
                    self.problem.return_index_task(x, offset=1)
                    for x in self.problem.successors.get(task, [])
                ]
            )
            for task in sorted_tasks
        ]

        instance["succ"] = succ
        keys += ["succ"]

        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.problem.return_index_task(task, offset=1)
            for task in self.problem.tasks_list
        }

        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
        else:
            strings, name_penalty = add_soft_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
            strings = define_second_part_objective(
                [100] * len(name_penalty), name_penalty
            )
            if len(name_penalty) > 0:
                for s in strings:
                    self.instance.add_string(s)

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

        start_times = kwargs["start"]
        mrun = kwargs["mrun"]
        unit_used = kwargs["unit_used"]
        skills_list = sorted(list(self.problem.skills_set))
        usage = {}
        modes_dict = {}
        for i in range(len(mrun)):
            if mrun[i]:
                modes_dict[self.modeindex_map[i + 1]["task"]] = self.modeindex_map[
                    i + 1
                ]["original_mode_index"]
        for w in range(len(unit_used)):
            for task in range(len(unit_used[w])):
                task_id = self.problem.tasks_list[task]
                if unit_used[w][task] == 1:
                    if "contrib" in kwargs:  # model="ms_no_multitasking"
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


class CP_MS_MRCPSP_MZN_PREEMPTIVE(MinizincCPSolver):
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
        self.unit_usage_preemptive = None

    def constraint_task_to_mode(self, task_id, mode):
        modes = self.problem.mode_details[task_id].keys()
        list_strings = []
        for m in modes:
            if m == mode:
                bool_str = "true"
            else:
                bool_str = "false"
            s = (
                "constraint mrun["
                + str(self.mode_dict_task_mode_to_index_minizinc[(task_id, m)])
                + "]="
                + bool_str
                + ";\n"
            )
            list_strings += [s]
        return list_strings

    def add_hard_special_constraints(self, partial_solution):
        return add_hard_special_constraints(partial_solution, self)

    def constraint_duration_to_min_duration_preemptive(self, task, min_duration):
        list_strings = []
        for i in range(2, self.nb_preemptive + 1):
            s = (
                "constraint d_preemptive["
                + str(self.index_in_minizinc[task])
                + ","
                + str(i)
                + "]>"
                + str(min_duration)
                + "\/"
                + "d_preemptive["
                + str(self.index_in_minizinc[task])
                + ","
                + str(i)
                + "]=0;\n"
            )
            list_strings += [s]
        return list_strings

    def constraint_is_paused(self, task, is_paused):
        is_paused_str = "true" if is_paused else "false"
        return (
            "constraint is_paused["
            + str(self.index_in_minizinc[task])
            + "]=="
            + is_paused_str
            + ";\n"
        )

    def constraint_start_time_string(
        self, task, start_time, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint s["
            + str(self.index_in_minizinc[task])
            + "]"
            + str(sign.value)
            + str(start_time)
            + ";\n"
        )

    def constraint_start_time_string_preemptive_i(
        self, task, start_time, part_id=1, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint s_preemptive["
            + str(self.index_in_minizinc[task])
            + ","
            + str(part_id)
            + "]"
            + str(sign.value)
            + str(start_time)
            + ";\n"
        )

    def constraint_duration_string_preemptive_i(
        self, task, duration, part_id=1, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint d_preemptive["
            + str(self.index_in_minizinc[task])
            + ","
            + str(part_id)
            + "]"
            + str(sign.value)
            + str(duration)
            + ";\n"
        )

    def constraint_used_employee(
        self, task, employee, part_id=1, indicator: bool = False
    ):
        id_task = self.index_in_minizinc[task]
        id_employee = self.index_employees_in_minizinc[employee]
        tag = "true" if indicator else "false"
        if self.unit_usage_preemptive:
            return [
                "constraint unit_used_preemptive["
                + str(id_employee)
                + ","
                + str(id_task)
                + ","
                + str(part_id)
                + "]=="
                + str(tag)
                + ";\n"
            ]
        else:
            return [
                "constraint unit_used["
                + str(id_employee)
                + ","
                + str(id_task)
                + "]=="
                + str(tag)
                + ";\n"
            ]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint s_preemptive["
                + str(self.index_in_minizinc[j])
                + ", nb_preemptive]<=objective;\n"
            ]
        return s

    def constraint_end_time_string(
        self, task, end_time, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint s_preemptive["
            + str(self.index_in_minizinc[task])
            + ", nb_preemptive]"
            + str(sign.value)
            + str(end_time)
            + ";\n"
        )

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = (
            "constraint (s_preemptive["
            + str(ind)
            + ", nb_preemptive]+d_preemptive["
            + str(ind)
            + ",nb_preemptive]==objective);\n"
        )
        return [s]

    def constraint_sum_of_ending_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        weights = [10 if s == self.problem.sink_task else 1 for s in set_subtasks]
        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of int: weights="""
            + str(weights)
            + """;\n
               array[1..nb_indexes] of Tasks: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in 1..nb_indexes)(weights[j]*(s_preemptive[index_tasks[j], nb_preemptive]+d_preemptive[index_tasks[j], nb_preemptive]));\n"""
        )
        return [s]

    def constraint_sum_of_starting_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        weights = [10 if s == self.problem.sink_task else 1 for s in set_subtasks]

        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of int: weights="""
            + str(weights)
            + """;\n
               array[1..nb_indexes] of Tasks: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in 1..nb_indexes)(weights[j]*s_preemptive[index_tasks[j], 1]);\n"""
        )
        return [s]

    def init_model(self, **args):
        model_type = "ms_rcpsp_preemptive"
        model = Model(files_mzn[model_type])
        exact_skills_need = args.get("exact_skills_need", False)
        strictly_disjunctive_subtasks = args.get("strictly_disjunctive_subtasks", True)
        fake_tasks = args.get(
            "fake_tasks", True
        )  # to modelize varying quantity of resource.
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        resources_list = self.problem.resources_list
        self.resources_index = resources_list
        instance = Instance(solver, model)
        n_res = len(resources_list)
        keys = []
        instance["nb_preemptive"] = args.get("nb_preemptive", 2)
        self.nb_preemptive = instance["nb_preemptive"]
        keys += ["nb_preemptive"]
        instance["possibly_preemptive"] = args.get(
            "possibly_preemptive", [True for task in self.problem.tasks_list]
        )
        keys += ["possibly_preemptive"]
        instance["max_preempted"] = args.get(
            "max_preempted", min(self.problem.n_jobs_non_dummy, 5)
        )
        keys += ["max_preempted"]
        instance["n_res"] = n_res
        keys += ["n_res"]
        instance["exact_skills_need"] = exact_skills_need
        keys += ["exact_skills_need"]
        instance["add_calendar_constraint_unit"] = args.get(
            "add_calendar_constraint_unit", True
        )
        keys += ["add_calendar_constraint_unit"]
        instance["unit_usage_preemptive"] = args.get("unit_usage_preemptive", False)
        instance["strictly_disjunctive"] = strictly_disjunctive_subtasks
        self.unit_usage_preemptive = instance["unit_usage_preemptive"]
        keys += ["unit_usage_preemptive"]
        instance["one_ressource_per_task"] = self.one_ressource_per_task
        keys += ["one_ressource_per_task"]
        n_tasks = self.problem.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = self.problem.tasks_list
        n_opt = sum(
            [len(list(self.problem.mode_details[key].keys())) for key in sorted_tasks]
        )
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
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
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["dur"] = dur
        keys += ["dur"]
        skills_set = sorted(list(self.problem.skills_set))
        nb_units = len(self.problem.employees)
        skill_required = [
            [int(all_modes[i][2].get(s, 0)) for i in range(len(all_modes))]
            for s in skills_set
        ]
        if "max_time" in args:
            instance["max_time"] = args["max_time"]
        else:
            instance["max_time"] = self.problem.horizon
        keys += ["max_time"]
        dict_to_add = add_fake_task_cp_data(
            rcpsp_model=self.problem,
            ignore_fake_task=not fake_tasks,
            max_time_to_consider=instance["max_time"],
        )
        for key in dict_to_add:
            instance[key] = dict_to_add[key]
            keys += [key]
        instance["nb_skill"] = len(self.problem.skills_set)
        instance["skillreq"] = skill_required
        instance["nb_units"] = nb_units
        keys += ["nb_skill", "skillreq", "nb_units"]

        skillunits = [
            [
                int(math.floor(self.problem.employees[j].dict_skill[s].skill_value))
                if s in self.problem.employees[j].dict_skill
                else 0
                for s in skills_set
            ]
            for j in self.problem.employees_list
        ]
        self.employees_position = self.problem.employees_list
        self.index_employees_in_minizinc = {
            self.problem.employees_list[i]: i + 1
            for i in range(len(self.problem.employees_list))
        }
        instance["skillunits"] = skillunits
        keys += ["skillunits"]
        logger.debug(f"Employee position CP {self.employees_position}")
        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in resources_list
        ]
        instance["rreq"] = rreq
        keys += ["rreq"]
        instance["rreq"] = rreq
        keys += ["rreq"]
        rcap = [
            int(max(self.problem.resources_availability[x])) for x in resources_list
        ]
        instance["rc"] = rcap
        keys += ["rc"]
        rtype = [
            2 if res in self.problem.non_renewable_resources else 1
            for res in resources_list
        ]
        instance["rtype"] = rtype
        keys += ["rtype"]
        succ = [
            set(
                [
                    self.problem.return_index_task(x, offset=1)
                    for x in self.problem.successors.get(task, [])
                ]
            )
            for task in sorted_tasks
        ]
        instance["suc"] = succ
        keys += ["suc"]

        instance["add_objective_makespan"] = args.get("add_objective_makespan", True)
        instance["ignore_sec_objective"] = args.get("ignore_sec_objective", True)
        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.problem.return_index_task(task, offset=1)
            for task in self.problem.tasks_list
        }
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        sec_objective_equal_penalty = args.get("sec_objective_equal_penalty", True)
        self.second_objectives = {"weights": [], "name_penalty": []}
        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
        else:
            strings, name_penalty = add_soft_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
            self.second_objectives = {
                "weights": [100] * len(name_penalty),
                "name_penalty": name_penalty,
            }
            if len(name_penalty) > 0:
                strings = define_second_part_objective(
                    [100] * len(name_penalty),
                    name_penalty,
                    equal=sec_objective_equal_penalty,
                )
                for s in strings:
                    self.instance.add_string(s)

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> MS_RCPSPSolution_Preemptive:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        _log_minzinc_result(_output_item=_output_item, **kwargs)

        starts_preemptive = kwargs["s_preemptive"]
        duration_preemptive = kwargs["d_preemptive"]
        mruns = kwargs["mrun"]
        units_used = kwargs["unit_used"]
        units_used_preemptive = kwargs["unit_used_preemptive"]
        rcpsp_schedule = {}
        modes_dict = {}
        for k in range(len(starts_preemptive)):
            starts_k = []
            ends_k = []
            for j in range(len(starts_preemptive[k])):
                if j == 0 or duration_preemptive[k][j] != 0:
                    starts_k += [starts_preemptive[k][j]]
                    ends_k += [starts_k[-1] + duration_preemptive[k][j]]
            rcpsp_schedule[self.problem.tasks_list[k]] = {
                "starts": starts_k,
                "ends": ends_k,
            }
        for m in range(len(mruns)):
            if mruns[m]:
                modes_dict[self.modeindex_map[m + 1]["task"]] = self.modeindex_map[
                    m + 1
                ]["original_mode_index"]
        if not self.unit_usage_preemptive:
            unit_used = units_used
            usage = {}
            for w in range(len(unit_used)):
                for task in range(len(unit_used[w])):
                    if unit_used[w][task] == 1:
                        task_id = self.problem.tasks_list[task]
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
            for task_id in usage:
                usage[task_id] = [
                    usage[task_id]
                    for i in range(len(rcpsp_schedule[task_id]["starts"]))
                ]
        else:
            unit_used = units_used_preemptive
            usage = {}
            for w in range(len(unit_used)):
                for task in range(len(unit_used[w])):
                    task_id = self.problem.tasks_list[task]
                    mode = modes_dict[task_id]
                    skills_needed = set(
                        [
                            s
                            for s in self.problem.skills_set
                            if s in self.problem.mode_details[task_id][mode]
                            and self.problem.mode_details[task_id][mode][s] > 0
                        ]
                    )
                    if task_id not in usage:
                        usage[task_id] = [
                            {} for j in range(len(rcpsp_schedule[task_id]["starts"]))
                        ]
                    for j in range(len(rcpsp_schedule[task_id]["starts"])):
                        if unit_used[w][task][j] == 1:
                            skills_worker = set(
                                [
                                    s
                                    for s in self.problem.employees[
                                        self.employees_position[w]
                                    ].dict_skill
                                    if self.problem.employees[
                                        self.employees_position[w]
                                    ]
                                    .dict_skill[s]
                                    .skill_value
                                    > 0
                                ]
                            )
                            intersection = skills_needed.intersection(skills_worker)
                            if len(intersection) > 0:
                                usage[task_id][j][
                                    self.employees_position[w]
                                ] = intersection

        return MS_RCPSPSolution_Preemptive(
            problem=self.problem,
            modes=modes_dict,
            schedule=rcpsp_schedule,
            employee_usage=usage,
        )


class CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE(CP_MS_MRCPSP_MZN_PREEMPTIVE):
    def init_model(self, **args):
        model_type = args.get("model_type", "ms_rcpsp_partial_preemptive")
        model = Model(files_mzn[model_type])
        exact_skills_need = args.get("exact_skills_need", False)
        fake_tasks = args.get(
            "fake_tasks", True
        )  # to modelize varying quantity of resource.
        include_constraint_on_start_value = args.get(
            "include_constraint_on_start_value", False
        )
        strictly_disjunctive_subtasks = args.get("strictly_disjunctive_subtasks", True)
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        resources_list = self.problem.resources_list
        self.resources_index = resources_list
        instance = Instance(solver, model)
        n_res = len(resources_list)
        keys = []
        instance["nb_preemptive"] = args.get("nb_preemptive", 2)
        instance[
            "include_constraint_on_start_value"
        ] = include_constraint_on_start_value
        self.nb_preemptive = instance["nb_preemptive"]
        keys += ["nb_preemptive"]
        instance["possibly_preemptive"] = args.get(
            "possibly_preemptive",
            [
                self.problem.preemptive_indicator.get(t, True)
                for t in self.problem.tasks_list
            ],
        )
        instance["strictly_disjunctive"] = strictly_disjunctive_subtasks
        keys += ["possibly_preemptive"]
        instance["max_preempted"] = args.get(
            "max_preempted", min(self.problem.n_jobs_non_dummy, 5)
        )
        keys += ["max_preempted"]
        instance["n_res"] = n_res
        keys += ["n_res"]
        instance["exact_skills_need"] = exact_skills_need
        keys += ["exact_skills_need"]
        instance["add_calendar_constraint_unit"] = args.get(
            "add_calendar_constraint_unit", True
        )
        keys += ["add_calendar_constraint_unit"]
        instance["unit_usage_preemptive"] = args.get("unit_usage_preemptive", False)
        self.unit_usage_preemptive = instance["unit_usage_preemptive"]
        keys += ["unit_usage_preemptive"]
        instance["one_ressource_per_task"] = self.one_ressource_per_task
        keys += ["one_ressource_per_task"]
        n_tasks = self.problem.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = self.problem.tasks_list
        n_opt = sum(
            [len(list(self.problem.mode_details[key].keys())) for key in sorted_tasks]
        )
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
        all_modes = [
            (
                act,
                mode,
                self.problem.mode_details[act][mode],
                self.problem.partial_preemption_data[act][mode],
            )
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
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["dur"] = dur
        keys += ["dur"]
        skills_set = sorted(list(self.problem.skills_set))
        nb_units = len(self.problem.employees)
        skill_required = [
            [int(all_modes[i][2].get(s, 0)) for i in range(len(all_modes))]
            for s in skills_set
        ]
        if "max_time" in args:
            instance["max_time"] = args["max_time"]
        else:
            instance["max_time"] = self.problem.horizon
        keys += ["max_time"]
        dict_to_add = add_fake_task_cp_data(
            rcpsp_model=self.problem,
            ignore_fake_task=not fake_tasks,
            max_time_to_consider=instance["max_time"],
        )
        for key in dict_to_add:
            instance[key] = dict_to_add[key]
            keys += [key]
        instance["nb_skill"] = len(self.problem.skills_set)
        instance["skillreq"] = skill_required
        instance["nb_units"] = nb_units
        keys += ["nb_skill", "skillreq", "nb_units"]
        skillunits = [
            [
                int(math.floor(self.problem.employees[j].dict_skill[s].skill_value))
                if s in self.problem.employees[j].dict_skill
                else 0
                for s in skills_set
            ]
            for j in self.problem.employees_list
        ]
        self.employees_position = self.problem.employees_list
        self.index_employees_in_minizinc = {
            self.problem.employees_list[i]: i + 1
            for i in range(len(self.problem.employees_list))
        }
        instance["skillunits"] = skillunits
        keys += ["skillunits"]
        logger.debug(f"Employee position CP {self.employees_position}")
        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in resources_list
        ]
        instance["rreq"] = rreq
        keys += ["rreq"]

        # New input data for the partially preemptive use case.
        is_releasable = [
            [1 if all_modes[i][3].get(res, True) else 0 for i in range(len(all_modes))]
            for res in resources_list
        ]
        instance["is_releasable"] = is_releasable
        keys += ["is_releasable"]

        rcap = [
            int(max(self.problem.resources_availability[x])) for x in resources_list
        ]
        instance["rc"] = rcap
        keys += ["rc"]
        rtype = [
            2 if res in self.problem.non_renewable_resources else 1
            for res in resources_list
        ]
        instance["rtype"] = rtype
        keys += ["rtype"]
        succ = [
            set(
                [
                    self.problem.return_index_task(x, offset=1)
                    for x in self.problem.successors.get(task, [])
                ]
            )
            for task in sorted_tasks
        ]
        instance["suc"] = succ
        keys += ["suc"]

        instance["add_objective_makespan"] = args.get("add_objective_makespan", True)
        instance["ignore_sec_objective"] = args.get("ignore_sec_objective", True)
        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.problem.return_index_task(task, offset=1)
            for task in self.problem.tasks_list
        }
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        self.second_objectives = {"weights": [], "name_penalty": []}
        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
        else:
            strings, name_penalty = add_soft_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
            self.second_objectives = {
                "weights": [100] * len(name_penalty),
                "name_penalty": name_penalty,
            }
            if len(name_penalty) > 0:
                strings = define_second_part_objective(
                    [100] * len(name_penalty), name_penalty, equal=False
                )
                for s in strings:
                    self.instance.add_string(s)


def stick_to_solution(solution: RCPSPSolution, cp_solver: CP_MS_MRCPSP_MZN):
    list_strings = []
    for task in solution.rcpsp_schedule:
        start = solution.rcpsp_schedule[task]["start_time"]
        list_strings += [
            cp_solver.constraint_start_time_string(
                task=task, start_time=start, sign=SignEnum.EQUAL
            )
        ]
    return list_strings


def stick_to_solution_preemptive(
    solution: RCPSPSolutionPreemptive, cp_solver: CP_MS_MRCPSP_MZN_PREEMPTIVE
):
    list_strings = []
    for task in solution.rcpsp_schedule:
        starts = solution.rcpsp_schedule[task]["starts"]
        ends = solution.rcpsp_schedule[task]["ends"]
        is_paused = len(starts) > 1
        list_strings += [cp_solver.constraint_is_paused(task=task, is_paused=is_paused)]
        for i in range(len(starts)):
            list_strings += [
                cp_solver.constraint_start_time_string_preemptive_i(
                    task, start_time=starts[i], part_id=i + 1, sign=SignEnum.EQUAL
                )
            ]
            list_strings += [
                cp_solver.constraint_duration_string_preemptive_i(
                    task,
                    duration=ends[i] - starts[i],
                    part_id=i + 1,
                    sign=SignEnum.EQUAL,
                )
            ]
        for k in range(len(starts), cp_solver.nb_preemptive):
            list_strings += [
                cp_solver.constraint_start_time_string_preemptive_i(
                    task, start_time=ends[-1], part_id=k + 1, sign=SignEnum.EQUAL
                )
            ]
            list_strings += [
                cp_solver.constraint_duration_string_preemptive_i(
                    task, duration=0, part_id=k + 1, sign=SignEnum.EQUAL
                )
            ]
    return list_strings


def hard_start_times(
    dict_start_times: Dict[Hashable, int],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    constraint_strings = []
    for task in dict_start_times:
        string = (
            "constraint s["
            + str(cp_solver.index_in_minizinc[task])
            + "] == "
            + str(dict_start_times[task])
            + ";\n"
        )
        constraint_strings += [string]
    return constraint_strings


def soft_start_times(
    dict_start_times: Dict[Hashable, int],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    list_task = list(dict_start_times.keys())
    s = (
        """
        var 0..max_time*nb_start_times: penalty_start_times;\n
        int: nb_start_times="""
        + str(len(list_task))
        + """;\n"""
        + """
        array[1..nb_start_times] of Tasks: st1_0="""
        + str([cp_solver.index_in_minizinc[t1] for t1 in list_task])
        + """;\n"""
        + """
        array[1..nb_start_times] of 0..max_time: array_start_0="""
        + str([dict_start_times[t1] for t1 in list_task])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_times)(abs(s[st1_0[i]]-array_start_0[i]))==penalty_start_times;\n"""
    )
    return [s], ["penalty_start_times"]


def soft_start_together(
    list_start_together: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    s = (
        """
        var 0..max_time*nb_start_together: penalty_start_together;\n
        int: nb_start_together="""
        + str(len(list_start_together))
        + """;\n"""
        + """
        array[1..nb_start_together] of Tasks: st1_3="""
        + str([cp_solver.index_in_minizinc[t1] for t1, t2 in list_start_together])
        + """;\n"""
        + """
        array[1..nb_start_together] of Tasks: st2_3="""
        + str([cp_solver.index_in_minizinc[t2] for t1, t2 in list_start_together])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_together)(abs(s[st1_3[i]]-s[st2_3[i]]))==penalty_start_together;\n
        """
    )
    return [s], ["penalty_start_together"]


def hard_start_together(
    list_start_together: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    constraint_strings = []
    for t1, t2 in list_start_together:
        string = (
            "constraint s["
            + str(cp_solver.index_in_minizinc[t1])
            + "] == s["
            + str(cp_solver.index_in_minizinc[t2])
            + "];\n"
        )
        constraint_strings += [string]
    return constraint_strings


def hard_start_after_nunit(
    list_start_after_nunit: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    constraint_strings = []
    for t1, t2, delta in list_start_after_nunit:
        string = (
            "constraint s["
            + str(cp_solver.index_in_minizinc[t2])
            + "] >= s["
            + str(cp_solver.index_in_minizinc[t1])
            + "]+"
            + str(delta)
            + ";\n"
        )
        constraint_strings += [string]
    return constraint_strings


def soft_start_after_nunit(
    list_start_after_nunit: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    s = (
        """
        var 0..max_time*nb_start_after_nunit: penalty_start_after_nunit;\n
        int: nb_start_after_nunit="""
        + str(len(list_start_after_nunit))
        + """;\n"""
        + """
        array[1..nb_start_after_nunit] of Tasks: st1_4="""
        + str(
            [
                cp_solver.index_in_minizinc[t1]
                for t1, t2, delta in list_start_after_nunit
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_after_nunit] of Tasks: st2_4="""
        + str(
            [
                cp_solver.index_in_minizinc[t2]
                for t1, t2, delta in list_start_after_nunit
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_after_nunit] of int: nunits_4="""
        + str([delta for t1, t2, delta in list_start_after_nunit])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_after_nunit)(max([0, s[st1_4[i]]+nunits_4[i]-s[st2_4[i]]]))==penalty_start_after_nunit;\n
        """
    )
    return [s], ["penalty_start_after_nunit"]


def hard_start_at_end_plus_offset(
    list_start_at_end_plus_offset,
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    constraint_strings = []
    for t1, t2, delta in list_start_at_end_plus_offset:
        if isinstance(cp_solver, CP_MS_MRCPSP_MZN):
            string = (
                "constraint s["
                + str(cp_solver.index_in_minizinc[t2])
                + "] >= s["
                + str(cp_solver.index_in_minizinc[t1])
                + "]+adur["
                + str(cp_solver.index_in_minizinc[t1])
                + "]+"
                + str(delta)
                + ";\n"
            )
        else:
            string = (
                "constraint s["
                + str(cp_solver.index_in_minizinc[t2])
                + "] >= s_preemptive["
                + str(cp_solver.index_in_minizinc[t1])
                + ", nb_preemptive]+"
                + str(delta)
                + ";\n"
            )
        constraint_strings += [string]
    return constraint_strings


def soft_start_at_end_plus_offset(
    list_start_at_end_plus_offset: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    s = (
        """
        var 0..max_time*nb_start_at_end_plus_offset: penalty_start_at_end_plus_offset;\n
        int: nb_start_at_end_plus_offset="""
        + str(len(list_start_at_end_plus_offset))
        + """;\n"""
        + """
        array[1..nb_start_at_end_plus_offset] of Tasks: st1_7="""
        + str(
            [
                cp_solver.index_in_minizinc[t1]
                for t1, t2, delta in list_start_at_end_plus_offset
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_at_end_plus_offset] of Tasks: st2_7="""
        + str(
            [
                cp_solver.index_in_minizinc[t2]
                for t1, t2, delta in list_start_at_end_plus_offset
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_at_end_plus_offset] of int: nunits="""
        + str([delta for t1, t2, delta in list_start_at_end_plus_offset])
        + """;\n"""
    )
    if isinstance(cp_solver, CP_MS_MRCPSP_MZN):
        s += """
             constraint sum(i in 1..nb_start_at_end_plus_offset)(max([0, s[st1_7[i]]+adur[st1_7[i]]+nunits[i]-s[st2_7[i]]]))==penalty_start_at_end_plus_offset;\n
             """
    else:
        s += """
             constraint sum(i in 1..nb_start_at_end_plus_offset)(max([0, s_preemptive[st1_7[i], nb_preemptive]+nunits[i]-s[st2_7[i]]]))==penalty_start_at_end_plus_offset;\n
             """
    return [s], ["penalty_start_at_end_plus_offset"]


def hard_start_at_end(
    list_start_at_end: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    constraint_strings = []
    for t1, t2 in list_start_at_end:
        if isinstance(cp_solver, CP_MS_MRCPSP_MZN):
            string = (
                "constraint s["
                + str(cp_solver.index_in_minizinc[t2])
                + "] == s["
                + str(cp_solver.index_in_minizinc[t1])
                + "]+adur["
                + str(cp_solver.index_in_minizinc[t1])
                + "];\n"
            )
        else:
            string = (
                "constraint s_preemptive["
                + str(cp_solver.index_in_minizinc[t2])
                + ", 1] == s_preemptive["
                + str(cp_solver.index_in_minizinc[t1])
                + ", nb_preemptive];\n"
            )
        constraint_strings += [string]
    return constraint_strings


def soft_start_at_end(
    list_start_at_end: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    s = (
        """
        var 0..max_time*nb_start_at_end: penalty_start_at_end;\n
        int: nb_start_at_end="""
        + str(len(list_start_at_end))
        + """;\n"""
        + """
        array[1..nb_start_at_end] of Tasks: st1_9="""
        + str([cp_solver.index_in_minizinc[t1] for t1, t2 in list_start_at_end])
        + """;\n"""
        + """
        array[1..nb_start_at_end] of Tasks: st2_9="""
        + str([cp_solver.index_in_minizinc[t2] for t1, t2 in list_start_at_end])
        + """;\n"""
    )
    if isinstance(cp_solver, CP_MS_MRCPSP_MZN_PREEMPTIVE):
        s += """
            %constraint forall(i in 1..nb_start_at_end)(s[st2_9[i]]-s_preemptive[st1_9[i], nb_preemptive]>=0);\n
            constraint sum(i in 1..nb_start_at_end)(abs(s[st2_9[i]]-s_preemptive[st1_9[i], nb_preemptive]))==penalty_start_at_end;\n
            """
    else:
        s += """
            %constraint forall(i in 1..nb_start_at_end)(s[st2_9[i]]-s[st1_9[i]]-adur[st1_9[i]]>=0);\n
            constraint sum(i in 1..nb_start_at_end)(abs(s[st2_9[i]]-s[st1_9[i]]-adur[st1_9[i]]))==penalty_start_at_end;\n
            """
    return [s], ["penalty_start_at_end"]


def soft_start_window(
    start_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    l_low = [
        (t, start_times_window[t][0])
        for t in start_times_window
        if start_times_window[t][0] is not None
    ]
    l_up = [
        (t, start_times_window[t][1])
        for t in start_times_window
        if start_times_window[t][1] is not None
    ]
    max_t_start = max(
        [max([x[1] for x in l_low], default=0), max([x[1] for x in l_up], default=0)]
    )

    s = (
        """
        var 0..max_time*nb_start_window_low: penalty_start_low;\n
        int: nb_start_window_low="""
        + str(len(l_low))
        + """;\n
        var 0..max_time*nb_start_window_up: penalty_start_up;\n
        int: nb_start_window_up="""
        + str(len(l_up))
        + """;\n
        array[1..nb_start_window_low] of Tasks: task_id_low_start="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_low])
        + """;\n
        array[1..nb_start_window_up] of Tasks:  task_id_up_start="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_up])
        + """;\n
        int: max_time_start="""
        + str(max_t_start)
        + """;\n
        array[1..nb_start_window_low] of 0..max_time_start: times_low_start="""
        + str([int(x[1]) for x in l_low])
        + """;\n
        array[1..nb_start_window_up] of 0..max_time_start: times_up_start="""
        + str([int(x[1]) for x in l_up])
        + """;\n"""
    )
    s += """
        constraint sum(i in 1..nb_start_window_low)(max([times_low_start[i]-s[task_id_low_start[i]], 0]))==penalty_start_low;\n
        constraint sum(i in 1..nb_start_window_up)(max([-times_up_start[i]+s[task_id_up_start[i]], 0]))==penalty_start_up;\n
        """
    return [s], ["penalty_start_low", "penalty_start_up"]


def soft_end_window(
    end_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    l_low = [
        (t, end_times_window[t][0])
        for t in end_times_window
        if end_times_window[t][0] is not None
    ]
    l_up = [
        (t, end_times_window[t][1])
        for t in end_times_window
        if end_times_window[t][1] is not None
    ]
    max_t_end = max(
        [max([x[1] for x in l_low], default=0), max([x[1] for x in l_up], default=0)]
    )
    s = (
        """
        var 0..max_time*nb_end_window_low: penalty_end_low;\n
        int: nb_end_window_low="""
        + str(len(l_low))
        + """;\n
        var 0..max_time*nb_end_window_up: penalty_end_up;\n
        int: nb_end_window_up="""
        + str(len(l_up))
        + """;\n
        array[1..nb_end_window_low] of Tasks: task_id_low_end="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_low])
        + """;\n
        array[1..nb_end_window_up] of Tasks:  task_id_up_end="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_up])
        + """;\n
        int: max_time_end="""
        + str(max_t_end)
        + """;\n
        array[1..nb_end_window_low] of 0..max_time_end: times_low_end="""
        + str([int(x[1]) for x in l_low])
        + """;\n
        array[1..nb_end_window_up] of 0..max_time_end: times_up_end="""
        + str([int(x[1]) for x in l_up])
        + """;\n"""
    )
    if isinstance(cp_solver, CP_MS_MRCPSP_MZN_PREEMPTIVE):
        s += """
            constraint sum(i in 1..nb_end_window_low)(max([times_low_end[i]-s_preemptive[task_id_low_end[i], nb_preemptive], 0]))==penalty_end_low;\n
            constraint sum(i in 1..nb_end_window_up)(max([-times_up_end[i]+s_preemptive[task_id_up_end[i], nb_preemptive], 0]))==penalty_end_up;\n
            """
    else:
        s += """
            constraint sum(i in 1..nb_end_window_low)(max([times_low_end[i]-s[task_id_low_end[i]]+adur[task_id_low_end[i]], 0]))==penalty_end_low;\n
            constraint sum(i in 1..nb_end_window_up)(max([-times_up_end[i]+s[task_id_up_end[i]]+adur[task_id_up_end[i]], 0]))==penalty_end_up;\n
            """

    return [s], ["penalty_end_low", "penalty_end_up"]


def hard_start_window(
    start_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    l = []
    for task in start_times_window:
        if start_times_window[task][0] is not None:
            l += [
                cp_solver.constraint_start_time_string(
                    task=task, start_time=start_times_window[task][0], sign=SignEnum.UEQ
                )
            ]
        if start_times_window[task][1] is not None:
            l += [
                cp_solver.constraint_start_time_string(
                    task=task, start_time=start_times_window[task][1], sign=SignEnum.LEQ
                )
            ]
    return l


def hard_end_window(
    end_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    l = []
    for task in end_times_window:
        if end_times_window[task][0] is not None:
            l += [
                cp_solver.constraint_end_time_string(
                    task=task, end_time=end_times_window[task][0], sign=SignEnum.UEQ
                )
            ]
        if end_times_window[task][1] is not None:
            l += [
                cp_solver.constraint_end_time_string(
                    task=task, end_time=end_times_window[task][1], sign=SignEnum.LEQ
                )
            ]
    return l


def add_hard_special_constraints(
    partial_solution: PartialSolution,
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    if partial_solution is None:
        return []
    constraint_strings = []
    if partial_solution.start_times is not None:
        constraint_strings += hard_start_times(
            dict_start_times=partial_solution.start_times, cp_solver=cp_solver
        )
    if partial_solution.partial_permutation is not None:
        for t1, t2 in zip(
            partial_solution.partial_permutation[:-1],
            partial_solution.partial_permutation[1:],
        ):
            string = (
                "constraint s[" + str(cp_solver.index_in_minizinc[t1]) + "] "
                "<= s[" + str(cp_solver.index_in_minizinc[t2]) + "];\n"
            )
            constraint_strings += [string]
    if partial_solution.list_partial_order is not None:
        for l in partial_solution.list_partial_order:
            for t1, t2 in zip(l[:-1], l[1:]):
                string = (
                    "constraint s[" + str(cp_solver.index_in_minizinc[t1]) + "] "
                    "<= s[" + str(cp_solver.index_in_minizinc[t2]) + "];\n"
                )
                constraint_strings += [string]
    if partial_solution.start_together is not None:
        constraint_strings += hard_start_together(
            list_start_together=partial_solution.start_together, cp_solver=cp_solver
        )
    if partial_solution.start_after_nunit is not None:
        constraint_strings += hard_start_after_nunit(
            list_start_after_nunit=partial_solution.start_after_nunit,
            cp_solver=cp_solver,
        )
    if partial_solution.start_at_end_plus_offset is not None:
        constraint_strings += hard_start_at_end_plus_offset(
            list_start_at_end_plus_offset=partial_solution.start_at_end_plus_offset,
            cp_solver=cp_solver,
        )
    if partial_solution.start_at_end is not None:
        constraint_strings += hard_start_at_end(
            list_start_at_end=partial_solution.start_at_end, cp_solver=cp_solver
        )

    if partial_solution.start_times_window is not None:
        constraint_strings += hard_start_window(
            start_times_window=partial_solution.start_times_window, cp_solver=cp_solver
        )
    if partial_solution.end_times_window is not None:
        constraint_strings += hard_end_window(
            end_times_window=partial_solution.end_times_window, cp_solver=cp_solver
        )
    return constraint_strings


def add_soft_special_constraints(
    partial_solution: PartialSolution,
    cp_solver: Union[CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE],
):
    if partial_solution is None:
        return [], []
    constraint_strings = []
    name_penalty = []
    if partial_solution.start_times is not None:
        c, n = soft_start_times(
            dict_start_times=partial_solution.start_times, cp_solver=cp_solver
        )
        constraint_strings += c
        name_penalty += n
    if partial_solution.partial_permutation is not None:
        for t1, t2 in zip(
            partial_solution.partial_permutation[:-1],
            partial_solution.partial_permutation[1:],
        ):
            string = (
                "constraint s["
                + str(cp_solver.index_in_minizinc[t1])
                + "] "
                + "<= s["
                + str(cp_solver.index_in_minizinc[t2])
                + "];\n"
            )
            constraint_strings += [string]
    if partial_solution.list_partial_order is not None:
        for l in partial_solution.list_partial_order:
            for t1, t2 in zip(l[:-1], l[1:]):
                string = (
                    "constraint s[" + str(cp_solver.index_in_minizinc[t1]) + "] "
                    "<= s[" + str(cp_solver.index_in_minizinc[t2]) + "];\n"
                )
                constraint_strings += [string]
    if partial_solution.start_together is not None:
        c, n = soft_start_together(
            list_start_together=partial_solution.start_together, cp_solver=cp_solver
        )
        constraint_strings += c
        name_penalty += n

    if partial_solution.start_after_nunit is not None:
        c, n = soft_start_after_nunit(
            list_start_after_nunit=partial_solution.start_after_nunit,
            cp_solver=cp_solver,
        )
        constraint_strings += c
        name_penalty += n
    if partial_solution.start_at_end_plus_offset is not None:
        c, n = soft_start_at_end_plus_offset(
            list_start_at_end_plus_offset=partial_solution.start_at_end_plus_offset,
            cp_solver=cp_solver,
        )
        constraint_strings += c
        name_penalty += n
    if partial_solution.start_at_end is not None:
        c, n = soft_start_at_end(
            list_start_at_end=partial_solution.start_at_end, cp_solver=cp_solver
        )
        constraint_strings += c
        name_penalty += n
    if partial_solution.start_times_window is not None:
        c, n = soft_start_window(
            partial_solution.start_times_window, cp_solver=cp_solver
        )
        constraint_strings += c
        constraint_strings += ["constraint " + str(n[0]) + "==0;\n"]
        name_penalty += n
    if partial_solution.end_times_window is not None:
        c, n = soft_end_window(partial_solution.end_times_window, cp_solver=cp_solver)
        constraint_strings += c
        name_penalty += n
    return constraint_strings, name_penalty


def define_second_part_objective(weights, name_penalty, equal=False):
    sum_string = "+".join(
        ["0"] + [str(weights[i]) + "*" + name_penalty[i] for i in range(len(weights))]
    )
    if equal:
        s = "constraint sec_objective==" + sum_string + ";\n"
    else:
        s = "constraint sec_objective>=" + sum_string + ";\n"
    return [s]


def add_constraints_string(child_instance, list_of_strings):
    for s in list_of_strings:
        child_instance.add_string(s)


class SolutionPrecomputeEmployeesForTasks:
    def __init__(
        self, unit_used, worker_type_used, mode_dict, overskill_unit, overskill_type
    ):
        self.unit_used = unit_used
        self.worker_type_used = worker_type_used
        self.mode_dict = mode_dict
        self.overskill_unit = overskill_unit
        self.overskill_type = overskill_type

    def __str__(self):
        return "-".join(
            [str(s) + " : " + str(getattr(self, s)) for s in self.__dict__.keys()]
        )


class PrecomputeEmployeesForTasks:
    def __init__(
        self,
        ms_rcpsp_model: MS_RCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
    ):
        self.ms_rcpsp_model = ms_rcpsp_model
        self.cp_solver_name = cp_solver_name
        self.instance: Instance = None
        (
            self.skills_representation_str,
            self.skills_dict,
        ) = cluster_employees_to_resource_types(self.ms_rcpsp_model)

    def init_model(self, **kwargs):
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        model = Model(files_mzn["compute_worker_for_tasks"])
        instance = Instance(solver, model)
        instance["one_ressource_per_task"] = kwargs.get("one_ressource_per_task", False)
        instance["one_worker_type_per_task"] = kwargs.get(
            "one_worker_type_per_task", False
        )
        instance["exact_skills_need"] = kwargs.get("exact_skills_need", False)
        instance["consider_worker_type"] = kwargs.get("consider_worker_type", True)
        instance["consider_units"] = kwargs.get("consider_units", False)
        tasks_of_interest = kwargs.get(
            "tasks_of_interest", [self.ms_rcpsp_model.tasks_list[0]]
        )
        instance["n_tasks"] = len(tasks_of_interest)
        n_opt = sum(
            [
                len(list(self.ms_rcpsp_model.mode_details[key].keys()))
                for key in tasks_of_interest
            ]
        )
        instance["n_opt"] = n_opt
        all_modes = [
            (act, mode, self.ms_rcpsp_model.mode_details[act][mode])
            for act in tasks_of_interest
            for mode in sorted(self.ms_rcpsp_model.mode_details[act])
        ]
        self.modeindex_map = {
            i + 1: {"task": all_modes[i][0], "original_mode_index": all_modes[i][1]}
            for i in range(len(all_modes))
        }
        modes = [set() for t in tasks_of_interest]
        for j in self.modeindex_map:
            modes[tasks_of_interest.index(self.modeindex_map[j]["task"])].add(j)
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        instance["dur"] = dur

        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in self.ms_rcpsp_model.resources_list
        ]
        instance["rreq"] = rreq
        instance["nb_res"] = len(self.ms_rcpsp_model.resources_list)
        rcap = [
            int(max(self.ms_rcpsp_model.resources_availability[x]))
            for x in self.ms_rcpsp_model.resources_list
        ]
        instance["rcap"] = rcap
        rtype = [
            2 if res in self.ms_rcpsp_model.non_renewable_resources else 1
            for res in self.ms_rcpsp_model.resources_list
        ]
        instance["rtype"] = rtype

        skills_set = self.ms_rcpsp_model.skills_list
        skill_required = [
            [int(all_modes[i][2].get(s, 0)) for i in range(len(all_modes))]
            for s in skills_set
        ]

        instance["nb_skill"] = len(self.ms_rcpsp_model.skills_set)
        instance["skillreq"] = skill_required
        instance["nb_units"] = len(self.ms_rcpsp_model.employees_list)

        skillunits = [
            [
                int(
                    math.floor(
                        self.ms_rcpsp_model.employees[j].dict_skill[s].skill_value
                    )
                )
                if s in self.ms_rcpsp_model.employees[j].dict_skill
                else 0
                for s in skills_set
            ]
            for j in self.ms_rcpsp_model.employees_list
        ]
        self.employees_position = self.ms_rcpsp_model.employees_list
        instance["skillunits"] = skillunits

        worker_type_list = sorted(self.skills_dict)
        instance["nb_worker_type"] = len(worker_type_list)
        instance["skills_worker_type"] = [
            [
                self.skills_dict[wt][s].skill_value if s in self.skills_dict[wt] else 0
                for s in skills_set
            ]
            for wt in worker_type_list
        ]
        instance["capacity_worker_type"] = [
            len(self.skills_representation_str[wt]) for wt in worker_type_list
        ]

        self.instance = instance

    def retrieve_solutions(self, result, parameters_cp: ParametersCP):
        intermediate_solutions = parameters_cp.intermediate_solution
        units_used = []
        workers_type_used = []
        overskills_units = []
        overskills_types = []
        mruns = []
        modes_dict = []

        if intermediate_solutions:
            for i in range(len(result)):
                mruns += [result[i, "mrun"]]
                overskills_types += [result[i, "overskill_type"]]
                overskills_units += [result[i, "overskill_unit"]]
                units_used += [result[i, "unit_used"]]
                workers_type_used += [result[i, "worker_type_used"]]
        else:
            mruns += [result["mrun"]]
            overskills_types += [result["overskill_type"]]
            overskills_units += [result["overskill_unit"]]
            units_used += [result["unit_used"]]
            workers_type_used += [result["worker_type_used"]]

        for k in range(len(mruns)):
            mode_dict = {}
            for i in range(len(mruns[k])):
                if mruns[k][i]:
                    mode_dict[self.modeindex_map[i + 1]["task"]] = self.modeindex_map[
                        i + 1
                    ]["original_mode_index"]
            modes_dict += [mode_dict]
        return [
            SolutionPrecomputeEmployeesForTasks(
                unit_used=u,
                worker_type_used=w,
                mode_dict=m,
                overskill_unit=ou,
                overskill_type=ow,
            )
            for u, w, m, ou, ow in zip(
                units_used,
                workers_type_used,
                modes_dict,
                overskills_units,
                overskills_types,
            )
        ]

    def solve(self, parameters_cp: Optional[ParametersCP] = None, **args):
        if self.instance is None:
            self.init_model(**args)
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        timeout = parameters_cp.time_limit
        intermediate_solutions = parameters_cp.intermediate_solution
        result = self.instance.solve(
            timeout=timedelta(seconds=timeout),
            nr_solutions=parameters_cp.nr_solutions,
            intermediate_solutions=intermediate_solutions,
        )
        logger.debug(result.status)
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)
