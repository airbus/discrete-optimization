#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import json
import logging
import os
from datetime import timedelta
from typing import Dict, Hashable, List, Optional, Set, Tuple, Union

from deprecation import deprecated
from minizinc import Instance, Model, Solver

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
from discrete_optimization.rcpsp.rcpsp_model import (
    PartialSolution,
    RCPSPModel,
    RCPSPModelCalendar,
    RCPSPSolution,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
)
from discrete_optimization.rcpsp.rcpsp_model_utils import create_fake_tasks

logger = logging.getLogger(__name__)
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {
    "single": os.path.join(this_path, "../minizinc/rcpsp_single_mode_mzn.mzn"),
    "single-resource": os.path.join(
        this_path, "../minizinc/rcpsp_minimize_resource.mzn"
    ),
    "single-no-search": os.path.join(
        this_path, "../minizinc/rcpsp_single_mode_mzn_no_search.mzn"
    ),
    "single-preemptive": os.path.join(
        this_path, "../minizinc/rcpsp_single_mode_mzn_preemptive.mzn"
    ),
    "multi-preemptive": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_preemptive.mzn"
    ),
    "multi-preemptive-calendar": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_preemptive_calendar.mzn"
    ),
    "single-preemptive-calendar": os.path.join(
        this_path, "../minizinc/rcpsp_single_mode_mzn_preemptive_calendar.mzn"
    ),
    "multi": os.path.join(this_path, "../minizinc/rcpsp_multi_mode_mzn.mzn"),
    "multi-faketasks": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_with_faketasks.mzn"
    ),
    "multi-no-bool": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_no_bool.mzn"
    ),
    "multi-calendar": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_mzn_calendar.mzn"
    ),
    "multi-calendar-boxes": os.path.join(
        this_path, "../minizinc/rcpsp_mzn_calendar_boxes.mzn"
    ),
    "multi-resource-feasibility": os.path.join(
        this_path, "../minizinc/rcpsp_multi_mode_resource_feasibility_mzn.mzn"
    ),
    "modes": os.path.join(this_path, "../minizinc/mrcpsp_mode_satisfy.mzn"),
    "single_resource": os.path.join(
        this_path, "../minizinc/rcpsp_single_mode_resource.mzn"
    ),
}


class RCPSPSolCP:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        logger.info(f"One solution {self.objective}")
        logger.info(f"Output {_output_item}")

    def check(self) -> bool:
        return True


def add_fake_task_cp_data(
    rcpsp_model: Union[RCPSPModel, RCPSPModelPreemptive],
    ignore_fake_task: bool = True,
    max_time_to_consider: int = None,
):
    if rcpsp_model.is_varying_resource() and not ignore_fake_task:
        fake_tasks = create_fake_tasks(rcpsp_problem=rcpsp_model)
        max_time_to_consider = (
            rcpsp_model.horizon
            if max_time_to_consider is None
            else max_time_to_consider
        )
        fake_tasks = [f for f in fake_tasks if f["start"] <= max_time_to_consider]
        n_fake_tasks = len(fake_tasks)
        fakestart = [fake_tasks[i]["start"] for i in range(len(fake_tasks))]
        fake_dur = [fake_tasks[i]["duration"] for i in range(len(fake_tasks))]
        max_duration_fake_task = max(fake_dur)
        fake_req = [
            [fake_tasks[i].get(res, 0) for i in range(len(fake_tasks))]
            for res in rcpsp_model.resources_list
        ]
        dict_to_add_in_instance = {
            "max_duration_fake_task": max_duration_fake_task,
            "n_fake_tasks": n_fake_tasks,
            "fakestart": fakestart,
            "fakedur": fake_dur,
            "fakereq": fake_req,
            "include_fake_tasks": True,
        }
        return dict_to_add_in_instance
    else:
        dict_to_add_in_instance = {
            "max_duration_fake_task": 0,
            "n_fake_tasks": 0,
            "fakestart": [],
            "fakedur": [],
            "fakereq": [[] for r in rcpsp_model.resources_list],
            "include_fake_tasks": False,
        }
        return dict_to_add_in_instance


class CP_RCPSP_MZN(MinizincCPSolver):
    def __init__(
        self,
        rcpsp_model: RCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        silent_solve_error: bool = True,
        **kwargs,
    ):
        self.silent_solve_error = silent_solve_error
        self.rcpsp_model = rcpsp_model
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = [
            "s"
        ]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.rcpsp_model, params_objective_function=params_objective_function
        )
        self.stats = []

    def init_model(self, **args):
        model_type = args.get("model_type", "single")
        max_time = args.get("max_time", self.rcpsp_model.horizon)
        fake_tasks = args.get(
            "fake_tasks", True
        )  # to modelize varying quantity of resource.
        add_objective_makespan = args.get("add_objective_makespan", True)
        if model_type == "single-resource":
            add_objective_makespan = False
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        custom_output_type = args.get("output_type", False)
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        model = Model(files_mzn[model_type])
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)
        self.keys_in_instance = []
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        if model_type == "single-resource":
            instance["add_objective_resource"] = args.get(
                "add_objective_resource", True
            )
            self.keys_in_instance += ["add_objective_resource"]
        self.keys_in_instance += ["add_objective_makespan", "ignore_sec_objective"]
        n_res = len(self.rcpsp_model.resources_list)
        instance["n_res"] = n_res
        self.keys_in_instance += ["n_res"]
        dict_to_add = add_fake_task_cp_data(
            rcpsp_model=self.rcpsp_model,
            ignore_fake_task=not fake_tasks,
            max_time_to_consider=max_time,
        )
        instance["max_time"] = max_time
        self.keys_in_instance += ["max_time"]
        for key in dict_to_add:
            instance[key] = dict_to_add[key]
            self.keys_in_instance += [key]

        sorted_resources = self.rcpsp_model.resources_list
        self.resources_index = sorted_resources
        rcap = [
            int(self.rcpsp_model.get_max_resource_capacity(x)) for x in sorted_resources
        ]
        instance["rc"] = rcap
        self.keys_in_instance += ["rc"]
        n_tasks = self.rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        self.keys_in_instance += ["n_tasks"]
        sorted_tasks = self.rcpsp_model.tasks_list
        d = [
            int(self.rcpsp_model.mode_details[key][1]["duration"])
            for key in sorted_tasks
        ]
        instance["d"] = d
        self.keys_in_instance += ["d"]
        all_modes = [
            (act, 1, self.rcpsp_model.mode_details[act][1]) for act in sorted_tasks
        ]
        rr = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in sorted_resources
        ]
        instance["rr"] = rr
        self.keys_in_instance += ["rr"]
        suc = [
            set(
                [
                    self.rcpsp_model.return_index_task(x, offset=1)
                    for x in self.rcpsp_model.successors[task]
                ]
            )
            for task in sorted_tasks
        ]
        instance["suc"] = suc
        self.keys_in_instance += ["suc"]
        self.instance = instance
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        self.strings_to_add = []
        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
                self.strings_to_add += [s]
        else:
            strings, name_penalty = add_soft_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
                self.strings_to_add += [s]
            strings = define_second_part_objective(
                [100] * len(name_penalty), name_penalty
            )
            for s in strings:
                self.instance.add_string(s)
                self.strings_to_add += [s]

    def add_hard_special_constraints(self, partial_solution):
        return add_hard_special_constraints(partial_solution, self)

    def constraint_objective_makespan(self):
        s = """constraint forall ( i in Tasks where suc[i] == {} )
                (s[i] + d[i] <= objective);\n"""
        return [s]

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = "constraint (s[" + str(ind) + "]+d[" + str(ind) + "]==objective);\n"
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
               constraint objective>=sum(j in index_tasks)(s[j]+d[j]);\n"""
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
               constraint objective>=sum(j in index_tasks)(s[j]);\n"""
        )
        return [s]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint s["
                + str(self.index_in_minizinc[j])
                + "]+d["
                + str(self.index_in_minizinc[j])
                + "]<=objective;\n"
            ]
        return s

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

    def constraint_end_time_string(
        self, task, end_time, sign: SignEnum = SignEnum.EQUAL
    ) -> str:
        return (
            "constraint s["
            + str(self.index_in_minizinc[task])
            + "]+d["
            + str(self.index_in_minizinc[task])
            + "]"
            + str(sign.value)
            + str(end_time)
            + ";\n"
        )

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts.append(result[i].dict["s"])
                else:
                    starts.append(result[i, "s"])
        else:
            if isinstance(result, RCPSPSolCP):
                starts.append(result.dict["s"])
            else:
                starts = [result["s"]]

        for start_times in starts:
            rcpsp_schedule = {}
            for k in range(len(start_times)):
                t = self.rcpsp_model.tasks_list[k]
                rcpsp_schedule[self.rcpsp_model.tasks_list[k]] = {
                    "start_time": start_times[k],
                    "end_time": start_times[k]
                    + self.rcpsp_model.mode_details[t][1]["duration"],
                }
            sol = RCPSPSolution(
                problem=self.rcpsp_model,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_modes=[1 for i in range(self.rcpsp_model.n_jobs_non_dummy)],
                rcpsp_schedule_feasible=True,
            )
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            list_solutions_fit.append((sol, objective))
        result_storage = ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=best_solution,
            mode_optim=self.params_objective_function.sense_function,
            limit_store=False,
        )
        return result_storage

    def get_stats(self):
        return self.stats


class CP_MRCPSP_MZN(MinizincCPSolver):
    def __init__(
        self,
        rcpsp_model: RCPSPModel,
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
        self.calendar = False
        if self.rcpsp_model.is_varying_resource():
            self.calendar = True

    def init_model(self, **args):
        model_type = args.get("model_type", "multi")
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type", False)
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        resources_list = self.rcpsp_model.resources_list
        self.resources_index = resources_list
        instance = Instance(solver, model)
        keys = []
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        keys += ["add_objective_makespan", "ignore_sec_objective"]
        max_time = args.get("max_time", self.rcpsp_model.horizon)
        if model_type != "multi-calendar":
            fake_tasks = args.get(
                "fake_tasks", True
            )  # to modelize varying quantity of resource.
            dict_to_add = add_fake_task_cp_data(
                rcpsp_model=self.rcpsp_model,
                ignore_fake_task=not fake_tasks,
                max_time_to_consider=max_time,
            )
            for key in dict_to_add:
                instance[key] = dict_to_add[key]
                keys += [key]
        instance["max_time"] = max_time
        keys += ["max_time"]

        n_res = len(resources_list)
        instance["n_res"] = n_res
        keys += ["n_res"]
        n_tasks = self.rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = self.rcpsp_model.tasks_list
        n_opt = sum([len(self.rcpsp_model.mode_details[key]) for key in sorted_tasks])
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
        all_modes = [
            (act, mode, self.rcpsp_model.mode_details[act][mode])
            for act in sorted_tasks
            for mode in sorted(self.rcpsp_model.mode_details[act])
        ]
        self.modeindex_map = {
            i + 1: {"task": all_modes[i][0], "original_mode_index": all_modes[i][1]}
            for i in range(len(all_modes))
        }
        self.modeindex_map = {
            i + 1: {"task": all_modes[i][0], "original_mode_index": all_modes[i][1]}
            for i in range(len(all_modes))
        }
        self.mode_dict_task_mode_to_index_minizinc = {}
        for ind in self.modeindex_map:
            task = self.modeindex_map[ind]["task"]
            mode = self.modeindex_map[ind]["original_mode_index"]
            self.mode_dict_task_mode_to_index_minizinc[(task, mode)] = ind
        modes = [set() for t in sorted_tasks]
        for j in self.modeindex_map:
            modes[self.rcpsp_model.index_task[self.modeindex_map[j]["task"]]].add(j)
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["dur"] = dur
        keys += ["dur"]
        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in resources_list
        ]
        instance["rreq"] = rreq
        keys += ["rreq"]
        if not self.calendar:
            rcap = [int(self.rcpsp_model.resources[x]) for x in resources_list]
        else:
            rcap = [int(max(self.rcpsp_model.resources[x])) for x in resources_list]
        if model_type != "multi-resource-feasibility":
            instance["rcap"] = rcap
            keys += ["rcap"]
        if model_type == "multi-resource-feasibility":
            instance["rweight"] = args.get("rweight", [1] * len(resources_list))
            instance["max_makespan"] = max_time
            instance["add_redundant_constraints"] = args.get(
                "add_redundant_constraints", False
            )
            instance["rcap_max"] = args.get(
                "rcap_max",
                [
                    self.rcpsp_model.get_max_resource_capacity(res)
                    for res in resources_list
                ],
            )
            instance["rcap_min"] = args.get("rcap_max", [0 for res in resources_list])
            keys += ["rweight", "max_makespan"]
        rtype = [
            2 if res in self.rcpsp_model.non_renewable_resources else 1
            for res in resources_list
        ]
        instance["rtype"] = rtype
        keys += ["rtype"]
        succ = [
            set(
                [
                    self.rcpsp_model.return_index_task(x, offset=1)
                    for x in self.rcpsp_model.successors.get(task, [])
                ]
            )
            for task in sorted_tasks
        ]
        instance["succ"] = succ
        keys += ["succ"]
        if self.calendar:
            one_ressource = list(self.rcpsp_model.resources.keys())[0]
            instance["max_time"] = len(self.rcpsp_model.resources[one_ressource])
            keys += ["max_time"]
            ressource_capacity_time = [
                [int(x) for x in self.rcpsp_model.resources[res]]
                for res in resources_list
            ]
            instance["ressource_capacity_time"] = ressource_capacity_time
            keys += ["ressource_capacity_time"]

        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints_mrcpsp(p_s, self)
            for s in strings:
                self.instance.add_string(s)
        else:
            strings, name_penalty = add_soft_special_constraints_mrcpsp(p_s, self)
            for s in strings:
                self.instance.add_string(s)
            strings = define_second_part_objective(
                [100] * len(name_penalty), name_penalty
            )
            for s in strings:
                self.instance.add_string(s)

    def add_hard_special_constraints(self, partial_solution):
        return add_hard_special_constraints(partial_solution, self)

    def constraint_task_to_mode(self, task_id, mode):
        modes = self.rcpsp_model.mode_details[task_id].keys()
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

    def constraint_objective_makespan(self):
        s = """constraint forall ( i in Act where suc[i] == {} )
                (start[i] + adur[i] <= objective);\n"""
        return [s]

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = "constraint (start[" + str(ind) + "]+adur[" + str(ind) + "]==objective);\n"
        return [s]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint start["
                + str(self.index_in_minizinc[j])
                + "]+adur["
                + str(self.index_in_minizinc[j])
                + "]<=objective;\n"
            ]
        return s

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

    def retrieve_solutions(self, result, parameters_cp: ParametersCP):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        mruns = []
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts.append(result[i].dict["start"])
                    mruns.append(result[i].dict["mrun"])
                else:
                    starts.append(result[i, "start"])
                    mruns.append(result[i, "mrun"])
        else:
            if isinstance(result, RCPSPSolCP):
                starts.append(result.dict["start"])
                mruns.append(result.dict["mrun"])
            else:
                starts.append(result["start"])
                mruns.append(result["mrun"])
        for start_times, mrun in zip(starts, mruns):
            modes_dict = {}
            for i in range(len(mrun)):
                if mrun[i]:
                    modes_dict[self.modeindex_map[i + 1]["task"]] = self.modeindex_map[
                        i + 1
                    ]["original_mode_index"]
            rcpsp_schedule = {}
            for i in range(len(start_times)):
                rcpsp_schedule[self.rcpsp_model.tasks_list[i]] = {
                    "start_time": start_times[i],
                    "end_time": start_times[i]
                    + self.rcpsp_model.mode_details[self.rcpsp_model.tasks_list[i]][
                        modes_dict[self.rcpsp_model.tasks_list[i]]
                    ]["duration"],
                }
            sol = RCPSPSolution(
                problem=self.rcpsp_model,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_modes=[
                    modes_dict[t] for t in self.rcpsp_model.tasks_list_non_dummy
                ],
                rcpsp_schedule_feasible=True,
            )
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            list_solutions_fit.append((sol, objective))
        result_storage = ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=best_solution,
            mode_optim=self.params_objective_function.sense_function,
            limit_store=False,
        )
        return result_storage


@deprecated(deprecated_in="0.1")
class CP_MRCPSP_MZN_WITH_FAKE_TASK(CP_MRCPSP_MZN):
    def __init__(
        self,
        rcpsp_model: RCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        silent_solve_error: bool = False,
        **kwargs,
    ):
        super().__init__(
            rcpsp_model=rcpsp_model,
            cp_solver_name=cp_solver_name,
            params_objective_function=params_objective_function,
            silent_solve_error=silent_solve_error,
            **kwargs,
        )
        self.fake_tasks = create_fake_tasks(rcpsp_problem=rcpsp_model)

    def init_model(self, **args):
        model_type = args.get("model_type", "multi-faketasks")
        if model_type is None:
            model_type = "multi-faketasks"
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type", False)
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        resources_list = self.rcpsp_model.resources_list
        self.resources_index = resources_list
        instance = Instance(solver, model)
        keys = []
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        keys += ["add_objective_makespan", "ignore_sec_objective"]
        n_res = len(resources_list)
        instance["n_res"] = n_res
        keys += ["n_res"]
        n_tasks = self.rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = self.rcpsp_model.tasks_list
        n_opt = sum([len(self.rcpsp_model.mode_details[key]) for key in sorted_tasks])
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
        all_modes = [
            (act, mode, self.rcpsp_model.mode_details[act][mode])
            for act in sorted_tasks
            for mode in sorted(self.rcpsp_model.mode_details[act])
        ]
        self.modeindex_map = {
            i + 1: {"task": all_modes[i][0], "original_mode_index": all_modes[i][1]}
            for i in range(len(all_modes))
        }

        max_time = args.get("max_time", self.rcpsp_model.horizon)
        fake_tasks = args.get(
            "fake_tasks", True
        )  # to modelize varying quantity of resource.
        dict_to_add = add_fake_task_cp_data(
            rcpsp_model=self.rcpsp_model,
            ignore_fake_task=not fake_tasks,
            max_time_to_consider=max_time,
        )
        instance["max_time"] = max_time
        keys += ["max_time"]
        for key in dict_to_add:
            instance[key] = dict_to_add[key]
            keys += [key]

        modes = [set() for t in sorted_tasks]
        for j in self.modeindex_map:
            modes[self.rcpsp_model.index_task[self.modeindex_map[j]["task"]]].add(j)
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["dur"] = dur
        keys += ["dur"]
        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in resources_list
        ]
        instance["rreq"] = rreq
        keys += ["rreq"]
        if not self.calendar:
            rcap = [int(self.rcpsp_model.resources[x]) for x in resources_list]
        else:
            rcap = [int(max(self.rcpsp_model.resources[x])) for x in resources_list]
        instance["rcap"] = rcap
        keys += ["rcap"]
        rtype = [
            2 if res in self.rcpsp_model.non_renewable_resources else 1
            for res in resources_list
        ]
        instance["rtype"] = rtype
        keys += ["rtype"]
        succ = [
            set(
                [
                    self.rcpsp_model.return_index_task(x, offset=1)
                    for x in self.rcpsp_model.successors.get(task, [])
                ]
            )
            for task in sorted_tasks
        ]
        instance["succ"] = succ
        keys += ["succ"]
        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints_mrcpsp(p_s, self)
            for s in strings:
                self.instance.add_string(s)
        else:
            strings, name_penalty = add_soft_special_constraints_mrcpsp(p_s, self)
            for s in strings:
                self.instance.add_string(s)
            strings = define_second_part_objective(
                [100] * len(name_penalty), name_penalty
            )
            for s in strings:
                self.instance.add_string(s)

    def retrieve_solutions(self, result, parameters_cp: ParametersCP):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        mruns = []
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts.append(result[i].dict["start"])
                    mruns.append(result[i].dict["mrun"])
                else:
                    starts.append(result[i, "start"])
                    mruns.append(result[i, "mrun"])
        else:
            if isinstance(result, RCPSPSolCP):
                starts.append(result.dict["start"])
                mruns.append(result.dict["mrun"])
            else:
                starts = [result["start"]]
                mruns = [result["mrun"]]
        for start_times, mrun in zip(starts, mruns):
            modes_dict = {}
            for i in range(len(mrun)):
                if mrun[i]:
                    modes_dict[self.modeindex_map[i + 1]["task"]] = self.modeindex_map[
                        i + 1
                    ]["original_mode_index"]
            rcpsp_schedule = {}
            for i in range(len(start_times)):
                rcpsp_schedule[self.rcpsp_model.tasks_list[i]] = {
                    "start_time": start_times[i],
                    "end_time": start_times[i]
                    + self.rcpsp_model.mode_details[self.rcpsp_model.tasks_list[i]][
                        modes_dict[self.rcpsp_model.tasks_list[i]]
                    ]["duration"],
                }
            sol = RCPSPSolution(
                problem=self.rcpsp_model,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_modes=[
                    modes_dict[t] for t in self.rcpsp_model.tasks_list_non_dummy
                ],
                rcpsp_schedule_feasible=True,
            )
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            list_solutions_fit.append((sol, objective))
        result_storage = ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=best_solution,
            mode_optim=self.params_objective_function.sense_function,
            limit_store=False,
        )
        return result_storage

    def constraint_objective_makespan(self):
        s = """constraint forall ( i in Act where suc[i] == {} )
                (start[i] + adur[i] <= objective);\n"""
        return [s]

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = "constraint (start[" + str(ind) + "]+adur[" + str(ind) + "]==objective);\n"
        return [s]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint start[" + str(self.index_in_minizinc[j]) + "]<=objective;\n"
            ]
        return s

    def constraint_sum_of_ending_time(self, set_subtasks: Set[Hashable]):
        indexes = [self.index_in_minizinc[s] for s in set_subtasks]
        s = (
            """int: nb_indexes="""
            + str(len(indexes))
            + """;\n
               array[1..nb_indexes] of Tasks: index_tasks="""
            + str(indexes)
            + """;\n
               constraint objective>=sum(j in index_tasks)(start[j]+adur[j, nb_preemptive]);\n"""
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


class CP_RCPSP_MZN_PREEMMPTIVE(MinizincCPSolver):
    def __init__(
        self,
        rcpsp_model: RCPSPModelPreemptive,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
        silent_solve_error: bool = True,
        **kwargs,
    ):
        self.silent_solve_error = silent_solve_error
        self.rcpsp_model = rcpsp_model
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = [
            "s"
        ]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.rcpsp_model, params_objective_function=params_objective_function
        )
        self.calendar = rcpsp_model.is_varying_resource()
        self.index_in_minizinc = None
        self.data_dict = None
        self.nb_preemptive = None

    def init_model(self, **args):
        model_type = args.get("model_type", None)
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        if model_type is None:
            model_type = (
                "single-preemptive"
                if not self.calendar
                else "single-preemptive-calendar"
            )
        keys = []
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type", False)
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        keys += ["add_objective_makespan", "ignore_sec_objective"]
        instance["nb_preemptive"] = args.get("nb_preemptive", 2)
        self.nb_preemptive = instance["nb_preemptive"]
        keys += ["nb_preemptive"]
        instance["possibly_preemptive"] = args.get(
            "possibly_preemptive",
            [
                self.rcpsp_model.can_be_preempted(task)
                for task in self.rcpsp_model.tasks_list
            ],
        )
        keys += ["possibly_preemptive"]
        instance["max_preempted"] = args.get(
            "max_preempted", min(self.rcpsp_model.n_jobs_non_dummy, 5)
        )
        keys += ["max_preempted"]

        if model_type != "single-preemptive-calendar":
            max_time = args.get("max_time", self.rcpsp_model.horizon)
            fake_tasks = args.get(
                "fake_tasks", True
            )  # to modelize varying quantity of resource.
            dict_to_add = add_fake_task_cp_data(
                rcpsp_model=self.rcpsp_model,
                ignore_fake_task=not fake_tasks,
                max_time_to_consider=max_time,
            )
            instance["max_time"] = max_time
            keys += ["max_time"]
            for key in dict_to_add:
                instance[key] = dict_to_add[key]
                keys += [key]

        n_res = len(list(self.rcpsp_model.resources.keys()))
        instance["n_res"] = n_res
        keys += ["n_res"]
        sorted_resources = self.rcpsp_model.resources_list
        self.resources_index = sorted_resources
        if not self.calendar:
            rc = [int(self.rcpsp_model.resources[x]) for x in sorted_resources]
        else:
            rc = [int(max(self.rcpsp_model.resources[x])) for x in sorted_resources]
        if self.calendar and model_type == "single-preemptive-calendar":
            one_ressource = list(self.rcpsp_model.resources.keys())[0]
            instance["max_time"] = len(self.rcpsp_model.resources[one_ressource])
            keys += ["max_time"]
            ressource_capacity_time = [
                [int(x) for x in self.rcpsp_model.resources[res]]
                for res in sorted_resources
            ]
            instance["ressource_capacity_time"] = ressource_capacity_time
            keys += ["ressource_capacity_time"]

        instance["rc"] = rc
        keys += ["rc"]

        n_tasks = self.rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]

        sorted_tasks = self.rcpsp_model.tasks_list
        d = [
            int(self.rcpsp_model.mode_details[key][1]["duration"])
            for key in sorted_tasks
        ]
        instance["d"] = d
        keys += ["d"]

        rr = []
        index = 0
        for res in sorted_resources:
            rr.append([])
            for task in sorted_tasks:
                rr[index].append(
                    int(self.rcpsp_model.mode_details[task][1].get(res, 0))
                )
            index += 1
        instance["rr"] = rr
        keys += ["rr"]

        suc = [
            set(
                [
                    self.rcpsp_model.return_index_task(x, offset=1)
                    for x in self.rcpsp_model.successors[task]
                ]
            )
            for task in sorted_tasks
        ]
        instance["suc"] = suc
        keys += ["suc"]
        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        self.data_dict = {key: self.instance[key] for key in keys}
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
            for s in strings:
                self.instance.add_string(s)

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

    def constraint_minduration_all_tasks(self):
        list_strings = []
        for task in self.rcpsp_model.tasks_list:
            if self.rcpsp_model.duration_subtask[task][0]:
                list_strings += self.constraint_duration_to_min_duration_preemptive(
                    task=task, min_duration=self.rcpsp_model.duration_subtask[task][1]
                )
        return list_strings

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

    def constraint_ressource_requirement_at_time_t(
        self, time, ressource, ressource_number, sign: SignEnum = SignEnum.LEQ
    ):
        index_ressource = self.resources_index.index(ressource) + 1
        s = (
            """constraint """
            + str(ressource_number)
            + str(sign.value)
            + """sum( i in Tasks, j in PREEMPTIVE) (
                                            bool2int(s_preemptive[i, j] <="""
            + str(time)
            + """ /\ """
            + str(time)
            + """< s_preemptive[i, j] + d_preemptive[i, j]) * rr["""
            + str(index_ressource)
            + """,i]);\n"""
        )
        return s

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        best_objectives_cp = float("inf")
        list_solutions_fit = []
        starts = []
        starts_preemptive = []
        duration_preemptive = []
        objectives_cp = []
        index_best = 0
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts.append(result[i].dict["s"])
                    starts_preemptive.append(result[i].dict["s_preemptive"])
                    duration_preemptive.append(result[i].dict["d_preemptive"])
                    objectives_cp.append(result[i].objective)
                else:
                    starts.append(result[i, "s"])
                    starts_preemptive.append(result[i, "s_preemptive"])
                    duration_preemptive.append(result[i, "d_preemptive"])
                    objectives_cp.append(result[i, "objective"])

        else:
            if isinstance(result, RCPSPSolCP):
                starts.append(result.dict["s"])
                starts_preemptive.append(result.dict["s_preemptive"])
                duration_preemptive.append(result.dict["d_preemptive"])
                objectives_cp.append(result.objective)
            else:
                starts = [result["s"]]
                starts_preemptive = [result["s_preemptive"]]
                duration_preemptive = [result["d_preemptive"]]
                objectives_cp.append(result["objective"])

        for i in range(len(starts)):
            rcpsp_schedule = {}
            for k in range(len(starts_preemptive[i])):
                starts_k = []
                ends_k = []
                for j in range(len(starts_preemptive[i][k])):
                    if j == 0 or duration_preemptive[i][k][j] != 0:
                        starts_k.append(starts_preemptive[i][k][j])
                        ends_k.append(starts_k[-1] + duration_preemptive[i][k][j])
                rcpsp_schedule[self.rcpsp_model.tasks_list[k]] = {
                    "starts": starts_k,
                    "ends": ends_k,
                }
            sol = RCPSPSolutionPreemptive(
                problem=self.rcpsp_model,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_schedule_feasible=True,
            )
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            if objectives_cp[i] < best_objectives_cp:
                index_best = i
                best_objectives_cp = objectives_cp[i]
            list_solutions_fit.append((sol, objective))
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

    def constraint_start_time_precomputed(self):
        intervals = precompute_posssible_starting_time_interval(self.rcpsp_model)
        list_strings = []
        for t in intervals:
            if self.rcpsp_model.mode_details[t][1]["duration"] == 0:
                continue
            s = "constraint "
            s_list = []
            for interv in intervals[t]:
                if interv[0] is not None and interv[1] is not None:
                    s_list += [
                        "(("
                        + str(interv[0])
                        + "<=s["
                        + str(self.index_in_minizinc[t])
                        + "])/\("
                        + "s["
                        + str(self.index_in_minizinc[t])
                        + "]<="
                        + str(interv[1] - 1)
                        + "))"
                    ]
                if interv[0] is not None and interv[1] is None:
                    s_list += [
                        "("
                        + str(interv[0])
                        + " <=s["
                        + str(self.index_in_minizinc[t])
                        + "])"
                    ]
            if len(s_list) > 0:
                s = s + "\/".join(s_list) + ";\n"
                list_strings += [s]
        return list_strings

    def constraint_objective_makespan(self):
        s = """constraint forall ( i in Tasks where suc[i] == {} )
                (s_preemptive[i, nb_preemptive] + d_preemptive[i, nb_preemptive] <= objective);\n"""
        return [s]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint s_preemptive["
                + str(self.index_in_minizinc[j])
                + ", nb_preemptive]<=objective;\n"
            ]
        return s

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
        weights = [10 if s == self.rcpsp_model.sink_task else 1 for s in set_subtasks]
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
        weights = [10 if s == self.rcpsp_model.sink_task else 1 for s in set_subtasks]

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


class CP_MRCPSP_MZN_PREEMMPTIVE(CP_RCPSP_MZN_PREEMMPTIVE):
    def init_model(self, **args):
        model_type = args.get("model_type", "multi-preemptive")
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        if model_type is None:
            model_type = (
                "multi-preemptive" if not self.calendar else "multi-preemptive-calendar"
            )
        keys = []
        model = Model(files_mzn[model_type])
        custom_output_type = args.get("output_type", True)
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        keys += ["add_objective_makespan", "ignore_sec_objective"]
        instance["nb_preemptive"] = args.get("nb_preemptive", 2)
        self.nb_preemptive = instance["nb_preemptive"]
        keys += ["nb_preemptive"]
        instance["possibly_preemptive"] = args.get(
            "possibly_preemptive",
            [
                self.rcpsp_model.can_be_preempted(task)
                for task in self.rcpsp_model.tasks_list
            ],
        )
        keys += ["possibly_preemptive"]
        instance["max_preempted"] = args.get(
            "max_preempted", min(self.rcpsp_model.n_jobs_non_dummy, 5)
        )
        keys += ["max_preempted"]

        n_res = len(list(self.rcpsp_model.resources.keys()))
        instance["n_res"] = n_res
        keys += ["n_res"]

        if model_type != "multi-preemptive-calendar":
            max_time = args.get("max_time", self.rcpsp_model.horizon)
            fake_tasks = args.get(
                "fake_tasks", True
            )  # to modelize varying quantity of resource.
            dict_to_add = add_fake_task_cp_data(
                rcpsp_model=self.rcpsp_model,
                ignore_fake_task=not fake_tasks,
                max_time_to_consider=max_time,
            )
            instance["max_time"] = max_time
            keys += ["max_time"]
            for key in dict_to_add:
                instance[key] = dict_to_add[key]
                keys += [key]

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
        self.mode_dict_task_mode_to_index_minizinc = {}
        for ind in self.modeindex_map:
            task = self.modeindex_map[ind]["task"]
            mode = self.modeindex_map[ind]["original_mode_index"]
            self.mode_dict_task_mode_to_index_minizinc[(task, mode)] = ind
        modes = [set() for t in sorted_tasks]
        for j in self.modeindex_map:
            modes[self.rcpsp_model.index_task[self.modeindex_map[j]["task"]]].add(j)
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["n_opt"] = len(all_modes)
        keys += ["n_opt"]
        instance["dur"] = dur
        keys += ["dur"]
        sorted_resources = self.rcpsp_model.resources_list
        self.resources_index = sorted_resources
        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in sorted_resources
        ]
        instance["rreq"] = rreq
        keys += ["rreq"]
        rtype = [
            2 if res in self.rcpsp_model.non_renewable_resources else 1
            for res in sorted_resources
        ]
        instance["rtype"] = rtype
        keys += ["rtype"]
        if not self.calendar:
            rc = [int(self.rcpsp_model.resources[x]) for x in sorted_resources]
        else:
            rc = [int(max(self.rcpsp_model.resources[x])) for x in sorted_resources]
        if self.calendar and model_type == "multi-preemptive-calendar":
            one_ressource = list(self.rcpsp_model.resources.keys())[0]
            instance["max_time"] = len(self.rcpsp_model.resources[one_ressource])
            keys += ["max_time"]
            ressource_capacity_time = [
                [int(x) for x in self.rcpsp_model.resources[res]]
                for res in sorted_resources
            ]
            instance["ressource_capacity_time"] = ressource_capacity_time
            keys += ["ressource_capacity_time"]

        instance["rc"] = rc
        keys += ["rc"]

        n_tasks = self.rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]

        suc = [
            set(
                [
                    self.rcpsp_model.return_index_task(x, offset=1)
                    for x in self.rcpsp_model.successors[task]
                ]
            )
            for task in sorted_tasks
        ]
        instance["suc"] = suc
        keys += ["suc"]
        self.instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        self.name_penalty = []

        if add_partial_solution_hard_constraint:
            strings = add_hard_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
        else:
            strings, name_penalty = add_soft_special_constraints(p_s, self)
            for s in strings:
                self.instance.add_string(s)
            self.name_penalty = name_penalty
            strings = define_second_part_objective(
                [100] * len(name_penalty), name_penalty, equal=False
            )
            for s in strings:
                self.instance.add_string(s)

    def constraint_task_to_mode(self, task_id, mode):
        modes = self.rcpsp_model.mode_details[task_id].keys()
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

    def constraint_ressource_requirement_at_time_t(
        self, time, ressource, ressource_number, sign: SignEnum = SignEnum.LEQ
    ):
        index_ressource = self.resources_index.index(ressource) + 1
        s = (
            """constraint """
            + str(ressource_number)
            + str(sign.value)
            + """sum( i in Tasks, j in PREEMPTIVE) (
                                            bool2int(s_preemptive[i, j] <="""
            + str(time)
            + """ /\ """
            + str(time)
            + """< s_preemptive[i, j] + d_preemptive[i, j]) * arreq["""
            + str(index_ressource)
            + """,i]);\n"""
        )
        return s

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        best_objectives_cp = float("inf")
        list_solutions_fit = []
        starts = []
        starts_preemptive = []
        duration_preemptive = []
        modes = []
        objectives_cp = []
        index_best = 0
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts.append(result[i].dict["s"])
                    starts_preemptive.append(result[i].dict["s_preemptive"])
                    duration_preemptive.append(result[i].dict["d_preemptive"])
                    modes.append(result[i].dict["mrun"])
                    objectives_cp.append(result[i].objective)
                else:
                    starts.append(result[i, "s"])
                    starts_preemptive.append(result[i, "s_preemptive"])
                    duration_preemptive.append(result[i, "d_preemptive"])
                    modes.append(result[i, "mrun"])
                    objectives_cp.append(result[i, "objective"])
        else:
            if isinstance(result, RCPSPSolCP):
                starts.append(result.dict["s"])
                starts_preemptive.append(result.dict["s_preemptive"])
                duration_preemptive.append(result.dict["d_preemptive"])
                modes.append(result.dict["mrun"])
                objectives_cp.append(result.objective)
            else:
                starts = [result["s"]]
                starts_preemptive = [result["s_preemptive"]]
                duration_preemptive = [result["d_preemptive"]]
                modes.append(result["mrun"])
                objectives_cp.append(result["objective"])
        for i in range(len(starts)):
            rcpsp_schedule = {}
            modes_dict = {}
            for k in range(len(modes[i])):
                if modes[i][k]:
                    modes_dict[self.modeindex_map[k + 1]["task"]] = self.modeindex_map[
                        k + 1
                    ]["original_mode_index"]
            for k in range(len(starts_preemptive[i])):
                starts_k = []
                ends_k = []
                for j in range(len(starts_preemptive[i][k])):
                    if j == 0 or duration_preemptive[i][k][j] != 0:
                        starts_k.append(starts_preemptive[i][k][j])
                        ends_k.append(starts_k[-1] + duration_preemptive[i][k][j])
                rcpsp_schedule[self.rcpsp_model.tasks_list[k]] = {
                    "starts": starts_k,
                    "ends": ends_k,
                }
            sol = RCPSPSolutionPreemptive(
                problem=self.rcpsp_model,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_modes=[
                    modes_dict[t] for t in self.rcpsp_model.tasks_list_non_dummy
                ],
                rcpsp_schedule_feasible=True,
            )
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            if objectives_cp[i] < best_objectives_cp:
                index_best = i
                best_objectives_cp = objectives_cp[i]
            list_solutions_fit.append((sol, objective))
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


class MRCPSP_Result:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        self.mode_chosen = json.loads(_output_item)

    def check(self) -> bool:
        return True


class CP_MRCPSP_MZN_NOBOOL(MinizincCPSolver):
    def __init__(
        self,
        rcpsp_model: RCPSPModel,
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
        self.calendar = False
        if isinstance(self.rcpsp_model, RCPSPModelCalendar):
            self.calendar = True

    def init_model(self, **args):
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get(
            "add_partial_solution_hard_constraint", True
        )
        model = Model(files_mzn["multi-no-bool"])
        model.output_type = MRCPSP_Result
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        resources_list = list(self.rcpsp_model.resources.keys())
        instance = Instance(solver, model)
        keys = []
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        keys += ["add_objective_makespan", "ignore_sec_objective"]
        n_res = len(resources_list)
        instance["n_res"] = n_res
        keys += ["n_res"]
        n_tasks = self.rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        keys += ["n_tasks"]
        sorted_tasks = self.rcpsp_model.tasks_list
        n_opt = sum([len(self.rcpsp_model.mode_details[key]) for key in sorted_tasks])
        instance["n_opt"] = n_opt
        keys += ["n_opt"]
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
        dur = [x[2]["duration"] for x in all_modes]
        instance["modes"] = modes
        keys += ["modes"]
        instance["dur"] = dur
        keys += ["dur"]
        rreq = [
            [all_modes[i][2].get(res, 0) for i in range(len(all_modes))]
            for res in resources_list
        ]
        instance["rreq"] = rreq
        keys += ["rreq"]

        if not self.calendar:
            rcap = [self.rcpsp_model.resources[x] for x in resources_list]
        else:
            rcap = [int(max(self.rcpsp_model.resources[x])) for x in resources_list]
        instance["rcap"] = rcap
        keys += ["rcap"]

        rtype = [
            2 if res in self.rcpsp_model.non_renewable_resources else 1
            for res in resources_list
        ]

        instance["rtype"] = rtype
        keys += ["rtype"]

        succ = [set(self.rcpsp_model.successors[task]) for task in sorted_tasks]

        instance["succ"] = succ
        keys += ["succ"]

        if self.calendar:
            one_ressource = list(self.rcpsp_model.resources.keys())[0]
            instance["max_time"] = len(self.rcpsp_model.resources[one_ressource])
            keys += ["max_time"]
            ressource_capacity_time = [
                [int(x) for x in self.rcpsp_model.resources[res]]
                for res in resources_list
            ]
            instance["ressource_capacity_time"] = ressource_capacity_time
            keys += ["ressource_capacity_time"]

        self.instance = instance
        self.index_in_minizinc = {
            task: self.rcpsp_model.return_index_task(task, offset=1)
            for task in self.rcpsp_model.tasks_list
        }
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        if p_s is not None and add_partial_solution_hard_constraint:
            constraint_strings = []
            if p_s.start_times is not None:
                for task in p_s.start_times:
                    string = (
                        "constraint start["
                        + str(self.index_in_minizinc[task])
                        + "] == "
                        + str(p_s.start_times[task])
                        + ";\n"
                    )
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.partial_permutation is not None:
                for t1, t2 in zip(
                    p_s.partial_permutation[:-1], p_s.partial_permutation[1:]
                ):
                    string = (
                        "constraint start["
                        + str(self.index_in_minizinc[t1])
                        + "] <= start["
                        + str(self.index_in_minizinc[t2])
                        + "];\n"
                    )
                    self.instance.add_string(string)
                    constraint_strings += [string]
            if p_s.list_partial_order is not None:
                for l in p_s.list_partial_order:
                    for t1, t2 in zip(l[:-1], l[1:]):
                        string = (
                            "constraint start["
                            + str(self.index_in_minizinc[t1])
                            + "] <= start["
                            + str(self.index_in_minizinc[t2])
                            + "];\n"
                        )
                        self.instance.add_string(string)
                        constraint_strings += [string]
            if p_s.task_mode is not None:
                for task in p_s.start_times:
                    indexes = [
                        i
                        for i in self.modeindex_map
                        if self.modeindex_map[i]["task"] == task
                        and self.modeindex_map[i]["original_mode_index"]
                        == p_s.task_mode[task]
                    ]
                    if len(indexes) >= 0:
                        string = "constraint mrun[" + str(indexes[0]) + "] == 1;"
                        self.instance.add_string(string)
                        constraint_strings += [string]

    def constraint_objective_makespan(self):
        s = """constraint forall ( i in Act where suc[i] == {} )
                (start[i] + adur[i] <= objective);\n"""
        return [s]

    def constraint_objective_equal_makespan(self, task_sink):
        ind = self.index_in_minizinc[task_sink]
        s = "constraint (start[" + str(ind) + "]+adur[" + str(ind) + "]==objective);\n"
        return [s]

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        s = []
        for j in set_of_jobs:
            s += [
                "constraint start["
                + str(self.index_in_minizinc[j])
                + "]+adur["
                + str(self.index_in_minizinc[j])
                + "]<=objective;\n"
            ]
        return s

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

    def retrieve_solutions(self, result, parameters_cp: ParametersCP):
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        object_result: List[MRCPSP_Result] = []
        if intermediate_solutions:
            for i in range(len(result)):
                object_result.append(result[i])
        else:
            object_result.append(result)
        for res in object_result:
            modes_dict = {}
            for j in range(len(res.mode_chosen)):
                modes_dict[
                    self.modeindex_map[res.mode_chosen[j]]["task"]
                ] = self.modeindex_map[res.mode_chosen[j]]["original_mode_index"]
            rcpsp_schedule = {}
            start_times = res.dict["start"]
            for i in range(len(start_times)):
                t = self.rcpsp_model.tasks_list[i]
                rcpsp_schedule[t] = {
                    "start_time": start_times[i],
                    "end_time": start_times[i]
                    + self.rcpsp_model.mode_details[t][modes_dict[t]]["duration"],
                }
            sol = RCPSPSolution(
                problem=self.rcpsp_model,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_modes=[
                    modes_dict[t] for t in self.rcpsp_model.tasks_list_non_dummy
                ],
                rcpsp_schedule_feasible=True,
            )
            objective = self.aggreg_from_dict_values(self.rcpsp_model.evaluate(sol))
            if objective > best_makespan:
                best_makespan = objective
                best_solution = sol.copy()
            list_solutions_fit.append((sol, objective))
        result_storage = ResultStorage(
            list_solution_fits=list_solutions_fit,
            best_solution=best_solution,
            mode_optim=self.params_objective_function.sense_function,
            limit_store=False,
        )
        return result_storage


class CP_MRCPSP_MZN_MODES:
    def __init__(
        self,
        rcpsp_model: RCPSPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        self.rcpsp_model = rcpsp_model
        self.instance: Instance = None
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

    def init_model(self, **args):
        model = Model(files_mzn["modes"])
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)

        n_res = len(list(self.rcpsp_model.resources.keys()))
        instance["n_res"] = n_res

        n_tasks = self.rcpsp_model.n_jobs_non_dummy + 2
        instance["n_tasks"] = n_tasks
        sorted_tasks = sorted(self.rcpsp_model.mode_details.keys())
        n_opt = sum(
            [
                len(list(self.rcpsp_model.mode_details[key].keys()))
                for key in sorted_tasks
            ]
        )
        instance["n_opt"] = n_opt

        modes = []
        counter = 0
        self.modeindex_map = {}

        for act in sorted_tasks:
            tmp = list(self.rcpsp_model.mode_details[act].keys())
            for i in range(len(tmp)):
                original_mode_index = tmp[i]
                mod_index = counter + tmp[i]
                tmp[i] = mod_index
                self.modeindex_map[mod_index] = {
                    "task": act,
                    "original_mode_index": original_mode_index,
                }
            modes.append(set(tmp))
            counter = tmp[-1]
        instance["modes"] = modes

        rreq = []
        index = 0
        for res in self.rcpsp_model.resources.keys():
            rreq.append([])
            for task in sorted_tasks:
                for mod in self.rcpsp_model.mode_details[task].keys():
                    rreq[index].append(
                        int(self.rcpsp_model.mode_details[task][mod][res])
                    )
            index += 1
        instance["rreq"] = rreq
        rcap = [val for val in self.rcpsp_model.resources.values()]
        if isinstance(rcap[0], list):
            rcap = [int(max(r)) for r in rcap]
        instance["rcap"] = rcap

        rtype = [
            2 if res in self.rcpsp_model.non_renewable_resources else 1
            for res in self.rcpsp_model.resources.keys()
        ]
        instance["rtype"] = rtype
        self.instance: Instance = instance
        p_s: Optional[PartialSolution] = args.get("partial_solution", None)
        if p_s is not None:
            constraint_strings = []
            if p_s.task_mode is not None:
                for task in p_s.start_times:
                    indexes = [
                        i
                        for i in self.modeindex_map
                        if self.modeindex_map[i]["task"] == task
                        and self.modeindex_map[i]["original_mode_index"]
                        == p_s.task_mode[task]
                    ]
                    if len(indexes) >= 0:
                        logger.debug(f"Index found : {len(indexes)}")
                        string = "constraint mrun[" + str(indexes[0]) + "] == 1;"
                        self.instance.add_string(string)
                        constraint_strings += [string]

    def retrieve_solutions(self, result, parameters_cp: ParametersCP):
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        intermediate_solutions = parameters_cp.intermediate_solution
        mruns = []
        if intermediate_solutions:
            for i in range(len(result)):
                mruns.append(result[i, "mrun"])
        else:
            mruns.append(result["mrun"])
        all_modes = []
        for mrun in mruns:
            modes = [1] * (self.rcpsp_model.n_jobs_non_dummy + 2)
            for i in range(len(mrun)):
                if (
                    mrun[i] == 1
                    and (self.modeindex_map[i + 1]["task"] != 1)
                    and (
                        self.modeindex_map[i + 1]["task"]
                        != self.rcpsp_model.n_jobs_non_dummy + 2
                    )
                ):
                    modes[self.modeindex_map[i + 1]["task"] - 1] = self.modeindex_map[
                        i + 1
                    ]["original_mode_index"]
            all_modes.append(modes)
        return all_modes

    def solve(self, parameters_cp: Optional[ParametersCP] = None, **args):
        if self.instance is None:
            self.init_model(**args)
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        timeout = parameters_cp.time_limit
        intermediate_solutions = parameters_cp.intermediate_solution
        result = self.instance.solve(
            timeout=timedelta(seconds=timeout),
            nr_solutions=parameters_cp.nr_solutions
            if not parameters_cp.all_solutions
            else None,
            all_solutions=parameters_cp.all_solutions,
            intermediate_solutions=intermediate_solutions,
        )
        logger.debug(result.status)
        return self.retrieve_solutions(result=result, parameters_cp=parameters_cp)


def precompute_posssible_starting_time_interval(problem: RCPSPModelPreemptive):
    interval_possible = {t: [] for t in problem.tasks_list}
    for task in interval_possible:
        time = 0
        while time is not None:
            time_1 = next(
                (
                    t
                    for t in range(time, problem.horizon)
                    if all(
                        problem.get_resource_available(r, t)
                        >= problem.mode_details[task][1].get(r, 0)
                        for r in problem.resources
                    )
                ),
                None,
            )
            time = time_1
            if time_1 is not None:
                time_2 = next(
                    (
                        t
                        for t in range(time_1, problem.horizon)
                        if any(
                            problem.get_resource_available(r, t)
                            < problem.mode_details[task][1].get(r, 0)
                            for r in problem.resources
                        )
                    ),
                    None,
                )
                time = time_2
            else:
                time_2 = None
            interval_possible[task] += [(time_1, time_2)]
    return interval_possible


def hard_start_times(
    dict_start_times: Dict[Hashable, int],
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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


def hard_start_times_mrcpsp(
    dict_start_times: Dict[Hashable, int],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    constraint_strings = []
    for task in dict_start_times:
        string = (
            "constraint start["
            + str(cp_solver.index_in_minizinc[task])
            + "] == "
            + str(dict_start_times[task])
            + ";\n"
        )
        constraint_strings += [string]
    return constraint_strings


def soft_start_times_mrcpsp(
    dict_start_times: Dict[Hashable, int],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    list_task = list(dict_start_times.keys())
    s = (
        """
        var 0..max_time*nb_start_times: penalty_start_times;\n
        int: nb_start_times="""
        + str(len(list_task))
        + """;\n"""
        + """
        array[1..nb_start_times] of Act: st1_1="""
        + str([cp_solver.index_in_minizinc[t1] for t1 in list_task])
        + """;\n"""
        + """
        array[1..nb_start_times] of 0..max_time: array_start_1="""
        + str([dict_start_times[t1] for t1 in list_task])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_times)(abs(start[st1_1[i]]-array_start_1[i]))==penalty_start_times;\n"""
    )
    return [s], ["penalty_start_times"]


def soft_start_together_mrcpsp(
    list_start_together: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    s = (
        """
        var 0..max_time*nb_start_together: penalty_start_together;\n
        int: nb_start_together="""
        + str(len(list_start_together))
        + """;\n"""
        + """
        array[1..nb_start_together] of Act: st1_2="""
        + str([cp_solver.index_in_minizinc[t1] for t1, t2 in list_start_together])
        + """;\n"""
        + """
        array[1..nb_start_together] of Act: st2_2="""
        + str([cp_solver.index_in_minizinc[t2] for t1, t2 in list_start_together])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_together)(abs(start[st1_2[i]]-start[st2_2[i]]))==penalty_start_together;\n
        """
    )
    return [s], ["penalty_start_together"]


def hard_start_together_mrcpsp(
    list_start_together: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    constraint_strings = []
    for t1, t2 in list_start_together:
        string = (
            "constraint start["
            + str(cp_solver.index_in_minizinc[t1])
            + "] == start["
            + str(cp_solver.index_in_minizinc[t2])
            + "];\n"
        )
        constraint_strings += [string]
    return constraint_strings


def soft_start_together(
    list_start_together: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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


def hard_start_after_nunit_mrcpsp(
    list_start_after_nunit: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    constraint_strings = []
    for t1, t2, delta in list_start_after_nunit:
        string = (
            "constraint start["
            + str(cp_solver.index_in_minizinc[t2])
            + "] >= start["
            + str(cp_solver.index_in_minizinc[t1])
            + "]+"
            + str(delta)
            + ";\n"
        )
        constraint_strings += [string]
    return constraint_strings


def soft_start_after_nunit_mrcpsp(
    list_start_after_nunit: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    s = (
        """
        var 0..max_time*nb_start_after_nunit: penalty_start_after_nunit;\n
        int: nb_start_after_nunit="""
        + str(len(list_start_after_nunit))
        + """;\n"""
        + """
        array[1..nb_start_after_nunit] of Act: st1_5="""
        + str(
            [
                cp_solver.index_in_minizinc[t1]
                for t1, t2, delta in list_start_after_nunit
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_after_nunit] of Act: st2_5="""
        + str(
            [
                cp_solver.index_in_minizinc[t2]
                for t1, t2, delta in list_start_after_nunit
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_after_nunit] of int: nunit_5="""
        + str([delta for t1, t2, delta in list_start_after_nunit])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_after_nunit)(max([0, start[st1_5[i]]+nunits_5[i]-start[st2_5[i]]]))==penalty_start_after_nunit;\n
        """
    )
    return [s], ["penalty_start_after_nunit"]


def hard_start_at_end_plus_offset_mrcpsp(
    list_start_at_end_plus_offset,
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    constraint_strings = []
    for t1, t2, delta in list_start_at_end_plus_offset:
        string = (
            "constraint start["
            + str(cp_solver.index_in_minizinc[t2])
            + "] >= start["
            + str(cp_solver.index_in_minizinc[t1])
            + "]+adur["
            + str(cp_solver.index_in_minizinc[t1])
            + "]+"
            + str(delta)
            + ";\n"
        )
        constraint_strings += [string]
    return constraint_strings


def hard_start_at_end_plus_offset(
    list_start_at_end_plus_offset,
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
):
    constraint_strings = []
    for t1, t2, delta in list_start_at_end_plus_offset:
        if isinstance(cp_solver, CP_RCPSP_MZN):
            string = (
                "constraint s["
                + str(cp_solver.index_in_minizinc[t2])
                + "] >= s["
                + str(cp_solver.index_in_minizinc[t1])
                + "]+d["
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
            )  # WARNING , i don't put the last duration d[i, nb_preemptive] that should be zero
        constraint_strings += [string]
    return constraint_strings


def soft_start_at_end_plus_offset_mrcpsp(
    list_start_at_end_plus_offset: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    s = (
        """
        var 0..max_time*nb_start_at_end_plus_offset: penalty_start_at_end_plus_offset;\n
        int: nb_start_at_end_plus_offset="""
        + str(len(list_start_at_end_plus_offset))
        + """;\n"""
        + """
        array[1..nb_start_at_end_plus_offset] of Act: st1_6="""
        + str(
            [
                cp_solver.index_in_minizinc[t1]
                for t1, t2, delta in list_start_at_end_plus_offset
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_at_end_plus_offset] of Act: st2_6="""
        + str(
            [
                cp_solver.index_in_minizinc[t2]
                for t1, t2, delta in list_start_at_end_plus_offset
            ]
        )
        + """;\n"""
        + """
        array[1..nb_start_at_end_plus_offset] of int: nunits_6="""
        + str([delta for t1, t2, delta in list_start_at_end_plus_offset])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_at_end_plus_offset)(max([0, start[st1_6[i]]+adur[st1_6[i]+nunits_6[i]-start[st2_6[i]]]))==penalty_start_at_end_plus_offset;\n
        """
    )
    return [s], ["penalty_start_at_end_plus_offset"]


def soft_start_at_end_plus_offset(
    list_start_at_end_plus_offset: List[Tuple[Hashable, Hashable, int]],
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    if isinstance(cp_solver, CP_RCPSP_MZN):
        s += """
             constraint sum(i in 1..nb_start_at_end_plus_offset)(max([0, s[st1_7[i]]+d[st1_7[i]]+nunits[i]-s[st2_7[i]]]))==penalty_start_at_end_plus_offset;\n
             """
    else:
        s += """
             constraint sum(i in 1..nb_start_at_end_plus_offset)(max([0, s_preemptive[st1_7[i], nb_preemptive]+nunits[i]-s[st2_7[i]]]))==penalty_start_at_end_plus_offset;\n
             """
    return [s], ["penalty_start_at_end_plus_offset"]


def hard_start_at_end_mrcpsp(
    list_start_at_end: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    constraint_strings = []
    for t1, t2 in list_start_at_end:
        string = (
            "constraint start["
            + str(cp_solver.index_in_minizinc[t2])
            + "] == start["
            + str(cp_solver.index_in_minizinc[t1])
            + "]+adur["
            + str(cp_solver.index_in_minizinc[t1])
            + "];\n"
        )
        constraint_strings += [string]
    return constraint_strings


def soft_start_at_end_mrcpsp(
    list_start_at_end: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    s = (
        """
        var 0..max_time*nb_start_at_end: penalty_start_at_end;\n
        int: nb_start_at_end="""
        + str(len(list_start_at_end))
        + """;\n"""
        + """
        array[1..nb_start_at_end] of Act: st1_8="""
        + str([cp_solver.index_in_minizinc[t1] for t1, t2 in list_start_at_end])
        + """;\n"""
        + """
        array[1..nb_start_at_end] of Act: st2_8="""
        + str([cp_solver.index_in_minizinc[t2] for t1, t2 in list_start_at_end])
        + """;\n"""
        + """
        constraint sum(i in 1..nb_start_at_end)(abs(start[st2_8[i]]-start[st1_8[i]]-adur[st1_8[i]]))==penalty_start_at_end;\n
        """
    )
    return [s], ["penalty_start_at_end"]


def hard_start_at_end(
    list_start_at_end: List[Tuple[Hashable, Hashable]],
    cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN_PREEMMPTIVE],
):
    constraint_strings = []
    for t1, t2 in list_start_at_end:
        if isinstance(cp_solver, CP_RCPSP_MZN):
            string = (
                "constraint s["
                + str(cp_solver.index_in_minizinc[t2])
                + "] == s["
                + str(cp_solver.index_in_minizinc[t1])
                + "]+d["
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
    cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN_PREEMMPTIVE],
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
    if isinstance(cp_solver, (CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE)):
        s += """
            %constraint forall(i in 1..nb_start_at_end)(s[st2_9[i]]-s_preemptive[st1_9[i], nb_preemptive]>=0);\n
            constraint sum(i in 1..nb_start_at_end)(abs(s[st2_9[i]]-s_preemptive[st1_9[i], nb_preemptive]))==penalty_start_at_end;\n
            """
    else:
        s += """
            %constraint forall(i in 1..nb_start_at_end)(s[st2_9[i]]-s[st1_9[i]]-d[st1_0[i]]>=0);\n
            constraint sum(i in 1..nb_start_at_end)(abs(s[st2_9[i]]-s[st1_9[i]]-d[st1_9[i]]))==penalty_start_at_end;\n
            """
    return [s], ["penalty_start_at_end"]


def soft_start_window(
    start_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN_PREEMMPTIVE],
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
    max_time_start = max(
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
        int: max_time_start="""
        + str(max_time_start)
        + """;\n
        array[1..nb_start_window_low] of Tasks: task_id_low_start="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_low])
        + """;\n
        array[1..nb_start_window_up] of Tasks:  task_id_up_start="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_up])
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
    cp_solver: Union[CP_RCPSP_MZN, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN_PREEMMPTIVE],
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
    max_time_end = max(
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
        int: max_time_end="""
        + str(max_time_end)
        + """;\n
        array[1..nb_end_window_low] of Tasks: task_id_low_end="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_low])
        + """;\n
        array[1..nb_end_window_up] of Tasks:  task_id_up_end="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_up])
        + """;\n
        array[1..nb_end_window_low] of 0..max_time_end: times_low_end="""
        + str([int(x[1]) for x in l_low])
        + """;\n
        array[1..nb_end_window_up] of 0..max_time_end: times_up_end="""
        + str([int(x[1]) for x in l_up])
        + """;\n"""
    )
    if isinstance(cp_solver, (CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE)):
        s += """
            constraint sum(i in 1..nb_end_window_low)(max([times_low_end[i]-s_preemptive[task_id_low_end[i], nb_preemptive], 0]))==penalty_end_low;\n
            constraint sum(i in 1..nb_end_window_up)(max([-times_up_end[i]+s_preemptive[task_id_up_end[i], nb_preemptive], 0]))==penalty_end_up;\n
            """
    else:
        s += """
            constraint sum(i in 1..nb_end_window_low)(max([times_low_end[i]-s[task_id_low_end[i]]+d[task_id_low_end[i]], 0]))==penalty_end_low;\n
            constraint sum(i in 1..nb_end_window_up)(max([-times_up_end[i]+s[task_id_up_end[i]]+d[task_id_up_end[i]], 0]))==penalty_end_up;\n
            """

    return [s], ["penalty_end_low", "penalty_end_up"]


def soft_start_window_mrcpsp(
    start_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
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
    max_time_start = max(
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
        int: max_time_start="""
        + str(max_time_start)
        + """;\n
        array[1..nb_start_window_low] of Tasks: task_id_low_start="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_low])
        + """;\n
        array[1..nb_start_window_up] of Tasks:  task_id_up_start="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_up])
        + """;\n
        array[1..nb_start_window_low] of 0..max_time_start: times_low_start="""
        + str([int(x[1]) for x in l_low])
        + """;\n
        array[1..nb_start_window_up] of 0..max_time_start: times_up_start="""
        + str([int(x[1]) for x in l_up])
        + """;\n"""
    )
    s += """
        constraint sum(i in 1..nb_start_window_low)(max([times_low_start[i]-start[task_id_low_start[i]], 0]))==penalty_start_low;\n
        constraint sum(i in 1..nb_start_window_up)(max([-times_up_start[i]+start[task_id_up_start[i]], 0]))==penalty_start_up;\n
        """
    return [s], ["penalty_start_low", "penalty_start_up"]


def soft_end_window_mrcpsp(
    end_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
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
    max_time_end = max(
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
        int: max_time_end="""
        + str(max_time_end)
        + """;\n
        array[1..nb_end_window_low] of Tasks: task_id_low_end="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_low])
        + """;\n
        array[1..nb_end_window_up] of Tasks:  task_id_up_end="""
        + str([cp_solver.index_in_minizinc[x[0]] for x in l_up])
        + """;\n
        array[1..nb_end_window_low] of 0..max_time_end: times_low_end="""
        + str([int(x[1]) for x in l_low])
        + """;\n
        array[1..nb_end_window_up] of 0..max_time_end: times_up_end="""
        + str([int(x[1]) for x in l_up])
        + """;\n"""
    )
    s += """
        constraint sum(i in 1..nb_end_window_low)(max([times_low_end[i]-start[task_id_low_end[i]]+adur[task_id_low_end[i]], 0]))==penalty_end_low;\n
        constraint sum(i in 1..nb_end_window_up)(max([-times_up_end[i]+start[task_id_up_end[i]]+adur[task_id_up_end[i]], 0]))==penalty_end_up;\n
        """
    return [s], ["penalty_end_low", "penalty_end_up"]


def hard_start_window(
    start_times_window: Dict[Hashable, Tuple[int, int]],
    cp_solver: Union[
        CP_RCPSP_MZN,
        CP_MRCPSP_MZN_PREEMMPTIVE,
        CP_RCPSP_MZN_PREEMMPTIVE,
        CP_MRCPSP_MZN,
        CP_MRCPSP_MZN_WITH_FAKE_TASK,
        CP_MRCPSP_MZN_NOBOOL,
    ],
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
    cp_solver: Union[
        CP_RCPSP_MZN,
        CP_MRCPSP_MZN_PREEMMPTIVE,
        CP_RCPSP_MZN_PREEMMPTIVE,
        CP_MRCPSP_MZN,
        CP_MRCPSP_MZN_WITH_FAKE_TASK,
        CP_MRCPSP_MZN_NOBOOL,
    ],
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


def add_hard_special_constraints_mrcpsp(
    partial_solution: PartialSolution,
    cp_solver: Union[CP_MRCPSP_MZN, CP_MRCPSP_MZN_WITH_FAKE_TASK, CP_MRCPSP_MZN_NOBOOL],
):
    if partial_solution is not None:
        constraint_strings = []
        if partial_solution.start_times is not None:
            constraint_strings += hard_start_times_mrcpsp(
                dict_start_times=partial_solution.start_times, cp_solver=cp_solver
            )
        if partial_solution.partial_permutation is not None:
            for t1, t2 in zip(
                partial_solution.partial_permutation[:-1],
                partial_solution.partial_permutation[1:],
            ):
                string = (
                    "constraint start["
                    + str(cp_solver.index_in_minizinc[t1])
                    + "] <= start["
                    + str(cp_solver.index_in_minizinc[t2])
                    + "];\n"
                )
                constraint_strings += [string]
        if partial_solution.list_partial_order is not None:
            for l in partial_solution.list_partial_order:
                for t1, t2 in zip(l[:-1], l[1:]):
                    string = (
                        "constraint start["
                        + str(cp_solver.index_in_minizinc[t1])
                        + "] <= start["
                        + str(cp_solver.index_in_minizinc[t2])
                        + "];\n"
                    )
                    constraint_strings += [string]
        if partial_solution.task_mode is not None:
            for task in partial_solution.task_mode:
                indexes = [
                    i
                    for i in cp_solver.modeindex_map
                    if cp_solver.modeindex_map[i]["task"] == task
                    and cp_solver.modeindex_map[i]["original_mode_index"]
                    == partial_solution.task_mode[task]
                ]
                if len(indexes) >= 0:
                    string = "constraint mrun[" + str(indexes[0]) + "] == 1;"
                    constraint_strings += [string]
        if partial_solution.start_together is not None:
            constraint_strings += hard_start_together_mrcpsp(
                list_start_together=partial_solution.start_together, cp_solver=cp_solver
            )
        if partial_solution.start_after_nunit is not None:
            constraint_strings += hard_start_after_nunit_mrcpsp(
                list_start_after_nunit=partial_solution.start_after_nunit,
                cp_solver=cp_solver,
            )
        if partial_solution.start_at_end_plus_offset is not None:
            constraint_strings += hard_start_at_end_plus_offset_mrcpsp(
                list_start_at_end_plus_offset=partial_solution.start_at_end_plus_offset,
                cp_solver=cp_solver,
            )
        if partial_solution.start_at_end is not None:
            constraint_strings += hard_start_at_end_mrcpsp(
                list_start_at_end=partial_solution.start_at_end, cp_solver=cp_solver
            )
        if partial_solution.start_times_window is not None:
            constraint_strings += hard_start_window(
                start_times_window=partial_solution.start_times_window,
                cp_solver=cp_solver,
            )
        if partial_solution.end_times_window is not None:
            constraint_strings += hard_end_window(
                end_times_window=partial_solution.end_times_window, cp_solver=cp_solver
            )
        return constraint_strings

    else:
        return []


def add_soft_special_constraints_mrcpsp(
    partial_solution: PartialSolution, cp_solver: Union[CP_MRCPSP_MZN]
):
    if partial_solution is not None:
        constraint_strings = []
        name_penalty = []
        if partial_solution.start_times is not None:
            c, n = soft_start_times_mrcpsp(
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
                    "constraint start["
                    + str(cp_solver.index_in_minizinc[t1])
                    + "] <= start["
                    + str(cp_solver.index_in_minizinc[t2])
                    + "];\n"
                )
                constraint_strings += [string]
        if partial_solution.list_partial_order is not None:
            for l in partial_solution.list_partial_order:
                for t1, t2 in zip(l[:-1], l[1:]):
                    string = (
                        "constraint start["
                        + str(cp_solver.index_in_minizinc[t1])
                        + "] <= start["
                        + str(cp_solver.index_in_minizinc[t2])
                        + "];\n"
                    )
                    constraint_strings += [string]
        if partial_solution.task_mode is not None:
            for task in partial_solution.task_mode:
                indexes = [
                    i
                    for i in cp_solver.modeindex_map
                    if cp_solver.modeindex_map[i]["task"] == task
                    and cp_solver.modeindex_map[i]["original_mode_index"]
                    == partial_solution.task_mode[task]
                ]
                if len(indexes) >= 0:
                    string = "constraint mrun[" + str(indexes[0]) + "] == 1;"
                    constraint_strings += [string]
        if partial_solution.start_together is not None:
            c, n = soft_start_together_mrcpsp(
                list_start_together=partial_solution.start_together, cp_solver=cp_solver
            )
            constraint_strings += c
            name_penalty += n
        if partial_solution.start_after_nunit is not None:
            c, n = soft_start_after_nunit_mrcpsp(
                list_start_after_nunit=partial_solution.start_after_nunit,
                cp_solver=cp_solver,
            )
            constraint_strings += c
            name_penalty += n
        if partial_solution.start_at_end_plus_offset is not None:
            c, n = soft_start_at_end_plus_offset_mrcpsp(
                list_start_at_end_plus_offset=partial_solution.start_at_end_plus_offset,
                cp_solver=cp_solver,
            )
            constraint_strings += c
            name_penalty += n
        if partial_solution.start_at_end is not None:
            c, n = soft_start_at_end_mrcpsp(
                list_start_at_end=partial_solution.start_at_end, cp_solver=cp_solver
            )
            constraint_strings += c
            name_penalty += n
        if partial_solution.start_times_window is not None:
            c, n = soft_start_window_mrcpsp(
                partial_solution.start_times_window, cp_solver=cp_solver
            )
            constraint_strings += c
            name_penalty += n
        if partial_solution.end_times_window is not None:
            c, n = soft_end_window_mrcpsp(
                partial_solution.end_times_window, cp_solver=cp_solver
            )
            constraint_strings += c
            name_penalty += n
        return constraint_strings, name_penalty

    else:
        return [], []


def add_hard_special_constraints(
    partial_solution: PartialSolution,
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE, CP_RCPSP_MZN],
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
    s = "constraint sec_objective>=" + sum_string + ";\n"
    if equal:
        s = "constraint sec_objective==" + sum_string + ";\n"
    return [s]


def add_constraints_string(child_instance, list_of_strings):
    for s in list_of_strings:
        child_instance.add_string(s)
