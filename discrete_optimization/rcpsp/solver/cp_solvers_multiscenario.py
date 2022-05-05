from dataclasses import InitVar
from typing import Union, List, Dict, Hashable, Tuple
from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP, CPSolverName, \
    map_cp_solver_name, SignEnum
from discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, ObjectiveHandling, \
    ParamsObjectiveFunction, build_aggreg_function_and_params_objective
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution,\
    SingleModeRCPSPModel, MultiModeRCPSPModel, \
    RCPSPModelCalendar, PartialSolution
from discrete_optimization.rcpsp.rcpsp_model_utils import create_fake_tasks

from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPSolutionPreemptive, RCPSPModelPreemptive,\
    PartialSolutionPreemptive


from minizinc import Instance, Model, Solver, Status, Result
import json
from datetime import timedelta
import os, random
this_path = os.path.dirname(os.path.abspath(__file__))

files_mzn = {"multiscenario": os.path.join(this_path, "../minizinc/rcpsp_multiscenario.mzn")}


class RCPSPSolCP:
    objective: int
    __output_item: InitVar[str] = None

    def __init__(self, objective, _output_item, **kwargs):
        self.objective = objective
        self.dict = kwargs
        print("One solution ", self.objective)
        print("Output ", _output_item)

    def check(self) -> bool:
        return True


def add_fake_task_cp_data(rcpsp_model: Union[RCPSPModel, RCPSPModelPreemptive],
                          ignore_fake_task: bool = True,
                          max_time_to_consider: int = None):
    if rcpsp_model.is_varying_resource() and not ignore_fake_task:
        fake_tasks = create_fake_tasks(rcpsp_problem=rcpsp_model)
        max_time_to_consider = rcpsp_model.horizon if max_time_to_consider is None else max_time_to_consider
        fake_tasks = [f for f in fake_tasks if f["start"] <= max_time_to_consider]
        n_fake_tasks = len(fake_tasks)
        fakestart = [fake_tasks[i]["start"] for i in range(len(fake_tasks))]
        fake_dur = [fake_tasks[i]["duration"] for i in range(len(fake_tasks))]
        max_duration_fake_task = max(fake_dur)
        fake_req = [[fake_tasks[i].get(res, 0) for i in range(len(fake_tasks))]
                    for res in rcpsp_model.resources_list]
        dict_to_add_in_instance = {}
        dict_to_add_in_instance["max_duration_fake_task"] = max_duration_fake_task
        dict_to_add_in_instance["n_fake_tasks"] = n_fake_tasks
        dict_to_add_in_instance["fakestart"] = fakestart
        dict_to_add_in_instance["fakedur"] = fake_dur
        dict_to_add_in_instance["fakereq"] = fake_req
        dict_to_add_in_instance["include_fake_tasks"] = True
        return dict_to_add_in_instance
    else:
        dict_to_add_in_instance = {}
        dict_to_add_in_instance["max_duration_fake_task"] = 0
        dict_to_add_in_instance["n_fake_tasks"] = 0
        dict_to_add_in_instance["fakestart"] = []
        dict_to_add_in_instance["fakedur"] = []
        dict_to_add_in_instance["fakereq"] = [[] for r in rcpsp_model.resources_list]
        dict_to_add_in_instance["include_fake_tasks"] = False
        return dict_to_add_in_instance


class CP_MULTISCENARIO(CPSolver):
    def __init__(self, list_rcpsp_model: List[RCPSPModel],
                 cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.list_rcpsp_model = list_rcpsp_model
        self.instance: Instance = None
        self.cp_solver_name = cp_solver_name
        self.base_rcpsp_model = list_rcpsp_model[0]
        self.key_decision_variable = ["s"]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.base_rcpsp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        model_type = args.get("model_type", "multiscenario")
        max_time = args.get("max_time", self.base_rcpsp_model.horizon)
        fake_tasks = args.get("fake_tasks", True)  # to modelize varying quantity of resource.
        add_objective_makespan = args.get("add_objective_makespan", True)
        ignore_sec_objective = args.get("ignore_sec_objective", True)
        add_partial_solution_hard_constraint = args.get("add_partial_solution_hard_constraint", True)
        custom_output_type = args.get("output_type",  False)
        p_s: Union[PartialSolution, None] = args.get("partial_solution", None)

        model = Model(files_mzn[model_type])
        if custom_output_type:
            model.output_type = RCPSPSolCP
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, model)
        instance["add_objective_makespan"] = add_objective_makespan
        instance["ignore_sec_objective"] = ignore_sec_objective
        n_res = len(self.base_rcpsp_model.resources_list)
        instance["n_res"] = n_res
        instance["n_scenario"] = len(self.list_rcpsp_model)
        dict_to_add = add_fake_task_cp_data(rcpsp_model=self.base_rcpsp_model,
                                            ignore_fake_task=not fake_tasks,
                                            max_time_to_consider=max_time)
        instance["max_time"] = max_time
        for key in dict_to_add:
            instance[key] = dict_to_add[key]

        sorted_resources = self.base_rcpsp_model.resources_list
        self.resources_index = sorted_resources
        rcap = [int(self.base_rcpsp_model.get_max_resource_capacity(x)) for x in sorted_resources]
        instance["rc"] = rcap
        n_tasks = self.base_rcpsp_model.n_jobs
        instance["n_tasks"] = n_tasks
        sorted_tasks = self.base_rcpsp_model.tasks_list
        d = [[int(self.list_rcpsp_model[j].mode_details[key][1]['duration'])
              for j in range(len(self.list_rcpsp_model))]
             for key in sorted_tasks]
        instance["d"] = d
        all_modes = [[(act, 1, self.list_rcpsp_model[j].mode_details[act][1])
                     for act in sorted_tasks]
                     for j in range(len(self.list_rcpsp_model))]
        rr = [[[all_modes[j][i][2].get(res, 0)
                for j in range(len(self.list_rcpsp_model))]
               for i in range(len(all_modes[0]))]
              for res in sorted_resources]
        instance["rr"] = rr
        suc = [set([self.base_rcpsp_model.return_index_task(x, offset=1)
                    for x in self.base_rcpsp_model.successors[task]])
               for task in sorted_tasks]
        instance["suc"] = suc
        instance["relax_ordering"] = args.get("relax_ordering", False)
        instance["nb_incoherence_limit"] = args.get("nb_incoherence_limit", 3)
        self.instance = instance
        self.index_in_minizinc = {task: self.base_rcpsp_model.return_index_task(task, offset=1)
                                  for task in self.base_rcpsp_model.tasks_list}
        self.instance["sink_task"] = self.index_in_minizinc[self.base_rcpsp_model.sink_task]

    def retrieve_solutions(self, result, parameters_cp: ParametersCP = ParametersCP.default()) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        best_solution = None
        best_makespan = -float("inf")
        list_solutions_fit = []
        starts = []
        orderings = []
        objectives = []
        if intermediate_solutions:
            for i in range(len(result)):
                if isinstance(result[i], RCPSPSolCP):
                    starts += [result[i].dict["s"]]
                    orderings += [result[i].dict["ordering"]]
                    objectives += [result[i].objective]
                else:
                    starts += [result[i, "s"]]
                    orderings += [result[i, "ordering"]]
                    objectives += [result[i, "objective"]]
        else:
            if isinstance(result, RCPSPSolCP):
                starts += [result.dict["s"]]
                orderings += [result.dict["ordering"]]
                objectives += [result.objective]
            else:
                starts = [result["s"]]
                orderings += [result["ordering"]]
                objectives += [result.objective]
        l = []
        for order, obj, start in zip(orderings, objectives, starts):
            oo = [self.base_rcpsp_model.index_task_non_dummy[self.base_rcpsp_model.tasks_list[j-1]]
                  for j in order if self.base_rcpsp_model.tasks_list[j-1]
                  in self.base_rcpsp_model.index_task_non_dummy]
            ll = [RCPSPSolution(problem=self.list_rcpsp_model[i],
                                rcpsp_permutation=oo)
                  for i in range(len(self.list_rcpsp_model))]
            eval = [self.aggreg_from_dict_values(self.list_rcpsp_model[i].evaluate(ll[i]))
                    for i in range(len(self.list_rcpsp_model))]
            sum_eval = sum(eval)
            if sum_eval > best_makespan:
                best_solution = tuple(ll)
                best_makespan = sum_eval
            print("starts found : ", start[-1])
            print("Sum of makespan : ",
                  eval,
                  sum_eval/len(self.list_rcpsp_model),
                  order, obj)
            list_solutions_fit += [((tuple(ll), obj), sum_eval)]
        result_storage = ResultStorage(list_solution_fits=list_solutions_fit,
                                       best_solution=best_solution,
                                       mode_optim=self.params_objective_function.sense_function,
                                       limit_store=False)
        return result_storage

    def solve(self, parameters_cp: ParametersCP, **args) -> ResultStorage:
        if self.instance is None:
            self.init_model(**args)
        timeout = parameters_cp.TimeLimit
        intermediate_solutions = parameters_cp.intermediate_solution
        try:
            result = self.instance.solve(timeout=timedelta(seconds=timeout),
                                         intermediate_solutions=intermediate_solutions)
        except Exception as e:
            print(e)
            return None
        verbose = args.get("verbose", True)
        self.result = result
        if verbose:
            print("Status : ", result.status)
            print("Solving time : ", result.statistics.get("solveTime", None))
        return self.retrieve_solutions(result, parameters_cp=parameters_cp)