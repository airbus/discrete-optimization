from typing import Union, Set, Hashable, Dict, List, Iterable, Any
import networkx as nx
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import GraphRCPSPSpecialConstraints, GraphRCPSP
from discrete_optimization.generic_tools.lns_cp import ConstraintHandler
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.rcpsp.rcpsp_solution_utils import get_max_time_solution
from discrete_optimization.rcpsp.solver.cp_solvers import CP_RCPSP_MZN_PREEMMPTIVE, CP_MRCPSP_MZN_PREEMMPTIVE
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPModelCalendar, MultiModeRCPSPModel, \
    SingleModeRCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints \
    import RCPSPModelSpecialConstraints, SpecialConstraintsDescription, \
    RCPSPModelSpecialConstraintsPreemptive, RCPSPSolutionSpecialPreemptive
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive, RCPSPSolutionPreemptive
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant, \
    MS_RCPSPSolution_Variant, MS_RCPSPSolution, MS_RCPSPSolution_Preemptive, employee_usage
from discrete_optimization.rcpsp.rcpsp_model_utils import create_fake_tasks
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import create_fake_tasks_multiskills
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import CP_MS_MRCPSP_MZN_PREEMPTIVE, CP_MS_MRCPSP_MZN
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import BasicConstraintBuilder, \
    ParamsConstraintBuilder, NeighborBuilder, constraint_unit_used_to_tasks_preemptive, \
    constraint_unit_used_subset_employees, constraint_unit_used_subset_employees_preemptive
import numpy as np
from discrete_optimization.generic_rcpsp_tools.neighbor_tools_rcpsp import BasicConstraintBuilder,\
    ParamsConstraintBuilder

ANY_RCPSP = Union[RCPSPModel,
                  RCPSPModelCalendar,
                  MultiModeRCPSPModel,
                  SingleModeRCPSPModel,
                  RCPSPModelPreemptive,
                  RCPSPModelSpecialConstraints,
                  RCPSPModelSpecialConstraintsPreemptive,
                  MS_RCPSPModel,
                  MS_RCPSPModel_Variant]


def compute_shift_extremities(model: ANY_RCPSP):
    ressource_names = model.get_resource_names()
    ressource_arrays = {r: model.get_resource_availability_array(r) for r in ressource_names}
    if isinstance(model, MS_RCPSPModel):
        fake_tasks, fake_tasks_unit = create_fake_tasks_multiskills(model)
        f = fake_tasks+fake_tasks_unit
    else:
        fake_tasks = create_fake_tasks(model)
        f = fake_tasks
    return f


def return_pauses_and_active_times(model: ANY_RCPSP,
                                   solution:
                                   Union[MS_RCPSPSolution_Preemptive,
                                         RCPSPSolutionPreemptive,
                                         RCPSPSolutionSpecialPreemptive]):
    tasks = model.get_tasks_list()
    dictionnary = {}
    for task in tasks:
        starts = solution.get_start_times_list(task)
        ends = solution.get_end_times_list(task)
        durations = [-starts[i]+ends[i] for i in range(len(starts))]
        dictionnary[task] = {"starts": starts, "ends": ends,
                             "durations": durations}
        total_duration = sum(durations)
        dictionnary[task]["total_duration"] = total_duration
        if len(starts) > 1:
            diff = [starts[i+1]-ends[i] for i in range(len(starts)-1)]
            dictionnary[task]["diff_start_end"] = diff
            # for j in range(len(diff)):
            #     if diff[j]<int(durations[j]/10) or diff[j]==1:
    return dictionnary


def find_possible_problems_preemptive(model: ANY_RCPSP,
                                      solution: Union[MS_RCPSPSolution_Preemptive,
                                                      RCPSPSolutionPreemptive,
                                                      RCPSPSolutionSpecialPreemptive]):
    dictionnary = return_pauses_and_active_times(model, solution)
    problems = {}
    for task in dictionnary:
        diff = dictionnary[task].get("diff_start_end", [])
        durations = dictionnary[task]["durations"]
        total_duration = dictionnary[task]["total_duration"]
        for j in range(len(diff)):
            if diff[j] < int(durations[j]/10) or diff[j] == 1:
                if task not in problems:
                    problems[task] = []
                problems[task] += [("end_start_problem", j, j+1,
                                    {"prev_end": dictionnary[task]["ends"][j],
                                     "next_start": dictionnary[task]["starts"][j+1]})]
        for j in range(len(durations)):
            if total_duration == 0:
                continue
            if (total_duration > 10 and durations[j] == 1) or (durations[j] <= 0.05*total_duration):
                if task not in problems:
                    problems[task] = []
                problems[task] += [("duration_problem", j, {"dur": durations[j],
                                                            "tot": total_duration})]
                # starts[j+1] should be higher than end[j]+delta...
                # typical case : starts = [0, 50], [49, 53]
    return problems


def problem_constraints(current_solution: Union[RCPSPSolutionPreemptive, MS_RCPSPSolution_Preemptive],
                        problems_output: Dict[Hashable, List],
                        minus_delta: int,
                        plus_delta: int,
                        cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE,
                                         CP_MRCPSP_MZN_PREEMMPTIVE,
                                         CP_MS_MRCPSP_MZN_PREEMPTIVE],
                        constraint_max_time=False,
                        minus_delta_2=0,
                        plus_delta_2=0):
    # max_time = get_max_time_solution(solution=current_solution)
    max_time = current_solution.get_end_time(task=current_solution.problem.sink_task)
    multimode = isinstance(cp_solver, (CP_MRCPSP_MZN_PREEMMPTIVE, CP_MS_MRCPSP_MZN))
    if multimode:
        modes_dict = current_solution.problem.build_mode_dict(current_solution.rcpsp_modes)
    subtasks = set(problems_output.keys())
    import random
    if len(subtasks) >= 0.2*current_solution.problem.n_jobs:
        subtasks = set(random.sample(subtasks,
                                     int(0.2*current_solution.problem.n_jobs)))
    else:
        subtasks = subtasks.union(set(random.sample(current_solution.problem.get_tasks_list(),
                                      int(0.2*current_solution.problem.n_jobs)-len(subtasks))))
    jobs_to_fix = [j for j in current_solution.problem.get_tasks_list()
                   if j not in subtasks]
    list_strings = []
    subtasks_2 = current_solution.problem.get_tasks_list()
    string = """
             int: nb_task_problems="""+str(len(subtasks_2))+""";\n
             array[1..nb_task_problems] of Tasks: task_list_problems="""+str([cp_solver.index_in_minizinc[task] for task in subtasks_2])+""";\n
             var int: nb_preemption_subtasks;\n
             var int: nb_small_tasks;\n
             constraint nb_preemption_subtasks=sum(i in 1..nb_task_problems)(sum(j in 2..nb_preemptive)(bool2int(d_preemptive[task_list_problems[i], j]>0)));\n
             constraint nb_small_tasks=sum(i in 1..nb_task_problems)(sum(j in 1..nb_preemptive)(bool2int(d_preemptive[task_list_problems[i], j]>0 /\ 
             ((d_preemptive[task_list_problems[i], j]<=adur[task_list_problems[i]] div 20) \/ (d_preemptive[task_list_problems[i], j]==1 /\ adur[task_list_problems[i]]>10)))));\n
             """
    list_strings += [string]
    weights = cp_solver.second_objectives["weights"]
    name_penalty = cp_solver.second_objectives["name_penalty"]
    def define_second_part_objective(weights_,
                                     name_penalty_):
        sum_string = "+".join(["0"] + [str(weights_[i]) + "*" + name_penalty_[i] for i in range(len(weights_))])
        s = "constraint sec_objective==" + sum_string + ";\n"
        return [s]
    s = define_second_part_objective(weights_=weights+[1000, 1000],
                                     name_penalty_=name_penalty+["nb_preemption_subtasks", "nb_small_tasks"])
    list_strings += s
    for job in subtasks:
        list_starts = current_solution.get_start_times_list(job)
        list_ends = current_solution.get_end_times_list(job)
        for j in range(len(list_starts)):
            start_time_j = list_starts[j]
            end_time_j = list_ends[j]
            duration_j = end_time_j - start_time_j
            string1_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                start_time=max(0,
                                                                                               start_time_j
                                                                                               - minus_delta),
                                                                                sign=SignEnum.UEQ,
                                                                                part_id=j + 1)
            string2_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                start_time=min(max_time,
                                                                                               start_time_j
                                                                                               + plus_delta)
                                                                                if constraint_max_time
                                                                                else start_time_j + plus_delta,
                                                                                sign=SignEnum.LEQ,
                                                                                part_id=j + 1)
            string1_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                            duration=max(duration_j-20, 0), # 0 for index 1
                                                                            sign=SignEnum.UEQ,
                                                                            part_id=j + 1)
            string2_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                            duration=duration_j + 20, #+50 for index 1
                                                                            sign=SignEnum.LEQ,
                                                                            part_id=j + 1)
            list_strings += [string1_dur, string1_start,
                             string2_dur, string2_start]
        for k in range(len(list_starts),
                       cp_solver.nb_preemptive):
            if isinstance(cp_solver, (CP_MS_MRCPSP_MZN, CP_MS_MRCPSP_MZN_PREEMPTIVE)):
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                                duration=0,
                                                                                sign=SignEnum.LEQ,
                                                                                part_id=k + 1)
            else:
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                                duration=0,
                                                                                sign=SignEnum.LEQ,
                                                                                part_id=k + 1)
            list_strings += [string1_dur]
    for job in jobs_to_fix:
        # continue
        # sif job in subtasks:
        #   continue
        is_paused = len(current_solution.get_start_times_list(job)) > 1
        is_paused_str = "true" if is_paused else "false"
        list_strings += ["constraint is_paused[" + str(cp_solver.index_in_minizinc[job]) + "]=="
                         + is_paused_str + ";\n"]
    for job in jobs_to_fix:
        if multimode:
            list_strings += cp_solver.constraint_task_to_mode(task_id=job, mode=modes_dict[job])
        if job in subtasks:
            continue
        list_starts = current_solution.get_start_times_list(job)
        list_ends = current_solution.get_end_times_list(job)
        for j in range(len(list_starts)):
            start_time_j = list_starts[j]
            end_time_j = list_ends[j]
            duration_j = end_time_j - start_time_j
            if minus_delta_2 == 0 and plus_delta_2 == 0:
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                    start_time=start_time_j,
                                                                                    sign=SignEnum.EQUAL,
                                                                                    part_id=j + 1)
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                                duration=duration_j,
                                                                                sign=SignEnum.EQUAL,
                                                                                part_id=j + 1)
                list_strings += [string1_dur, string1_start]
            else:
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                    start_time=max(0,
                                                                                                   start_time_j - minus_delta_2),
                                                                                    sign=SignEnum.UEQ,
                                                                                    part_id=j + 1)
                string2_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                    start_time=min(max_time,
                                                                                                   start_time_j + plus_delta_2)
                                                                                    if constraint_max_time
                                                                                    else start_time_j + plus_delta_2,
                                                                                    sign=SignEnum.LEQ,
                                                                                    part_id=j + 1)
                string1_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                                duration=max(duration_j - 5, 0),
                                                                                sign=SignEnum.UEQ,
                                                                                part_id=j + 1)
                string2_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                                duration=duration_j + 5,
                                                                                sign=SignEnum.LEQ,
                                                                                part_id=j + 1)
                list_strings += [string1_dur, string1_start, string2_dur, string2_start]
        for k in range(len(list_starts),
                       cp_solver.nb_preemptive):
            if minus_delta_2 == 0 and plus_delta_2 == 0:
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                    start_time=
                                                                                    current_solution.get_end_time(job),
                                                                                    sign=SignEnum.EQUAL,
                                                                                    part_id=k + 1)
                list_strings += [string1_start]
            else:
                string1_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                    start_time=
                                                                                    current_solution.get_end_time(job)
                                                                                    - minus_delta_2,
                                                                                    sign=SignEnum.UEQ,
                                                                                    part_id=k + 1)
                string2_start = cp_solver.constraint_start_time_string_preemptive_i(task=job,
                                                                                    start_time=
                                                                                    min(max_time,
                                                                                        current_solution.get_end_time(job) + plus_delta_2)
                                                                                    if constraint_max_time
                                                                                    else current_solution.get_end_time(job) + plus_delta_2,
                                                                                    sign=SignEnum.LEQ,
                                                                                    part_id=k + 1)
                # list_strings += [string2_start]
                list_strings += [string2_start, string1_start]
            string1_dur = cp_solver.constraint_duration_string_preemptive_i(task=job,
                                                                            duration=0,
                                                                            sign=SignEnum.EQUAL,
                                                                            part_id=k + 1)
            list_strings += [string1_dur]

    exceptions = []
    for task in problems_output:
        for l in problems_output[task]:
            if l[0] == "end_start_problem":
                exceptions += [(task, l[1]), (task, l[2])]
            else:
                exceptions += [(task, l[1])]
    # exceptions = random.sample(exceptions, int(0.1*len(exceptions)))
    if random.random() < 0.99:
        print("Method 1")
        print(len(exceptions))
        exceptions = random.sample(exceptions, min(len(exceptions), 10))
        list_strings += constraint_unit_used_to_tasks_preemptive(tasks_set=
                                                                 set(random.sample(current_solution.problem.
                                                                                   get_tasks_list(),
                                                                                   min(int(0.99 * current_solution.
                                                                                       problem.n_jobs),
                                                                                       current_solution.problem.n_jobs-1))),
                                                                 current_solution=current_solution,
                                                                 cp_solver=cp_solver,
                                                                 exceptions=exceptions)
    else:
        print("Method 2")
        employee_usage_matrix, sum_usage, employees_usage_dict = employee_usage(solution=current_solution,
                                                                                problem=current_solution.problem)
        sorted_employee = np.argsort(sum_usage)
        set_employees_to_fix = set([current_solution.problem.employees_list[i]
                                    for i in sorted_employee[:int(len(sorted_employee)/2)]])
        set_employees_to_fix = set(random.sample(current_solution.problem.employees_list,
                                                 int(len(current_solution.problem.employees_list)-2)))
        list_strings += constraint_unit_used_subset_employees_preemptive(employees_set=set_employees_to_fix,
                                                                         current_solution=current_solution,
                                                                         cp_solver=cp_solver,
                                                                         employees_usage_dict=employees_usage_dict,
                                                                         exceptions=exceptions)
    return list_strings


def post_process_solution(model, solution):
    problem = find_possible_problems_preemptive(model, solution)


class NeighborRepairProblems(ConstraintHandler):
    def __init__(self, problem: Union[RCPSPModel,
                                      RCPSPModelPreemptive,
                                      RCPSPSolutionPreemptive,
                                      RCPSPModelSpecialConstraints,
                                      RCPSPModelSpecialConstraintsPreemptive,
                                      MS_RCPSPModel],
                 params_list: List[ParamsConstraintBuilder] = None):
        self.problem = problem
        if isinstance(self.problem, (RCPSPModelSpecialConstraintsPreemptive,
                                     RCPSPModelSpecialConstraints)):
            self.graph_rcpsp = GraphRCPSPSpecialConstraints(problem=self.problem)
            self.special_constraints = True
        else:
            self.graph_rcpsp = GraphRCPSP(problem=self.problem)
            self.special_constraints = False
        params = ParamsConstraintBuilder(minus_delta_primary=6000, plus_delta_primary=6000, minus_delta_secondary=1,
                                         plus_delta_secondary=1)
        params_2 = ParamsConstraintBuilder(minus_delta_primary=6000, plus_delta_primary=6000, minus_delta_secondary=300,
                                           plus_delta_secondary=300)
        if params_list is None:
            self.params_list = [params, params_2]
        else:
            self.params_list = params_list

    def adding_constraint_from_results_store(self, cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE,
                                                                    CP_MS_MRCPSP_MZN_PREEMPTIVE,
                                                                    CP_MRCPSP_MZN_PREEMMPTIVE],
                                             child_instance,
                                             result_storage: ResultStorage,
                                             last_result_store: ResultStorage = None) -> Iterable[Any]:
        if last_result_store is not None:
            current_solution, fit = next((last_result_store.list_solution_fits[j]
                                          for j in range(len(last_result_store.list_solution_fits))
                                          if "opti_from_cp"
                                          in last_result_store.list_solution_fits[j][0].__dict__.keys()), (None, None))
            print(current_solution)
            print("Index opti :", [j for j in range(len(last_result_store.list_solution_fits))
                                   if "opti_from_cp" in last_result_store.list_solution_fits[j][0].__dict__.keys()])
        else:
            current_solution, fit = next((result_storage.list_solution_fits[j]
                                          for j in range(len(result_storage.list_solution_fits))
                                          if "opti_from_cp" in result_storage.list_solution_fits[j][0].__dict__.keys()),
                                         (None, None))
            print(current_solution)
            print("Index opti :", [j for j in range(len(result_storage.list_solution_fits))
                                   if "opti_from_cp" in result_storage.list_solution_fits[j][0].__dict__.keys()])

        if current_solution is None: # or fit != result_storage.get_best_solution_fit()[1]:
            current_solution, fit = result_storage.get_last_best_solution()
        current_solution: RCPSPSolutionPreemptive = current_solution
        print(current_solution.get_nb_task_preemption(), " task preempted ")
        print(current_solution.get_max_preempted(), " most preempted task ")
        print(current_solution.total_number_of_cut(), " total number of cut ")
        eval = self.problem.evaluate(current_solution)
        print("Current Eval :", eval)
        if eval.get("constraint_penalty", 0) == 0:
            p = self.params_list[1]
        else:
            p = self.params_list[1]
        problems = find_possible_problems_preemptive(model=self.problem,
                                                     solution=current_solution)
        print(sum([len(problems[t]) for t in problems]), " problems ?")
        print("Problems : ", problems)
        list_strings = problem_constraints(current_solution=current_solution,
                                           problems_output=problems,
                                           minus_delta=p.minus_delta_primary,
                                           plus_delta=p.plus_delta_primary,
                                           cp_solver=cp_solver,
                                           constraint_max_time=p.constraint_max_time_to_current_solution,
                                           minus_delta_2=p.minus_delta_secondary,
                                           plus_delta_2=p.plus_delta_secondary)
        for s in list_strings:
            child_instance.add_string(s)
        if eval.get("constraint_penalty", 0) > 0:
            child_instance.add_string("constraint objective=" + str(eval["makespan"]) + ";\n")
        else:
            string = cp_solver.constraint_start_time_string(task=self.problem.sink_task,
                                                            start_time=current_solution
                                                            .get_start_time(self.problem.sink_task)+200,
                                                            sign=SignEnum.LEQ)
            child_instance.add_string(string)
            string = cp_solver.constraint_start_time_string_preemptive_i(task=self.problem.sink_task,
                                                                         start_time=current_solution
                                                                         .get_end_time(self.problem.sink_task) + 200,
                                                                         part_id=cp_solver.nb_preemptive,
                                                                         sign=SignEnum.LEQ)
            child_instance.add_string(string)
        weights = cp_solver.second_objectives["weights"]
        name_penalty = cp_solver.second_objectives["name_penalty"]
        sum_string = "+".join(["0"] + [str(weights[i]) + "*" + name_penalty[i] for i in range(len(weights))])
        child_instance.add_string("constraint "+sum_string+"<="+str(int(1.01*100*eval.get("constraint_penalty", 0)))+ ";\n")
        if eval.get("constraint_penalty", 0) > 0:
            # strings = cp_solver.constraint_objective_max_time_set_of_jobs([self.problem.sink_task])
            strings = []
        else:
            # strings = cp_solver.constraint_objective_max_time_set_of_jobs(set(self.problem.get_tasks_list()))
            strings = cp_solver.constraint_objective_equal_makespan(self.problem.sink_task)
        for s in strings:
            child_instance.add_string(s)
            list_strings += [s]
        return list_strings

    def remove_constraints_from_previous_iteration(self,
                                                   cp_solver: Union[CP_RCPSP_MZN_PREEMMPTIVE,
                                                                    CP_MS_MRCPSP_MZN_PREEMPTIVE,
                                                                    CP_MRCPSP_MZN_PREEMMPTIVE],
                                                   child_instance,
                                                   previous_constraints: Iterable[Any]):
        pass