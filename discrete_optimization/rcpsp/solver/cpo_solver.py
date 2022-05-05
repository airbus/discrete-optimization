from typing import List

from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.cp_tools import CPSolver, CPSolverName, ParametersCP, SignEnum
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from docplex.cp.model import *
import os


class CPOptSolver(CPSolver):
    def __init__(self, rcpsp_model: RCPSPModel,
                 cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction=None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["s"]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        filename = os.path.dirname(os.path.abspath(__file__)) + '/data/rcpsp_default.data'
        NB_TASKS, NB_RESOURCES = self.rcpsp_model.n_jobs, len(self.rcpsp_model.resources_list)
        CAPACITIES = [self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources]
        tasks_list = self.rcpsp_model.tasks_list
        self.index_task = {tasks_list[i]: i for i in range(len(tasks_list))}
        # -----------------------------------------------------------------------------
        # Prepare the data for modeling
        # -----------------------------------------------------------------------------

        # Extract duration of each task
        DURATIONS = [self.rcpsp_model.mode_details[t][1]["duration"] for t in tasks_list]

        # Extract demand of each task
        DEMANDS = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list]
                   for t in tasks_list]

        # Extract successors of each task
        SUCCESSORS = [[self.index_task[tt] for tt in self.rcpsp_model.successors[t]]
                      for t in tasks_list]

        # -----------------------------------------------------------------------------
        # Build the model
        # -----------------------------------------------------------------------------

        # Create model
        mdl = CpoModel()
        # Create task interval variables
        tasks = [interval_var(name='T{}'.format(i + 1), size=DURATIONS[i]) for i in range(NB_TASKS)]
        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(NB_TASKS) for s in SUCCESSORS[t])
        # Constrain capacity of resources
        mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
                for r in range(NB_RESOURCES))

        # Minimize end of all tasks
        self.objective_expression = minimize(max(end_of(t) for t in tasks))
        mdl.add(self.objective_expression)
        self.model = mdl
        self.variables = {"tasks": tasks}

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        list_solution_fit = []
        schedule = {}
        for i in range(len(self.variables["tasks"])):
            itv = result.get_var_solution(self.variables["tasks"][i])
            schedule[self.rcpsp_model.tasks_list[i]] = {"start_time": itv.get_start(),
                                                        "end_time": itv.get_end()}
        solution = RCPSPSolution(problem=self.rcpsp_model, rcpsp_schedule=schedule,
                                 rcpsp_modes=[1]*self.rcpsp_model.n_jobs_non_dummy)
        fit = self.aggreg_sol(solution)
        return ResultStorage(list_solution_fits=[(solution, fit)],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_cp: ParametersCP, **args) -> ResultStorage:
        result: CpoSolveResult = self.model.solve(TimeLimit=parameters_cp.TimeLimit,
                                  execfile="/Applications/CPLEX_Studio201/cpoptimizer/bin/x86-64_osx/cpoptimizer")
        self.results = result
        self.solver_infos = result.solver_infos
        return self.retrieve_solutions(result, parameters_cp)

    def get_stats(self):
        return self.solver_infos


    # Function to run LNS
    def constraint_objective_makespan(self):
        self.model.add(minimize(max(end_of(t) for t in self.variables["tasks"])))

    def constraint_objective_equal_makespan(self, task_sink):
        #TODO
        pass

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        expression = minimize(max(end_of(self.variables["tasks"][self.index_task[t]])
                              for t in set_of_jobs))
        self.model.add(expression)
        return [expression]

    def constraint_start_time_string(self, task, start_time, sign: SignEnum = SignEnum.EQUAL) -> List[CpoExpr]:
        expression = None
        if sign == SignEnum.EQUAL:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) == start_time
        elif sign == SignEnum.LEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) <= start_time
        elif sign == SignEnum.LESS:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) < start_time
        elif sign == SignEnum.UP:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) > start_time
        elif sign == SignEnum.UEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) >= start_time
        self.model.add(expression)
        return [expression]

    def constraint_end_time_string(self, task, start_time, sign: SignEnum = SignEnum.EQUAL) -> List[CpoExpr]:
        expression = None
        if sign == SignEnum.EQUAL:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) == start_time
        elif sign == SignEnum.LEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) <= start_time
        elif sign == SignEnum.LESS:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) < start_time
        elif sign == SignEnum.UP:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) > start_time
        elif sign == SignEnum.UEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) >= start_time
        self.model.add(expression)
        return [expression]


import docplex.cp.utils_visu as visu

def vizu(result_storage: ResultStorage):
    if visu.is_visu_enabled():
        solution: RCPSPSolution = result_storage.get_best_solution_fit()[0]
        problem: RCPSPModel = solution.problem
        load = {r: CpoStepFunction() for r in problem.resources_list}
        visu.timeline('Solution for RCPSP ')
        visu.panel('Tasks')
        for i in range(problem.n_jobs):
            visu.interval(solution.rcpsp_schedule[problem.tasks_list[i]]["start_time"],
                          solution.rcpsp_schedule[problem.tasks_list[i]]["end_time"],
                          i, str(problem.tasks_list[i]))
            for r in problem.resources_list:
                if problem.mode_details[problem.tasks_list[i]][1][r]>0:
                    load[r].add_value(solution.rcpsp_schedule[problem.tasks_list[i]]["start_time"],
                                      solution.rcpsp_schedule[problem.tasks_list[i]]["end_time"],
                                      problem.mode_details[problem.tasks_list[i]][1][r])

        j = 0
        for r in load:
            visu.panel('R-'+str(r))
            visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, problem.resources[r])], style='area', color='lightgrey')
            visu.function(segments=load[r], style='area', color=j)
            j += 1
        visu.show()





