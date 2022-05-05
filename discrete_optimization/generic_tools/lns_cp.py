from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.do_problem import Solution, Problem, build_evaluate_function_aggregated, \
    ParamsObjectiveFunction, ModeOptim, build_aggreg_function_and_params_objective
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.generic_tools.lns_mip import InitialSolution, PostProcessSolution, TrivialPostProcessSolution
from abc import abstractmethod
from typing import Any, Iterable, Optional
from datetime import timedelta
from minizinc import Status
import random
from typing import Union, Iterable, Any, List
import numpy as np

import time


class ConstraintHandler:
    @abstractmethod
    def adding_constraint_from_results_store(self,
                                             cp_solver: CPSolver,
                                             child_instance,
                                             result_storage: ResultStorage,
                                             last_result_store: ResultStorage) -> Iterable[Any]:
        ...

    @abstractmethod
    def remove_constraints_from_previous_iteration(self,
                                                   cp_solver: CPSolver,
                                                   child_instance,
                                                   previous_constraints: Iterable[Any]):
        ...


class LNS_CP(SolverDO):
    def __init__(self,
                 problem: Problem,
                 cp_solver: CPSolver,
                 initial_solution_provider: InitialSolution,
                 constraint_handler: ConstraintHandler,
                 post_process_solution: PostProcessSolution = None,
                 params_objective_function: ParamsObjectiveFunction = None):
        self.problem = problem
        self.cp_solver = cp_solver
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.post_process_solution = post_process_solution
        if self.post_process_solution is None:
            self.post_process_solution = TrivialPostProcessSolution()
        self.params_objective_function = params_objective_function
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=
                                                       self.params_objective_function)

    def solve_lns(self,
                  parameters_cp: ParametersCP,
                  nb_iteration_lns: int,
                  nb_iteration_no_improvement: Optional[int] = None,
                  max_time_seconds: Optional[int] = None,
                  skip_first_iteration: bool=False,
                  stop_first_iteration_if_optimal: bool=True,
                  **args)->ResultStorage:
        sense = self.params_objective_function.sense_function
        if max_time_seconds is None:
            max_time_seconds = 3600*24  # One day
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2*nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        deb_time = time.time()
        if not skip_first_iteration:
            store_lns = self.initial_solution_provider.get_starting_solution()
            store_lns = self.post_process_solution.build_other_solution(store_lns)
            init_solution, objective = store_lns.get_best_solution_fit()
            satisfy = self.problem.satisfy(init_solution)
            print("Satisfy Initial solution", satisfy)
            try:
                print("Nb task preempted = ", init_solution.get_nb_task_preemption())
                print("Nb max preemption = ", init_solution.get_max_preempted())
            except:
                pass
            best_objective = objective
        else:
            best_objective = float('inf') if sense == ModeOptim.MINIMIZATION else -float("inf")
            store_lns = None
        for iteration in range(nb_iteration_lns):
            print('Starting iteration n°', iteration,
                  " current objective ", best_objective)
            with self.cp_solver.instance.branch() as child:
                if iteration == 0 and not skip_first_iteration or iteration >= 1:
                    constraint_iterable = self.constraint_handler \
                        .adding_constraint_from_results_store(cp_solver=self.cp_solver,
                                                              child_instance=child,
                                                              result_storage=store_lns,
                                                              last_result_store=
                                                              store_lns if iteration == 0
                                                              else result_store)
                #if True:
                try:
                    if iteration == 0:
                        result = child.solve(timeout=
                                             timedelta(seconds=parameters_cp.TimeLimit_iter0),
                                             intermediate_solutions=parameters_cp.intermediate_solution,
                                             free_search=parameters_cp.free_search,
                                             processes=None if not parameters_cp.multiprocess
                                             else parameters_cp.nb_process)
                    else:
                        result = child.solve(timeout=
                                             timedelta(seconds=parameters_cp.TimeLimit),
                                             intermediate_solutions=parameters_cp.intermediate_solution,
                                             free_search=parameters_cp.free_search,
                                             processes=None if not parameters_cp.multiprocess
                                             else parameters_cp.nb_process)
                    result_store = self.cp_solver.retrieve_solutions(result,
                                                                     parameters_cp=parameters_cp)
                    print("iteration n°", iteration, "Solved !!!")
                    print(result.status)
                    if len(result_store.list_solution_fits) > 0:
                        print("Solved !!!")
                        bsol, fit = result_store.get_best_solution_fit()
                        print("Fitness Before = ", fit)
                        print("Satisfaction Before = ", self.problem.satisfy(bsol))
                        print("Post Process..")
                        result_store = self.post_process_solution.build_other_solution(result_store)
                        bsol, fit = result_store.get_best_solution_fit()
                        print("Satisfy after : ", self.problem.satisfy(bsol))
                        if sense == ModeOptim.MAXIMIZATION and fit >= best_objective:
                            if fit > best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_objective = fit
                        elif sense == ModeOptim.MAXIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        elif sense == ModeOptim.MINIMIZATION and fit <= best_objective:
                            if fit < best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_objective = fit
                        elif sense == ModeOptim.MINIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        if skip_first_iteration and iteration == 0:
                            store_lns = result_store
                        else:
                            for s, f in list(result_store.list_solution_fits):
                                store_lns.add_solution(solution=s, fitness=f)
                    else:
                        current_nb_iteration_no_improvement += 1
                    if skip_first_iteration \
                            and result.status == Status.OPTIMAL_SOLUTION \
                            and iteration == 0\
                            and self.problem.satisfy(bsol)\
                            and stop_first_iteration_if_optimal:
                        print("Finish LNS because found optimal solution")
                        break
                # else:
                except Exception as e:
                    current_nb_iteration_no_improvement += 1
                    print("Failed ! reason : ", e)
                if time.time() - deb_time > max_time_seconds:
                    print("Finish LNS with time limit reached")
                    break
                print(current_nb_iteration_no_improvement, "/", nb_iteration_no_improvement)
                if current_nb_iteration_no_improvement > nb_iteration_no_improvement:
                    print("Finish LNS with maximum no improvement iteration ")
                    break
        return store_lns

    def solve(self, **kwargs) -> ResultStorage:
        return self.solve_lns(**kwargs)

# TODO : continue working on this
class LNS_CPlex(SolverDO):
    def __init__(self,
                 problem: Problem,
                 cp_solver: CPSolver,
                 initial_solution_provider: InitialSolution,
                 constraint_handler: ConstraintHandler,
                 post_process_solution: PostProcessSolution = None,
                 params_objective_function: ParamsObjectiveFunction = None):
        self.problem = problem
        self.cp_solver = cp_solver
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.post_process_solution = post_process_solution
        if self.post_process_solution is None:
            self.post_process_solution = TrivialPostProcessSolution()
        self.params_objective_function = params_objective_function
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=
                                                       self.params_objective_function)

    def solve_lns(self,
                  parameters_cp: ParametersCP,
                  nb_iteration_lns: int,
                  nb_iteration_no_improvement: Optional[int] = None,
                  max_time_seconds: Optional[int] = None,
                  skip_first_iteration: bool=False,
                  stop_first_iteration_if_optimal: bool=True,
                  **args)->ResultStorage:
        sense = self.params_objective_function.sense_function
        if max_time_seconds is None:
            max_time_seconds = 3600*24  # One day
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2*nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        deb_time = time.time()
        if not skip_first_iteration:
            store_lns = self.initial_solution_provider.get_starting_solution()
            store_lns = self.post_process_solution.build_other_solution(store_lns)
            init_solution, objective = store_lns.get_best_solution_fit()
            satisfy = self.problem.satisfy(init_solution)
            print("Satisfy Initial solution", satisfy)
            try:
                print("Nb task preempted = ", init_solution.get_nb_task_preemption())
                print("Nb max preemption = ", init_solution.get_max_preempted())
            except:
                pass
            best_objective = objective
        else:
            best_objective = float('inf') if sense == ModeOptim.MINIMIZATION else -float("inf")
            store_lns = None
        for iteration in range(nb_iteration_lns):
            print('Starting iteration n°', iteration,
                  " current objective ", best_objective)
            if iteration == 0 and not skip_first_iteration or iteration >= 1:
                constraint_iterable = self.constraint_handler \
                    .adding_constraint_from_results_store(cp_solver=self.cp_solver,
                                                          child_instance=None,
                                                          result_storage=store_lns,
                                                          last_result_store=
                                                          store_lns if iteration == 0
                                                          else result_store)

            try:
                if iteration == 0:
                    p = parameters_cp.default()
                    p.TimeLimit = parameters_cp.TimeLimit_iter0
                    result_store = self.cp_solver.solve(parameters_cp=parameters_cp)
                else:
                    result_store = self.cp_solver.solve(parameters_cp=parameters_cp)
                print("iteration n°", iteration, "Solved !!!")
                if len(result_store.list_solution_fits) > 0:
                    print("Solved !!!")
                    bsol, fit = result_store.get_best_solution_fit()
                    print("Fitness Before = ", fit)
                    print("Satisfaction Before = ", self.problem.satisfy(bsol))
                    print("Post Process..")
                    result_store = self.post_process_solution.build_other_solution(result_store)
                    bsol, fit = result_store.get_best_solution_fit()
                    print("Satisfy after : ", self.problem.satisfy(bsol))
                    if sense == ModeOptim.MAXIMIZATION and fit >= best_objective:
                        if fit > best_objective:
                            current_nb_iteration_no_improvement = 0
                        else:
                            current_nb_iteration_no_improvement += 1
                        best_objective = fit
                    elif sense == ModeOptim.MAXIMIZATION:
                        current_nb_iteration_no_improvement += 1
                    elif sense == ModeOptim.MINIMIZATION and fit <= best_objective:
                        if fit < best_objective:
                            current_nb_iteration_no_improvement = 0
                        else:
                            current_nb_iteration_no_improvement += 1
                        best_objective = fit
                    elif sense == ModeOptim.MINIMIZATION:
                        current_nb_iteration_no_improvement += 1
                    if skip_first_iteration and iteration == 0:
                        store_lns = result_store
                    else:
                        for s, f in list(result_store.list_solution_fits):
                            store_lns.add_solution(solution=s, fitness=f)
                else:
                    current_nb_iteration_no_improvement += 1
                # TODO : retrieve the solving status of cplex
                if skip_first_iteration \
                        and False \
                        and iteration == 0 \
                        and self.problem.satisfy(bsol) \
                        and stop_first_iteration_if_optimal:
                    print("Finish LNS because found optimal solution")
                    break
                # else:
            except Exception as e:
                current_nb_iteration_no_improvement += 1
                print("Failed ! reason : ", e)
            if time.time() - deb_time > max_time_seconds:
                print("Finish LNS with time limit reached")
                break
            print(current_nb_iteration_no_improvement, "/", nb_iteration_no_improvement)
            if current_nb_iteration_no_improvement > nb_iteration_no_improvement:
                print("Finish LNS with maximum no improvement iteration ")
                break
        return store_lns

    def solve(self, **kwargs) -> ResultStorage:
        return self.solve_lns(**kwargs)


class ConstraintHandlerMix(ConstraintHandler):
    def __init__(self, problem: Problem,
                 list_constraints_handler: List[ConstraintHandler],
                 list_proba: List[float],
                 update_proba: bool = True,
                 tag_constraint_handler: List[str] = None,
                 sequential: bool = False):
        self.problem = problem
        self.list_constraints_handler = list_constraints_handler
        self.tag_constraint_handler = tag_constraint_handler
        self.sequential = sequential
        if self.tag_constraint_handler is None:
            self.tag_constraint_handler = [str(i) for i in range(len(self.list_constraints_handler))]
        self.list_proba = list_proba
        if isinstance(self.list_proba, list):
            self.list_proba = np.array(self.list_proba)
        self.list_proba = self.list_proba / np.sum(self.list_proba)
        self.index_np = np.array(range(len(self.list_proba)), dtype=np.int)
        self.current_iteration = 0
        self.status = {i: {"nb_usage": 0,
                           "nb_improvement": 0,
                           "name": self.tag_constraint_handler[i]}
                       for i in range(len(self.list_constraints_handler))}
        self.last_index_param = None
        self.last_fitness = None
        self.update_proba = update_proba

    def adding_constraint_from_results_store(self, cp_solver: CPSolver,
                                             child_instance,
                                             result_storage: ResultStorage,
                                             last_result_store: ResultStorage = None) -> Iterable[Any]:
        new_fitness = result_storage.get_best_solution_fit()[1]
        if self.last_index_param is not None:
            if new_fitness != self.last_fitness:
                self.status[self.last_index_param]["nb_improvement"] += 1
                self.last_fitness = new_fitness
                if self.update_proba:
                    self.list_proba[self.last_index_param] *= 1.05
                    self.list_proba = self.list_proba / np.sum(self.list_proba)
            else:
                if self.update_proba:
                    self.list_proba[self.last_index_param] *= 0.95
                    self.list_proba = self.list_proba / np.sum(self.list_proba)
        else:
            self.last_fitness = new_fitness
        if self.sequential:
            if self.last_index_param is not None:
                choice = (self.last_index_param+1) % len(self.list_constraints_handler)
            else:
                choice = 0
        else:
            if random.random() <= 0.95:
                choice = np.random.choice(self.index_np,
                                          size=1,
                                          p=self.list_proba)[0]
            else:
                max_improvement = max([self.status[x]["nb_improvement"]/max(self.status[x]["nb_usage"], 1) for x in self.status])
                choice = random.choice([x for x in self.status
                                        if self.status[x]["nb_improvement"]/max(self.status[x]["nb_usage"], 1)
                                        == max_improvement])
        ch = self.list_constraints_handler[int(choice)]
        self.current_iteration += 1
        self.last_index_param = choice
        self.status[self.last_index_param]["nb_usage"] += 1
        print("Status ", self.status)
        return ch.adding_constraint_from_results_store(cp_solver,
                                                       child_instance,
                                                       result_storage,
                                                       last_result_store)

    def remove_constraints_from_previous_iteration(self, cp_solver: CPSolver,
                                                   child_instance,
                                                   previous_constraints: Iterable[Any]):
        pass









