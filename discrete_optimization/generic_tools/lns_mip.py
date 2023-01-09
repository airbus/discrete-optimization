#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time
from abc import abstractmethod
from typing import Any, Hashable, Mapping, Optional

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import MilpSolver, ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class ConstraintHandler:
    @abstractmethod
    def adding_constraint_from_results_store(
        self, milp_solver: MilpSolver, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        ...

    @abstractmethod
    def remove_constraints_from_previous_iteration(
        self, milp_solver: MilpSolver, previous_constraints: Mapping[Hashable, Any]
    ) -> None:
        ...


class InitialSolution:
    @abstractmethod
    def get_starting_solution(self) -> ResultStorage:
        ...


class InitialSolutionFromSolver(InitialSolution):
    def __init__(self, solver: SolverDO, **kwargs: Any):
        self.solver = solver
        self.dict = kwargs

    @abstractmethod
    def get_starting_solution(self) -> ResultStorage:
        return self.solver.solve(**self.dict)


class TrivialInitialSolution(InitialSolution):
    def __init__(self, solution: ResultStorage, **kwargs: Any):
        self.solution = solution
        self.dict = kwargs

    @abstractmethod
    def get_starting_solution(self) -> ResultStorage:
        return self.solution


class PostProcessSolution:
    # From solution from MIP or CP you can build other solution.
    # Here you can have many different approaches:
    # if solution from mip/cp are not feasible you can code a repair function
    # you can also do mall changes (filling gap in a schedule) to try to improve the solution
    # you can also run algorithms from the new found solution.
    @abstractmethod
    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        ...


class TrivialPostProcessSolution(PostProcessSolution):
    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        return result_storage


class LNS_MILP(SolverDO):
    def __init__(
        self,
        problem: Problem,
        milp_solver: MilpSolver,
        initial_solution_provider: InitialSolution,
        constraint_handler: ConstraintHandler,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.problem = problem
        self.milp_solver = milp_solver
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.post_process_solution: PostProcessSolution
        if post_process_solution is None:
            self.post_process_solution = TrivialPostProcessSolution()
        else:
            self.post_process_solution = post_process_solution

        (
            self.aggreg_from_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=params_objective_function,
        )

    def solve_lns(
        self,
        parameters_milp: ParametersMilp,
        nb_iteration_lns: int,
        nb_iteration_no_improvement: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
        skip_first_iteration: Optional[bool] = False,
        **args: Any,
    ) -> ResultStorage:
        sense = self.params_objective_function.sense_function
        if max_time_seconds is None:
            max_time_seconds = 3600 * 24  # One day
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2 * nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        deb_time = time.time()
        if not skip_first_iteration:
            store_lns = self.initial_solution_provider.get_starting_solution()
            init_solution, objective = store_lns.get_best_solution_fit()
            constraint_iterable = (
                self.constraint_handler.adding_constraint_from_results_store(
                    milp_solver=self.milp_solver, result_storage=store_lns
                )
            )
            best_objective = objective
        else:
            best_objective = float("inf")
            constraint_iterable = {"empty": []}
            store_lns = None
        for iteration in range(nb_iteration_lns):
            result_store = self.milp_solver.solve(
                parameters_milp=parameters_milp, **args
            )
            logger.debug("Solved !!!")
            bsol, fit = result_store.get_best_solution_fit()
            logger.debug(f"Fitness = {fit}")
            logger.debug("Post Process..")
            result_store = self.post_process_solution.build_other_solution(result_store)
            bsol, fit = result_store.get_best_solution_fit()
            logger.debug(f"After postpro = {fit}")
            if sense == ModeOptim.MAXIMIZATION and fit >= best_objective:
                if fit > best_objective:
                    current_nb_iteration_no_improvement = 0
                else:
                    current_nb_iteration_no_improvement += 1
                best_objective = fit
            if sense == ModeOptim.MINIMIZATION and fit <= best_objective:
                if fit < best_objective:
                    current_nb_iteration_no_improvement = 0
                else:
                    current_nb_iteration_no_improvement += 1
                best_objective = fit
            if skip_first_iteration and iteration == 0:
                store_lns = result_store
            for s, f in result_store.list_solution_fits:
                if store_lns is None:  # for mypy, should never happen
                    raise RuntimeError("store_lns should have been initialized for now")
                store_lns.add_solution(solution=s, fitness=f)
            logger.debug("Removing constraint:")
            self.constraint_handler.remove_constraints_from_previous_iteration(
                milp_solver=self.milp_solver, previous_constraints=constraint_iterable
            )
            logger.debug("Adding constraint:")
            constraint_iterable = (
                self.constraint_handler.adding_constraint_from_results_store(
                    milp_solver=self.milp_solver, result_storage=result_store
                )
            )
            if time.time() - deb_time > max_time_seconds:
                logger.info("Finish LNS with time limit reached")
                break
            if current_nb_iteration_no_improvement > nb_iteration_no_improvement:
                logger.info("Finish LNS with maximum no improvement iteration ")
                break
        if store_lns is None:  # for mypy, should never happen
            raise RuntimeError("store_lns should have been initialized for now")
        return store_lns
