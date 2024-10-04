#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from collections.abc import Iterable
from enum import Enum
from typing import Any, Union

import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_evaluate_function_aggregated,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import (
    GurobiConstraintHandler,
    InitialSolution,
    OrtoolsMathOptConstraintHandler,
)
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.mutations.mutation_rcpsp import (
    PermutationMutationRCPSP,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.solver import CP_MRCPSP_MZN
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import (
    LP_MRCPSP_GUROBI,
    LP_MRCPSP_MATHOPT,
    LP_RCPSP_MATHOPT,
)
from discrete_optimization.rcpsp.solver.rcpsp_pile import (
    GreedyChoice,
    PileSolverRCPSP,
    PileSolverRCPSP_Calendar,
)

logger = logging.getLogger(__name__)


class InitialMethodRCPSP(Enum):
    DUMMY = 0
    PILE = 1
    PILE_CALENDAR = 2
    LS = 3
    GA = 4
    CP = 5


class InitialSolutionRCPSP(InitialSolution):
    def __init__(
        self,
        problem: RCPSPModel,
        params_objective_function: ParamsObjectiveFunction = None,
        initial_method: InitialMethodRCPSP = InitialMethodRCPSP.PILE,
    ):
        self.problem = problem
        self.params_objective_function = params_objective_function
        if self.params_objective_function is None:
            self.params_objective_function = get_default_objective_setup(
                problem=self.problem
            )
        self.aggreg, _ = build_evaluate_function_aggregated(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )
        self.initial_method = initial_method

    def get_starting_solution(self) -> ResultStorage:
        if self.initial_method == InitialMethodRCPSP.PILE:
            logger.info("Compute greedy")
            greedy_solver = PileSolverRCPSP(self.problem)
            store_solution = greedy_solver.solve(
                greedy_choice=GreedyChoice.MOST_SUCCESSORS
            )
        if self.initial_method == InitialMethodRCPSP.PILE_CALENDAR:
            logger.info("Compute greedy")
            greedy_solver = PileSolverRCPSP_Calendar(self.problem)
            store_solution = greedy_solver.solve(
                greedy_choice=GreedyChoice.MOST_SUCCESSORS
            )
        elif self.initial_method == InitialMethodRCPSP.DUMMY:
            logger.info("Compute dummy")
            solution = self.problem.get_dummy_solution()
            fit = self.aggreg(solution)
            store_solution = ResultStorage(
                mode_optim=self.params_objective_function.sense_function,
                list_solution_fits=[(solution, fit)],
            )
        elif self.initial_method == InitialMethodRCPSP.CP:
            solver = CP_MRCPSP_MZN(
                problem=self.problem,
                params_objective_function=self.params_objective_function,
            )
            store_solution = solver.solve(parameters_cp=ParametersCP.default())
        elif self.initial_method == InitialMethodRCPSP.LS:
            dummy = self.problem.get_dummy_solution()
            _, mutations = get_available_mutations(self.problem, dummy)
            list_mutation = [
                mutate[0].build(self.problem, dummy, **mutate[1])
                for mutate in mutations
                if mutate[0] == PermutationMutationRCPSP
            ]
            mixed_mutation = BasicPortfolioMutation(
                list_mutation, np.ones((len(list_mutation)))
            )
            res = RestartHandlerLimit(500)
            sa = SimulatedAnnealing(
                problem=self.problem,
                mutator=mixed_mutation,
                restart_handler=res,
                temperature_handler=TemperatureSchedulingFactor(2, res, 0.9999),
                mode_mutation=ModeMutation.MUTATE,
                params_objective_function=self.params_objective_function,
                store_solution=True,
            )
            store_solution = sa.solve(initial_variable=dummy, nb_iteration_max=10000)
        return store_solution


class ConstraintHandlerFixStartTime_MathOpt(OrtoolsMathOptConstraintHandler):
    def __init__(self, problem: RCPSPModel, fraction_fix_start_time: float = 0.9):
        self.problem = problem
        self.fraction_fix_start_time = fraction_fix_start_time

    def adding_constraint_from_results_store(
        self, solver: LP_RCPSP_MATHOPT, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:

        nb_jobs = self.problem.n_jobs + 2
        lns_constraints = []
        current_solution, fit = result_storage.get_best_solution_fit()
        # Starting point
        solver.set_warm_start(current_solution)

        # Fix start time for a subset of task.
        jobs_to_fix = set(
            random.sample(
                list(current_solution.rcpsp_schedule),
                int(self.fraction_fix_start_time * nb_jobs),
            )
        )
        for job_to_fix in jobs_to_fix:
            for t in solver.index_time:
                if current_solution.rcpsp_schedule[job_to_fix]["start_time"] == t:
                    lns_constraints.append(
                        solver.add_linear_constraint(
                            solver.x[solver.index_in_var[job_to_fix]][t] == 1
                        )
                    )
                else:
                    lns_constraints.append(
                        solver.add_linear_constraint(
                            solver.x[solver.index_in_var[job_to_fix]][t] == 0
                        )
                    )
        return lns_constraints


class ConstraintHandlerStartTimeInterval_MathOpt(OrtoolsMathOptConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModel,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta

    def adding_constraint_from_results_store(
        self, solver: LP_RCPSP_MATHOPT, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        lns_constraints = []
        current_solution, fit = result_storage.get_best_solution_fit()
        # Starting point
        solver.set_warm_start(current_solution)
        # Avoid fixing last jobs to allow makespan reduction
        max_time = max(
            [
                current_solution.rcpsp_schedule[x]["end_time"]
                for x in current_solution.rcpsp_schedule
            ]
        )
        last_jobs = [
            x
            for x in current_solution.rcpsp_schedule
            if current_solution.rcpsp_schedule[x]["end_time"] >= max_time - 5
        ]
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                list(current_solution.rcpsp_schedule),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        # add constraints
        for job in jobs_to_fix:
            start_time_j = current_solution.rcpsp_schedule[job]["start_time"]
            min_st = max(start_time_j - self.minus_delta, 0)
            max_st = min(start_time_j + self.plus_delta, max_time)
            for t in solver.index_time:
                if t < min_st or t > max_st:
                    lns_constraints.append(
                        solver.add_linear_constraint(
                            solver.x[solver.index_in_var[job]][t] == 0
                        )
                    )
        return lns_constraints


class _BaseConstraintHandlerStartTimeIntervalMRCPSP(ConstraintHandler):
    def __init__(
        self,
        problem: RCPSPModel,
        fraction_to_fix: float = 0.9,
        minus_delta: int = 2,
        plus_delta: int = 2,
    ):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.minus_delta = minus_delta
        self.plus_delta = plus_delta

    def adding_constraint_from_results_store(
        self,
        solver: Union[LP_MRCPSP_GUROBI, LP_MRCPSP_MATHOPT],
        result_storage: ResultStorage,
        **kwargs: Any
    ) -> Iterable[Any]:
        # set a good start: either solver.start_solution or result_storage.best_solution()
        current_solution, fit = result_storage.get_best_solution_fit()
        st = solver.start_solution
        if (
            self.problem.evaluate(st)["makespan"]
            < self.problem.evaluate(current_solution)["makespan"]
        ):
            current_solution = st
        solver.set_warm_start(current_solution)

        # add constraints
        constraints = []
        max_time = max(
            [
                current_solution.rcpsp_schedule[x]["end_time"]
                for x in current_solution.rcpsp_schedule
            ]
        )
        last_jobs = [
            x
            for x in current_solution.rcpsp_schedule
            if current_solution.rcpsp_schedule[x]["end_time"] >= max_time - 5
        ]
        nb_jobs = self.problem.n_jobs
        jobs_to_fix = set(
            random.sample(
                list(current_solution.rcpsp_schedule),
                int(self.fraction_to_fix * nb_jobs),
            )
        )
        for lj in last_jobs:
            if lj in jobs_to_fix:
                jobs_to_fix.remove(lj)
        for job in jobs_to_fix:
            start_time_j = current_solution.rcpsp_schedule[job]["start_time"]
            min_st = max(start_time_j - self.minus_delta, 0)
            max_st = min(start_time_j + self.plus_delta, max_time)
            for key in solver.variable_per_task[job]:
                t = key[2]
                if t < min_st or t > max_st:
                    constraints.append(solver.add_linear_constraint(solver.x[key] == 0))
        return constraints


class ConstraintHandlerStartTimeIntervalMRCPSP_GRB(
    GurobiConstraintHandler, _BaseConstraintHandlerStartTimeIntervalMRCPSP
):
    def adding_constraint_from_results_store(
        self, solver: LP_MRCPSP_GUROBI, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        constraints = _BaseConstraintHandlerStartTimeIntervalMRCPSP.adding_constraint_from_results_store(
            self, solver=solver, result_storage=result_storage, **kwargs
        )
        solver.model.update()
        return constraints


class ConstraintHandlerStartTimeIntervalMRCPSP_MathOpt(
    OrtoolsMathOptConstraintHandler, _BaseConstraintHandlerStartTimeIntervalMRCPSP
):
    def adding_constraint_from_results_store(
        self, solver: LP_MRCPSP_MATHOPT, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        constraints = _BaseConstraintHandlerStartTimeIntervalMRCPSP.adding_constraint_from_results_store(
            self, solver=solver, result_storage=result_storage, **kwargs
        )
        return constraints
