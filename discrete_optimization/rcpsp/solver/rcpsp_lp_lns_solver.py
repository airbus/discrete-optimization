#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from enum import Enum
from typing import Any, Hashable, List, Mapping, Optional

import mip
import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_evaluate_function_aggregated,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import (
    LNS_MILP,
    ConstraintHandler,
    InitialSolution,
)
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
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
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.solver import CP_MRCPSP_MZN
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import (
    LP_MRCPSP,
    LP_MRCPSP_GUROBI,
    LP_RCPSP,
)
from discrete_optimization.rcpsp.solver.rcpsp_pile import (
    GreedyChoice,
    PileSolverRCPSP,
    PileSolverRCPSP_Calendar,
)
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP

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
                list_solution_fits=[(solution, fit)],
                best_solution=solution,
                mode_optim=self.params_objective_function.sense_function,
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
            store_solution = sa.solve(dummy, nb_iteration_max=10000)
        return store_solution


class ConstraintHandlerFixStartTime(ConstraintHandler):
    def __init__(self, problem: RCPSPModel, fraction_fix_start_time: float = 0.9):
        self.problem = problem
        self.fraction_fix_start_time = fraction_fix_start_time

    def adding_constraint_from_results_store(
        self, milp_solver: LP_RCPSP, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:

        nb_jobs = self.problem.n_jobs + 2
        constraints_dict = {}
        current_solution, fit = result_storage.get_best_solution_fit()
        current_solution: RCPSPSolution = current_solution
        # Starting point :
        start = []
        for j in self.problem.tasks_list:
            start_time_j = current_solution.rcpsp_schedule[j]["start_time"]
            for t in milp_solver.index_time:
                if start_time_j == t:
                    start += [(milp_solver.x[milp_solver.index_in_var[j]][t], 1)]
                else:
                    start += [(milp_solver.x[milp_solver.index_in_var[j]][t], 0)]
        milp_solver.model.start = start

        # Fix start time for a subset of task.
        jobs_to_fix = set(
            random.sample(
                list(current_solution.rcpsp_schedule),
                int(self.fraction_fix_start_time * nb_jobs),
            )
        )
        constraints_dict["fix_start_time"] = []
        for job_to_fix in jobs_to_fix:
            for t in milp_solver.index_time:
                if current_solution.rcpsp_schedule[job_to_fix]["start_time"] == t:
                    constraints_dict["fix_start_time"].append(
                        milp_solver.model.add_constr(
                            milp_solver.x[milp_solver.index_in_var[job_to_fix]][t] == 1
                        )
                    )
                else:
                    constraints_dict["fix_start_time"].append(
                        milp_solver.model.add_constr(
                            milp_solver.x[milp_solver.index_in_var[job_to_fix]][t] == 0
                        )
                    )
            if milp_solver.solver_name == mip.GRB:
                milp_solver.model.solver.update()
        return constraints_dict

    def remove_constraints_from_previous_iteration(
        self, milp_solver: LP_RCPSP, previous_constraints: Mapping[Hashable, Any]
    ):
        milp_solver.model.remove(previous_constraints["fix_start_time"])
        if milp_solver.solver_name == mip.GRB:
            milp_solver.model.solver.update()


class ConstraintHandlerStartTimeInterval(ConstraintHandler):
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
        self, milp_solver: LP_RCPSP, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        constraints_dict = {}
        current_solution, fit = result_storage.get_best_solution_fit()
        # Starting point :
        current_solution: RCPSPSolution = current_solution
        # Starting point :
        start = []
        for j in self.problem.tasks_list:
            start_time_j = current_solution.rcpsp_schedule[j]["start_time"]
            for t in milp_solver.index_time:
                if start_time_j == t:
                    start += [(milp_solver.x[milp_solver.index_in_var[j]][t], 1)]
                else:
                    start += [(milp_solver.x[milp_solver.index_in_var[j]][t], 0)]
        milp_solver.model.start = start
        constraints_dict["range_start_time"] = []
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
            for t in milp_solver.index_time:
                if t < min_st or t > max_st:
                    constraints_dict["range_start_time"].append(
                        milp_solver.model.add_constr(
                            milp_solver.x[milp_solver.index_in_var[job]][t] == 0
                        )
                    )
        if milp_solver.solver_name == mip.GRB:
            milp_solver.model.solver.update()
        return constraints_dict

    def remove_constraints_from_previous_iteration(
        self, milp_solver: LP_RCPSP, previous_constraints: Mapping[Hashable, Any]
    ):
        milp_solver.model.remove(previous_constraints["range_start_time"])
        if milp_solver.solver_name == mip.GRB:
            milp_solver.model.solver.update()


class ConstraintHandlerStartTimeIntervalMRCPSP(ConstraintHandler):
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
        self, milp_solver: LP_MRCPSP, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        current_solution, fit = result_storage.get_best_solution_fit()
        current_solution: RCPSPSolution = current_solution
        st = milp_solver.start_solution
        if (
            self.problem.evaluate(st)["makespan"]
            < self.problem.evaluate(current_solution)["makespan"]
        ):
            current_solution = st
        start = []
        modes_dict = self.problem.build_mode_dict(current_solution.rcpsp_modes)
        for j in current_solution.rcpsp_schedule:
            start_time_j = current_solution.rcpsp_schedule[j]["start_time"]
            mode_j = modes_dict[j]
            start += [
                (
                    milp_solver.durations[j],
                    self.problem.mode_details[j][mode_j]["duration"],
                )
            ]
            for k in milp_solver.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == mode_j:
                    start += [(milp_solver.x[k], 1)]
                else:
                    start += [(milp_solver.x[k], 0)]
        milp_solver.model.start = start
        constraints_dict = {"range_start_time": []}
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
            for key in milp_solver.variable_per_task[job]:
                t = key[2]
                if t < min_st or t > max_st:
                    constraints_dict["range_start_time"].append(
                        milp_solver.model.add_constr(milp_solver.x[key] == 0)
                    )
        if milp_solver.solver_name == mip.GRB:
            milp_solver.model.solver.update()
        return constraints_dict

    def remove_constraints_from_previous_iteration(
        self, milp_solver: LP_MRCPSP, previous_constraints: Mapping[Hashable, Any]
    ):
        milp_solver.model.remove(previous_constraints["range_start_time"])
        if milp_solver.solver_name == mip.GRB:
            milp_solver.model.solver.update()


class ConstraintHandlerStartTimeIntervalMRCPSP_GRB(ConstraintHandler):
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
        self, milp_solver: LP_MRCPSP_GUROBI, result_storage: ResultStorage
    ) -> Mapping[Hashable, Any]:
        current_solution, fit = result_storage.get_best_solution_fit()
        st = milp_solver.start_solution
        if (
            self.problem.evaluate(st)["makespan"]
            < self.problem.evaluate(current_solution)["makespan"]
        ):
            current_solution = st
        start = []
        modes_dict = self.problem.build_mode_dict(current_solution.rcpsp_modes)
        for j in current_solution.rcpsp_schedule:
            start_time_j = current_solution.rcpsp_schedule[j]["start_time"]
            mode_j = modes_dict[j]
            start += [
                (
                    milp_solver.durations[j],
                    self.problem.mode_details[j][mode_j]["duration"],
                )
            ]
            for k in milp_solver.variable_per_task[j]:
                task, mode, time = k
                if start_time_j == time and mode == mode_j:
                    milp_solver.x[k].start = 1
                    milp_solver.starts[j].start = start_time_j
                else:
                    milp_solver.x[k].start = 0
        constraints_dict = {"range_start_time": []}
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
            for key in milp_solver.variable_per_task[job]:
                t = key[2]
                if t < min_st or t > max_st:
                    constraints_dict["range_start_time"].append(
                        milp_solver.model.addLConstr(milp_solver.x[key] == 0)
                    )
        milp_solver.model.update()
        return constraints_dict

    def remove_constraints_from_previous_iteration(
        self,
        milp_solver: LP_MRCPSP_GUROBI,
        previous_constraints: Mapping[Hashable, Any],
    ):
        milp_solver.model.remove(previous_constraints.get("range_start_time", []))
        milp_solver.model.update()


class LNS_LP_RCPSP_SOLVER(SolverRCPSP):
    problem: RCPSPModel

    def __init__(
        self,
        problem: RCPSPModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        solver = LP_MRCPSP(problem=problem, **kwargs)
        solver.init_model(greedy_start=False)
        self.parameters_milp = ParametersMilp(
            time_limit=100,
            pool_solutions=1000,
            mip_gap_abs=0.001,
            mip_gap=0.001,
            retrieve_all_solution=True,
            n_solutions_max=100,
        )
        self.constraint_handler = ConstraintHandlerStartTimeIntervalMRCPSP(
            problem=problem, fraction_to_fix=0.6, minus_delta=5, plus_delta=5
        )
        self.initial_solution_provider = InitialSolutionRCPSP(
            problem=problem,
            initial_method=InitialMethodRCPSP.DUMMY,
            params_objective_function=self.params_objective_function,
        )
        self.lns_solver = LNS_MILP(
            problem=problem,
            milp_solver=solver,
            initial_solution_provider=self.initial_solution_provider,
            constraint_handler=self.constraint_handler,
            params_objective_function=self.params_objective_function,
        )

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs
    ) -> ResultStorage:
        return self.lns_solver.solve_lns(
            parameters_milp=self.parameters_milp,
            nb_iteration_lns=kwargs.get("nb_iteration_lns", 100),
            callbacks=callbacks,
        )
