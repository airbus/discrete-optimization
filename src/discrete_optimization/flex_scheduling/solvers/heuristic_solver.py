from typing import Any, Hashable, Optional

import numpy as np

from discrete_optimization.flex_scheduling.problem import (
    FlexProblem,
    ScheduleSolution,
    ScheduleSolutionPreemptive,
)
from discrete_optimization.flex_scheduling.simulator import PostprocessTool
from discrete_optimization.flex_scheduling.solvers.cpsat import CpSatFlexSolver
from discrete_optimization.flex_scheduling.solvers.cpsat_preempt import (
    CPSatFlexSPPreempt,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class WSCallback(Callback):
    def __init__(self, objectives: list[str]):
        self.objectives = objectives

    def on_step_end(self, step: int, res: ResultStorage, solver: SolverDO):
        solver.subsolver.set_warm_start_from_sol(res[-1][0])
        return False


class HeuristicSolverFlexProblem(SolverDO, WarmstartMixin):
    hyperparameters = []

    def __init__(
        self,
        problem: FlexProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem, params_objective_function=params_objective_function, **kwargs
        )
        self.problem: FlexProblem = self.problem
        self.previous_calendar: dict[Hashable, np.ndarray] = {}
        self.ws: ScheduleSolution = None

    def modify_calendars(self):
        fsp: FlexProblem = self.problem
        for resource in fsp.resources:
            calendar = resource.calendar_availability
            indices = np.nonzero(calendar)
            non_zero_values = set(calendar[indices])
            max_value = np.max(calendar)
            self.previous_calendar[resource.id] = np.copy(calendar)
            if len(non_zero_values) > 0 and max_value >= 2:
                # if 0<res_availability<max_capacity, cut to 0 !
                indices_ = np.nonzero(
                    np.logical_and(calendar > 0, calendar != max_value)
                )
                calendar[indices_] = 0
        fsp.update_data_placeholders()

    def put_back_calendar(self):
        for resource in self.problem.resources:
            resource.calendar_availability = self.previous_calendar[resource.id]
        self.problem.update_data_placeholders()

    def set_warm_start(self, solution: Solution) -> None:
        self.ws = solution

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        params_cp_first_solve: ParametersCp = None,
        params_cp_second_solve: ParametersCp = None,
        time_limit_per_objectives_first_solve: int = 150,
        time_limit_per_objectives_second_solve: int = 150,
        objectives_first_solve: list[str] = None,
        objectives_second_solve: list[str] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        self.modify_calendars()
        subsolver = CpSatFlexSolver(self.problem)
        subsolver.init_model()
        if self.ws is not None:
            subsolver.set_warm_start_from_sol(self.ws)
        if objectives_first_solve is None:
            objectives_first_solve = [
                "tardiness",
                "earliness",
                "resource_cost",
                "makespan",
            ]
        lexico_solver = LexicoSolver(subsolver=subsolver, problem=self.problem)
        if params_cp_first_solve is None:
            params_cp_first_solve = ParametersCp.default_cpsat()
            params_cp_first_solve.nb_process = 16
        res = lexico_solver.solve(
            callbacks=[WSCallback(objectives_first_solve)],
            objectives=objectives_first_solve,
            subsolver_callbacks=[],
            time_limit=time_limit_per_objectives_first_solve,
            parameters_cp=params_cp_first_solve,
            ortools_cpsat_solver_kwargs={"log_search_progress": True},
        )
        sol = res[-1][0]
        # Put back real calendar
        self.put_back_calendar()
        # Post process schedule left.
        pptool = PostprocessTool(flex_problem=self.problem, solution=sol)
        schedule = pptool.post_process_left(
            flex_problem=self.problem,
            solution=sol,
            keep_min_time=False,
            keep_strict_order_task=False,
        )
        sol_ = ScheduleSolutionPreemptive(
            problem=self.problem,
            schedule=[
                schedule[self.problem.index_to_task_id[i]]
                for i in range(self.problem.nb_tasks)
            ],
            modes=np.ones(self.problem.nb_tasks),
        )
        solver = CPSatFlexSPPreempt(problem=self.problem)
        solver.init_model()
        solver.set_warm_start_from_sol(sol_)
        if params_cp_second_solve is None:
            params_cp_second_solve = ParametersCp.default_cpsat()
            params_cp_second_solve.nb_process = 16
        if objectives_second_solve is None:
            objectives_second_solve = [
                "tardiness",
                "earliness",
                "resource_cost",
                "makespan",
            ]
        lexico_solver = LexicoSolver(subsolver=solver, problem=self.problem)
        res_preemptive = lexico_solver.solve(
            callbacks=[WSCallback(objectives_second_solve)],
            objectives=objectives_second_solve,
            subsolver_callbacks=kwargs.get("subsolver_callbacks", []),
            time_limit=time_limit_per_objectives_second_solve,
            parameters_cp=params_cp_second_solve,
            ortools_cpsat_solver_kwargs=dict(
                log_search_progress=True,
                # probing_deterministic_time_limit=0,
                # use_sat_inprocessing=False,
                # linearization_level=0,
                # fix_variables_to_their_hinted_value=False,
                debug_crash_on_bad_hint=False,
            ),
        )
        res.extend(res_preemptive.list_solution_fits)
        return res
