#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from typing import Any, List, Optional

from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNKNOWN,
    CpModel,
    CpSolver,
    CpSolverSolutionCallback,
)

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolver,
    ParametersCP,
    StatusSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class OrtoolsCPSatSolver(CPSolver):
    """Generic ortools cp-sat solver."""

    cp_model: Optional[CpModel] = None
    early_stopping_exception: Optional[Exception] = None

    @abstractmethod
    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        """Construct a do solution from the cpsat solver internal solution.

        It will be called each time the cpsat solver find a new solution.
        At that point, value of internal variables are accessible via `cpsolvercb.Value(VARIABLE_NAME)`.

        Args:
            cpsolvercb: the ortools callback called when the cpsat solver finds a new solution.

        Returns:
            the intermediate solution, at do format.

        """
        ...

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        parameters_cp: Optional[ParametersCP] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the problem with a CPSat solver drom ortools library.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            parameters_cp: parameters specific to cp solvers.
                We use here only `parameters_cp.time_limit` and `parameters_cp.nb_process`.
            **kwargs: keyword arguments passed to `self.init_model()`

        Returns:

        A dedicated ortools callback is used to:
        - update a resultstorage each time a new solution is found by the cpsat solver.
        - call the user (do) callbacks at each new solution, with the possibility of early stopping if the callback return True.

        This ortools callback use the method `self.retrieve_solution()` to reconstruct a do Solution from the cpsat solve internal state.

        """
        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        if self.cp_model is None:
            self.init_model(**kwargs)
        if parameters_cp is None:
            parameters_cp = ParametersCP.default_cpsat()
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = parameters_cp.time_limit
        solver.parameters.num_workers = parameters_cp.nb_process
        ortools_callback = OrtoolsCallback(do_solver=self, callback=callbacks_list)
        status = solver.Solve(self.cp_model, ortools_callback)
        self.status_solver = cpstatus_to_dostatus(status_from_cpsat=status)
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception
        res = ortools_callback.res
        callbacks_list.on_solve_end(res=res, solver=self)
        return res


class OrtoolsCallback(CpSolverSolutionCallback):
    def __init__(self, do_solver: OrtoolsCPSatSolver, callback: Callback):
        super().__init__()
        self.do_solver = do_solver
        self.callback = callback
        self.res = ResultStorage(
            [],
            mode_optim=self.do_solver.params_objective_function.sense_function,
            limit_store=False,
        )
        self.nb_solutions = 0

    def on_solution_callback(self) -> None:
        self.store_current_solution()
        self.nb_solutions += 1
        # end of step callback: stopping?
        try:
            stopping = self.callback.on_step_end(
                step=self.nb_solutions, res=self.res, solver=self.do_solver
            )
        except Exception as e:
            self.do_solver.early_stopping_exception = e
            stopping = True
        else:
            if stopping:
                self.do_solver.early_stopping_exception = SolveEarlyStop(
                    f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                )
        if stopping:
            self.StopSearch()

    def store_current_solution(self):
        sol = self.do_solver.retrieve_solution(cpsolvercb=self)
        fit = self.do_solver.aggreg_from_sol(sol)
        self.res.add_solution(solution=sol, fitness=fit)


def cpstatus_to_dostatus(status_from_cpsat) -> StatusSolver:
    """

    :param status_from_cpsat: either [UNKNOWN,INFEASIBLE,OPTIMAL,FEASIBLE] from ortools.cp api.
    :return: Status
    """
    if status_from_cpsat == UNKNOWN:
        return StatusSolver.UNKNOWN
    if status_from_cpsat == INFEASIBLE:
        return StatusSolver.UNSATISFIABLE
    if status_from_cpsat == OPTIMAL:
        return StatusSolver.OPTIMAL
    if status_from_cpsat == FEASIBLE:
        return StatusSolver.SATISFIED
