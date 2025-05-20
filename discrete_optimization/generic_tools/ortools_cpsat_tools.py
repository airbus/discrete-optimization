#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNKNOWN,
    Constraint,
    CpModel,
)
from ortools.sat.python.cp_model import CpSolver as OrtoolsInternalCpSolver
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import CpSolver, ParametersCp
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import (
    BoundsProviderMixin,
    StatusSolver,
)
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class OrtoolsCpSatSolver(CpSolver, BoundsProviderMixin):
    """Generic ortools cp-sat solver."""

    cp_model: Optional[CpModel] = None
    solver: Optional[OrtoolsInternalCpSolver] = None
    clb: Optional[CpSolverSolutionCallback] = None
    early_stopping_exception: Optional[Exception] = None

    _current_internal_objective_best_value: Optional[float] = None
    _current_internal_objective_best_bound: Optional[float] = None

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
        callbacks: Optional[list[Callback]] = None,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: Optional[float] = 100.0,
        ortools_cpsat_solver_kwargs: Optional[dict[str, Any]] = None,
        retrieve_stats: bool = False,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the problem with a CpSat solver drom ortools library.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            time_limit: the solve process stops after this time limit (in seconds).
                If None, no time limit is applied.
            parameters_cp: parameters specific to cp solvers.
                We use here only `parameters_cp.nb_process`.
            ortools_cpsat_solver_kwargs: used to customize the underlying ortools solver.
                Each key/value will update the corresponding attribute from the ortools.sat.python.cp_model.CpSolver
            retrieve_stats: retrieve detailed stats of cpsat solving in the cpsat callback
            and store it in the res object.
            **kwargs: keyword arguments passed to `self.init_model()`

        Returns:

        A dedicated ortools callback is used to:
        - update a resultstorage each time a new solution is found by the cpsat solver.
        - call the user (do) callbacks at each new solution, with the possibility of early stopping if the callback return True.

        This ortools callback use the method `self.retrieve_solution()` to reconstruct a do Solution from the cpsat solve internal state.

        """
        self.early_stopping_exception = None
        self._current_internal_objective_best_bound = None
        self._current_internal_objective_best_value = None
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        if self.cp_model is None:
            self.init_model(**kwargs)
        if parameters_cp is None:
            parameters_cp = ParametersCp.default_cpsat()
        solver = OrtoolsInternalCpSolver()
        self.solver = solver
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_workers = parameters_cp.nb_process
        if ortools_cpsat_solver_kwargs is not None:
            # customize solver
            for k, v in ortools_cpsat_solver_kwargs.items():
                setattr(solver.parameters, k, v)
        ortools_callback = OrtoolsCpSatCallback(
            do_solver=self, callback=callbacks_list, retrieve_stats=retrieve_stats
        )
        self.clb = ortools_callback
        status = solver.Solve(self.cp_model, ortools_callback)
        self.status_solver = cpstatus_to_dostatus(status_from_cpsat=status)
        res = ortools_callback.res

        # update current obj and bound
        if (
            len(res) > 0
        ):  # do not update when no solution (obj and bound set to 0. by cpsat which seems wrong)
            self._current_internal_objective_best_bound = (
                self.solver.BestObjectiveBound()
            )
            self._current_internal_objective_best_value = self.solver.ObjectiveValue()
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception
        callbacks_list.on_solve_end(res=res, solver=self)
        return res

    def remove_constraints(self, constraints: Iterable[Any]) -> None:
        """Remove the internal model constraints.

        Args:
            constraints: constraints created for instance with `add_lexico_constraint()`

        Returns:

        """
        for cstr in constraints:
            if not isinstance(cstr, Constraint):
                raise RuntimeError()
            cstr.proto.Clear()

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self._current_internal_objective_best_bound

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self._current_internal_objective_best_value


class OrtoolsCpSatCallback(CpSolverSolutionCallback):
    def __init__(
        self,
        do_solver: OrtoolsCpSatSolver,
        callback: Callback,
        retrieve_stats: bool = False,
    ):
        super().__init__()
        self.do_solver = do_solver
        self.callback = callback
        self.retrieve_stats = retrieve_stats
        self.res = do_solver.create_result_storage()
        if retrieve_stats:
            self.res.stats = []
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
        self.res.append((sol, fit))
        if self.retrieve_stats:
            self.res.stats.append(
                {
                    "bound": self.BestObjectiveBound(),
                    "obj": self.ObjectiveValue(),
                    "time": self.UserTime(),
                    "num_conflicts": self.NumConflicts(),
                }
            )
        # update current bound and value
        self.do_solver._current_internal_objective_best_value = self.ObjectiveValue()
        self.do_solver._current_internal_objective_best_bound = (
            self.BestObjectiveBound()
        )


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
