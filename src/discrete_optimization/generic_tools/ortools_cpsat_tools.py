#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, Sequence

import numpy as np
from ortools.sat.python.cp_model import (
    FEASIBLE,
    INFEASIBLE,
    OPTIMAL,
    UNKNOWN,
    Constraint,
    CpModel,
    CpSolverSolutionCallback,
    Domain,
)
from ortools.sat.python.cp_model import CpSolver as OrtoolsInternalCpSolver
from ortools.sat.python.cp_model_helper import Literal

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    ParametersCp,
)
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
        """Solve the problem with a CpSat solver from ortools library.

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
            try:
                # lowercase starting from ortools 9.16
                cstr.proto.clear()
            except AttributeError:
                # Uppercase for ortools < 9.15  (NB: method missing for ortools 9.15)
                cstr.proto.Clear()

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self._current_internal_objective_best_bound

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self._current_internal_objective_best_value

    def set_warm_start_from_previous_run(self):
        """Set warm start from previous run of the solver."""
        if self.solver is None:
            raise RuntimeError(
                "You cannot call `set_warm_start_from_previous_run()` before a first call to `solve()`"
            )
        self.cp_model.ClearHints()
        response = self.solver.ResponseProto()  # Get the raw response
        for i in range(len(response.solution)):
            var = self.cp_model.GetIntVarFromProtoIndex(i)
            self.cp_model.AddHint(var, response.solution[i])

    def init_model(self, **kwargs: Any) -> None:
        """Init cp model and reset stored variables if any."""
        self.cp_model = CpModel()

    def add_bool_constraint(self, cstr: Any) -> Any:
        return self.cp_model.add(cstr)

    def create_int_var(self, lb: int, ub: int, name: str) -> Any:
        return self.cp_model.new_int_var(lb=lb, ub=ub, name=name)

    def create_bool_var(self, name: str) -> Any:
        return self.cp_model.new_bool_var(name)

    def add_max_equality(self, target: Any, variables: Iterable[Any]) -> list[Any]:
        return [self.cp_model.add_max_equality(target=target, exprs=variables)]

    def minimize_variable(self, var: Any) -> None:
        self.cp_model.minimize(var)

    def maximize_variable(self, var: Any) -> None:
        self.cp_model.maximize(var)

    def create_int_var_from_possible_values(
        self, possible_values: Iterable[int], name: str
    ) -> Any:
        return self.cp_model.new_int_var_from_domain(
            domain=Domain.from_values(list(possible_values)), name=name
        )

    def create_int_var_from_flat_intervals(
        self, flat_intervals: list[int], name: str
    ) -> Any:
        return self.cp_model.new_int_var_from_domain(
            domain=Domain.from_flat_intervals(flat_intervals), name=name
        )

    def add_min_equality(self, target: Any, variables: Iterable[Any]) -> list[Any]:
        return [self.cp_model.add_min_equality(target, variables)]

    def add_abs_equality(self, target: Any, expr: Any) -> list[Any]:
        return [self.cp_model.add_abs_equality(target, expr)]

    def add_element(
        self, index: Any, expressions: Sequence[Any], target: Any
    ) -> list[Any]:
        return [
            self.cp_model.add_element(
                index=index, expressions=expressions, target=target
            )
        ]

    def add_modulo_equality(self, target: Any, expr: Any, mod: Any) -> list[Any]:
        return [self.cp_model.add_modulo_equality(target=target, expr=expr, mod=mod)]

    def add_implication(self, left_bool_expr: Any, right_bool_expr: Any) -> list[Any]:
        if isinstance(left_bool_expr, Literal) and isinstance(right_bool_expr, Literal):
            return [self.cp_model.add_implication(left_bool_expr, right_bool_expr)]
        elif isinstance(right_bool_expr, Constraint) or (
            isinstance(right_bool_expr, list)
            and all(isinstance(cstr, Constraint) for cstr in right_bool_expr)
        ):
            if isinstance(right_bool_expr, Constraint):
                right_bool_expr = [right_bool_expr]
            if isinstance(left_bool_expr, Literal) or np.isscalar(left_bool_expr):
                # single condition
                return [
                    cstr.only_enforce_if(left_bool_expr) for cstr in right_bool_expr
                ]
            else:
                # left hand side is an iterable of conditions
                return [
                    cstr.only_enforce_if(*left_bool_expr) for cstr in right_bool_expr
                ]
        else:
            constraint = self.cp_model.add(right_bool_expr)
            return self.add_implication(left_bool_expr, constraint)

    def add_implication_bool_and(
        self, left_bool_expr: Any, right_bool_exprs: Sequence[Any]
    ) -> list[Any]:
        return [
            self.cp_model.add_bool_and(right_bool_exprs).only_enforce_if(left_bool_expr)
        ]

    def add_implication_exactly_one(
        self, left_bool_expr: Any, right_bool_exprs: Iterable[Any]
    ) -> list[Any]:
        return [
            self.cp_model.add_exactly_one(right_bool_exprs).only_enforce_if(
                left_bool_expr
            )
        ]

    def add_implication_at_least_one(
        self, left_bool_expr: Any, right_bool_exprs: Iterable[Any]
    ) -> list[Any]:
        return [
            self.cp_model.add_at_least_one(right_bool_exprs).only_enforce_if(
                left_bool_expr
            )
        ]

    def add_implication_in_intervals(
        self, left_bool_expr: Any, right_expr: Any, intervals: Iterable[tuple[int, int]]
    ) -> list[Any]:
        return [
            self.cp_model.add_linear_expression_in_domain(
                linear_expr=right_expr,
                domain=Domain.from_intervals(
                    list(list(interval) for interval in intervals)
                ),
            ).only_enforce_if(left_bool_expr)
        ]

    def add_allowed_assignments(
        self, variables: Sequence[Any], values: Iterable[Sequence[int]]
    ) -> list[Any]:
        pass

    def add_exactly_one(self, exprs: Iterable[Any]) -> list[Any]:
        return [self.cp_model.add_exactly_one(exprs)]

    def add_at_most_one(self, exprs: Iterable[Any]) -> list[Any]:
        return [self.cp_model.add_at_most_one(exprs)]

    def add_at_least_one(self, exprs: Iterable[Any]) -> list[Any]:
        return [self.cp_model.add_at_least_one(exprs)]

    def hint_solution(self, hints: dict[Any, int | bool]) -> None:
        self.cp_model.clear_hints()
        for var, value in hints.items():
            self.cp_model.add_hint(var, value)


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
