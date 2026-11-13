# Generic optal-cp python wrapper.
import asyncio
import threading
from typing import Awaitable, Iterable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")
import logging
from abc import abstractmethod
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    ParametersCp,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import (
    BoundsProviderMixin,
    StatusSolver,
)
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
    optalcp_available = False
else:
    optalcp_available = True

logger = logging.getLogger(__name__)


class OptalCpSolver(CpSolver, BoundsProviderMixin):
    hyperparameters = [
        CategoricalHyperparameter(
            name="searchType",
            choices=["Auto", "LNS", "FDS", "FDSDual", "SetTimes", "FDSLB"],
            default="Auto",
        )
    ]
    cp_model: Optional["cp.Model"] = None
    early_stopping_exception: Optional[Exception] = None
    current_bound: int | float | None = None
    current_obj: int | float | None = None
    current_solution: Optional["cp.Solution"] = None
    status_solver: StatusSolver = None
    use_warm_start: bool = False
    warm_start_solution: Optional["cp.Solution"] = None

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        if not optalcp_available:
            raise RuntimeError("You must install optalcp to use this solver.")
        super().__init__(problem, params_objective_function, **kwargs)

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self.current_bound

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self.current_obj

    def _solve_sync(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[int] = 60,
        parameters_cp: Optional[ParametersCp] = None,
        **args: Any,
    ) -> ResultStorage:
        args = self.complete_with_default_hyperparameters(args)
        kw = {"timeLimit": time_limit, "nbWorkers": parameters_cp.nb_process}
        kw.update(args)
        result = self.cp_model.solve(parameters=cp.Parameters(**kw))
        if result.solution is not None:
            if result.proof:
                self.status_solver = StatusSolver.OPTIMAL
            else:
                self.status_solver = StatusSolver.SATISFIED
            self.current_solution = result.solution
            sol = self.retrieve_solution(result.solution)
            fit = self.aggreg_from_sol(sol)
            return self.create_result_storage([(sol, fit)])
        else:
            if result.proof:
                self.status_solver = StatusSolver.UNSATISFIABLE
            else:
                self.status_solver = StatusSolver.UNKNOWN
            return self.create_result_storage()

    async def _solve_async(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[int] = 60,
        parameters_cp: Optional[ParametersCp] = None,
        do_not_retrieve_solutions: bool = False,
        **args: Any,
    ):
        if parameters_cp is None:
            parameters_cp = ParametersCp.default_cpsat()
        args = self.complete_with_default_hyperparameters(args)
        self.current_bound = None
        callback_do = CallbackList(callbacks)
        solver = cp.Solver()
        callback_optal = OptalSolutionCallback(
            do_solver=self,
            optal_solver=solver,
            callback=callback_do,
            do_not_retrieve_solutions=do_not_retrieve_solutions,
        )
        callback_do.on_solve_start(self)
        solver.on_solution = callback_optal.on_solution
        solver.on_objective_bound = callback_optal.on_bound
        kw = {"timeLimit": time_limit, "nbWorkers": parameters_cp.nb_process}
        kw.update(args)
        kw_params = {
            key: kw[key]
            for key in kw
            if key not in {"lower_bound_method"}
            and key in cp.Parameters.__annotations__.keys()
        }
        logger.debug(kw_params)
        if self.use_warm_start and self.warm_start_solution is not None:
            result = await solver.solve(
                model=self.cp_model,
                params=cp.Parameters(**kw_params),
                warm_start=self.warm_start_solution,
            )
        else:
            result = await solver.solve(
                model=self.cp_model, params=cp.Parameters(**kw_params)
            )
        if result.solution is not None:
            if result.proof:
                self.status_solver = StatusSolver.OPTIMAL
            else:
                self.status_solver = StatusSolver.SATISFIED
        else:
            if result.proof:
                self.status_solver = StatusSolver.UNSATISFIABLE
            else:
                self.status_solver = StatusSolver.UNKNOWN
        callback_do.on_solve_end(callback_optal.res, self)
        return callback_optal.res

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[int] = 60,
        parameters_cp: Optional[ParametersCp] = None,
        do_not_retrieve_solutions: bool = False,
        **args: Any,
    ) -> ResultStorage:
        """Solve the cp model

        Args:
            callbacks:
            time_limit:
            parameters_cp:
            do_not_retrieve_solutions: if True, do no try to retrieve solutions.
                Useful when using preview version of optalcp.
            **args:

        Returns:

        """
        if self.cp_model is None:
            self.init_model(**args)
        return run_async_synchronously(
            self._solve_async(
                callbacks=callbacks,
                time_limit=time_limit,
                parameters_cp=parameters_cp,
                do_not_retrieve_solutions=do_not_retrieve_solutions,
                **args,
            )
        )

    @abstractmethod
    def retrieve_solution(self, result: "cp.Solution") -> Solution:
        """Construct a d-o solution from OptalCP solver solution.

        Args:
            solution: OptalCP solution

        Returns:
            d-o solution

        """
        ...

    def minimize_variable(self, var: Any) -> None:
        self.cp_model.minimize(var)

    def maximize_variable(self, var: Any) -> None:
        self.cp_model.maximize(var)

    def add_bool_constraint(self, cstr: Any) -> Any:
        self.cp_model.enforce(cstr)
        return cstr

    def create_int_var(self, lb: int, ub: int, name: str) -> Any:
        return self.cp_model.int_var(min=lb, max=ub, name=name)

    def create_int_var_from_possible_values(
        self, possible_values: Iterable[int], name: str
    ) -> Any:
        lb = min(possible_values)
        ub = max(possible_values)
        var = self.cp_model.int_var(min=lb, max=ub, name=name)
        self.cp_model.enforce(
            self.cp_model.sum(var == value for value in possible_values) >= 1
        )

        return var

    def create_int_var_from_flat_intervals(
        self, flat_intervals: list[int], name: str
    ) -> Any:
        lb = min(flat_intervals)
        ub = max(flat_intervals)
        var = self.cp_model.int_var(min=lb, max=ub, name=name)
        self.cp_model.enforce(
            self.cp_model.sum(
                var.in_range(flat_intervals[i * 2], flat_intervals[i * 2 + 1])
                for i in range(len(flat_intervals) // 2)
            )
            >= 1
        )
        return var

    def create_bool_var(self, name: str) -> Any:
        return self.cp_model.bool_var(name=name)

    def add_max_equality(self, target: Any, variables: Iterable[Any]) -> list[Any]:
        return [self.cp_model.enforce(target == self.cp_model.max(variables))]

    def add_min_equality(self, target: Any, variables: Iterable[Any]) -> list[Any]:
        return [self.cp_model.enforce(target == self.cp_model.min(variables))]

    def add_abs_equality(self, target: Any, expr: Any) -> list[Any]:
        return [self.cp_model.enforce(target == self.cp_model.abs(expr))]

    def add_element(
        self, index: Any, expressions: Sequence[Any], target: Any
    ) -> list[Any]:
        return [
            self.cp_model.enforce(
                target == self.cp_model.element(array=expressions, subscript=index)
            )
        ]

    def add_modulo_equality(self, target: Any, expr: Any, mod: Any) -> list[Any]:
        raise NotImplementedError()

    def add_implication(self, left_bool_expr: Any, right_bool_expr: Any) -> list[Any]:
        if right_bool_expr is None or isinstance(right_bool_expr, cp.Constraint):
            raise NotImplementedError(
                "Cannot imply a constraint already added to the model."
            )
        if isinstance(right_bool_expr, list):
            # add implication for each right member
            return [
                cstr
                for right in right_bool_expr
                for cstr in self.add_implication(left_bool_expr, right)
            ]
        # From now, right hand is a bool expression or a literal
        if not isinstance(right_bool_expr, cp.ModelElement):
            right_bool_expr = bool(right_bool_expr)
        if isinstance(left_bool_expr, cp.ModelElement):
            cstr = self.cp_model.implies(left_bool_expr, right_bool_expr)
        elif np.isscalar(left_bool_expr):
            left_eval = bool(left_bool_expr)
            cstr = self.cp_model.implies(left_eval, right_bool_expr)
        else:
            # multiple left hand: all must be true to imply right hand
            aggregated_left_hand = self.bool_and(
                tuple(
                    expr if isinstance(expr, cp.ModelElement) else bool(expr)
                    for expr in left_bool_expr
                )
            )
            return self.add_implication(aggregated_left_hand, right_bool_expr)
        return [self.cp_model.enforce(cstr)]

    def add_implication_bool_and(
        self, left_bool_expr: Any, right_bool_exprs: Sequence[Any]
    ) -> list[Any]:
        return super().add_implication_bool_and(
            left_bool_expr=left_bool_expr, right_bool_exprs=right_bool_exprs
        )

    def add_implication_exactly_one(
        self, left_bool_expr: Any, right_bool_exprs: Iterable[Any]
    ) -> list[Any]:
        return super().add_implication_exactly_one(
            left_bool_expr=left_bool_expr, right_bool_exprs=right_bool_exprs
        )

    def add_implication_at_least_one(
        self, left_bool_expr: Any, right_bool_exprs: Iterable[Any]
    ) -> list[Any]:
        return super().add_implication_at_least_one(
            left_bool_expr=left_bool_expr, right_bool_exprs=right_bool_exprs
        )

    def add_implication_in_intervals(
        self, left_bool_expr: Any, right_expr: Any, intervals: Iterable[tuple[int, int]]
    ) -> list[Any]:
        return super().add_implication_in_intervals(
            left_bool_expr=left_bool_expr, right_expr=right_expr, intervals=intervals
        )

    def add_allowed_assignments(
        self, variables: Sequence[Any], values: Iterable[Sequence[int]]
    ) -> list[Any]:
        return super().add_allowed_assignments(variables=variables, values=values)

    def sum(self, exprs: Iterable[Any]) -> Any:
        return self.cp_model.sum(exprs)

    def add_exactly_one(self, exprs: Iterable[Any]) -> list[Any]:
        return super().add_exactly_one(exprs)

    def add_at_most_one(self, exprs: Iterable[Any]) -> list[Any]:
        return super().add_at_most_one(exprs)

    def add_at_least_one(self, exprs: Iterable[Any]) -> list[Any]:
        return super().add_at_least_one(exprs)

    def set_warm_start_from_previous_run(self):
        """Set warm start from the previous solve run."""
        if self.current_solution is not None:
            self.warm_start_solution = self.current_solution
            self.use_warm_start = True
            logger.info("Warm start set from previous run")


class OptalSolutionCallback:
    def __init__(
        self,
        do_solver: OptalCpSolver,
        optal_solver: "cp.Solver",
        callback: Callback,
        do_not_retrieve_solutions: bool = False,
    ):
        super().__init__()
        self.do_solver = do_solver
        self.optal_solver = optal_solver
        self.callback = callback
        self.res = do_solver.create_result_storage([])
        self.nb_solutions = 0
        self.current_solve_result: Optional[cp.SolveResult] = None
        self.do_not_retrieve_solutions = do_not_retrieve_solutions
        self.stopping = False

    def on_solution(self, event: "cp.SolutionEvent") -> None:
        self.do_solver.current_obj = event.solution.get_objective()
        if not self.do_not_retrieve_solutions:
            self.do_solver.current_solution = event.solution
            sol = self.do_solver.retrieve_solution(event.solution)
            fit = self.do_solver.aggreg_from_sol(sol)
            self.res.append((sol, fit))
        self.nb_solutions += 1
        try:
            stopping = self.callback.on_step_end(
                step=self.nb_solutions, res=self.res, solver=self.do_solver
            )
        except Exception as e:
            self.do_solver.early_stopping_exception = e
            stopping = True
            logger.debug("should stop")
        else:
            if stopping:
                self.do_solver.early_stopping_exception = SolveEarlyStop(
                    f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                )
        if stopping and not self.stopping:
            logger.debug("stopping")
            self.optal_solver.stop("stopped by usr callback")
            self.stopping = (
                True  # avoid retrying to stop afterwards (would result in a kill -9)
            )

    def on_bound(self, event: "cp.ObjectiveBoundEntry") -> None:
        self.do_solver.current_bound = event.value
        logger.debug("on_bound")
        stopping = self.callback.on_step_end(
            step=self.nb_solutions, res=self.res, solver=self.do_solver
        )


def run_async_synchronously(coroutine: Awaitable[T]) -> T:
    """
    Runs a coroutine synchronously, blocking until it completes.

    This helper handles two cases:
    1. No event loop is running (standard script): Uses asyncio.run()
    2. An event loop is running (e.g., Jupyter): Runs the coroutine in a
       separate thread to avoid blocking the existing loop/raising runtime errors.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result = None
        exception = None

        def runner():
            nonlocal result, exception
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coroutine)
            except Exception as e:
                exception = e
            finally:
                new_loop.close()

        t = threading.Thread(target=runner)
        t.start()
        t.join()

        if exception:
            raise exception
        return result
    else:
        # Case 1: No loop running. Standard run.
        return asyncio.run(coroutine)
