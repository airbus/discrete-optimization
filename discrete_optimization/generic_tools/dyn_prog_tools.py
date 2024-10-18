#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import inspect
import logging
from abc import abstractmethod
from typing import Any, List, Optional

import didppy as dp

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

solvers = {
    x.__name__: x
    for x in [
        dp.ForwardRecursion,
        dp.CABS,
        dp.CAASDy,
        dp.LNBS,
        dp.DFBB,
        dp.CBFS,
        dp.ACPS,
        dp.APPS,
        dp.DBDFS,
        dp.BreadthFirstSearch,
        dp.DDLNS,
        dp.WeightedAstar,
        dp.ExpressionBeamSearch,
    ]
}

logger = logging.getLogger(__name__)


class DpCallback:
    def __init__(self, do_solver: "DpSolver", callback: Callback):
        super().__init__()
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0

    def on_solution_callback(self, sol: dp.Solution) -> bool:
        self.nb_solutions += 1
        self.store_current_solution(sol)
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
            return stopping

    def store_current_solution(self, sol: dp.Solution):
        solution = self.do_solver.retrieve_solution(sol)
        fit = self.do_solver.aggreg_from_sol(solution)
        self.res.append((solution, fit))


class DpSolver(SolverDO):
    early_stopping_exception: Optional[Exception] = None
    model: dp.Model = None
    hyperparameters = [
        CategoricalHyperparameter(name="solver", choices=solvers, default=dp.CABS)
    ]
    initial_solution: Optional[list[dp.Transition]] = None

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        ...

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        time_limit: Optional[float] = 100.0,
        retrieve_intermediate_solutions: bool = True,
        **kwargs: Any,
    ) -> ResultStorage:
        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        if self.model is None:
            self.init_model(**kwargs)
        did_callback = DpCallback(do_solver=self, callback=callbacks_list)
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if self.initial_solution is not None:
            kwargs["initial_solution"] = self.initial_solution
        if "quiet" not in kwargs:
            kwargs["quiet"] = False
        solver_cls = kwargs["solver"]
        try:
            solver_allowed_params = inspect.signature(solver_cls).parameters
            kwargs_solver = {
                k: v for k, v in kwargs.items() if k in solver_allowed_params
            }
        except:
            # Previous mode, for python<=3.9
            for k in list(kwargs.keys()):
                if k not in {"threads", "initial_solution"}:
                    kwargs.pop(k)
                if k == "threads" and solver_cls in {dp.DDLNS, dp.DFBB}:
                    kwargs.pop(k)
            kwargs_solver = kwargs

        solver = solver_cls(self.model, time_limit=time_limit, **kwargs_solver)
        if retrieve_intermediate_solutions:
            while True:
                solution, terminated = solver.search_next()
                logger.info(f"Objective = {solution.cost}, {solution.is_infeasible}")
                stopping = did_callback.on_solution_callback(solution)
                if terminated or stopping:
                    break
        else:
            solution = solver.search()
            did_callback.on_solution_callback(solution)
        logger.info(f"Is optimal {solution.is_optimal}")
        logger.info(f"Is infeasible {solution.is_infeasible}")
        logger.info(f"Best bound {solution.best_bound}")
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception
        res = did_callback.res
        callbacks_list.on_solve_end(res=res, solver=self)
        return res
