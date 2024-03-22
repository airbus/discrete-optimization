#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from abc import abstractmethod
from typing import Any, List, Optional

import clingo
from clingo import Symbol

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)
cur_folder = os.path.abspath(os.path.dirname(__file__))


class ASPClingoSolver(SolverDO):
    """Base class for solver based on Answer Set Programming formulation and clingo solver."""

    ctl: Optional[clingo.Control] = None
    early_stopping_exception: Optional[Exception] = None

    @abstractmethod
    def retrieve_solution(self, model: clingo.Model) -> Solution:
        """Construct a do solution from a clingo model.

        Args:
            model: the current constructed clingo model

        Returns:
            the intermediate solution, at do format.

        """
        ...

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the problem with a CPSat solver drom ortools library.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: keyword arguments passed to `self.init_model()`

        Returns:

        A dedicated clingo callback is used to:
        - update a resultstorage each time a new solution is found by the clingo solver.
        - call the user (do) callbacks at each new solution, with the possibility of early stopping if the callback return True.

        This clingo callback use the method `self.retrieve_solution()` to reconstruct a do Solution from the current clingo model.

        """
        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        if self.ctl is None:
            self.init_model(**kwargs)
        self.ctl.ground([("base", [])])

        asp_callback = ASPCallback(
            do_solver=self,
            callback=callbacks_list,
            dump_model_in_folders=kwargs.get("dump_model_in_folders", False),
        )
        timeout_seconds = kwargs.get("timeout_seconds", 100)
        with self.ctl.solve(on_model=asp_callback.on_model, async_=True) as handle:
            handle.wait(timeout_seconds)
            handle.cancel()
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception
        res = asp_callback.res
        callbacks_list.on_solve_end(res=res, solver=self)
        return res


class ASPCallback:
    def __init__(
        self,
        do_solver: ASPClingoSolver,
        callback: Callback,
        dump_model_in_folders: bool = False,
    ):
        super().__init__()
        self.dump_model_in_folders = dump_model_in_folders
        self.do_solver = do_solver
        self.callback = callback
        self.res = ResultStorage(
            [],
            mode_optim=self.do_solver.params_objective_function.sense_function,
            limit_store=False,
        )
        self.nb_solutions = 0

    def on_model(self, model: clingo.Model) -> bool:
        # debug: store model
        if self.dump_model_in_folders:
            folder = os.path.join(
                cur_folder, f"output-folder/model_{self.nb_solutions}"
            )
            if os.path.exists(folder):
                os.removedirs(folder)
            os.makedirs(folder)
            with open(os.path.join(folder, "model.txt"), "w") as model_file:
                model_file.write(str(model))
        # translate into do solution
        sol = self.do_solver.retrieve_solution(model=model)
        fit = self.do_solver.aggreg_from_sol(sol)
        self.res.add_solution(solution=sol, fitness=fit)
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
        # optimality?
        stopping = stopping or model.optimality_proven

        # return go-on status (=not stopping)
        return not stopping
