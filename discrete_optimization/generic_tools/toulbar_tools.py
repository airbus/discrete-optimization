#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Any, Optional, Type

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import pytoulbar2

    toulbar_available = True
except ImportError as e:
    toulbar_available = True

import logging

from discrete_optimization.generic_tools.callbacks.callback import CallbackList
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)

logger = logging.getLogger(__name__)


class ToulbarSolver(SolverDO):
    model: "pytoulbar2.CFN" = None
    hyperparameters = [
        CategoricalHyperparameter(
            name="vns", choices=[None, -4, -3, -2, -1, 0], default=None
        )
    ]

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        ...

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[int] = 10,
        **kwargs: Any,
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
        callback = CallbackList(callbacks)
        callback.on_solve_start(solver=self)
        solution = self.model.Solve(showSolutions=1, timeLimit=int(time_limit))
        if solution is None:
            return self.create_result_storage([])
        logger.info(
            f"Solution value = {solution[1]}, bound={self.model.GetDDualBound()}"
        )
        sol = self.retrieve_solution(solution)
        res = self.create_result_storage([(sol, self.aggreg_from_sol(sol))])
        callback.on_solve_end(res=res, solver=self)
        return res


def to_lns_toulbar(cls: Type[ToulbarSolver]):
    class ToulbarSolverLns(cls):
        depth: int
        init_ub: float

        def init_model(self, **kwargs: Any) -> None:
            super().init_model(**kwargs)
            self.model.CFN.timer(100)
            self.model.SolveFirst()
            self.depth = self.model.Depth()
            self.model.Store()
            self.init_ub = self.model.GetUB()

        def solve(self, time_limit: Optional[int] = 20, **kwargs: Any) -> ResultStorage:
            try:
                solution = self.model.SolveNext(
                    showSolutions=1, timeLimit=int(time_limit)
                )
                logger.info(f"=== Solution === \n {solution}")
                logger.info(
                    f"Best solution = {solution[1]}, Bound = {self.model.GetDDualBound()}"
                )
                self.model.Restore(self.depth)
                self.model.Store()
                self.model.SetUB(self.init_ub)
                if solution is not None:
                    sol = self.retrieve_solution(solution)
                    fit = self.aggreg_from_sol(sol)
                    return self.create_result_storage(
                        [(sol, fit)],
                    )
                else:
                    return self.create_result_storage()
            except Exception as e:
                self.model.ClearPropagationQueues()
                self.model.Restore(self.depth)
                self.model.Store()
                self.model.SetUB(self.init_ub)
                logger.info(f"Solve failed in given time {e}")
                return self.create_result_storage()

    return ToulbarSolverLns
