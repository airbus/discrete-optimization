#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Any, List, Optional

import didppy as dp

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
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


class DidSolver(SolverDO):

    model: dp.Model = None
    hyperparameters = [
        CategoricalHyperparameter(name="solver", choices=solvers, default=dp.CABS)
    ]

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
        **kwargs: Any
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        solver_cls = kwargs["solver"]
        if "initial_solution" in self.__dict__.keys():
            kwargs["initial_solution"] = self.initial_solution
        for k in list(kwargs.keys()):
            if k not in {"threads", "initial_solution"}:
                kwargs.pop(k)
        quiet = kwargs.get("quiet", False)
        solver = solver_cls(self.model, time_limit=time_limit, quiet=quiet, **kwargs)
        solution = solver.search()
        sol = self.retrieve_solution(solution)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])
