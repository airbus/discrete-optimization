#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

import cpmpy as cp

from discrete_optimization.generic_tools.cpmpy_tools import CpmpySolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.shop.base import CommonShopProblem


class CpmpyShopSolver(CpmpySolver):
    problem: CommonShopProblem

    def __init__(
        self,
        problem: CommonShopProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self._max_time = None

    def get_makespan_upper_bound(self) -> int:
        if self._max_time is None:
            return self.problem.get_makespan_upper_bound()
        else:
            return min(self._max_time, self.problem.get_makespan_upper_bound())

    def init_model(self, **kwargs: Any) -> None:
        # optional parameters
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self._max_time: int | None = kwargs.get(
            "max_time", None
        )  # update the upper bound for makespan
        # Decision `start[j]`: integer start time for each job `j`
        start = cp.intvar(
            0,
            self.get_makespan_upper_bound(),
            shape=self.problem.n_all_jobs,
            name="start",
        )
        # Decision `end[j]`: integer end time for each job `j`
        end = cp.intvar(
            0,
            self.get_makespan_upper_bound(),
            shape=self.problem.n_all_jobs,
            name="end",
        )
        pass

    def create_vars(self):
        pass
