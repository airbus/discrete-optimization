#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class RetrieveSubRes(Callback):
    """
    Utility to store separately solutions found at each step of a lexico or Sequential solve.
    WARNING : This will work only if the solution object can be stored in a set.
    """

    def __init__(self):
        self.sol_per_step: list[list[Solution]] = []
        self.all_sols = set()

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        self.sol_per_step.append(
            [s for s, _ in res.list_solution_fits if s not in self.all_sols]
        )
        self.all_sols.update(set(self.sol_per_step[-1]))
        return False
