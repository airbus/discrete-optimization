#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class NbIterationTracker(Callback):
    """
    Log the number of iteration of a given solver
    """

    def __init__(
        self,
        step_verbosity_level: int = logging.DEBUG,
        end_verbosity_level: int = logging.INFO,
    ):
        """

        Args:
            step_verbosity_level:
            end_verbosity_level:
        """
        self.step_verbosity_level = step_verbosity_level
        self.end_verbosity_level = end_verbosity_level
        self.nb_iteration = 0

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        logger.log(
            msg=f"Solve finished after {self.nb_iteration} iterations",
            level=self.end_verbosity_level,
        )

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        self.nb_iteration += 1
        if len(res) > 0:
            logger.log(
                msg=f"Iteration #{self.nb_iteration}", level=self.step_verbosity_level
            )


class ObjectiveLogger(Callback):
    """
    Log the current best objective function at every iteration/new solution found by the solver
    """

    def __init__(
        self,
        step_verbosity_level: int = logging.DEBUG,
        end_verbosity_level: int = logging.INFO,
    ):
        """

        Args:
            step_verbosity_level:
            end_verbosity_level:
        """
        self.step_verbosity_level = step_verbosity_level
        self.end_verbosity_level = end_verbosity_level
        self.nb_iteration = 0

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        logger.log(
            msg=f"Solve finished after {self.nb_iteration} iterations",
            level=self.end_verbosity_level,
        )

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        self.nb_iteration += 1
        if len(res) > 0:
            logger.log(
                msg=f"Iteration #{self.nb_iteration}, objective={res.get_best_solution_fit()[1]}",
                level=self.step_verbosity_level,
            )


class ProblemEvaluateLogger(Callback):
    """
    Log the output of problem.evaluate(sol) and problem.satisfy(sol)
    """

    def __init__(
        self,
        step_verbosity_level: int = logging.DEBUG,
        end_verbosity_level: int = logging.INFO,
    ):
        """

        Args:
            step_verbosity_level:
            end_verbosity_level:
        """
        self.step_verbosity_level = step_verbosity_level
        self.end_verbosity_level = end_verbosity_level
        self.nb_iteration = 0
        self.current_res_len = 0

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        logger.log(
            msg=f"Solve finished after {self.nb_iteration} iterations,"
            f"last sol evaluate={solver.problem.evaluate(res[-1][0])}, "
            f" Satisfy={solver.problem.satisfy(res[-1][0])}",
            level=self.end_verbosity_level,
        )

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        self.nb_iteration += 1
        nb_res = len(res)
        if nb_res > 0:
            if nb_res > self.current_res_len:
                logger.log(
                    msg=f"Iteration #{self.nb_iteration}, \nlast sol evaluate={solver.problem.evaluate(res[-1][0])}, "
                    f"\n Satisfy={solver.problem.satisfy(res[-1][0])}",
                    level=self.step_verbosity_level,
                )
                self.current_res_len = nb_res
