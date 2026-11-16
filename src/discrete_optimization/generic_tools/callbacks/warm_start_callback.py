#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import (
    SolverDO,
    WarmstartFromPreviousRunMixin,
    WarmstartMixin,
)
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.lns_cp import BaseLnsCp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class WarmStartCallback(Callback):
    def __init__(
        self,
        warm_start_best_solution: bool = True,
        warm_start_last_solution: bool = False,
    ):
        self.warm_start_best_solution = warm_start_best_solution
        self.warm_start_last_solution = warm_start_last_solution

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        solver_: Optional[WarmstartMixin] = None
        if isinstance(solver, LexicoSolver) or isinstance(solver, BaseLnsCp):
            if isinstance(solver.subsolver, WarmstartMixin):
                solver_ = solver.subsolver
        elif isinstance(solver, WarmstartMixin):
            solver_ = solver
        if solver_ is not None:
            sol = None
            if self.warm_start_best_solution:
                sol, _ = res.get_best_solution_fit()
            if self.warm_start_last_solution and len(res) > 0:
                sol, _ = res[-1]
            if sol is not None:
                solver_.set_warm_start(sol)
                logger.info(f"Warm-start done")


class WarmStartCallbackLastRun(WarmStartCallback):
    """Only works for Cp-sat solver"""

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        solver_ = None
        if isinstance(solver, LexicoSolver) or isinstance(solver, BaseLnsCp):
            if isinstance(solver.subsolver, WarmstartFromPreviousRunMixin):
                solver_ = solver.subsolver
        elif isinstance(solver, WarmstartFromPreviousRunMixin):
            solver_ = solver
        if solver_ is not None:
            if solver_.was_previously_run():
                solver_.set_warm_start_from_previous_run()
        else:
            return super().on_step_end(step, res, solver)
