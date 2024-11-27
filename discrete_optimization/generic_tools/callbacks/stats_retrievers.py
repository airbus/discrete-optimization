#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from time import perf_counter
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class BasicStatsCallback(Callback):
    """
    This callback is storing the computation time at different step of the solving process,
    this can help to display the evolution of the best solution through time, and compare easily different solvers.
    """

    def __init__(self):
        self.starting_time: int = None
        self.end_time: int = None
        self.stats: list[dict] = []

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        t = perf_counter()
        best_sol, fit = res.get_best_solution_fit()
        self.stats.append({"sol": best_sol, "fit": fit, "time": t - self.starting_time})

    def on_solve_start(self, solver: SolverDO):
        self.starting_time = perf_counter()

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        """Called at the end of solve.
        Args:
        res: current result storage
        solver: solvers using the callback
        """
        self.on_step_end(None, res, solver)


class StatsCpsatCallback(BasicStatsCallback):
    """
    This callback is specific to cpsat solver.
    """

    def __init__(self):
        super().__init__()
        self.final_status: str = None

    def on_step_end(
        self, step: int, res: ResultStorage, solver: OrtoolsCpSatSolver
    ) -> Optional[bool]:
        super().on_step_end(step=step, res=res, solver=solver)
        self.stats[-1].update(
            {
                "obj": solver.clb.ObjectiveValue(),
                "bound": solver.clb.BestObjectiveBound(),
                "time-cpsat": {
                    "user-time": solver.clb.UserTime(),
                    "wall-time": solver.clb.WallTime(),
                },
            }
        )
        if solver.clb.ObjectiveValue() == solver.clb.BestObjectiveBound():
            return False

    def on_solve_start(self, solver: OrtoolsCpSatSolver):
        self.starting_time = perf_counter()

    def on_solve_end(self, res: ResultStorage, solver: OrtoolsCpSatSolver):
        # super().on_solve_end(res=res, solver=solver)
        status_name = solver.solver.status_name()
        if len(self.stats) > 0:
            self.stats[-1].update(
                {
                    "obj": solver.solver.ObjectiveValue(),
                    "bound": solver.solver.BestObjectiveBound(),
                    "time-cpsat": {
                        "user-time": solver.clb.UserTime(),
                        "wall-time": solver.clb.WallTime(),
                    },
                }
            )
        self.final_status = status_name
