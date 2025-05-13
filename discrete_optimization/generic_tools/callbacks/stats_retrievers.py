#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from time import perf_counter
from typing import Optional

import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import BoundsProviderMixin, SolverDO
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class BasicStatsCallback(Callback):
    """
    This callback is storing the computation time at different step of the solving process,
    this can help to display the evolution of the best solution through time, and compare easily different solvers.
    """

    time_column = "time"
    metric_columns = ["fit"]

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

    def get_df_metrics(self) -> pd.DataFrame:
        """Construct a dataframe indexed by time of the recorded metrics (fitness, bounds...)."""
        column_names = [self.time_column] + self.metric_columns
        df = pd.DataFrame(
            [{k: v for k, v in st.items() if k in column_names} for st in self.stats]
        ).set_index(self.time_column)
        df.columns.name = "metric"
        return df


class StatsWithBoundsCallback(BasicStatsCallback):
    """
    This callback is specific to BoundsProviderMixin solvers.
    """

    metric_columns = BasicStatsCallback.metric_columns + ["obj", "bound"]

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        if not isinstance(solver, BoundsProviderMixin):
            raise ValueError(
                "The ObjectiveGapStopper can be applied only to a solver deriving from BoundsProviderMixin."
            )
        super().on_step_end(step=step, res=res, solver=solver)
        self.stats[-1].update(
            {
                "obj": solver.get_current_best_internal_objective_value(),
                "bound": solver.get_current_best_internal_objective_bound(),
            }
        )


class StatsCpsatCallback(StatsWithBoundsCallback):
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
                "time-cpsat": {
                    "user-time": solver.clb.UserTime(),
                    "wall-time": solver.clb.WallTime(),
                },
            }
        )

    def on_solve_end(self, res: ResultStorage, solver: OrtoolsCpSatSolver):
        super().on_solve_end(res=res, solver=solver)
        status_name = solver.solver.status_name()
        self.final_status = status_name
