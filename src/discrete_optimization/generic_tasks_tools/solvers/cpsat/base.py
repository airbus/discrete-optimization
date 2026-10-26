#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Iterable

import numpy as np

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.base import (
    TasksCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.typing import (
    Constraint,
    IntervalVar,
    LinearExpr,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class TasksCpSatSolver(OrtoolsCpSatSolver, TasksCpBaseSolver[Task]):
    def create_interval_var(
        self,
        start: LinearExpr,
        duration: LinearExpr,
        name: str,
        end: LinearExpr | None = None,
        presence: LinearExpr = True,
    ) -> IntervalVar:
        if isinstance(presence, bool) and presence:
            if end is None:
                assert np.isscalar(duration), (
                    "if end is missing, duration must be fixed."
                )
                return self.cp_model.new_fixed_size_interval_var(
                    start=start, size=duration, name=name
                )
            else:
                return self.cp_model.new_interval_var(
                    start=start, size=duration, end=end, name=name
                )
        else:
            if end is None:
                assert np.isscalar(duration), (
                    "if end is missing, duration must be fixed."
                )
                return self.cp_model.new_optional_fixed_size_interval_var(
                    start=start, size=duration, is_present=presence, name=name
                )
            else:
                return self.cp_model.new_optional_interval_var(
                    start=start, size=duration, end=end, is_present=presence, name=name
                )

        pass

    def add_no_overlap(self, intervals: Iterable[IntervalVar]) -> list[Constraint]:
        return [self.cp_model.add_no_overlap(intervals)]

    def add_cumulative(
        self,
        intervals: Iterable[IntervalVar],
        demands: Iterable[LinearExpr],
        capacity: LinearExpr,
    ) -> list[Constraint]:
        return [
            self.cp_model.add_cumulative(
                intervals=intervals, demands=demands, capacity=capacity
            )
        ]
