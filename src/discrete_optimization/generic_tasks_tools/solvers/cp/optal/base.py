#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Iterable

import numpy as np

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.cp.base.base import (
    TasksCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    BoolExpr,
    Constraint,
    IntervalVar,
    LinearExpr,
)
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)


class TasksOptalSolver(OptalCpSolver, TasksCpSolver[Task]):
    def enforce_only_if_tasks_present(self, expr: BoolExpr, tasks: Iterable[Task]):
        # Optal handle already by itself absence of tasks in constraints
        return [self.add_bool_constraint(expr)]

    def create_interval_var(
        self,
        start: LinearExpr,
        duration: LinearExpr,
        name: str,
        end: LinearExpr | None = None,
        presence: LinearExpr = True,
    ) -> IntervalVar:
        if (
            isinstance(presence, bool)
            and presence
            and end is None
            and np.isscalar(duration)
            and np.isscalar(start)
        ):
            # create fixed interval (e.g. to be used in cumulative/no overlap constraints)
            return self.cp_model.interval_var(
                start=int(start), length=int(duration), name=name
            )
        else:
            raise NotImplementedError(
                "Optal does not create interval variables from integer variables, "
                "but rather use them as core variables."
            )

    def add_no_overlap(self, intervals: Iterable[IntervalVar]) -> list[Constraint]:
        return [self.cp_model.no_overlap(intervals)]

    def add_cumulative(
        self,
        intervals: Iterable[IntervalVar],
        demands: Iterable[LinearExpr],
        capacity: LinearExpr,
    ) -> list[Constraint]:
        # create pulses for each intervals
        pulses = [
            self.cp_model.pulse(
                interval=interval,
                height=demand,
            )
            for interval, demand in zip(intervals, demands)
        ]
        return [self.cp_model.enforce(self.cp_model.sum(pulses) <= capacity)]
