#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Iterable

import cpmpy
from cpmpy.expressions.core import Expression
from cpmpy.expressions.utils import implies

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.base import (
    TasksCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.typing import (
    Constraint,
    LinearExpr,
)
from discrete_optimization.generic_tasks_tools.solvers.cpmpy.interval_var import (
    IntervalVar,
)
from discrete_optimization.generic_tools.cpmpy_tools import CpmpySolver


class TasksCpmpySolver(CpmpySolver, TasksCpBaseSolver[Task]):
    def create_interval_var(
        self,
        start: LinearExpr,
        duration: LinearExpr,
        name: str,
        end: LinearExpr | None = None,
        presence: LinearExpr = True,
    ) -> IntervalVar:
        var = IntervalVar(
            start=start, duration=duration, name=name, end=end, presence=presence
        )
        if end is not None:
            # create the corresponding constraint between start, end, and duration
            expr = duration == end - start
            # optional?
            if isinstance(presence, Expression):
                cstr = presence.implies(expr)
            else:  # bool or int
                presence_bool = bool(presence)
                cstr = implies(presence_bool, expr)
            self.cp_model.add(cstr)

        return var

    def add_no_overlap(self, intervals: Iterable[IntervalVar]) -> list[Constraint]:
        optional = not all(
            (isinstance(interval.presence, bool) and interval.presence)
            for interval in intervals
        )
        if optional:
            start, duration, is_present = tuple(
                zip(
                    *(
                        (interval.start, interval.duration, interval.presence)
                        for interval in intervals
                    )
                )
            )
            cstr = cpmpy.NoOverlapOptional(
                start=start, duration=duration, is_present=is_present
            )
        else:
            start, duration = tuple(
                zip(*((interval.start, interval.duration) for interval in intervals))
            )
            cstr = cpmpy.NoOverlap(
                start=start,
                duration=duration,
            )

        self.cp_model.add(cstr)
        return [cstr]

    def add_cumulative(
        self,
        intervals: Iterable[IntervalVar],
        demands: Iterable[LinearExpr],
        capacity: LinearExpr,
    ) -> list[Constraint]:
        optional = not all(
            (isinstance(interval.presence, bool) and interval.presence)
            for interval in intervals
        )
        if optional:
            start, duration, is_present = tuple(
                zip(
                    *(
                        (interval.start, interval.duration, interval.presence)
                        for interval in intervals
                    )
                )
            )
            cstr = cpmpy.CumulativeOptional(
                start=start,
                duration=duration,
                demand=tuple(demands),
                capacity=capacity,
                is_present=is_present,
            )
        else:
            start, duration = tuple(
                zip(*((interval.start, interval.duration) for interval in intervals))
            )
            cstr = cpmpy.Cumulative(
                start=start,
                duration=duration,
                demand=tuple(demands),
                capacity=capacity,
            )

        self.cp_model.add(cstr)
        return [cstr]
