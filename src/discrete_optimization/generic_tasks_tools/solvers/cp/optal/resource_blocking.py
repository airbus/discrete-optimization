#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cp.base.cumulative_resource import (
    CumulativeResource,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.resource_blocking import (
    ResourceBlockingCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.cumulative_resource import (
    CumulativeResourceSchedulingOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.entity import (
    EntityIntervalsOptalMixin,
)

try:
    import optalcp as cp
except ImportError:
    cp = None


class ResourceBlockingOptalSolver(
    ResourceBlockingCpSolver[Task, CumulativeResource, OtherCalendarResource],
    EntityIntervalsOptalMixin[Task],
    CumulativeResourceSchedulingOptalSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    def _create_flexible_gap_blocking_interval(
        self, constraint, i_constraint
    ) -> cp.IntervalVar:
        entity1 = constraint.left_entity
        ref1 = constraint.start_or_end_left_entity
        entity2 = constraint.right_entity
        ref2 = constraint.start_or_end_right_entity
        itv1 = self._intervals_entity[entity1]
        itv2 = self._intervals_entity[entity2]

        gap_interval = self.cp_model.interval_var(
            optional=True,
            name=f"blocking_gap_{entity1.entity_id}_{ref1}_to_{entity2.entity_id}_{ref2}",
        )
        self.add_implication(
            (itv1.presence(), itv2.presence()), gap_interval.presence()
        )
        if ref1 == StartOrEnd.START:
            self.cp_model.start_at_start(gap_interval, itv1)
        else:
            self.cp_model.start_at_end(gap_interval, itv1)
        if ref2 == StartOrEnd.START:
            self.cp_model.end_at_start(gap_interval, itv2)
        else:
            self.cp_model.end_at_end(gap_interval, itv2)

        return gap_interval
