#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    CompositeEntity,
    ConstantDurationEntity,
    GroupEntity,
    ModuloTaskEntity,
    MultiplyTaskEntity,
    SchedulingEntity,
    TaskEntity,
    TaskModeEntity,
)
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

try:
    import optalcp as cp
except ImportError:
    cp = None


class ResourceBlockingOptalSolver(
    ResourceBlockingCpSolver[Task, CumulativeResource, OtherCalendarResource],
    CumulativeResourceSchedulingOptalSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    _intervals_entity: dict[SchedulingEntity, cp.IntervalVar]

    def create_task_entity_interval(
        self, entity: TaskEntity[Task] | TaskModeEntity[Task]
    ):
        if isinstance(entity, TaskEntity):
            self._intervals_entity[entity] = self.get_task_interval(entity.task)
        elif isinstance(entity, TaskModeEntity):
            self._intervals_entity[entity] = self.get_task_mode_interval(
                task=entity.task, mode=entity.mode
            )

    def create_constant_duration_interval(self, entity: ConstantDurationEntity[Task]):
        self.define_start_and_end_var_and_constraint(entity.other_entity)
        other_itv = self._intervals_entity[entity.other_entity]
        itv = self.cp_model.interval_var(
            length=entity.constant_duration,
            name=f"interval_{entity.entity_id}",
            optional=other_itv.optional,
        )
        # synchro start or end with other entity
        if entity.start_or_end == StartOrEnd.START:
            self.cp_model.start_at_start(
                predecessor=other_itv, successor=itv, delay=entity.offset
            )
        else:
            self.cp_model.end_at_start(
                predecessor=other_itv, successor=itv, delay=entity.offset
            )
        self._intervals_entity[entity] = itv

    def create_group_entity_interval(self, entity: GroupEntity[Task]):
        task_intervals = tuple(self.get_task_interval(task) for task in entity.tasks)
        optional = any(interval.optional for interval in task_intervals)
        itv = self.cp_model.interval_var(
            name=f"interval_{entity.entity_id}", optional=optional
        )
        self.cp_model.span(
            itv, tuple(self.get_task_interval(task) for task in entity.tasks)
        )
        self._intervals_entity[entity] = itv

    def create_composite_entity_interval(self, entity: CompositeEntity[Task]):
        for subentity in entity.entities:
            self.define_start_and_end_var_and_constraint(subentity)
        optional = any(
            self._intervals_entity[subentity].optional for subentity in entity.entities
        )
        itv = self.cp_model.interval_var(
            name=f"interval_{entity.entity_id}", optional=optional
        )
        self.cp_model.span(
            itv,
            tuple(self._intervals_entity[subentity] for subentity in entity.entities),
        )
        self._intervals_entity[entity] = itv

    def create_modulo_entity_interval(self, entity: ModuloTaskEntity[Task]):
        raise not NotImplementedError()

    def create_multiply_entity_interval(self, entity: MultiplyTaskEntity[Task]):
        base_itv = self.get_task_interval(entity.task)
        if entity.start_or_end == StartOrEnd.START:
            start_bounds = (
                base_itv.start_min * entity.multiply_factor,
                base_itv.start_max * entity.multiply_factor,
            )
            end_bounds = (base_itv.end_min, base_itv.end_max)
        else:
            start_bounds = (base_itv.start_min, base_itv.start_max)
            end_bounds = (
                base_itv.end_min * entity.multiply_factor,
                base_itv.end_max * entity.multiply_factor,
            )
        itv = self.cp_model.interval_var(
            start=start_bounds,
            end=end_bounds,
            name=f"interval_{entity.entity_id}",
            optional=base_itv.optional,
        )
        if entity.start_or_end == StartOrEnd.START:
            self.cp_model.enforce(
                itv.start() == base_itv.start() * entity.multiply_factor
            )
        else:
            self.cp_model.enforce(itv.end() == base_itv.end() * entity.multiply_factor)
        super().create_multiply_entity_interval(entity)

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
