from __future__ import annotations

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    CompositeEntity,
    ConstantDurationEntity,
    GroupEntity,
    ModuloTaskEntity,
    ModuloTwoEntity,
    MultiplyTaskEntity,
    SchedulingEntity,
    TaskEntity,
    TaskModeEntity,
    VariableBatchEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd, UndefinedData
from discrete_optimization.generic_tasks_tools.solvers.cp.base.entity import (
    EntityIntervalsMixin,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    LinearExpr,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.multimode_scheduling import (
    MultimodeSchedulingOptalSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None


class EntityIntervalsOptalMixin(
    EntityIntervalsMixin[Task], MultimodeSchedulingOptalSolver[Task], Generic[Task]
):
    """Optal mixin for creating interval variables for any SchedulingEntity."""

    _intervals_entity: dict[SchedulingEntity, cp.IntervalVar]

    def get_entity_is_active_variable(
        self, entity: SchedulingEntity
    ) -> LinearExpr | int:
        """Get presence/active variable for an entity.

        Args:
            entity: The scheduling entity

        Returns:
            IntVar (BoolVar) if entity is optional, or int 1 if always active
        """
        return self._intervals_entity[entity].presence()

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
        raise NotImplementedError()

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
        self._intervals_entity[entity] = itv
        if entity.start_or_end == StartOrEnd.START:
            self.cp_model.enforce(
                itv.start() == base_itv.start() * entity.multiply_factor
            )
        else:
            self.cp_model.enforce(itv.end() == base_itv.end() * entity.multiply_factor)

    def create_modulo_two_entity_interval(self, entity: ModuloTwoEntity[Task]):
        raise NotImplementedError()

    def create_variable_batch_entity_interval(self, entity: VariableBatchEntity[Task]):
        """Create interval for VariableBatchEntity (batch with variable membership).

        The batch interval spans from the earliest start to latest end of tasks
        assigned to this batch. Assignment is determined by task_to_batch_vars.
        """
        batch_id = entity.batch_id
        candidate_tasks = list(entity.candidate_tasks)

        if not candidate_tasks:
            itv = self.cp_model.interval_var(
                name=f"interval_{entity.entity_id}", start=0, length=0
            )
        else:
            itv = self.cp_model.interval_var(
                name=f"interval_{entity.entity_id}", optional=True
            )
            self.add_bool_constraint(
                itv.presence() == self.batch_used_vars[entity.batch_id]
            )
        self._intervals_entity[entity] = itv

        # Link batch span to task assignments
        # Batch start = min(task start for tasks assigned to batch)
        # Batch end = max(task end for tasks assigned to batch)
        if candidate_tasks:
            for task in candidate_tasks:
                # Get assignment variable for this task to this batch
                if (
                    task in self.task_to_batch_vars
                    and batch_id in self.task_to_batch_vars[task]
                ):
                    task_assigned = self.task_to_batch_vars[task][batch_id]
                    task_interval = self.get_task_interval(task)

                    # If task assigned to batch: batch_start <= task_start
                    self.add_implication(
                        task_assigned,
                        # cannot use start_at_start() which adds it to the model w/o the condition task_assigned
                        itv.start() == task_interval.start(),
                    )
                    # If task assigned to batch: batch_end >= task_end
                    self.add_implication(
                        task_assigned, itv.end() >= task_interval.end()
                    )
                    task_min_dur = self.problem.get_min_task_duration_in_batch(task)
                    if task_min_dur != UndefinedData.UNDEFINED:
                        self.add_implication(
                            task_assigned, itv.length() >= task_min_dur
                        )
                    task_max_dur = self.problem.get_max_task_duration_in_batch(task)
                    if task_max_dur != UndefinedData.UNDEFINED:
                        self.add_implication(
                            task_assigned, itv.length() <= task_max_dur
                        )
