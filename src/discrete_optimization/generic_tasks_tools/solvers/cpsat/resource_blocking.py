#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Any, Generic

from ortools.sat.python.cp_model import Domain, IntervalVar, IntVar, LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    CompositeEntity,
    ConstantDurationEntity,
    GroupEntity,
    SchedulingEntity,
    TaskEntity,
    TaskModeEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    CumulativeResource,
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceSchedulingCpSatSolver,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    SpanModeling,
    create_span_start_end_from_vars,
    create_span_start_end_variables,
)


class ResourceBlockingCpSatSolver(
    CumulativeResourceSchedulingCpSatSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """CP-SAT mixin for handling resource blocking constraints.

    This mixin adds support for:
    - Flexible gap blocking: resources blocked between two scheduling entities
    - Span blocking: resources blocked during the span of a set of tasks

    The mixin handles overlaps between blocking intervals and:
    - Calendar unavailability periods (fake tasks)
    - Actual task execution intervals
    - Other blocking intervals

    It creates appropriate cumulative constraints per resource, properly accounting
    for blocking intervals without double-counting consumption.
    """

    problem: GenericSchedulingProblem
    _blocking_intervals: dict
    _starts_entity: dict[SchedulingEntity, IntVar]
    _ends_entity: dict[SchedulingEntity, IntVar]
    _durations_entity: dict[SchedulingEntity, IntVar]
    _intervals_entity: dict[SchedulingEntity, IntVar]
    _bounds_entity: dict[SchedulingEntity, IntVar]
    _choices_blocking_constraint_vars: dict[str, dict[int, IntVar]]
    _choices_demands_variable: dict[str, dict[CumulativeResource, IntVar]]

    def init_model(self, **kwargs: Any) -> None:
        """Initialize model and reset blocking interval storage."""
        super().init_model(**kwargs)
        self._blocking_intervals: dict[
            CumulativeResource,
            list[tuple[IntervalVar, int, BlockingConstraintMetadata]],
        ] = {}
        self._entity_active: dict[SchedulingEntity[Task], LinearExprT] = {}
        self._choices_blocking_constraint_vars = {}
        self._choices_demands_variable = {}

    def _get_tasks_from_entity(self, entity: SchedulingEntity) -> set[Task]:
        """Extract all tasks involved in a scheduling entity.

        Args:
            entity: The scheduling entity

        Returns:
            Set of tasks involved in this entity
        """
        if isinstance(entity, TaskEntity):
            return {entity.task}
        elif isinstance(entity, TaskModeEntity):
            return {entity.task}
        elif isinstance(entity, GroupEntity):
            return set(entity.tasks)
        else:
            return set()

    def get_lb_ub_entity(self, entity: SchedulingEntity) -> tuple[int, int, int, int]:
        """Return lbstart, ubstart, lbend, ubend for an entity."""
        if isinstance(entity, (TaskEntity, TaskModeEntity)):
            lbs = self.problem.get_task_start_or_end_lower_bound(
                entity.task, StartOrEnd.START
            )
            ubs = self.problem.get_task_start_or_end_upper_bound(
                entity.task, StartOrEnd.START
            )
            lbe = self.problem.get_task_start_or_end_lower_bound(
                entity.task, StartOrEnd.END
            )
            ube = self.problem.get_task_start_or_end_upper_bound(
                entity.task, StartOrEnd.END
            )
            return lbs, ubs, lbe, ube
        elif isinstance(entity, GroupEntity):
            lb_start = [
                self.problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.START
                )
                for task in entity.get_tasks()
            ]
            ub_start = [
                self.problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.START
                )
                for task in entity.get_tasks()
            ]
            lb_end = [
                self.problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in entity.get_tasks()
            ]
            ub_end = [
                self.problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in entity.get_tasks()
            ]
            return min(lb_start), max(ub_start), min(lb_end), max(ub_end)
        elif isinstance(entity, CompositeEntity):
            array = [self.get_lb_ub_entity(ent) for ent in entity.entities]
            lb_start = min(x[0] for x in array)
            ub_start = max(x[1] for x in array)
            lb_end = min(x[2] for x in array)
            ub_end = max(x[3] for x in array)
            return lb_start, ub_start, lb_end, ub_end
        elif isinstance(entity, ConstantDurationEntity):
            lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(
                entity.other_entity
            )
            match entity.start_or_end:
                case StartOrEnd.START:
                    return (
                        lb_start + entity.offset,
                        ub_start + entity.offset,
                        lb_start + entity.offset + entity.constant_duration,
                        ub_start + entity.offset + entity.constant_duration,
                    )
                case StartOrEnd.END:
                    return (
                        lb_end + entity.offset,
                        ub_end + entity.offset,
                        lb_end + entity.offset + entity.constant_duration,
                        ub_end + entity.offset + entity.constant_duration,
                    )
        else:
            raise NotImplementedError

    def get_lb_ub_size(
        self,
        entity1: SchedulingEntity,
        start_or_end1: StartOrEnd,
        entity2: SchedulingEntity,
        start_or_end2: StartOrEnd,
    ):
        lbs1, ubs1, lbe1, ube1 = self.get_lb_ub_entity(entity1)
        lbs2, ubs2, lbe2, ube2 = self.get_lb_ub_entity(entity2)
        if start_or_end1 == StartOrEnd.START:
            if start_or_end2 == StartOrEnd.START:
                return max(0, lbs2 - ubs1), max(0, ubs2 - lbs1)
            if start_or_end2 == StartOrEnd.END:
                return max(0, lbe2 - ubs1), max(0, ube2 - lbs1)
        if start_or_end1 == StartOrEnd.END:
            if start_or_end2 == StartOrEnd.START:
                return max(0, lbs2 - ube1), max(0, ubs2 - lbe1)
            if start_or_end2 == StartOrEnd.END:
                return max(0, lbe2 - ube1), max(0, ube2 - lbe1)
        return None, None

    def define_start_and_end_var_and_constraint(self, entity: SchedulingEntity):
        if entity in self._starts_entity:
            # Already defined.
            return
        match entity:
            case TaskEntity():
                self.create_task_entity_interval(entity)
            case GroupEntity():
                self.create_group_entity_interval(entity)
            case TaskModeEntity():
                self.create_task_entity_interval(entity)
            case CompositeEntity():
                self.create_composite_entity_interval(entity)
            case ConstantDurationEntity():
                self.create_constant_duration_interval(entity)
            case _:
                raise NotImplementedError()

    def create_task_entity_interval(
        self, entity: TaskEntity[Task] | TaskModeEntity[Task]
    ):
        task = entity.task
        self._starts_entity[entity] = self.get_task_start_or_end_variable(
            task=task, start_or_end=StartOrEnd.START
        )
        self._ends_entity[entity] = self.get_task_start_or_end_variable(
            task=task, start_or_end=StartOrEnd.END
        )
        self._durations_entity[entity] = self.get_duration_variable(task)
        is_present = self._get_entity_is_active_var(entity)
        if isinstance(is_present, int) and is_present == 1:
            self._intervals_entity[entity] = self.get_task_interval(task)
        else:
            self._intervals_entity[entity] = self.cp_model.new_optional_interval_var(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )
        self._intervals_entity[entity] = self.get_task_interval(task)

    def create_constant_duration_interval(self, entity: ConstantDurationEntity[Task]):
        # Defines first the other entity, it will be recursive.
        self.define_start_and_end_var_and_constraint(entity.other_entity)
        # Now we have access to start/end of the other entity :)
        if entity.start_or_end == StartOrEnd.START:
            self._starts_entity[entity] = self._starts_entity[entity.other_entity]
            self._ends_entity[entity] = (
                self._starts_entity[entity.other_entity] + entity.constant_duration
            )
            self._durations_entity[entity] = entity.constant_duration
            is_present = self._get_entity_is_active_var(entity)
            if isinstance(is_present, int) and is_present == 1:
                self._intervals_entity[entity] = self.cp_model.new_interval_var(
                    start=self._starts_entity[entity],
                    size=self._durations_entity[entity],
                    end=self._ends_entity[entity],
                    name=f"interval_{entity.entity_id}",
                )
            else:
                self._intervals_entity[entity] = (
                    self.cp_model.new_optional_interval_var(
                        start=self._starts_entity[entity],
                        size=self._durations_entity[entity],
                        end=self._ends_entity[entity],
                        is_present=is_present,
                        name=f"interval_{entity.entity_id}",
                    )
                )

    def create_group_entity_interval(self, entity: GroupEntity[Task]):
        lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(entity)
        self._starts_entity[entity] = self.cp_model.NewIntVar(
            lb=lb_start, ub=ub_start, name=f"start_{entity.entity_id}"
        )
        self._ends_entity[entity] = self.cp_model.NewIntVar(
            lb=lb_end, ub=ub_end, name=f"end_{entity.entity_id}"
        )
        self._durations_entity[entity] = self.cp_model.NewIntVar(
            lb=max(0, lb_end - ub_start),
            ub=max(0, ub_end - lb_start),
            name=f"duration_{entity.entity_id}",
        )
        is_present = self._get_entity_is_active_var(entity)
        if isinstance(is_present, int) and is_present == 1:
            self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                name=f"interval_{entity.entity_id}",
            )
        else:
            self._intervals_entity[entity] = self.cp_model.new_optional_interval_var(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )
        create_span_start_end_variables(
            solver=self,
            set_tasks=entity.get_tasks(),
            name_span=f"{entity.entity_id}",
            start_span=self._starts_entity[entity],
            end_span=self._ends_entity[entity],
            span_modeling=SpanModeling.INEQUALITIES,
        )

    def create_composite_entity_interval(self, entity: CompositeEntity[Task]):
        unfolded_entities = list(entity.unfold_entities())
        only_task_based = all(
            isinstance(x, (GroupEntity, TaskEntity, TaskModeEntity, CompositeEntity))
            for x in unfolded_entities
        )
        if only_task_based:
            self.create_group_entity_interval(entity)
        else:
            for entity in entity.entities:
                self.define_start_and_end_var_and_constraint(entity)
            entities = list(entity.entities)
            start, end = create_span_start_end_from_vars(
                solver=self,
                starts_list=[self._starts_entity[entity] for entity in entities],
                ends_list=[self._ends_entity[entity] for entity in entities],
                is_present_list=[
                    self._get_entity_is_active_var(entity) for entity in entities
                ],
                name_span=f"{entity.entity_id}",
                span_modeling=SpanModeling.INEQUALITIES,
            )
            lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(entity)
            self._starts_entity[entity] = start
            self._ends_entity[entity] = end
            self._durations_entity[entity] = self.cp_model.new_int_var(
                lb=max(0, lb_end - ub_start),
                ub=max(0, ub_end - lb_start),
                name=f"duration_{entity.entity_id}",
            )
            self._durations_entity[entity] = end
            is_present = self._get_entity_is_active_var(entity)
            if isinstance(is_present, int) and is_present == 1:
                self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
                    start=self._starts_entity[entity],
                    end=self._ends_entity[entity],
                    size=self._durations_entity[entity],
                    name=f"interval_{entity.entity_id}",
                )
            else:
                self._intervals_entity[entity] = (
                    self.cp_model.new_optional_interval_var(
                        start=self._starts_entity[entity],
                        end=self._ends_entity[entity],
                        size=self._durations_entity[entity],
                        is_present=is_present,
                        name=f"interval_{entity.entity_id}",
                    )
                )

    def create_entity_intervals(self) -> None:
        self._starts_entity = {}
        self._ends_entity = {}
        self._durations_entity = {}
        self._intervals_entity: dict[SchedulingEntity[Task], IntervalVar] = {}
        all_entities = []
        for constraint in self.problem.get_flexible_gap_blocking_constraints():
            all_entities.append(constraint.left_entity)
            all_entities.append(constraint.right_entity)
        for span_constraint in self.problem.get_span_blocking_constraints():
            all_entities.append(span_constraint.entity)

        for entity in all_entities:
            if entity not in self._starts_entity:
                self.define_start_and_end_var_and_constraint(entity)

    def constrain_group_entity_times(self, entity: GroupEntity) -> None:
        """Add constraints for group entity start/end times.

        A group's start is the minimum start of its tasks.
        A group's end is the maximum end of its tasks.

        Args:
            entity: The group entity
        """
        group_start = self._starts_entity[entity]
        group_end = self._ends_entity[entity]
        # TODO : extract this into an entity module
        if any(self.problem.is_optional(t) for t in entity.tasks):
            for task in entity.tasks:
                if self.problem.is_optional(task):
                    (
                        self.cp_model.add(
                            group_start
                            <= self.get_task_start_or_end_variable(
                                task, StartOrEnd.START
                            )
                        ).only_enforce_if(self.get_task_is_present_variable(task))
                    )

                    (
                        self.cp_model.add(
                            group_end
                            >= self.get_task_start_or_end_variable(task, StartOrEnd.END)
                        ).only_enforce_if(self.get_task_is_present_variable(task))
                    )
                else:
                    self.cp_model.add(
                        group_start
                        <= self.get_task_start_or_end_variable(task, StartOrEnd.START)
                    )
                    self.cp_model.add(
                        group_end
                        >= self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    )
        else:
            self.cp_model.AddMinEquality(
                group_start,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.START)
                    for task in entity.tasks
                ],
            )
            self.cp_model.AddMaxEquality(
                group_end,
                [
                    self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    for task in entity.tasks
                ],
            )

    def _get_entity_is_active_var(self, entity: SchedulingEntity[Task]) -> LinearExprT:
        if entity not in self._entity_active:
            match entity:
                case TaskEntity():
                    if self.problem.is_optional(entity.task):
                        self._entity_active[entity] = self.get_task_is_present_variable(
                            task=entity.task
                        )
                    else:
                        self._entity_active[entity] = 1
                case GroupEntity():
                    optional_tasks_in_group = [
                        task for task in entity.tasks if self.problem.is_optional(task)
                    ]
                    if len(optional_tasks_in_group) > 0:
                        # active if at least one task is active
                        var = self.cp_model.new_bool_var(
                            name=f"is_active_{entity.entity_id}"
                        )
                        is_present_vars = [
                            self.get_task_is_present_variable(task=task)
                            for task in entity.tasks
                        ]
                        self.cp_model.add_max_equality(var, is_present_vars)
                        self._entity_active[entity] = var
                    else:
                        self._entity_active[entity] = 1
                case TaskModeEntity():
                    self._entity_active[entity] = (
                        self.get_task_mode_is_present_variable(
                            task=entity.task, mode=entity.mode
                        )
                    )
                case CompositeEntity():
                    is_active_subentity_list = [
                        is_active_subentity
                        for subentity in entity.entities
                        # discard always active subentities
                        if not (
                            isinstance(
                                (
                                    is_active_subentity
                                    := self._get_entity_is_active_var(subentity)
                                ),
                                int,
                            )
                            and is_active_subentity == 1
                        )
                    ]
                    if len(is_active_subentity_list) > 0:
                        # active if at least one subentity is active
                        var = self.cp_model.new_bool_var(
                            name=f"is_active_{entity.entity_id}"
                        )
                        self.cp_model.add_max_equality(var, is_active_subentity_list)
                        self._entity_active[entity] = var
                    else:
                        self._entity_active[entity] = 1
                case ConstantDurationEntity():
                    return self._get_entity_is_active_var(entity.other_entity)
                case _:
                    raise NotImplementedError()
        return self._entity_active[entity]

    def create_flexible_gap_blocking_intervals(self) -> None:
        """Create interval variables for flexible gap blocking constraints.

        For each flexible gap constraint, creates an interval from the first entity's
        reference point to the second entity's reference point, with the specified
        resource demands.
        """
        for i_constraint, constraint in enumerate(
            self.problem.get_flexible_gap_blocking_constraints()
        ):
            entity1 = constraint.left_entity
            ref1 = constraint.start_or_end_left_entity
            entity2 = constraint.right_entity
            ref2 = constraint.start_or_end_right_entity
            metadata = constraint.metadata
            default_resources = constraint.default_resource_blocked
            # Get time variables for the gap boundaries
            gap_start = (
                self._starts_entity[entity1]
                if ref1 == StartOrEnd.START
                else self._ends_entity[entity1]
            )
            gap_end = (
                self._starts_entity[entity2]
                if ref2 == StartOrEnd.START
                else self._ends_entity[entity2]
            )
            lb_size, ub_size = self.get_lb_ub_size(entity1, ref1, entity2, ref2)
            # Create a variable for the gap size
            # The gap may be 0 or positive (we'll enforce positive later)
            gap_size = self.cp_model.NewIntVar(
                lb=lb_size,
                ub=ub_size,
                name=f"gap_size_{i_constraint}",
            )
            # Create a boolean variable for gap validity (positive duration)
            gap_is_present = self.cp_model.NewBoolVar(f"gap_valid_{i_constraint}")
            # Constrain gap_size = gap_end - gap_start when gap is present
            self.cp_model.Add(gap_size == gap_end - gap_start).OnlyEnforceIf(
                gap_is_present
            )
            # Create interval variable for the gap
            gap_interval = self.cp_model.NewOptionalIntervalVar(
                start=gap_start,
                size=gap_size,
                end=gap_end,
                is_present=gap_is_present,
                name=f"blocking_gap_{entity1.entity_id}_{ref1}_to_{entity2.entity_id}_{ref2}",
            )
            is_active_entity_list = [
                is_active_entity
                for entity in (entity1, entity2)
                # discard always active entities
                if not (
                    isinstance(
                        (is_active_entity := self._get_entity_is_active_var(entity)),
                        int,
                    )
                    and is_active_entity == 1
                )
            ]
            if len(is_active_entity_list) > 0:
                # gap present <=> all entities are active
                self.cp_model.add(gap_is_present == 1).only_enforce_if(
                    *is_active_entity_list
                )
                for is_active_var in is_active_entity_list:
                    self.cp_model.add(gap_is_present == 0).only_enforce_if(
                        ~is_active_var
                    )
            else:
                # both entities always active
                self.cp_model.add(gap_is_present == 1)

            # Store blocking intervals per resource with metadata and involved tasks
            for resource, demand in default_resources.items():
                if resource not in self._blocking_intervals:
                    self._blocking_intervals[resource] = []
                # Store interval with its metadata and tasks for later processing
                self._blocking_intervals[resource].append(
                    (gap_interval, demand, metadata)
                )
            if constraint.has_a_choice():
                name_choice = metadata.name_choice
                self._choices_blocking_constraint_vars[name_choice] = {}
                self._choices_demands_variable[name_choice] = {}
                resources = set()
                possible_values_per_resource = defaultdict(set)
                for value in constraint.choice_resource_blocked:
                    var = self.cp_model.new_bool_var(f"{name_choice}_{value}")
                    self._choices_blocking_constraint_vars[name_choice][value] = var
                    resources.update(
                        set(constraint.choice_resource_blocked[value].keys())
                    )
                    for r in constraint.choice_resource_blocked[value]:
                        possible_values_per_resource[r].add(
                            constraint.choice_resource_blocked[value][r]
                        )
                self.cp_model.add_exactly_one(
                    self._choices_blocking_constraint_vars[name_choice].values()
                )
                for r in resources:
                    self._choices_demands_variable[name_choice][r] = (
                        self.cp_model.new_int_var_from_domain(
                            domain=Domain.from_values(
                                list(possible_values_per_resource[r]) + [0]
                            ),
                            name=f"{name_choice}_{r}",
                        )
                    )
                for value in constraint.choice_resource_blocked:
                    for r in constraint.choice_resource_blocked[value]:
                        self.cp_model.add(
                            self._choices_demands_variable[name_choice][r]
                            == constraint.choice_resource_blocked[value][r]
                        ).only_enforce_if(
                            self._choices_blocking_constraint_vars[name_choice][value]
                        )
                for r in resources:
                    if r not in self._blocking_intervals:
                        self._blocking_intervals[r] = []
                    self._blocking_intervals[r].append(
                        (
                            gap_interval,
                            self._choices_demands_variable[name_choice][r],
                            metadata,
                        )
                    )

    def create_span_blocking_intervals(self) -> None:
        """Create interval variables for span blocking constraints.

        For each span constraint, creates an interval from the minimum start time
        to the maximum end time of the specified tasks, with the specified resource demands.
        """
        for i_constraint, constraint in enumerate(
            self.problem.get_span_blocking_constraints()
        ):
            entity = constraint.entity
            resources = constraint.default_resource_blocked
            metadata = constraint.metadata
            # Store blocking intervals per resource with metadata
            for resource, demand in resources.items():
                if resource not in self._blocking_intervals:
                    self._blocking_intervals[resource] = []
                # Store interval with its metadata and task set for overlap handling
                self._blocking_intervals[resource].append(
                    (self._intervals_entity[entity], demand, metadata)
                )

            if constraint.has_a_choice():
                name_choice = metadata.name_choice
                self._choices_blocking_constraint_vars[name_choice] = {}
                self._choices_demands_variable[name_choice] = {}
                resources = set()
                possible_values_per_resource = defaultdict(set)
                for value in constraint.choice_resource_blocked:
                    var = self.cp_model.new_bool_var(f"{name_choice}_{value}")
                    self._choices_blocking_constraint_vars[name_choice][value] = var
                    resources.update(
                        set(constraint.choice_resource_blocked[value].keys())
                    )
                    for r in constraint.choice_resource_blocked[value]:
                        possible_values_per_resource[r].add(
                            constraint.choice_resource_blocked[value][r]
                        )
                self.cp_model.add_exactly_one(
                    self._choices_blocking_constraint_vars[name_choice].values()
                )
                for r in resources:
                    self._choices_demands_variable[name_choice][r] = (
                        self.cp_model.new_int_var_from_domain(
                            domain=Domain.from_values(
                                list(possible_values_per_resource[r]) + [0]
                            ),
                            name=f"{name_choice}_{r}",
                        )
                    )
                for value in constraint.choice_resource_blocked:
                    for r in constraint.choice_resource_blocked[value]:
                        self.cp_model.add(
                            self._choices_demands_variable[name_choice][r]
                            == constraint.choice_resource_blocked[value][r]
                        ).only_enforce_if(
                            self._choices_blocking_constraint_vars[name_choice][value]
                        )
                for r in resources:
                    if r not in self._blocking_intervals:
                        self._blocking_intervals[r] = []
                    self._blocking_intervals[r].append(
                        (
                            self._intervals_entity[entity],
                            self._choices_demands_variable[name_choice][r],
                            metadata,
                        )
                    )

    def get_blocking_intervals_and_demands(
        self, resource: CumulativeResource
    ) -> tuple[list[tuple[IntervalVar, int]], list[tuple[IntervalVar, int]]]:
        blocking_data = self._blocking_intervals.get(resource, [])
        reservation_blocking = []
        active_blocking = []

        for blocking_entry in blocking_data:
            interval, demand, metadata = blocking_entry
            if metadata.mode == BlockingMode.RESERVATION:
                reservation_blocking.append((interval, demand))
            else:  # ACTIVE
                active_blocking.append((interval, demand))
        return reservation_blocking, active_blocking

    def create_cumulative_constraint_including_blocking(
        self, resource: CumulativeResource
    ) -> None:
        """Create cumulative constraints including blocking intervals.

        Blocking is ALWAYS ADDITIVE: task consumption + blocking consumption.

        Creates TWO cumulative constraints to properly handle BlockingMode:

        Constraint 1 (WITHOUT calendar):
            - All task intervals
            - ALL blocking intervals (RESERVATION + ACTIVE)
            - NO fake tasks (calendar gaps)
            Purpose: Enforces RESERVATION blocking even during unavailable periods

        Constraint 2 (WITH calendar):
            - All task intervals
            - ONLY ACTIVE blocking intervals
            - Fake tasks (calendar gaps)
            Purpose: Enforces ACTIVE blocking only during available periods

        Args:
            resource: The cumulative resource to constrain
        """
        # Get task consumption intervals
        task_intervals = self.get_resource_consumption_intervals(resource)

        # Get fake tasks for calendar gaps
        fake_tasks_intervals = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=start,
                    size=end - start,
                    name=f"fake_task_{resource}_{i_task}",
                ),
                value,
            )
            for i_task, (start, end, value) in enumerate(
                self.problem.get_fake_tasks(resource=resource)
            )
        ]
        # Separate blocking intervals by mode
        reservation_blocking, active_blocking = self.get_blocking_intervals_and_demands(
            resource
        )
        # Get resource capacity
        capacity = self.problem.get_resource_max_capacity(resource)
        # CONSTRAINT 1: Tasks + ALL blocking (RESERVATION + ACTIVE) - NO calendar
        # This enforces RESERVATION blocking even during unavailable periods
        intervals_no_calendar = []
        intervals_no_calendar.extend(task_intervals)
        intervals_no_calendar.extend(reservation_blocking)
        intervals_no_calendar.extend(active_blocking)

        intervals_1 = [
            interval
            for interval, demand in intervals_no_calendar
            if not isinstance(demand, int) or demand > 0
        ]
        demands_1 = [
            demand
            for interval, demand in intervals_no_calendar
            if not isinstance(demand, int) or demand > 0
        ]

        if len(intervals_1) > 0:
            if capacity == 1 and all(isinstance(v, int) and v == 1 for v in demands_1):
                if self.use_no_overlap_for_capa_1 or not self.use_cumulative_for_capa_1:
                    self.cp_model.add_no_overlap(intervals_1)
                if self.use_cumulative_for_capa_1:
                    self.cp_model.add_cumulative(
                        intervals=intervals_1, demands=demands_1, capacity=capacity
                    )
            else:
                self.cp_model.add_cumulative(
                    intervals=intervals_1, demands=demands_1, capacity=capacity
                )

        # CONSTRAINT 2: Tasks + ACTIVE blocking + calendar - NO RESERVATION blocking
        # This enforces ACTIVE blocking only during available periods (constrained by calendar)
        if active_blocking or fake_tasks_intervals:
            intervals_with_calendar = []
            intervals_with_calendar.extend(task_intervals)
            intervals_with_calendar.extend(active_blocking)
            intervals_with_calendar.extend(fake_tasks_intervals)

            intervals_2 = [
                interval
                for interval, demand in intervals_with_calendar
                if not isinstance(demand, int) or demand > 0
            ]
            demands_2 = [
                demand
                for interval, demand in intervals_with_calendar
                if not isinstance(demand, int) or demand > 0
            ]

            if len(intervals_2) > 0:
                if capacity == 1 and all(
                    isinstance(v, int) and v == 1 for v in demands_2
                ):
                    if (
                        self.use_no_overlap_for_capa_1
                        or not self.use_cumulative_for_capa_1
                    ):
                        self.cp_model.add_no_overlap(intervals_2)
                    if self.use_cumulative_for_capa_1:
                        self.cp_model.add_cumulative(
                            intervals=intervals_2, demands=demands_2, capacity=capacity
                        )
                else:
                    self.cp_model.add_cumulative(
                        intervals=intervals_2, demands=demands_2, capacity=capacity
                    )

    def create_resource_blocking_constraints(self) -> None:
        """Create all resource blocking interval variables.

        This should be called during model initialization, before cumulative
        resource constraints are created (so that create_calendar_resources_constraint
        can check for blocking intervals).
        """
        self.create_entity_intervals()
        self.create_flexible_gap_blocking_intervals()
        self.create_span_blocking_intervals()

    def create_calendar_resources_constraint(
        self, resource: CumulativeResource
    ) -> None:
        """Create calendar resource constraint, using blocking-aware version if needed.

        Overrides the parent method to automatically use blocking-aware cumulative
        constraints when blocking intervals exist for this resource.

        Args:
            resource: The resource to constrain
        """
        # Check if this resource has blocking constraints
        has_blocking = (
            resource in self._blocking_intervals
            and len(self._blocking_intervals[resource]) > 0
        )

        if has_blocking:
            # Use specialized method that includes blocking intervals
            self.create_cumulative_constraint_including_blocking(resource=resource)
        else:
            # Use standard parent method
            super().create_calendar_resources_constraint(resource=resource)
