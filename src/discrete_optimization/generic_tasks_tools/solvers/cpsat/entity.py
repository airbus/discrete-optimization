from typing import Any, Generic

from ortools.sat.python.cp_model import IntervalVar, IntVar, LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    CompositeEntity,
    ConstantDurationEntity,
    DummyEntity,
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
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.batching_cpsat import (
    BatchingCPSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    SpanModeling,
    create_span_start_end_from_vars,
    create_span_start_end_variables,
)


class EntityIntervalsMixin(BatchingCPSatSolver[Task], Generic[Task]):
    """Pure mixin for creating CP-SAT interval variables for any SchedulingEntity.

    This mixin provides methods to create start/end/duration/interval variables
    for all entity types, including VariableBatchEntity which requires batch
    assignment variables.

    Can be reused by EntityConstraintsCpSatSolver, ResourceBlockingCpSatSolver, etc.

    Attributes stored:
        _starts_entity[entity]: IntVar for entity start time
        _ends_entity[entity]: IntVar for entity end time
        _durations_entity[entity]: IntVar for entity duration
        _intervals_entity[entity]: IntervalVar for entity interval
        _entity_active[entity]: BoolVar/int for entity presence

    Required methods (must be provided by inheriting class):
        get_task_start_or_end_variable(task, start_or_end) -> IntVar
        get_duration_variable(task) -> IntVar
        get_task_interval(task) -> IntervalVar
        get_task_is_present_variable(task) -> IntVar
        get_task_mode_is_present_variable(task, mode) -> IntVar
    """

    problem: GenericSchedulingProblem
    # cp_model: Any  # CP model

    # Entity interval variables
    _starts_entity: dict[SchedulingEntity, IntVar]
    _ends_entity: dict[SchedulingEntity, IntVar]
    _durations_entity: dict[SchedulingEntity, IntVar]
    _intervals_entity: dict[SchedulingEntity, IntervalVar]
    _entity_active: dict[SchedulingEntity, LinearExprT]

    # From BatchingCpSatSolver (if used with batching)
    task_to_batch_vars: dict[Task, dict[Any, IntVar]]
    batch_used_vars: dict[Any, IntVar]

    # Methods expected from scheduling solver (provided by inheriting class)
    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> IntVar:
        """Get task start or end variable (provided by inheriting solver)."""
        ...

    def get_duration_variable(self, task: Task) -> IntVar:
        """Get task duration variable (provided by inheriting solver)."""
        ...

    def get_task_interval(self, task: Task) -> IntervalVar:
        """Get task interval variable (provided by inheriting solver)."""
        ...

    def get_task_is_present_variable(self, task: Task) -> IntVar:
        """Get task presence variable (provided by inheriting solver)."""
        ...

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> IntVar:
        """Get task-mode presence variable (provided by inheriting solver)."""
        ...

    def init_entity_intervals_storage(self) -> None:
        """Initialize entity interval storage dictionaries."""
        self._starts_entity = {}
        self._ends_entity = {}
        self._durations_entity = {}
        self._intervals_entity = {}
        self._entity_active = {}

    def get_entity_start_or_end_variable(
        self, entity: SchedulingEntity, start_or_end: StartOrEnd
    ) -> IntVar:
        """Get start or end time variable for an entity.

        Args:
            entity: The scheduling entity
            start_or_end: Whether to get start or end variable

        Returns:
            IntVar for entity's start or end time
        """
        if entity not in self._starts_entity:
            self.define_start_and_end_var_and_constraint(entity)
        if start_or_end == StartOrEnd.START:
            return self._starts_entity[entity]
        else:
            return self._ends_entity[entity]

    def get_entity_is_active_variable(self, entity: SchedulingEntity) -> IntVar | int:
        """Get presence/active variable for an entity.

        Args:
            entity: The scheduling entity

        Returns:
            IntVar (BoolVar) if entity is optional, or int 1 if always active
        """
        return self._get_entity_is_active_var(entity)

    # =====================================================================
    # ENTITY INTERVAL CREATION METHODS
    # =====================================================================
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
        elif isinstance(entity, ModuloTaskEntity):
            modulo = entity.modulo_value
            return 0, modulo - 1, 0, self.problem.get_makespan_lower_bound()
        elif isinstance(entity, MultiplyTaskEntity):
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
            if entity.start_or_end == StartOrEnd.START:
                min_d = max(0, lbe - ubs)
                max_d = max(0, ube - lbs)
                return (
                    lbs * entity.multiply_factor,
                    ubs * entity.multiply_factor,
                    lbs * entity.multiply_factor + min_d,
                    ubs * entity.multiply_factor + max_d,
                )
            else:
                min_d = max(0, lbe - ubs)
                max_d = max(0, ube - lbs)
                return (
                    lbe * entity.multiply_factor - max_d,
                    ube * entity.multiply_factor - min_d,
                    lbe * entity.multiply_factor,
                    ube * entity.multiply_factor,
                )
        elif isinstance(entity, ModuloTwoEntity):
            lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(entity.entity)
            lb_start_mod, ub_start_mod, lb_end_mod, ub_end_mod = self.get_lb_ub_entity(
                entity.modulo_entity
            )
            max_dur = ub_end - lb_start
            match entity.modulo_value_start_or_end:
                case StartOrEnd.START:
                    return (
                        0,
                        min(ub_start, ub_start_mod - 1),
                        0,
                        min(ub_start, ub_start_mod - 1) + max_dur,
                    )
                case StartOrEnd.END:
                    return (
                        0,
                        min(ub_start, ub_end_mod - 1),
                        0,
                        min(ub_start, ub_end_mod - 1) + max_dur,
                    )
        else:
            raise NotImplementedError(f"Entity type {type(entity)} not supported")

    def _get_entity_is_active_var(self, entity: SchedulingEntity[Task]) -> LinearExprT:
        """Get or create the presence/active variable for an entity.

        Returns:
            LinearExprT: Either an IntVar (BoolVar) or the constant 1 if always active
        """
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
                        var = self.cp_model.NewBoolVar(
                            name=f"is_active_{entity.entity_id}"
                        )
                        is_present_vars = [
                            self.get_task_is_present_variable(task=task)
                            for task in entity.tasks
                        ]
                        self.cp_model.AddMaxEquality(var, is_present_vars)
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
                        var = self.cp_model.NewBoolVar(
                            name=f"is_active_{entity.entity_id}"
                        )
                        self.cp_model.AddMaxEquality(var, is_active_subentity_list)
                        self._entity_active[entity] = var
                    else:
                        self._entity_active[entity] = 1
                case ConstantDurationEntity():
                    return self._get_entity_is_active_var(entity.other_entity)
                case ModuloTaskEntity():
                    if self.problem.is_optional(entity.task):
                        self._entity_active[entity] = self.get_task_is_present_variable(
                            task=entity.task
                        )
                    else:
                        self._entity_active[entity] = 1
                case MultiplyTaskEntity():
                    if self.problem.is_optional(entity.task):
                        self._entity_active[entity] = self.get_task_is_present_variable(
                            task=entity.task
                        )
                    else:
                        self._entity_active[entity] = 1
                case ModuloTwoEntity():
                    self._entity_active[entity] = self._get_entity_is_active_var(
                        entity.entity
                    )
                case VariableBatchEntity():
                    self._entity_active[entity] = self.batch_used_vars[entity.batch_id]
                case DummyEntity():
                    self._entity_active[entity] = 1
                case _:
                    raise NotImplementedError(
                        f"Entity type {type(entity)} not supported"
                    )
        return self._entity_active[entity]

    def define_start_and_end_var_and_constraint(self, entity: SchedulingEntity):
        """Create start/end/duration/interval variables for an entity.

        Dispatches to appropriate creation method based on entity type.
        """
        if entity in self._starts_entity:
            # Already defined
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
            case ModuloTaskEntity():
                self.create_modulo_entity_interval(entity)
            case MultiplyTaskEntity():
                self.create_multiply_entity_interval(entity)
            case ModuloTwoEntity():
                self.create_modulo_two_entity_interval(entity)
            case VariableBatchEntity():
                self.create_variable_batch_entity_interval(entity)
            case DummyEntity():
                self.create_dummy_entity_interval(entity)
            case _:
                raise NotImplementedError(f"Entity type {type(entity)} not supported")

    def create_dummy_entity_interval(self, entity: DummyEntity):
        self._starts_entity[entity] = entity.start
        self._ends_entity[entity] = entity.end
        self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
            start=0, end=0, size=0, name=f"interval_{entity.entity_id}"
        )

    def create_task_entity_interval(
        self, entity: TaskEntity[Task] | TaskModeEntity[Task]
    ):
        """Create interval for TaskEntity or TaskModeEntity."""
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
            self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )

    def create_group_entity_interval(self, entity: GroupEntity[Task]):
        """Create interval for GroupEntity (span of its tasks)."""
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
            self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )
        # Link span to task start/end times
        create_span_start_end_variables(
            solver=self,
            set_tasks=entity.get_tasks(),
            name_span=f"{entity.entity_id}",
            start_span=self._starts_entity[entity],
            end_span=self._ends_entity[entity],
            span_modeling=SpanModeling.INEQUALITIES,
        )

    def create_composite_entity_interval(self, entity: CompositeEntity[Task]):
        """Create interval for CompositeEntity (composition of entities)."""
        unfolded_entities = list(entity.unfold_entities())
        only_task_based = all(
            isinstance(x, (GroupEntity, TaskEntity, TaskModeEntity, CompositeEntity))
            for x in unfolded_entities
        )
        if only_task_based:
            self.create_group_entity_interval(entity)
        else:
            # Create intervals for sub-entities first
            for ent in entity.entities:
                self.define_start_and_end_var_and_constraint(ent)
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
                self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                    start=self._starts_entity[entity],
                    end=self._ends_entity[entity],
                    size=self._durations_entity[entity],
                    is_present=is_present,
                    name=f"interval_{entity.entity_id}",
                )

    def create_constant_duration_interval(self, entity: ConstantDurationEntity[Task]):
        """Create interval for ConstantDurationEntity (fixed offset from another entity)."""
        # Define the other entity first (recursive)
        self.define_start_and_end_var_and_constraint(entity.other_entity)
        # Now create this entity's interval
        if entity.start_or_end == StartOrEnd.START:
            self._starts_entity[entity] = self._starts_entity[entity.other_entity]
            self._ends_entity[entity] = (
                self._starts_entity[entity.other_entity] + entity.constant_duration
            )
            self._durations_entity[entity] = entity.constant_duration
            is_present = self._get_entity_is_active_var(entity)
            if isinstance(is_present, int) and is_present == 1:
                self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
                    start=self._starts_entity[entity],
                    size=self._durations_entity[entity],
                    end=self._ends_entity[entity],
                    name=f"interval_{entity.entity_id}",
                )
            else:
                self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                    start=self._starts_entity[entity],
                    size=self._durations_entity[entity],
                    end=self._ends_entity[entity],
                    is_present=is_present,
                    name=f"interval_{entity.entity_id}",
                )

    def create_modulo_entity_interval(self, entity: ModuloTaskEntity[Task]):
        """Create interval for ModuloTaskEntity (task start modulo value)."""
        task = entity.task
        lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(entity)
        self._starts_entity[entity] = self.cp_model.NewIntVar(
            lb=lb_start, ub=ub_start, name=f"start_{entity.entity_id}"
        )
        self._ends_entity[entity] = self.cp_model.NewIntVar(
            lb=lb_end, ub=ub_end, name=f"end_{entity.entity_id}"
        )
        self._durations_entity[entity] = self.get_duration_variable(task)
        is_present = self._get_entity_is_active_var(entity)
        self.cp_model.AddModuloEquality(
            self._starts_entity[entity],
            self.get_task_start_or_end_variable(task, StartOrEnd.START),
            entity.modulo_value,
        )
        if isinstance(is_present, int) and is_present == 1:
            self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
                start=self._starts_entity[entity],
                size=self._durations_entity[entity],
                end=self._ends_entity[entity],
                name=f"interval_{entity.entity_id}",
            )
        else:
            self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )

    def create_multiply_entity_interval(self, entity: MultiplyTaskEntity[Task]):
        """Create interval for MultiplyTaskEntity (task time * factor)."""
        task = entity.task
        lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(entity)
        if entity.start_or_end == StartOrEnd.END:
            self._starts_entity[entity] = self.cp_model.NewIntVar(
                lb=lb_start, ub=ub_start, name=f"start_{entity.entity_id}"
            )
        if entity.start_or_end == StartOrEnd.START:
            self._ends_entity[entity] = self.cp_model.NewIntVar(
                lb=lb_end, ub=ub_end, name=f"end_{entity.entity_id}"
            )
        self._durations_entity[entity] = self.get_duration_variable(task)
        is_present = self._get_entity_is_active_var(entity)
        match entity.start_or_end:
            case StartOrEnd.START:
                self._starts_entity[entity] = (
                    entity.multiply_factor
                    * self.get_task_start_or_end_variable(task, StartOrEnd.START)
                )
            case StartOrEnd.END:
                self._ends_entity[entity] = (
                    entity.multiply_factor
                    * self.get_task_start_or_end_variable(task, StartOrEnd.END)
                )
        if isinstance(is_present, int) and is_present == 1:
            self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
                start=self._starts_entity[entity],
                size=self._durations_entity[entity],
                end=self._ends_entity[entity],
                name=f"interval_{entity.entity_id}",
            )
        else:
            self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )

    def create_modulo_two_entity_interval(self, entity: ModuloTwoEntity[Task]):
        """Create interval for ModuloTwoEntity (entity start modulo another entity)."""
        self.define_start_and_end_var_and_constraint(entity.entity)
        self.define_start_and_end_var_and_constraint(entity.modulo_entity)
        lb_start, ub_start, lb_end, ub_end = self.get_lb_ub_entity(entity)
        self._starts_entity[entity] = self.cp_model.NewIntVar(
            lb=lb_start, ub=ub_start, name=f"start_{entity.entity_id}"
        )
        self._ends_entity[entity] = self.cp_model.NewIntVar(
            lb=lb_end, ub=ub_end, name=f"end_{entity.entity_id}"
        )
        self._durations_entity[entity] = self._durations_entity[entity.entity]
        if entity.modulo_value_start_or_end == StartOrEnd.START:
            self.cp_model.AddModuloEquality(
                self._starts_entity[entity],
                self._starts_entity[entity.entity],
                mod=self._starts_entity[entity.modulo_entity],
            )
        else:
            self.cp_model.AddModuloEquality(
                self._starts_entity[entity],
                self._starts_entity[entity.entity],
                mod=self._ends_entity[entity.modulo_entity],
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
            self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )

    def create_variable_batch_entity_interval(self, entity: VariableBatchEntity[Task]):
        """Create interval for VariableBatchEntity (batch with variable membership).

        The batch interval spans from the earliest start to latest end of tasks
        assigned to this batch. Assignment is determined by task_to_batch_vars.
        """
        batch_id = entity.batch_id
        candidate_tasks = list(entity.candidate_tasks)

        if not candidate_tasks:
            # Empty batch - shouldn't happen but handle gracefully
            lb_start, ub_start, lb_end, ub_end = 0, 0, 0, 0
        else:
            # Get bounds from candidate tasks
            lb_start = min(
                self.problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.START
                )
                for task in candidate_tasks
            )
            ub_start = max(
                self.problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.START
                )
                for task in candidate_tasks
            )
            lb_end = min(
                self.problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in candidate_tasks
            )
            ub_end = max(
                self.problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in candidate_tasks
            )
        # Create variables
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
            self._intervals_entity[entity] = self.cp_model.NewOptionalIntervalVar(
                start=self._starts_entity[entity],
                end=self._ends_entity[entity],
                size=self._durations_entity[entity],
                is_present=is_present,
                name=f"interval_{entity.entity_id}",
            )

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

                    # If task assigned to batch: batch_start <= task_start
                    self.cp_model.Add(
                        self._starts_entity[entity]
                        == self.get_task_start_or_end_variable(task, StartOrEnd.START)
                    ).OnlyEnforceIf(task_assigned)
                    # If task assigned to batch: batch_end >= task_end
                    self.cp_model.Add(
                        self._ends_entity[entity]
                        >= self.get_task_start_or_end_variable(task, StartOrEnd.END)
                    ).OnlyEnforceIf(task_assigned)
                    task_min_dur = self.problem.get_min_task_duration_in_batch(task)
                    if task_min_dur != UndefinedData.UNDEFINED:
                        self.cp_model.Add(
                            self._durations_entity[entity] >= task_min_dur
                        ).OnlyEnforceIf(task_assigned)
                    task_max_dur = self.problem.get_max_task_duration_in_batch(task)
                    if task_max_dur != UndefinedData.UNDEFINED:
                        self.cp_model.Add(
                            self._durations_entity[entity] <= task_max_dur
                        ).OnlyEnforceIf(task_assigned)
