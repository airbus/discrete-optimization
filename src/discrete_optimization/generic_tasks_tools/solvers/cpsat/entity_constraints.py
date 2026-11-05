#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CP-SAT solver for entity-based constraints.

This module provides a mixin for handling entity constraints (precedence, time lags,
time windows) on any SchedulingEntity, not just tasks.

It provides complete entity interval creation logic that can be reused by
resource_blocking.py and other modules.
"""

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    SchedulingEntity,
)
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.entity import (
    EntityIntervalsMixin,
)


class EntityConstraintsCpSatSolver(
    EntityIntervalsMixin[Task],
):
    """CP-SAT solver mixin for entity-based constraints.

    Combines:
    - EntityIntervalsMixin: Provides entity interval creation for all entity types
    - BatchingCPSatSolver: Provides batch assignment variables for VariableBatchEntity

    Adds:
    - Precedence constraints between any SchedulingEntities
    - Time lag constraints (8 types: start/end × start/end × min/max)
    - Time window constraints on entity start/end times

    Pure mixin - does not inherit from base solver classes.
    """

    problem: GenericSchedulingProblem

    def init_entity_constraints(self) -> None:
        """Initialize entity constraints (batching + entity intervals).

        Should be called from the concrete solver's init_model() method.
        """
        self.init_batching_variables_and_constraints()
        self.init_entity_intervals_storage()
        self.create_entity_constraints()

    def collect_entities_from_constraints(self) -> set[SchedulingEntity]:
        """Collect all entities mentioned in entity constraints.

        Returns:
            Set of all entities that need interval variables created
        """
        entities = set()

        # From precedence constraints
        for (
            entity1,
            successors,
        ) in self.problem.get_entity_precedence_constraints().items():
            entities.add(entity1)
            entities.update(successors)

        # From time lag constraints (all 8 types)
        for entity1_se, entity2_se, min_max in [
            (StartOrEnd.START, StartOrEnd.START, MinOrMax.MIN),
            (StartOrEnd.START, StartOrEnd.START, MinOrMax.MAX),
            (StartOrEnd.START, StartOrEnd.END, MinOrMax.MIN),
            (StartOrEnd.START, StartOrEnd.END, MinOrMax.MAX),
            (StartOrEnd.END, StartOrEnd.START, MinOrMax.MIN),
            (StartOrEnd.END, StartOrEnd.START, MinOrMax.MAX),
            (StartOrEnd.END, StartOrEnd.END, MinOrMax.MIN),
            (StartOrEnd.END, StartOrEnd.END, MinOrMax.MAX),
        ]:
            for entity1, entity2, _ in self.problem.get_entity_time_lags(
                entity1_se, entity2_se, min_max
            ):
                entities.add(entity1)
                entities.add(entity2)

        return entities

    def create_entity_constraint_intervals(self) -> None:
        """Create interval variables for all entities in entity constraints."""
        entities = self.collect_entities_from_constraints()
        for entity in self.problem.get_batch_entities():
            entities.add(entity)
        for entity in entities:
            if entity not in self._starts_entity:
                self.define_start_and_end_var_and_constraint(entity)

    def create_entity_precedence_constraints(self) -> None:
        """Add CP-SAT constraints for entity precedence.

        For each entity1 → entity2 precedence:
            end(entity1) <= start(entity2)

        Only enforced when both entities are active.
        """
        for (
            entity1,
            successors,
        ) in self.problem.get_entity_precedence_constraints().items():
            if entity1 not in self._starts_entity:
                continue  # Entity not created (shouldn't happen if create_entity_constraint_intervals was called)

            for entity2 in successors:
                if entity2 not in self._starts_entity:
                    continue

                # Get time variables
                end1 = self._ends_entity[entity1]
                start2 = self._starts_entity[entity2]

                # Get presence variables
                is_active1 = self._get_entity_is_active_var(entity1)
                is_active2 = self._get_entity_is_active_var(entity2)

                # Build enforcement list (only non-trivial presence vars)
                enforcement_list = []
                if not (isinstance(is_active1, int) and is_active1 == 1):
                    enforcement_list.append(is_active1)
                if not (isinstance(is_active2, int) and is_active2 == 1):
                    enforcement_list.append(is_active2)

                # Add constraint
                if enforcement_list:
                    # Only enforce if both entities are active
                    self.cp_model.Add(end1 <= start2).OnlyEnforceIf(enforcement_list)
                else:
                    # Both entities always active
                    self.cp_model.Add(end1 <= start2)

    def create_entity_time_lag_constraints(self) -> None:
        """Add CP-SAT constraints for entity time lags.

        For each time lag type (start/end × start/end × min/max):
            MIN: time1 + offset <= time2
            MAX: time1 + offset >= time2

        Only enforced when both entities are active.
        """
        # All 8 time lag combinations
        for entity1_se, entity2_se, min_max in [
            (StartOrEnd.START, StartOrEnd.START, MinOrMax.MIN),
            (StartOrEnd.START, StartOrEnd.START, MinOrMax.MAX),
            (StartOrEnd.START, StartOrEnd.END, MinOrMax.MIN),
            (StartOrEnd.START, StartOrEnd.END, MinOrMax.MAX),
            (StartOrEnd.END, StartOrEnd.START, MinOrMax.MIN),
            (StartOrEnd.END, StartOrEnd.START, MinOrMax.MAX),
            (StartOrEnd.END, StartOrEnd.END, MinOrMax.MIN),
            (StartOrEnd.END, StartOrEnd.END, MinOrMax.MAX),
        ]:
            for entity1, entity2, offset in self.problem.get_entity_time_lags(
                entity1_se, entity2_se, min_max
            ):
                if (
                    entity1 not in self._starts_entity
                    or entity2 not in self._starts_entity
                ):
                    continue

                # Get appropriate time variables
                time1 = (
                    self._starts_entity[entity1]
                    if entity1_se == StartOrEnd.START
                    else self._ends_entity[entity1]
                )
                time2 = (
                    self._starts_entity[entity2]
                    if entity2_se == StartOrEnd.START
                    else self._ends_entity[entity2]
                )

                # Get presence variables
                is_active1 = self._get_entity_is_active_var(entity1)
                is_active2 = self._get_entity_is_active_var(entity2)

                # Build enforcement list
                enforcement_list = []
                if not (isinstance(is_active1, int) and is_active1 == 1):
                    enforcement_list.append(is_active1)
                if not (isinstance(is_active2, int) and is_active2 == 1):
                    enforcement_list.append(is_active2)

                # Add constraint
                if min_max == MinOrMax.MIN:
                    # time1 + offset <= time2
                    if enforcement_list:
                        self.cp_model.Add(time1 + offset <= time2).OnlyEnforceIf(
                            enforcement_list
                        )
                    else:
                        self.cp_model.Add(time1 + offset <= time2)
                else:  # MAX
                    # time1 + offset >= time2
                    if enforcement_list:
                        self.cp_model.Add(time1 + offset >= time2).OnlyEnforceIf(
                            enforcement_list
                        )
                    else:
                        self.cp_model.Add(time1 + offset >= time2)

    def create_entity_time_window_constraints(self) -> None:
        """Add CP-SAT constraints for entity time windows.

        For each entity with time window bounds:
            lower_bound <= entity_time <= upper_bound

        Only enforced when entity is active.
        """
        # Collect all entities that might have time windows
        entities = self.collect_entities_from_constraints()

        for entity in entities:
            if entity not in self._starts_entity:
                continue

            for start_or_end in [StartOrEnd.START, StartOrEnd.END]:
                # Get appropriate time variable
                time_var = (
                    self._starts_entity[entity]
                    if start_or_end == StartOrEnd.START
                    else self._ends_entity[entity]
                )

                # Check lower bound
                lb = self.problem.get_entity_time_window_lower_bound(
                    entity, start_or_end
                )
                if lb is not None:
                    is_active = self._get_entity_is_active_var(entity)
                    if isinstance(is_active, int) and is_active == 1:
                        self.cp_model.Add(time_var >= lb)
                    else:
                        self.cp_model.Add(time_var >= lb).OnlyEnforceIf(is_active)

                # Check upper bound
                ub = self.problem.get_entity_time_window_upper_bound(
                    entity, start_or_end
                )
                if ub is not None:
                    is_active = self._get_entity_is_active_var(entity)
                    if isinstance(is_active, int) and is_active == 1:
                        self.cp_model.Add(time_var <= ub)
                    else:
                        self.cp_model.Add(time_var <= ub).OnlyEnforceIf(is_active)

    def create_entity_constraints(self) -> None:
        """Create all entity constraint intervals and constraints.

        This is the main entry point that should be called during model initialization.
        """
        # Create intervals for entities
        self.create_entity_constraint_intervals()

        # Add constraints
        self.create_entity_precedence_constraints()
        self.create_entity_time_lag_constraints()
        self.create_entity_time_window_constraints()
