#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Optal solver for entity-based constraints.

This module provides a mixin for handling entity constraints (precedence, time lags,
time windows) on any SchedulingEntity, not just tasks.

It provides complete entity interval creation logic that can be reused by
resource_blocking.py and other modules.
"""

from __future__ import annotations

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cp.base.entity_constraints import (
    EntityConstraintsCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.entity import (
    EntityIntervalsOptalMixin,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.utils import (
    get_interval_start_or_end_variable,
)


class EntityConstraintsOptalSolver(
    EntityConstraintsCpSolver[Task], EntityIntervalsOptalMixin[Task]
):
    """Optal solver mixin for entity-based constraints.

    Adds:
    - Precedence constraints between any SchedulingEntities
    - Time lag constraints (8 types: start/end × start/end × min/max)
    - Time window constraints on entity start/end times

    """

    def create_entity_precedence_constraints(self) -> None:
        """Add constraints for entity precedence.

        For each entity1 → entity2 precedence:
            end(entity1) <= start(entity2)

        Only enforced when both entities are active.
        """
        for (
            entity1,
            successors,
        ) in self.problem.get_entity_precedence_constraints().items():
            if entity1 not in self._intervals_entity:
                continue  # Entity not created (shouldn't happen if create_entity_constraint_intervals was called)

            for entity2 in successors:
                if entity2 not in self._intervals_entity:
                    continue

                # Get time variables
                itv1 = self._intervals_entity[entity1]
                itv2 = self._intervals_entity[entity2]

                self.cp_model.end_before_start(itv1, itv2)

    def create_entity_time_lag_constraints(self) -> None:
        """Add constraints for entity time lags.

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
                    entity1 not in self._intervals_entity
                    or entity2 not in self._intervals_entity
                ):
                    continue

                itv1 = self._intervals_entity[entity1]
                itv2 = self._intervals_entity[entity2]

                # Get appropriate time variables
                time1 = get_interval_start_or_end_variable(
                    itv=itv1, start_or_end=entity1_se
                )
                time2 = get_interval_start_or_end_variable(
                    itv=itv2, start_or_end=entity2_se
                )

                # Add constraint
                if min_max == MinOrMax.MIN:
                    # time1 + offset <= time2
                    self.add_bool_constraint(time1 + offset <= time2)
                else:  # MAX
                    # time1 + offset >= time2
                    self.add_bool_constraint(time1 + offset >= time2)

    def create_entity_time_window_constraints(self) -> None:
        """Add constraints for entity time windows.

        For each entity with time window bounds:
            lower_bound <= entity_time <= upper_bound

        Only enforced when entity is active.
        """
        # Collect all entities that might have time windows
        entities = self.collect_entities_from_constraints()

        for entity in entities:
            if entity not in self._intervals_entity:
                continue
            itv = self._intervals_entity[entity]

            for start_or_end in [StartOrEnd.START, StartOrEnd.END]:
                # Get appropriate time variable
                time_var = get_interval_start_or_end_variable(
                    itv=itv, start_or_end=start_or_end
                )

                # Check lower bound
                lb = self.problem.get_entity_time_window_lower_bound(
                    entity, start_or_end
                )
                if lb is not None:
                    self.add_bool_constraint(time_var >= lb)

                # Check upper bound
                ub = self.problem.get_entity_time_window_upper_bound(
                    entity, start_or_end
                )
                if ub is not None:
                    self.add_bool_constraint(time_var <= ub)
