#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from typing import Iterable

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.base import TasksSolver
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    BoolExpr,
    Constraint,
    IntervalVar,
    LinearExpr,
    Model,
)
from discrete_optimization.generic_tasks_tools.utils import optional_override

logger = logging.getLogger(__name__)


class TasksCpSolver(TasksSolver[Task]):
    cp_model: Model

    @optional_override
    def get_task_is_present_variable(self, task: Task) -> LinearExpr:
        """Get the boolean variable whether the (optional) task is present.

        Default implementation returns True for mandatory task.
        To be overriden when optional tasks to get an actual boolean variable.

        """
        if self.problem.is_optional(task):
            raise NotImplementedError()
        return True

    def enforce_only_if_tasks_present(
        self,
        expr: BoolExpr,
        tasks: Iterable[Task],
    ) -> list[Constraint]:
        """Enforce given constraints only if all given tasks are present.

        Args:
            expr:
            tasks:

        Returns:

        """
        is_present_tasks_variables = tuple(
            self.get_task_is_present_variable(task)
            for task in tasks
            if self.problem.is_optional(task)
        )
        if len(is_present_tasks_variables) > 0:
            return self.add_implication(is_present_tasks_variables, expr)
        else:
            return [self.add_bool_constraint(expr)]

    @abstractmethod
    def create_interval_var(
        self,
        start: LinearExpr,
        duration: LinearExpr,
        name: str,
        end: LinearExpr | None = None,
        presence: LinearExpr = True,
    ) -> IntervalVar:
        """Create an interval variable in the underlying cp model.

        The interval can be
        - fixed or not
        - with fixed duration or not
        - optional (with associated `presence` bool var)


        Args:
            start:
            duration:
            end:
            presence: should be a boolean variable if the interval is optional
            name:

        Returns:
            the created interval variable

        """
        ...

    @abstractmethod
    def add_no_overlap(self, intervals: Iterable[IntervalVar]) -> list[Constraint]:
        """Add constraint so that no intervals overlap.

        Args:
            intervals:

        Returns:

        """
        ...

    @abstractmethod
    def add_cumulative_with_scheduling(
        self,
        intervals: Iterable[IntervalVar],
        demands: Iterable[LinearExpr],
        capacity: LinearExpr,
    ) -> list[Constraint]:
        """Add cumulative constraint taking into account scheduled intervals

        Enforces that:

            for all t:
              self.sum(demands[i]
                if (start(intervals[i]) <= t < end(intervals[i])) and
                (intervals[i] is present)) <= capacity

        Args:
            intervals:
            demands:
            capacity:

        Returns:

        """
        ...

    def add_cumulative(
        self,
        intervals: Iterable[IntervalVar],
        demands: Iterable[LinearExpr],
        capacity: LinearExpr,
        no_scheduling: bool = False,
    ) -> list[Constraint]:
        """Add cumulative constraint

        Either takes into account scheduled intervals or not, accordind to `no_scheduling` flag.

        Enforces that:

            for all t:
              self.sum(demands[i]
                if (start(intervals[i]) <= t < end(intervals[i])) and
                (intervals[i] is present)) <= capacity

        Args:
            intervals:
            demands: demand variables. Should take account presence of tasks if no_scheduling is True
                (i.e. be 0 or absent if task is not scheduled) .
                Else, the presence of tasks can be taken into account by intervals instead
            capacity:
            no_scheduling: if `True`, assume that all tasks happen simultaneously
                and ignore interval variables.

        Returns:

        """
        if no_scheduling:
            return self.add_cumulative_wo_scheduling(
                demands=demands,
                capacity=capacity,
            )
        else:
            return self.add_cumulative_with_scheduling(
                intervals=intervals,
                demands=demands,
                capacity=capacity,
            )

    def add_cumulative_wo_scheduling(
        self,
        demands: Iterable[LinearExpr],
        capacity: LinearExpr,
    ) -> list[Constraint]:
        """Add cumulative constraint when no scheduling is considered.

        We do as if all tasks are happening simultaneously.
        It amounts to constraints for non-renewable resources i.e.
            sum(demands[i]) <= capacity

        """
        return [self.add_bool_constraint(self.sum(demands) <= capacity)]
