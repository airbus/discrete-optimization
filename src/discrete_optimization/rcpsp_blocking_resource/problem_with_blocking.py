#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""RCPSP with Resource Blocking Constraints.

This module provides an extension to standard RCPSP that includes resource blocking
constraints such as setup times, changeover periods, and safety monitoring spans.
"""

import logging
from collections.abc import Hashable
from typing import Any, Optional, Union

from discrete_optimization.generic_tasks_tools.resource_blocking import (
    FlexibleGapBlockingConstraint,
    SpanBlockingConstraint,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


class RcpspWithResourceBlocking(RcpspProblem):
    """RCPSP with Resource Blocking Constraints.

    Extends standard RCPSP with two types of blocking constraints:
    1. Flexible gap blocking: Resources blocked between two task events
       (e.g., setup time between task A end and task B start)
    2. Span blocking: Resources blocked for entire span of task group
       (e.g., project reservation blocking resource from first to last task)

    Attributes:
        All attributes from RcpspProblem, plus:
        flexible_gap_blocking_constraints: List of gap blocking constraints
        span_blocking_constraints: List of span blocking constraints

    Example:
        >>> from discrete_optimization.rcpsp.problem_with_blocking import RcpspWithResourceBlocking
        >>> from discrete_optimization.generic_tasks_tools.resource_blocking import (
        ...     BlockingMode, BlockingConstraintMetadata,
        ... )
        >>> from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
        >>> from discrete_optimization.generic_tasks_tools.entities import TaskEntity
        >>> # Define standard RCPSP parameters
        >>> resources = {"R1": 5}
        >>> mode_details = {
        ...     1: {1: {"duration": 0, "R1": 0}},
        ...     2: {1: {"duration": 4, "R1": 2}},
        ...     3: {1: {"duration": 3, "R1": 3}},
        ...     4: {1: {"duration": 0, "R1": 0}},
        ... }
        >>> successors = {1: [2, 3], 2: [4], 3: [4], 4: []}
        >>> # Add setup time blocking between tasks
        >>> blocking_constraints = [
        ...     (
        ...         TaskEntity(2), StartOrEnd.END,
        ...         TaskEntity(3), StartOrEnd.START,
        ...         {"R1": 1},  # 1 unit of R1 blocked during setup
        ...         BlockingConstraintMetadata(
        ...             mode=BlockingMode.RESERVATION,
        ...             description="Setup time between task 2 and 3"
        ...         ),
        ...     )
        ... ]
        >>> problem = RcpspWithResourceBlocking(
        ...     resources=resources,
        ...     non_renewable_resources=[],
        ...     mode_details=mode_details,
        ...     successors=successors,
        ...     horizon=20,
        ...     flexible_gap_blocking_constraints=blocking_constraints,
        ... )
    """

    def __init__(
        self,
        resources: dict[str, Union[int, list[int]]],
        non_renewable_resources: list[str],
        mode_details: dict[Hashable, dict[int, dict[str, int]]],
        successors: dict[Hashable, list[Hashable]],
        horizon: int,
        tasks_list: Optional[list[Hashable]] = None,
        source_task: Optional[Hashable] = None,
        sink_task: Optional[Hashable] = None,
        name_task: Optional[dict[Hashable, str]] = None,
        calendar_details: Optional[dict[str, list[list[int]]]] = None,
        special_constraints: Optional[SpecialConstraintsDescription] = None,
        fixed_permutation: Optional[list[int]] = None,
        fixed_modes: Optional[list[int]] = None,
        flexible_gap_blocking_constraints: Optional[
            list[FlexibleGapBlockingConstraint]
        ] = None,
        span_blocking_constraints: Optional[list[SpanBlockingConstraint]] = None,
        **kwargs: Any,
    ):
        """Initialize RCPSP with resource blocking constraints.

        Args:
            resources: Resource capacities
            non_renewable_resources: List of non-renewable resource names
            mode_details: Task modes with durations and resource requirements
            successors: Precedence constraints
            horizon: Maximum time horizon
            tasks_list: List of task IDs (optional)
            source_task: Source/dummy start task (optional)
            sink_task: Sink/dummy end task (optional)
            name_task: Task names mapping (optional)
            calendar_details: Resource calendar availability (optional)
            special_constraints: Additional special constraints (optional)
            fixed_permutation: Fixed task permutation for solutions (optional)
            fixed_modes: Fixed task modes for solutions (optional)
            flexible_gap_blocking_constraints: Gap blocking constraints (optional)
            span_blocking_constraints: Span blocking constraints (optional)
            **kwargs: Additional arguments
        """
        # Call parent RcpspProblem initialization
        super().__init__(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            tasks_list=tasks_list,
            source_task=source_task,
            sink_task=sink_task,
            name_task=name_task,
            calendar_details=calendar_details,
            special_constraints=special_constraints,
            fixed_permutation=fixed_permutation,
            fixed_modes=fixed_modes,
            **kwargs,
        )

        # Store blocking constraints
        self._flexible_gap_blocking_constraints = (
            flexible_gap_blocking_constraints or []
        )
        self._span_blocking_constraints = span_blocking_constraints or []

    def get_flexible_gap_blocking_constraints(
        self,
    ) -> list[FlexibleGapBlockingConstraint]:
        """Return flexible gap blocking constraints."""
        return self._flexible_gap_blocking_constraints

    def get_span_blocking_constraints(self) -> list[SpanBlockingConstraint]:
        """Return span blocking constraints."""
        return self._span_blocking_constraints

    def satisfy(self, variable) -> bool:  # type: ignore
        """Check if solution satisfies all constraints including blocking.

        Args:
            variable: The solution to check

        Returns:
            True if solution satisfies all constraints
        """
        # Check standard RCPSP constraints first
        if not super().satisfy(variable):
            return False

        # Check blocking constraints (from ResourceBlockingSolution mixin)
        return variable.check_blocking_constraints()
