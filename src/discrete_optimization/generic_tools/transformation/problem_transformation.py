#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Base class for problem transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tools.do_problem import Problem, Solution

P1 = TypeVar("P1", bound=Problem)
P2 = TypeVar("P2", bound=Problem)
S1 = TypeVar("S1", bound=Solution)
S2 = TypeVar("S2", bound=Solution)


class ProblemTransformation(ABC, Generic[P1, S1, P2, S2]):
    """Base class for transforming between two problem types.

    A transformation defines:
    1. How to convert problem P1 → P2 (required)
    2. How to convert solutions S2 → S1 (required, for back-transformation)
    3. How to convert solutions S1 → S2 (optional, for warmstart support)

    The transformation is stateless and can be reused across multiple problem instances.

    Example:
        Transform RCPSP to RCPSPMultiskill by creating dummy employees:

        >>> class RcpspToMultiskillTransformation(ProblemTransformation):
        ...     def transform_problem(self, source_problem: RcpspProblem):
        ...         # Create MultiskillRcpspProblem from source_problem
        ...         ...
        ...     def back_transform_solution(
        ...         self, solution: MultiskillRcpspSolution, source_problem: RcpspProblem
        ...     ):
        ...         # Extract schedule/modes, discard employee assignments
        ...         return RcpspSolution(problem=source_problem, ...)

    """

    @abstractmethod
    def transform_problem(self, source_problem: P1) -> P2:
        """Transform source problem to target problem.

        This method should be deterministic: same source → same target.

        Args:
            source_problem: The original problem to transform

        Returns:
            Transformed problem instance

        """
        ...

    @abstractmethod
    def back_transform_solution(self, solution: S2, source_problem: P1) -> S1:
        """Convert solution from target problem back to source problem.

        This is REQUIRED for all transformations.

        Args:
            solution: Solution in target problem space
            source_problem: Original problem (to associate with back-transformed solution)

        Returns:
            Corresponding solution in source problem space

        """
        ...

    def forward_transform_solution(
        self, solution: S1, target_problem: P2
    ) -> Optional[S2]:
        """Convert solution from source problem to target problem.

        This is OPTIONAL - only needed for warmstart support.
        Return None if transformation not supported/meaningful.

        Args:
            solution: Solution in source problem space
            target_problem: Transformed problem (to associate with forward-transformed solution)

        Returns:
            Corresponding solution in target problem space, or None if not supported

        """
        return None

    def is_bidirectional(self, source_problem: P1) -> bool:
        """Check if transformation supports both directions.

        Args:
            source_problem: Problem to check bidirectionality for

        Returns:
            True if forward_transform_solution is implemented

        """
        try:
            target = self.transform_problem(source_problem)
            dummy_source = source_problem.get_dummy_solution()
            dummy_target = self.forward_transform_solution(dummy_source, target)
            return dummy_target is not None
        except (NotImplementedError, AttributeError):
            return False

    def __repr__(self) -> str:
        """String representation of transformation."""
        return f"{type(self).__name__}()"
