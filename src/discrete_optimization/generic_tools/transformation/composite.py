#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Composite transformations for chaining multiple transformations."""

from __future__ import annotations

from typing import Optional

from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)


class CompositeTransformation(ProblemTransformation):
    """Compose multiple transformations into a single transformation.

    Example:
        Chain transformations T1: RCPSP → Multiskill and T2: Multiskill → Preemptive

        >>> t1 = RcpspToMultiskillTransformation()
        >>> t2 = MultiskillToPreemptiveTransformation()
        >>> composite = CompositeTransformation([t1, t2])
        >>> # Now composite: RCPSP → Preemptive

    Back-transformation automatically chains in reverse:
        S_preemptive → T2⁻¹ → S_multiskill → T1⁻¹ → S_rcpsp

    """

    transformations: list[ProblemTransformation]
    _intermediate_problems: dict[int, Problem]  # Cache for problem instances

    def __init__(self, transformations: list[ProblemTransformation]):
        """Initialize composite transformation.

        Args:
            transformations: List of transformations to chain.

        Raises:
            ValueError: If transformations list is empty

        """
        if len(transformations) == 0:
            raise ValueError("Must provide at least one transformation")

        self.transformations = transformations
        self._intermediate_problems = {}

    def transform_problem(self, source_problem: Problem) -> Problem:
        """Apply all transformations in sequence.

        Args:
            source_problem: Original problem

        Returns:
            Final transformed problem

        """
        current_problem = source_problem
        self._intermediate_problems.clear()
        self._intermediate_problems[0] = source_problem

        for i, transformation in enumerate(self.transformations):
            current_problem = transformation.transform_problem(current_problem)
            self._intermediate_problems[i + 1] = current_problem

        return current_problem

    def back_transform_solution(
        self, solution: Solution, source_problem: Problem
    ) -> Solution:
        """Back-transform through the chain in reverse order.

        If we have transformations [T1, T2, T3]:
        - T1: P1 → P2
        - T2: P2 → P3
        - T3: P3 → P4

        Then back-transform: S4 → T3⁻¹ → S3 → T2⁻¹ → S2 → T1⁻¹ → S1

        Args:
            solution: Solution in final target problem space
            source_problem: Original source problem

        Returns:
            Solution in original source problem space

        """
        current_solution = solution

        # Apply back-transformations in reverse order
        for i, transformation in enumerate(reversed(self.transformations)):
            # Get the source problem for this back-transformation
            # (which is the intermediate problem from the forward pass)
            intermediate_idx = len(self.transformations) - i - 1
            intermediate_problem = self._intermediate_problems.get(
                intermediate_idx, source_problem
            )

            current_solution = transformation.back_transform_solution(
                current_solution, intermediate_problem
            )

        return current_solution

    def forward_transform_solution(
        self, solution: Solution, target_problem: Problem
    ) -> Optional[Solution]:
        """Forward-transform through the chain.

        Only works if ALL transformations support forward transformation.

        Args:
            solution: Solution in source problem space
            target_problem: Final target problem

        Returns:
            Solution in final target problem space, or None if any transformation
            doesn't support forward transformation

        """
        current_solution = solution

        for i, transformation in enumerate(self.transformations):
            # Get intermediate target problem
            intermediate_target = self._intermediate_problems.get(i + 1, target_problem)

            current_solution = transformation.forward_transform_solution(
                current_solution, intermediate_target
            )

            # If any transformation doesn't support forward, abort
            if current_solution is None:
                return None

        return current_solution

    def is_bidirectional(self, source_problem: Problem) -> bool:
        """Check if all transformations are bidirectional.

        Args:
            source_problem: Source problem to check

        Returns:
            True if all transformations support forward transformation

        """
        current_problem = source_problem
        for t in self.transformations:
            if not t.is_bidirectional(current_problem):
                return False
            current_problem = t.transform_problem(current_problem)
        return True

    def __repr__(self) -> str:
        """Nice representation showing the transformation chain."""
        if not self.transformations:
            return "CompositeTransformation(empty)"

        chain = " → ".join([type(t).__name__ for t in self.transformations])
        return f"CompositeTransformation({chain})"


def chain_transformations(
    *transformations: ProblemTransformation,
) -> CompositeTransformation:
    """Chain multiple transformations into a composite.

    Example:
        >>> t1 = RcpspToMultiskillTransformation()
        >>> t2 = MultiskillToPreemptiveTransformation()
        >>> t3 = PreemptiveToMultiskillTransformation()
        >>>
        >>> composite = chain_transformations(t1, t2, t3)
        >>> # Equivalent to: RCPSP → Multiskill → Preemptive → Multiskill

    Args:
        *transformations: Transformations to chain

    Returns:
        CompositeTransformation instance

    """
    return CompositeTransformation(list(transformations))
