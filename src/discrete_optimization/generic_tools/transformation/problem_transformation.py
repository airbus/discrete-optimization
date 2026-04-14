#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Base class for problem transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)

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
    4. Metadata documenting what's lost/gained in EACH direction (recommended)

    The transformation is stateless and can be reused across multiple problem instances.

    Important: Transformations can be asymmetric!
    - Forward problem transformation (P1 → P2) may be lossy
    - Backward solution transformation (S2 → S1) may be exact (or vice versa)

    Example:
        BinPack → SALBP transformation:
        - Forward (problem): Lossy (incompatibility constraints lost)
        - Backward (solution): Exact (SALBP solutions map perfectly to BinPack)

        >>> class BinpackToSalbpTransformation(ProblemTransformation):
        ...     def get_forward_metadata(self):
        ...         # Problem transformation: BinPack → SALBP (lossy)
        ...         return lossy_transformation(
        ...             losses=[InformationLoss(
        ...                 name="incompatibility_constraints",
        ...                 loss_type=LossType.CONSTRAINT,
        ...                 description="Item incompatibility constraints",
        ...                 reason="SALBP has no incompatibility concept",
        ...                 impact=LossImpact.MAJOR
        ...             )]
        ...         )
        ...
        ...     def get_backward_metadata(self):
        ...         # Solution transformation: SALBP solution → BinPack solution (exact)
        ...         return exact_transformation(
        ...             use_cases=["Direct mapping: stations → bins"]
        ...         )

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

    def get_forward_metadata(self) -> TransformationMetadata:
        """Get metadata for problem transformation (P1 → P2).

        Documents what information is lost when transforming the PROBLEM.

        Override this method to provide detailed information about:
        - Constraints that cannot be represented in target problem
        - Objectives that are ignored or approximated
        - Assumptions made during transformation

        Returns:
            TransformationMetadata documenting losses in problem transformation

        Default:
            Returns exact_transformation() (no losses documented)

        Note:
            Solutions from the target problem always map back MECHANICALLY via
            back_transform_solution(), but may not satisfy all constraints from
            the original source problem if this transformation is lossy.

            Example: BinPack → SALBP loses incompatibility constraints.
                     Solutions from SALBP solvers map back to BinPack allocations,
                     but may violate incompatibility if that constraint was present.

            Always verify solutions in the original problem after solving via transformation!

        """
        return exact_transformation()

    def get_metadata(self) -> TransformationMetadata:
        """Get overall transformation metadata (for backward compatibility).

        DEPRECATED: Use get_forward_metadata() and get_backward_metadata() instead.

        Returns forward metadata by default for compatibility.

        Returns:
            TransformationMetadata (forward transformation)

        """
        return self.get_forward_metadata()

    def is_forward_exact(self) -> bool:
        """Check if forward problem transformation is exact.

        Returns:
            True if problem transformation preserves all information

        """
        return self.get_forward_metadata().is_exact()

    def is_backward_exact(self) -> bool:
        """Check if backward solution transformation is exact.

        Returns:
            True if solution back-transformation preserves all information

        """
        return self.get_backward_metadata().is_exact()

    def is_symmetric(self) -> bool:
        """Check if transformation has same characteristics in both directions.

        Returns:
            True if forward and backward have same completeness level

        """
        forward = self.get_forward_metadata()
        backward = self.get_backward_metadata()
        return forward.completeness == backward.completeness

    def has_constraint_loss(self) -> bool:
        """Check if transformation loses any constraints (in either direction).

        Returns:
            True if some constraints cannot be represented

        """
        return (
            self.get_forward_metadata().has_constraint_loss()
            or self.get_backward_metadata().has_constraint_loss()
        )

    def has_objective_loss(self) -> bool:
        """Check if transformation loses any objectives (in either direction).

        Returns:
            True if some objectives cannot be represented

        """
        return (
            self.get_forward_metadata().has_objective_loss()
            or self.get_backward_metadata().has_objective_loss()
        )

    def get_max_impact(self):
        """Get the maximum impact level of information loss (either direction).

        Returns:
            LossImpact enum indicating worst-case impact

        """
        from discrete_optimization.generic_tools.transformation.transformation_metadata import (
            LossImpact,
        )

        forward_impact = self.get_forward_metadata().get_max_impact()
        backward_impact = self.get_backward_metadata().get_max_impact()

        impact_order = [
            LossImpact.NONE,
            LossImpact.MINOR,
            LossImpact.MODERATE,
            LossImpact.MAJOR,
            LossImpact.CRITICAL,
        ]

        if impact_order.index(forward_impact) > impact_order.index(backward_impact):
            return forward_impact
        return backward_impact

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

    def print_metadata(self) -> None:
        """Print human-readable transformation metadata (both directions)."""
        print(f"\n{type(self).__name__}")
        print("=" * 80)

        # Forward metadata
        print("\nFORWARD: Problem Transformation (source → target)")
        print("-" * 80)
        print(self.get_forward_metadata())

        # Backward metadata
        print("\nBACKWARD: Solution Transformation (target solution → source solution)")
        print("-" * 80)
        print(self.get_backward_metadata())

        # Symmetry check
        if self.is_symmetric():
            print("\n✓ Symmetric transformation (same characteristics both directions)")
        else:
            print(
                "\n⚠ Asymmetric transformation (different characteristics in each direction)"
            )

        print()

    def __repr__(self) -> str:
        """String representation of transformation."""
        forward = self.get_forward_metadata()
        backward = self.get_backward_metadata()

        forward_str = "exact" if forward.is_exact() else "lossy"
        backward_str = "exact" if backward.is_exact() else "lossy"

        if forward_str == backward_str:
            return f"{type(self).__name__}({forward_str})"
        else:
            return (
                f"{type(self).__name__}(forward={forward_str}, backward={backward_str})"
            )
