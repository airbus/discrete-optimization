#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Metadata for documenting transformation characteristics and losses.

This module provides classes to explicitly document:
- Whether transformations are exact or lossy
- What information is lost (constraints, objectives)
- Why the loss occurs
- Impact on solution quality
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TransformationCompleteness(Enum):
    """Classification of transformation completeness.

    EXACT: Perfect bidirectional mapping, no information loss
    LOSSY_CONSTRAINTS: Some constraints cannot be represented in target
    LOSSY_OBJECTIVES: Some objectives cannot be represented in target
    LOSSY_BOTH: Both constraints and objectives are lost
    SUBSET: Source is a strict subset of target (generalization)
    """

    EXACT = "exact"
    LOSSY_CONSTRAINTS = "lossy_constraints"
    LOSSY_OBJECTIVES = "lossy_objectives"
    LOSSY_BOTH = "lossy_both"
    SUBSET = "subset"  # Source ⊂ Target (e.g., JSP ⊂ FJSP)


class LossType(Enum):
    """Type of information that can be lost in transformation."""

    CONSTRAINT = "constraint"
    OBJECTIVE = "objective"
    PARAMETER = "parameter"
    STRUCTURE = "structure"


class LossImpact(Enum):
    """Impact of information loss on solution quality.

    NONE: No loss, exact transformation
    MINOR: Loss unlikely to affect practical solutions
    MODERATE: May affect solution quality, case-dependent
    MAJOR: Significant impact, solutions may be infeasible in original problem
    CRITICAL: Transformation not recommended without manual verification
    """

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

    def severity(self) -> int:
        """Get numeric severity level (0=NONE, 4=CRITICAL)."""
        severity_map = {
            LossImpact.NONE: 0,
            LossImpact.MINOR: 1,
            LossImpact.MODERATE: 2,
            LossImpact.MAJOR: 3,
            LossImpact.CRITICAL: 4,
        }
        return severity_map[self]

    def __lt__(self, other):
        """Compare by severity."""
        if not isinstance(other, LossImpact):
            return NotImplemented
        return self.severity() < other.severity()

    def __le__(self, other):
        """Compare by severity."""
        if not isinstance(other, LossImpact):
            return NotImplemented
        return self.severity() <= other.severity()

    def __gt__(self, other):
        """Compare by severity."""
        if not isinstance(other, LossImpact):
            return NotImplemented
        return self.severity() > other.severity()

    def __ge__(self, other):
        """Compare by severity."""
        if not isinstance(other, LossImpact):
            return NotImplemented
        return self.severity() >= other.severity()


@dataclass
class InformationLoss:
    """Documents a specific piece of information lost in transformation.

    Attributes:
        name: Name of the lost element (e.g., "incompatibility_constraints")
        loss_type: Type of loss (constraint, objective, parameter)
        description: Human-readable description of what's lost
        reason: Why this information cannot be represented in target
        workaround: Optional suggestion for handling the loss
        impact: Impact on solution quality

    Example:
        # >>> loss = InformationLoss(
        # ...     name="incompatibility_constraints",
        # ...     loss_type=LossType.CONSTRAINT,
        # ...     description="Item incompatibility constraints (items that cannot be in same bin)",
        # ...     reason="SALBP has no concept of task incompatibility",
        # ...     impact=LossImpact.MAJOR,
        # ...     workaround="Pre-filter incompatible items or use BinPack→RCPSP with virtual resources"
        # ... )

    """

    name: str
    loss_type: LossType
    description: str
    reason: str
    impact: LossImpact
    workaround: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"  ⚠ {self.name} ({self.loss_type.value}, impact: {self.impact.value})",
            f"     Description: {self.description}",
            f"     Reason: {self.reason}",
        ]
        if self.workaround:
            lines.append(f"     Workaround: {self.workaround}")
        return "\n".join(lines)


@dataclass
class TransformationMetadata:
    """Complete metadata for a problem transformation.

    Attributes:
        completeness: Overall classification of transformation completeness
        losses: List of specific information losses
        gains: Information added by target (may be ignored)
        assumptions: Assumptions made during transformation
        use_cases: Recommended use cases for this transformation
        warnings: Important warnings for users

    Example:
        # >>> metadata = TransformationMetadata(
        # ...     completeness=TransformationCompleteness.LOSSY_CONSTRAINTS,
        # ...     losses=[
        # ...         InformationLoss(
        # ...             name="incompatibility_constraints",
        # ...             loss_type=LossType.CONSTRAINT,
        # ...             description="Item incompatibility constraints",
        # ...             reason="SALBP has no incompatibility concept",
        # ...             impact=LossImpact.MAJOR
        # ...         )
        # ...     ],
        # ...     assumptions=["No item incompatibility constraints"],
        # ...     use_cases=["Pure bin packing without incompatibility"],
        # ...     warnings=["Solutions may violate incompatibility if present"]
        # ... )

    """

    completeness: TransformationCompleteness
    losses: list[InformationLoss] = field(default_factory=list)
    gains: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_exact(self) -> bool:
        """Check if transformation is exact (no losses)."""
        return self.completeness in (
            TransformationCompleteness.EXACT,
            TransformationCompleteness.SUBSET,
        )

    def has_constraint_loss(self) -> bool:
        """Check if any constraints are lost."""
        return self.completeness in (
            TransformationCompleteness.LOSSY_CONSTRAINTS,
            TransformationCompleteness.LOSSY_BOTH,
        ) or any(loss.loss_type == LossType.CONSTRAINT for loss in self.losses)

    def has_objective_loss(self) -> bool:
        """Check if any objectives are lost."""
        return self.completeness in (
            TransformationCompleteness.LOSSY_OBJECTIVES,
            TransformationCompleteness.LOSSY_BOTH,
        ) or any(loss.loss_type == LossType.OBJECTIVE for loss in self.losses)

    def get_max_impact(self) -> LossImpact:
        """Get the maximum impact level among all losses."""
        if not self.losses:
            return LossImpact.NONE

        impact_order = [
            LossImpact.NONE,
            LossImpact.MINOR,
            LossImpact.MODERATE,
            LossImpact.MAJOR,
            LossImpact.CRITICAL,
        ]

        max_impact = LossImpact.NONE
        for loss in self.losses:
            if impact_order.index(loss.impact) > impact_order.index(max_impact):
                max_impact = loss.impact

        return max_impact

    def get_losses_by_type(self, loss_type: LossType) -> list[InformationLoss]:
        """Get all losses of a specific type."""
        return [loss for loss in self.losses if loss.loss_type == loss_type]

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [f"Transformation Completeness: {self.completeness.value}"]

        if self.is_exact():
            lines.append("  ✓ Exact transformation (no information loss)")
        else:
            max_impact = self.get_max_impact()
            lines.append(f"  ⚠ Lossy transformation (max impact: {max_impact.value})")

        if self.losses:
            lines.append(f"\nLosses ({len(self.losses)}):")
            for loss in self.losses:
                lines.append(str(loss))

        if self.gains:
            lines.append(f"\nGains (available but may be ignored):")
            for gain in self.gains:
                lines.append(f"  + {gain}")

        if self.assumptions:
            lines.append(f"\nAssumptions:")
            for assumption in self.assumptions:
                lines.append(f"  • {assumption}")

        if self.use_cases:
            lines.append(f"\nRecommended use cases:")
            for use_case in self.use_cases:
                lines.append(f"  ✓ {use_case}")

        if self.warnings:
            lines.append(f"\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)


# Convenience constructors for common patterns
def exact_transformation(
    use_cases: Optional[list[str]] = None,
) -> TransformationMetadata:
    """Create metadata for an exact transformation."""
    return TransformationMetadata(
        completeness=TransformationCompleteness.EXACT,
        use_cases=use_cases or ["Exact problem equivalence"],
    )


def subset_transformation(
    use_cases: Optional[list[str]] = None,
    assumptions: Optional[list[str]] = None,
) -> TransformationMetadata:
    """Create metadata for a subset transformation (source ⊂ target)."""
    return TransformationMetadata(
        completeness=TransformationCompleteness.SUBSET,
        use_cases=use_cases or ["Generalization of source problem"],
        assumptions=assumptions or [],
    )


def lossy_transformation(
    losses: list[InformationLoss],
    assumptions: Optional[list[str]] = None,
    use_cases: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> TransformationMetadata:
    """Create metadata for a lossy transformation."""
    # Determine completeness from losses
    has_constraints = any(loss.loss_type == LossType.CONSTRAINT for loss in losses)
    has_objectives = any(loss.loss_type == LossType.OBJECTIVE for loss in losses)

    if has_constraints and has_objectives:
        completeness = TransformationCompleteness.LOSSY_BOTH
    elif has_constraints:
        completeness = TransformationCompleteness.LOSSY_CONSTRAINTS
    elif has_objectives:
        completeness = TransformationCompleteness.LOSSY_OBJECTIVES
    else:
        completeness = TransformationCompleteness.EXACT

    return TransformationMetadata(
        completeness=completeness,
        losses=losses,
        assumptions=assumptions or [],
        use_cases=use_cases or [],
        warnings=warnings or [],
    )
