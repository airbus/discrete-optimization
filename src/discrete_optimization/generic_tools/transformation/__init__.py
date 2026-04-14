"""Problem transformation framework for discrete optimization.

This module provides tools for transforming problems between different formulations
and solving via transformed representations.
"""

from discrete_optimization.generic_tools.transformation.composite import (
    CompositeTransformation,
    chain_transformations,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    InformationLoss,
    LossImpact,
    LossType,
    TransformationCompleteness,
    TransformationMetadata,
    exact_transformation,
    lossy_transformation,
    subset_transformation,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)

__all__ = [
    "ProblemTransformation",
    "TransformationSolver",
    "CompositeTransformation",
    "chain_transformations",
    "TransformationMetadata",
    "TransformationCompleteness",
    "InformationLoss",
    "LossType",
    "LossImpact",
    "exact_transformation",
    "lossy_transformation",
    "subset_transformation",
]
