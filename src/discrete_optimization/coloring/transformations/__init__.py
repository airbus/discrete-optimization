#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformations between ColoringProblem and ListColoringProblem."""

from discrete_optimization.coloring.transformations.from_list_coloring import (
    ListColoringToColoringTransformation,
)
from discrete_optimization.coloring.transformations.to_list_coloring import (
    ColoringToListColoringTransformation,
)

__all__ = [
    "ColoringToListColoringTransformation",
    "ListColoringToColoringTransformation",
]
