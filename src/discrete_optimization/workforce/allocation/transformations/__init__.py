#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformations from Workforce Allocation to other problems."""

from discrete_optimization.workforce.allocation.transformations.to_coloring import (
    WorkforceAllocationToColoringTransformation,
)
from discrete_optimization.workforce.allocation.transformations.to_list_coloring import (
    WorkforceAllocationToListColoringTransformation,
)

__all__ = [
    "WorkforceAllocationToColoringTransformation",
    "WorkforceAllocationToListColoringTransformation",
]
