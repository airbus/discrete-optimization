#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformations from Workforce Scheduling to other problems."""

from discrete_optimization.workforce.scheduling.transformations.to_rcpsp import (
    WorkforceSchedulingToRcpspTransformation,
)

__all__ = ["WorkforceSchedulingToRcpspTransformation"]
