#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    NonRenewableResource,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto_allocation_impl import (
    GenericAllocationAutoCpImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto_allocation import (
    GenericAllocationAutoOptalSolver,
)

logger = logging.getLogger(__name__)


class GenericAllocationAutoOptalImplSolver(
    GenericAllocationAutoCpImplSolver,
    GenericAllocationAutoOptalSolver[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic optal solver for allocation problems (without scheduling) applied to `GenericAllocationImplProblem`."""

    ...
