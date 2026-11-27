#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_allocation import (
    NoNonSkillCumulativeResource,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    Task,
)
from discrete_optimization.generic_tasks_tools.skill import (
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto_allocation import (
    GenericAllocationAutoCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.auto import (
    GenericSchedulingAutoOptalSolver,
)

logger = logging.getLogger(__name__)

AnyResource = NonRenewableResource | UnaryResource | Skill


class GenericAllocationAutoOptalSolver(
    GenericAllocationAutoCpSolver[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
    GenericSchedulingAutoOptalSolver[
        Task,
        UnaryResource,
        Skill,
        NoNonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    Generic[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic optal solver for allocation problems (without scheduling).

    The needed variables are automatically created, with common constraints (e.g. resource capacity).
    The objective is set to match the problem objective according to its list of objective computers.

    This solver class needs still to be derived to create a solution from the proper class.
    You will need at least to implement the conversion from the task variables to the actual solution object.

    If custom constraints are needed, override `init_model()`.
    If a custom objective is needed,
    - if the original problem was using the CustomObjectiveComputer` class, implement `get_custom_objective_expr()`
    which will be used to translate the custom objective
    - or simply override `init_model()` by setting the objective "a la mano".

    """

    ...
