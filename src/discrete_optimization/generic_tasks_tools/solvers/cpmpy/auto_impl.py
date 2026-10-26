#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    NonRenewableResource,
    NonSkillCumulativeResource,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.solvers.base.auto_impl import (
    GenericSchedulingAutoCpBaseImplSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpmpy.auto import (
    GenericSchedulingAutoCpmpySolver,
)

logger = logging.getLogger(__name__)


class GenericSchedulingAutoCpmpyImplSolver(
    GenericSchedulingAutoCpBaseImplSolver,
    GenericSchedulingAutoCpmpySolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic implementation of cpmpy solver for scheduling problems (with or without allocation).

    It implements abstract class `GenericSchedulingAutoCpmpySolver`.

    """

    ...
