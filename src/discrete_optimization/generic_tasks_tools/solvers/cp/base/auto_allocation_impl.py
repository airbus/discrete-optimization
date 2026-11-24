#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_allocation_impl import (
    GenericAllocationImplProblem,
    GenericAllocationImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    NonRenewableResource,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto_allocation import (
    GenericAllocationAutoCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.auto_impl import (
    GenericSchedulingAutoCpImplSolver,
)

logger = logging.getLogger(__name__)


class GenericAllocationAutoCpImplSolver(
    GenericAllocationAutoCpSolver[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
    GenericSchedulingAutoCpImplSolver,
):
    """Generic cp solver for allocation problems (without scheduling) applied to `GenericAllocationImplProblem`."""

    problem: GenericAllocationImplProblem

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> GenericAllocationImplSolution:
        return GenericAllocationImplSolution(problem=self.problem, raw_sol=raw_sol)
