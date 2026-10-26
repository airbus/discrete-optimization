#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Generic

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
)

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    Task,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.base.auto import (
    GenericSchedulingAutoCpBaseSolver,
    SinglemodeGenericSchedulingAutoCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.base import (
    TasksCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    Solution,
)

logger = logging.getLogger(__name__)

AnyResource = NonRenewableResource | UnaryResource | Skill | NonSkillCumulativeResource


class GenericSchedulingAutoCpSatSolver(
    GenericSchedulingAutoCpBaseSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    TasksCpSatSolver[Task],
    Generic[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic cpsat solver for scheduling problems (with or without allocation).

    The needed variables are automatically created, with common constraints (precedence, resource capacity).
    The objective is set by default to global makespan but can be changed by modifying `solver.default_objective` value.

    This solver class needs still to be derived to create a solution from the proper class.
    You will need at least to implement the conversion from the task variables to the actual solution object.

    If custom constraints are needed, override `init_model()`.
    If a custom objective is needed, set `solver.default_objective` to `Objective.CUSTOM` so that no objective is set
    by default and override `init_model()` to define your objective.

    """

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        # construct generic tasks variables
        def solution_value_fn(var):
            return cpsolvercb.Value(var)

        raw_sol = self.retrieve_tasks_variables(solution_value_fn=solution_value_fn)
        # convert to specific solution type
        return self.convert_task_variables_to_solution(raw_sol=raw_sol)


class SinglemodeGenericSchedulingAutoCpSatSolver(
    SinglemodeGenericSchedulingAutoCpBaseSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
    GenericSchedulingAutoCpSatSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Subclass of GenericSchedulingAutoCpSatSolver for single mode problems.

    Give access to task intervals without dealing with modes.

    """

    ...
