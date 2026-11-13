#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from/to generic scheduling to/from RCPSP."""

from __future__ import annotations

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.transformations.generic_scheduling_impl import (
    ToGenericSchedulingImpl,
)
from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
    RcpspResourceDependentSolution,
)


def transform_solution_from_raw_generic_to_rcpsp_resource_dependent(
    raw_sol: RawSolution[Task, UnaryResource, Skill],
    problem: RcpspResourceDependentProblem,
) -> RcpspResourceDependentSolution:
    """Convert generic solution to RCPSP resource dependent solution.

    Args:
        raw_sol:
        problem:

    Returns:

    """
    return RcpspResourceDependentSolution(
        problem=problem,
        schedule={
            t: (
                raw_sol.task_variables[t].get_start_or_end(
                    start_or_end=StartOrEnd.START
                ),
                raw_sol.task_variables[t].get_start_or_end(start_or_end=StartOrEnd.END),
            )
            for t in problem.tasks_list
        },
        modes={t: raw_sol.task_variables[t].mode for t in problem.tasks_list},
    )


class RcpspResourceDependentToGenericSchedulingTransformation(
    ToGenericSchedulingImpl[
        RcpspResourceDependentProblem, RcpspResourceDependentSolution
    ]
):
    """Transform RcpspResourceDependentProblem to GenericSchedulingImplProblem."""

    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: RcpspResourceDependentProblem,
    ) -> RcpspResourceDependentSolution:
        return transform_solution_from_raw_generic_to_rcpsp_resource_dependent(
            raw_sol=raw_sol, problem=source_problem
        )
