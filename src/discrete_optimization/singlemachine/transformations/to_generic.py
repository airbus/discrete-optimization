#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

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
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


def transform_solution_from_raw_generic_to_wt(
    raw_sol: RawSolution[Task, UnaryResource, Skill], problem: WeightedTardinessProblem
) -> WTSolution:
    """ """
    return WTSolution(
        problem=problem,
        schedule=[
            (raw_sol.task_variables[t].start, raw_sol.task_variables[t].end)
            for t in problem.tasks_list
        ],
    )


class SingleMachineToGenericSchedulingTransformation(
    ToGenericSchedulingImpl[WeightedTardinessProblem, WTSolution]
):
    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: WeightedTardinessProblem,
    ) -> WTSolution:
        return transform_solution_from_raw_generic_to_wt(
            raw_sol=raw_sol, problem=source_problem
        )
