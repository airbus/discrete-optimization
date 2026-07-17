#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from CommonShopProblem (JSP/FJSP/OSP) to GenericSchedulingImpl."""

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    Skill,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.transformations.generic_scheduling_impl import (
    ToGenericSchedulingImpl,
)
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem, Task


def transform_solution_from_raw_generic_to_shop(
    raw_sol: RawSolution[Task, UnaryResource, Skill], problem: CommonShopProblem
) -> AnyShopSolution:
    schedule_and_machine = [
        [
            (
                (task_var := raw_sol.task_variables[j, k]).start,
                task_var.end,
                problem.mode2machine[j, k][task_var.mode],
                task_var.mode,
            )
            for k, sub_job in enumerate(job.subjobs)
        ]
        for j, job in enumerate(problem.list_jobs)
    ]
    return AnyShopSolution(
        problem=problem,
        schedule=[[(x[0], x[1]) for x in sched_i] for sched_i in schedule_and_machine],
        machine_index=[[x[2] for x in sched_i] for sched_i in schedule_and_machine],
        recipe_index=[[x[3] for x in sched_i] for sched_i in schedule_and_machine],
    )


class ShopToGenericSchedulingTransformation(
    ToGenericSchedulingImpl[
        CommonShopProblem,
        AnyShopSolution,
    ]
):
    """Transform CommonShopProblem to GenericSchedulingImplProblem.

    This transformation works for JSP, FJSP, and OSP problems:
    - JSP: Single mode per task (one recipe per subjob)
    - FJSP: Multiple modes per task (multiple recipe options per subjob)
    - OSP: Single mode per task, no precedence constraints

    Mapping:
    - Tasks: (job_index, subjob_index) tuples
    - Modes: Recipe options for each subjob
    - Cumulative resources: Machines (capacity 1 each)
    - Precedence: From problem's get_precedence_constraints()
    - No-overlap: Tasks within same job (from get_no_overlap())

    """

    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: CommonShopProblem,
    ) -> AnyShopSolution:
        return transform_solution_from_raw_generic_to_shop(
            raw_sol=raw_sol, problem=source_problem
        )
