#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from CommonShopProblem (JSP/FJSP/OSP) to GenericSchedulingImpl."""

from typing import Optional

from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem, Task


class ShopToGenericSchedulingTransformation(
    ProblemTransformation[
        CommonShopProblem,
        AnyShopSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
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

    def __init__(self):
        """Initialize transformation."""
        pass

    def transform_problem(
        self, source_problem: CommonShopProblem
    ) -> GenericSchedulingImplProblem:
        """Transform CommonShopProblem to GenericSchedulingImplProblem.

        Args:
            source_problem: CommonShopProblem instance (JSP/FJSP/OSP)

        Returns:
            Equivalent GenericSchedulingImplProblem

        """
        # Build durations_per_mode: task -> mode -> duration
        durations_per_mode: dict[Task, dict[int, int]] = {}

        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx, subjob in enumerate(job.subjobs):
                task = (job_idx, subjob_idx)
                durations_per_mode[task] = {}

                # Each recipe option becomes a mode
                for mode_idx, recipe in enumerate(subjob.recipes):
                    durations_per_mode[task][mode_idx] = recipe.processing_time

        # Build resource_consumptions: task -> mode -> resource -> consumption
        resource_consumptions: dict[Task, dict[int, dict[str, int]]] = {}

        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx, subjob in enumerate(job.subjobs):
                task = (job_idx, subjob_idx)
                resource_consumptions[task] = {}

                # Each recipe option consumes the corresponding machine
                for mode_idx, recipe in enumerate(subjob.recipes):
                    machine_resource = f"M{recipe.machine_index}"
                    resource_consumptions[task][mode_idx] = {machine_resource: 1}

        # Build non_skill_cumulative_resources: machines with capacity 1
        non_skill_cumulative_resources = {
            f"M{machine}": 1 for machine in range(source_problem.n_machines)
        }

        # Get successors from problem's precedence constraints
        successors = source_problem.get_precedence_constraints()

        # Get no-overlap sets (tasks within same job)
        no_overlap_sets = source_problem.get_no_overlap()

        return GenericSchedulingImplProblem(
            horizon=source_problem.horizon,
            durations_per_mode=durations_per_mode,
            resource_consumptions=resource_consumptions,
            successors=successors,
            non_skill_cumulative_resources=non_skill_cumulative_resources,
            no_overlap_sets=no_overlap_sets,
        )

    def back_transform_solution(
        self,
        solution: GenericSchedulingImplSolution,
        source_problem: CommonShopProblem,
    ) -> AnyShopSolution:
        """Transform GenericSchedulingImplSolution back to CommonShopProblem solution.

        Args:
            solution: GenericSchedulingImplSolution
            source_problem: Original CommonShopProblem

        Returns:
            Equivalent CommonShopProblem solution

        """
        # Build shop schedule from GenericSchedulingImplSolution
        shop_schedule = [[None] * len(job.subjobs) for job in source_problem.list_jobs]
        machine_index = [[None] * len(job.subjobs) for job in source_problem.list_jobs]

        for job_idx, job in enumerate(source_problem.list_jobs):
            for subjob_idx in range(len(job.subjobs)):
                task = (job_idx, subjob_idx)

                start = solution.get_start_time(task)
                end = solution.get_end_time(task)
                mode = solution.get_mode(task)

                # Get machine from the chosen mode/recipe
                machine_id = job.subjobs[subjob_idx].recipes[mode].machine_index

                shop_schedule[job_idx][subjob_idx] = (start, end)
                machine_index[job_idx][subjob_idx] = machine_id

        return AnyShopSolution(
            problem=source_problem,
            schedule=shop_schedule,
            machine_index=machine_index,
        )

    def forward_transform_solution(
        self,
        solution: AnyShopSolution,
        target_problem: GenericSchedulingImplProblem,
    ) -> Optional[GenericSchedulingImplSolution]:
        """Transform CommonShopProblem solution to GenericSchedulingImplSolution.

        Args:
            solution: CommonShopProblem solution
            target_problem: Target GenericSchedulingImplProblem

        Returns:
            Equivalent GenericSchedulingImplSolution for warm-start

        """
        # Build task_variables dict
        task_variables: dict[Task, TaskVariable] = {}

        for job_idx, job_schedule in enumerate(solution.schedule):
            for subjob_idx, (start, end) in enumerate(job_schedule):
                task = (job_idx, subjob_idx)
                mode = solution.get_mode(task)

                task_variables[task] = TaskVariable(
                    start=start,
                    end=end,
                    mode=mode,
                    allocated={},  # No unary resources in shop problems
                )

        raw_sol = RawSolution(task_variables=task_variables)

        return GenericSchedulingImplSolution(problem=target_problem, raw_sol=raw_sol)
