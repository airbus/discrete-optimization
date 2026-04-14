#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from JobShop to FlexibleJobShop."""

from typing import Optional

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution, Job
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution


class JspToFjspTransformation(
    ProblemTransformation[
        JobShopProblem, JobShopSolution, FJobShopProblem, FJobShopSolution
    ]
):
    """Transform JobShop to FlexibleJobShop.

    Mapping:
    - Each subjob with fixed machine → subjob with 1 machine option
    - Job structure preserved
    - Precedence within jobs preserved

    JobShop is a special case of FlexibleJobShop where each operation
    can only be processed on exactly one machine.
    """

    def transform_problem(self, source_problem: JobShopProblem) -> FJobShopProblem:
        """Transform JobShop to FlexibleJobShop.

        Args:
            source_problem: JobShop problem instance

        Returns:
            Equivalent FlexibleJobShop problem

        """
        # Convert each job: list[Subjob] → Job with list[SubjobOptions]
        fjsp_jobs = []

        for job_idx, jsp_job in enumerate(source_problem.list_jobs):
            # Each subjob in JSP has 1 fixed machine
            # In FJSP, this becomes 1 option (list with single Subjob)
            sub_jobs_options = [[subjob] for subjob in jsp_job]

            fjsp_jobs.append(Job(job_id=job_idx, sub_jobs=sub_jobs_options))

        return FJobShopProblem(
            list_jobs=fjsp_jobs,
            n_jobs=source_problem.n_jobs,
            n_machines=source_problem.n_machines,
            horizon=source_problem.horizon,
        )

    def back_transform_solution(
        self, solution: FJobShopSolution, source_problem: JobShopProblem
    ) -> JobShopSolution:
        """Transform FlexibleJobShop solution back to JobShop solution.

        Args:
            solution: FlexibleJobShop solution
            source_problem: Original JobShop problem

        Returns:
            Equivalent JobShop solution

        """
        # FJSP schedule: list[list[tuple[start, end, machine, option]]]
        # JSP schedule: list[list[tuple[start, end]]]
        jsp_schedule = [
            [(start, end) for start, end, machine, option in job_schedule]
            for job_schedule in solution.schedule
        ]

        return JobShopSolution(problem=source_problem, schedule=jsp_schedule)

    def forward_transform_solution(
        self, solution: JobShopSolution, target_problem: FJobShopProblem
    ) -> Optional[FJobShopSolution]:
        """Transform JobShop solution to FlexibleJobShop solution (for warmstart).

        Args:
            solution: JobShop solution
            target_problem: Target FlexibleJobShop problem

        Returns:
            Equivalent FlexibleJobShop solution for warmstart

        """
        # JSP schedule: list[list[tuple[start, end]]]
        # FJSP schedule: list[list[tuple[start, end, machine, option]]]
        # We need to add machine_id and option=0

        fjsp_schedule = []

        for job_idx, job_schedule in enumerate(solution.schedule):
            fjsp_job_schedule = []
            for subjob_idx, (start, end) in enumerate(job_schedule):
                # Get the machine from the source problem
                machine_id = solution.problem.list_jobs[job_idx][subjob_idx].machine_id
                option = 0  # Only 1 option in transformed FJSP

                fjsp_job_schedule.append((start, end, machine_id, option))

            fjsp_schedule.append(fjsp_job_schedule)

        return FJobShopSolution(problem=target_problem, schedule=fjsp_schedule)
