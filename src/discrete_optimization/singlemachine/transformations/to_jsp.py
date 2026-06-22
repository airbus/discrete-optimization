#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Single Machine Scheduling to Job Shop Problem.

This is a LOSSY transformation because:
- Due dates are lost (JSP doesn't model deadlines)
- Weights are lost (JSP optimizes makespan, not weighted tardiness)
- Release dates may not be fully enforced (JSP doesn't have built-in release constraints)
"""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    InformationLoss,
    LossImpact,
    LossType,
    TransformationMetadata,
    lossy_transformation,
)
from discrete_optimization.shop.base import Job, Subjob, SubjobRecipe
from discrete_optimization.shop.jsp.problem import JobShopProblem, JobShopSolution
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


class SingleMachineToJspTransformation(
    ProblemTransformation[
        WeightedTardinessProblem,
        WTSolution,
        JobShopProblem,
        JobShopSolution,
    ]
):
    """Transform Single Machine Scheduling to Job Shop Problem.

    Mapping:
    - Single machine → JSP with 1 machine
    - Each job → JSP job with 1 subjob
    - All subjobs assigned to machine 0
    - Processing times map directly

    Information Loss:
    - Due dates: Lost (JSP doesn't model deadlines)
    - Weights: Lost (JSP only optimizes makespan)
    - Release dates: Partially lost (not enforced in standard JSP)

    This transformation is LOSSY but still useful for:
    - Reusing powerful JSP solvers for single machine scheduling
    - Focusing on minimizing makespan (ignoring tardiness)
    - Benchmarking JSP solvers on single-machine instances
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward transformation (SingleMachine → JSP)."""
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="due_dates",
                    loss_type=LossType.CONSTRAINT,
                    description="Job due dates / deadlines",
                    reason="JSP doesn't model deadlines or due dates",
                    impact=LossImpact.MAJOR,
                    workaround="Post-process solution to evaluate tardiness",
                ),
                InformationLoss(
                    name="weights",
                    loss_type=LossType.OBJECTIVE,
                    description="Job weights for tardiness penalty",
                    reason="JSP optimizes makespan, not weighted tardiness",
                    impact=LossImpact.MAJOR,
                    workaround="Use makespan as proxy or post-process to compute tardiness",
                ),
                InformationLoss(
                    name="release_dates",
                    loss_type=LossType.CONSTRAINT,
                    description="Job release dates (earliest start times)",
                    reason="Standard JSP doesn't enforce release dates",
                    impact=LossImpact.MODERATE,
                    workaround="Use JSP variant with release constraints if available",
                ),
            ],
            use_cases=[
                "Reuse JSP solvers for single machine scheduling",
                "Focus on makespan minimization instead of weighted tardiness",
                "Benchmark JSP algorithms on single machine instances",
            ],
            warnings=[
                "Due dates and weights are lost",
                "Release dates not enforced in standard JSP",
                "Optimal JSP solution may not be optimal for weighted tardiness",
            ],
        )

    def transform_problem(
        self, source_problem: WeightedTardinessProblem
    ) -> JobShopProblem:
        """Transform Single Machine problem to JSP.

        Args:
            source_problem: WeightedTardinessProblem instance

        Returns:
            Equivalent JobShopProblem (single machine, single subjob per job)
        """
        # Each single-machine job becomes a JSP job with 1 subjob on machine 0
        list_jobs = [
            Job(
                job_index=j,
                subjobs=[
                    Subjob(
                        job_index=j,
                        subjob_index=0,
                        recipes=[
                            SubjobRecipe(
                                machine_index=0,
                                processing_time=source_problem.processing_times[j],
                            )
                        ],
                    )
                ],
            )
            for j in range(source_problem.num_jobs)
        ]

        return JobShopProblem(
            list_jobs=list_jobs,
            n_jobs=source_problem.num_jobs,
            n_machines=1,
            horizon=source_problem.get_makespan_upper_bound(),
        )

    def back_transform_solution(
        self, solution: JobShopSolution, source_problem: WeightedTardinessProblem
    ) -> WTSolution:
        """Transform JSP solution back to Single Machine solution.

        Args:
            solution: JSP solution (schedule for jobs on 1 machine)
            source_problem: Original WeightedTardinessProblem

        Returns:
            Equivalent WTSolution (permutation + schedule)
        """
        # Extract permutation: order jobs by start time
        jobs_with_start_times = [
            (job_idx, solution.schedule[job_idx][0][0])  # (job, start_time)
            for job_idx in range(source_problem.num_jobs)
        ]

        # Sort by start time
        sorted_jobs = sorted(jobs_with_start_times, key=lambda x: x[1])
        permutation = [job_idx for job_idx, _ in sorted_jobs]

        # Build schedule: (start, end) for each job
        schedule = [
            solution.schedule[job_idx][0]  # (start, end) from JSP
            for job_idx in range(source_problem.num_jobs)
        ]

        return WTSolution(
            problem=source_problem, schedule=schedule, permutation=permutation
        )

    def forward_transform_solution(
        self, solution: WTSolution, target_problem: JobShopProblem
    ) -> Optional[JobShopSolution]:
        """Transform Single Machine solution to JSP solution (for warmstart).

        Args:
            solution: WTSolution (permutation + schedule)
            target_problem: Target JobShopProblem

        Returns:
            Equivalent JobShopSolution
        """
        # Build JSP schedule from single-machine schedule
        # Each job has 1 subjob on machine 0
        jsp_schedule = [
            [solution.schedule[job_idx]]  # Wrap in list (single subjob)
            for job_idx in range(solution.problem.num_jobs)
        ]

        return JobShopSolution(problem=target_problem, schedule=jsp_schedule)
