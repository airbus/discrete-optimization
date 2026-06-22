#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Workforce Scheduling to Flexible Job Shop (FJSP).

Teams are mapped to machines, tasks to operations.
"""

from typing import Optional

import numpy as np

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
from discrete_optimization.shop.fjsp.problem import (
    FJobShopProblem,
    FJobShopSolution,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
)


class WorkforceSchedulingToFjspTransformation(
    ProblemTransformation[
        AllocSchedulingProblem,
        AllocSchedulingSolution,
        FJobShopProblem,
        FJobShopSolution,
    ]
):
    """Transform Workforce Scheduling to Flexible Job Shop.

    Mapping:
    - Tasks → Operations (each task becomes a single-operation job)
    - Teams → Machines
    - Available teams for task → Eligible machines for operation
    - Team calendars → Machine availability
    - Precedence → Job/operation precedence

    This transformation is LOSSY:
    - Jobs are artificially created (one per task)
    - Some workforce-specific constraints lost
    - Cumulative resources ignored
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (WorkforceScheduling → FJSP).

        This direction is LOSSY.
        """
        losses = [
            InformationLoss(
                name="same_allocation_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Tasks that must be assigned to the same team",
                reason="FJSP has no same-machine constraint for operations across jobs",
                impact=LossImpact.MAJOR,
                workaround="Post-process to check same-team constraints",
            ),
            InformationLoss(
                name="cumulative_resources",
                loss_type=LossType.CONSTRAINT,
                description="Cumulative resource consumption",
                reason="FJSP only models machine assignment, not resource consumption",
                impact=LossImpact.MODERATE,
                workaround="Use RCPSP transformation for full resource modeling",
            ),
            InformationLoss(
                name="task_precedence",
                loss_type=LossType.STRUCTURE,
                description="the arbitrary precedence constraint in the workforce scheduling problem "
                "can't be mapped to the well structured precedence constraint of flexible job shop.",
                reason="FJSP requires job-operation hierarchy",
                impact=LossImpact.MAJOR,
                workaround="Accept artificial structure for compatibility",
            ),
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "Each task becomes a single-operation job",
                "Teams map to machines",
                "Precedence constraints approximate job ordering",
            ],
            use_cases=[
                "Workforce scheduling with team flexibility",
                "Access to FJSP solvers and algorithms",
                "Scheduling problems with machine-like resources",
            ],
            warnings=[
                "Same_allocation constraints not enforced",
                "Job structure is artificial",
                "Cumulative resources ignored",
            ],
        )

    def transform_problem(
        self, source_problem: AllocSchedulingProblem
    ) -> FJobShopProblem:
        """Transform Workforce Scheduling to FJSP.

        Args:
            source_problem: AllocSchedulingProblem instance

        Returns:
            Equivalent FJobShopProblem
        """
        # Create jobs: one job per task
        n_jobs = len(source_problem.tasks_list)
        n_machines = len(source_problem.team_names)

        # Build job data
        # Each job has one operation
        jobs = {}
        job_idx = 0

        for task in source_problem.tasks_list:
            task_desc = source_problem.tasks_data[task]
            duration = task_desc.duration_task

            # Get eligible teams (machines)
            eligible_teams = source_problem.available_team_for_activity.get(task, set())
            eligible_machine_ids = []

            for team in eligible_teams:
                if team in source_problem.teams_to_index:
                    team_idx = source_problem.teams_to_index[team]
                    eligible_machine_ids.append(team_idx)

            # If no eligible teams, make all teams eligible
            if not eligible_machine_ids:
                eligible_machine_ids = list(range(n_machines))

            # Create job with single operation
            # operations = [(eligible_machines, duration), ...]
            jobs[job_idx] = {
                "operations": [(eligible_machine_ids, duration)],
                "original_task": task,  # Store for back-transformation
            }

            job_idx += 1

        # Note: FJSP precedence is typically within jobs
        # We lose cross-job precedence in this transformation
        # This is documented in the metadata as a loss

        return FJobShopProblem(
            n_jobs=n_jobs,
            n_machines=n_machines,
            list_jobs=[
                Job(
                    idx,
                    subjobs=[
                        Subjob(
                            job_index=idx,
                            subjob_index=0,
                            recipes=[
                                SubjobRecipe(
                                    machine_index=m,
                                    processing_time=jobs[idx]["operations"][0][1],
                                )
                                for m in jobs[idx]["operations"][0][0]
                            ],
                        )
                    ],
                )
                for idx in range(len(jobs))
            ],
        )

    def back_transform_solution(
        self,
        solution: FJobShopSolution,
        source_problem: AllocSchedulingProblem,
    ) -> AllocSchedulingSolution:
        """Transform FJSP solution back to Workforce Scheduling.

        Args:
            solution: FJobShopSolution
            source_problem: Original AllocSchedulingProblem

        Returns:
            Equivalent AllocSchedulingSolution
        """
        n_tasks = len(source_problem.tasks_list)

        # Create schedule and allocation arrays
        schedule = np.zeros((n_tasks, 2), dtype=int)
        allocation = np.zeros(n_tasks, dtype=int)

        # Map from job_idx to task_idx
        for task_idx, task in enumerate(source_problem.tasks_list):
            job_idx = task_idx  # One-to-one mapping
            start, end = solution.schedule[job_idx][0]
            machine = solution.machine_index[job_idx][0]
            schedule[task_idx, 0] = start
            schedule[task_idx, 1] = end
            allocation[task_idx] = machine
        return AllocSchedulingSolution(
            problem=source_problem,
            schedule=schedule,
            allocation=allocation,
        )

    def forward_transform_solution(
        self,
        solution: AllocSchedulingSolution,
        target_problem: FJobShopProblem,
    ) -> Optional[FJobShopSolution]:
        """Transform Workforce Scheduling solution to FJSP (for warmstart).

        Args:
            solution: AllocSchedulingSolution
            target_problem: Target FJobShopProblem

        Returns:
            Equivalent FJobShopSolution
        """
        # Build FJSP schedule
        schedule = []
        for task_idx, task in enumerate(solution.problem.tasks_list):
            start = int(solution.schedule[task_idx, 0])
            end = int(solution.schedule[task_idx, 1])
            machine_id = int(solution.allocation[task_idx])
            option = next(
                i
                for i in range(
                    len(target_problem.list_jobs[task_idx].subjobs[0].recipes)
                )
                if target_problem.list_jobs[task_idx]
                .subjobs[0]
                .recipes[i]
                .machine_index
                == machine_id
            )
            # Create operation schedule
            schedule.append([(start, end, machine_id, option)])
        return FJobShopSolution(
            problem=target_problem,
            schedule=schedule,
        )
