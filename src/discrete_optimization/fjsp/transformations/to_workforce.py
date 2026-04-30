#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Flexible Job Shop to Workforce Scheduling.

This is the inverse of WorkforceScheduling → FJSP.
Operations are mapped to tasks, machines to teams.
"""

from typing import Optional

import numpy as np

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    TasksDescription,
)


class FjspToWorkforceSchedulingTransformation(
    ProblemTransformation[
        FJobShopProblem,
        FJobShopSolution,
        AllocSchedulingProblem,
        AllocSchedulingSolution,
    ]
):
    """Transform Flexible Job Shop to Workforce Scheduling.

    Mapping:
    - Operations → Tasks
    - Machines → Teams
    - Eligible machines → Available teams for task
    - Operation duration → Task duration
    - Job precedence → Task precedence

    This transformation is EXACT:
    - All FJSP constraints preserved in workforce scheduling
    - Machines become teams (resources)
    - Operations become tasks with team eligibility
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (FJSP → WorkforceScheduling).

        This direction is EXACT: all FJSP information preserved.
        """
        return exact_transformation(
            use_cases=[
                "Convert FJSP to workforce scheduling formulation",
                "Use workforce scheduling solvers on FJSP instances",
                "Model machine flexibility as team assignment",
            ]
        )

    def transform_problem(
        self, source_problem: FJobShopProblem
    ) -> AllocSchedulingProblem:
        """Transform FJSP to Workforce Scheduling.

        Args:
            source_problem: FJobShopProblem instance

        Returns:
            Equivalent AllocSchedulingProblem
        """
        # Create teams from machines
        team_names = [f"machine_{i}" for i in range(source_problem.n_machines)]

        # Machine calendars (always available in FJSP)
        horizon = 10000  # Large horizon
        calendar_team = {team: [(0, horizon)] for team in team_names}

        # Create tasks from operations
        tasks_list = []
        tasks_data = {}
        available_team_for_activity = {}
        precedence_constraints = {}

        task_id = 0
        job_task_mapping = {}  # job_id -> list of task_ids

        for job_id, job_data in source_problem.jobs_data.items():
            job_tasks = []

            for op_idx, (eligible_machines, duration) in enumerate(
                job_data["operations"]
            ):
                task_name = f"job_{job_id}_op_{op_idx}"
                tasks_list.append(task_name)

                # Task description
                tasks_data[task_name] = TasksDescription(
                    duration_task=duration,
                    resource_consumption={},
                )

                # Eligible teams (machines)
                eligible_teams = {team_names[m_id] for m_id in eligible_machines}
                available_team_for_activity[task_name] = eligible_teams

                # Precedence within job (operation sequence)
                if op_idx > 0:
                    prev_task = f"job_{job_id}_op_{op_idx - 1}"
                    if prev_task not in precedence_constraints:
                        precedence_constraints[prev_task] = set()
                    precedence_constraints[prev_task].add(task_name)

                job_tasks.append(task_name)
                task_id += 1

            job_task_mapping[job_id] = job_tasks

        # No same_allocation constraints by default
        same_allocation = []

        # Time windows (unbounded)
        start_window = {task: (None, None) for task in tasks_list}
        end_window = {task: (None, None) for task in tasks_list}
        original_start = {task: 0 for task in tasks_list}
        original_end = {task: horizon for task in tasks_list}

        return AllocSchedulingProblem(
            team_names=team_names,
            calendar_team=calendar_team,
            horizon=horizon,
            tasks_list=tasks_list,
            tasks_data=tasks_data,
            same_allocation=same_allocation,
            precedence_constraints=precedence_constraints,
            available_team_for_activity=available_team_for_activity,
            start_window=start_window,
            end_window=end_window,
            original_start=original_start,
            original_end=original_end,
        )

    def back_transform_solution(
        self,
        solution: AllocSchedulingSolution,
        source_problem: FJobShopProblem,
    ) -> FJobShopSolution:
        """Transform Workforce Scheduling solution back to FJSP.

        Args:
            solution: AllocSchedulingSolution
            source_problem: Original FJobShopProblem

        Returns:
            Equivalent FJobShopSolution
        """
        # Build FJSP schedule
        schedule = {}

        # Parse task names to extract job and operation indices
        for task_idx, task in enumerate(solution.problem.tasks_list):
            # Task name format: "job_{job_id}_op_{op_idx}"
            parts = task.split("_")
            if len(parts) >= 4:
                job_id = int(parts[1])
                op_idx = int(parts[3])

                start = int(solution.schedule[task_idx, 0])
                end = int(solution.schedule[task_idx, 1])
                team_idx = int(solution.allocation[task_idx])

                # Machine ID from team index
                machine_id = team_idx

                if job_id not in schedule:
                    schedule[job_id] = {}

                schedule[job_id][op_idx] = {
                    "start": start,
                    "end": end,
                    "machine": machine_id,
                }

        return FJobShopSolution(
            problem=source_problem,
            schedule=schedule,
        )

    def forward_transform_solution(
        self,
        solution: FJobShopSolution,
        target_problem: AllocSchedulingProblem,
    ) -> Optional[AllocSchedulingSolution]:
        """Transform FJSP solution to Workforce Scheduling (for warmstart).

        Args:
            solution: FJobShopSolution
            target_problem: Target AllocSchedulingProblem

        Returns:
            Equivalent AllocSchedulingSolution
        """
        n_tasks = len(target_problem.tasks_list)

        # Create schedule and allocation arrays
        schedule_arr = np.zeros((n_tasks, 2), dtype=int)
        allocation_arr = np.zeros(n_tasks, dtype=int)

        # Map tasks to schedule
        for task_idx, task in enumerate(target_problem.tasks_list):
            # Parse task name: "job_{job_id}_op_{op_idx}"
            parts = task.split("_")
            if len(parts) >= 4:
                job_id = int(parts[1])
                op_idx = int(parts[3])

                # Get from FJSP solution
                if hasattr(solution, "schedule") and job_id in solution.schedule:
                    if op_idx in solution.schedule[job_id]:
                        op_data = solution.schedule[job_id][op_idx]
                        start = op_data.get("start", 0)
                        end = op_data.get("end", 0)
                        machine_id = op_data.get("machine", 0)

                        schedule_arr[task_idx, 0] = start
                        schedule_arr[task_idx, 1] = end
                        allocation_arr[task_idx] = machine_id

        return AllocSchedulingSolution(
            problem=target_problem,
            schedule=schedule_arr,
            allocation=allocation_arr,
        )
