#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from RCPSP to Preemptive RCPSP."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_preemptive import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
    get_rcpsp_problemp_preemptive,
)
from discrete_optimization.rcpsp.solution import RcpspSolution


class RcpspToPreemptiveTransformation(
    ProblemTransformation[
        RcpspProblem,
        RcpspSolution,
        PreemptiveRcpspProblem,
        PreemptiveRcpspSolution,
    ]
):
    """Transform RCPSP to Preemptive RCPSP.

    Mapping:
    - All tasks → Preemptive tasks (can be interrupted)
    - All resources and constraints preserved
    - Same mode details and precedence
    - All tasks marked as preemptive

    RCPSP is a special case of Preemptive RCPSP where tasks cannot be preempted.
    The reverse is also true when solutions don't use preemption.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (RCPSP → Preemptive RCPSP).

        This direction is EXACT: RCPSP is a subset of Preemptive RCPSP.
        """
        return exact_transformation(
            use_cases=[
                "Use Preemptive RCPSP solvers to solve RCPSP problems",
                "RCPSP is exactly Preemptive RCPSP where tasks are not preempted",
                "Preemption can lead to better schedules by allowing interruptions",
            ]
        )

    def transform_problem(self, source_problem: RcpspProblem) -> PreemptiveRcpspProblem:
        """Transform RCPSP to Preemptive RCPSP.

        Args:
            source_problem: RCPSP problem instance

        Returns:
            Equivalent Preemptive RCPSP problem

        """
        # Use existing transformation function
        return get_rcpsp_problemp_preemptive(source_problem)

    def back_transform_solution(
        self, solution: PreemptiveRcpspSolution, source_problem: RcpspProblem
    ) -> RcpspSolution:
        """Transform Preemptive RCPSP solution back to RCPSP solution.

        Args:
            solution: Preemptive RCPSP solution
            source_problem: Original RCPSP problem

        Returns:
            Equivalent RCPSP solution

        Note:
            If the preemptive solution uses preemption (tasks with multiple parts),
            the RCPSP solution will use the first start time and last end time,
            potentially with idle time in between. This may not be feasible
            in the non-preemptive problem if resources are occupied during idle periods.

        """
        # Build RCPSP schedule from preemptive schedule
        # For each task, take first start and last end
        rcpsp_schedule = {}

        for task, task_details in solution.rcpsp_schedule.items():
            if "starts" in task_details:
                # Preemptive format with multiple parts
                starts_list = task_details["starts"]
                ends_list = task_details["ends"]

                if starts_list and ends_list:
                    # Use first start and last end
                    start_time = starts_list[0]
                    end_time = ends_list[-1]
                else:
                    start_time = 0
                    end_time = 0
            else:
                # Already in simple format
                start_time = task_details.get("start_time", 0)
                end_time = task_details.get("end_time", 0)

            rcpsp_schedule[task] = {"start_time": start_time, "end_time": end_time}

        return RcpspSolution(
            problem=source_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=solution.rcpsp_modes,
        )

    def forward_transform_solution(
        self, solution: RcpspSolution, target_problem: PreemptiveRcpspProblem
    ) -> Optional[PreemptiveRcpspSolution]:
        """Transform RCPSP solution to Preemptive RCPSP solution (for warmstart).

        Args:
            solution: RCPSP solution
            target_problem: Target Preemptive RCPSP problem

        Returns:
            Equivalent Preemptive RCPSP solution (without preemption)

        """
        # Build preemptive schedule from RCPSP schedule
        # Each task will have a single continuous part (no preemption)
        preemptive_schedule = {}

        for task, task_details in solution.rcpsp_schedule.items():
            start_time = task_details["start_time"]
            end_time = task_details["end_time"]

            # Single continuous part (no preemption)
            preemptive_schedule[task] = {
                "starts": [start_time],
                "ends": [end_time],
            }

        return PreemptiveRcpspSolution(
            problem=target_problem,
            rcpsp_schedule=preemptive_schedule,
            rcpsp_modes=solution.rcpsp_modes,
        )
