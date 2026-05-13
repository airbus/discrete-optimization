#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from RCPSP to RCPSP Multiskill."""

from __future__ import annotations

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
)


class RcpspToMultiskillTransformation(
    ProblemTransformation[
        RcpspProblem, RcpspSolution, MultiskillRcpspProblem, MultiskillRcpspSolution
    ]
):
    """Transform RCPSP to RCPSPMultiskill.

    Mapping strategy:
    - RCPSP resources map directly to MultiskillRCPSP cumulative resources
    - No employees or skills needed (MultiskillRCPSP can work with just resources)
    - Same mode details and precedence constraints

    Example:
        # >>> problem = RcpspProblem(
        # ...     resources={"R1": 5, "R2": 2},
        # ...     ...
        # ... )
        # >>> transformation = RcpspToMultiskillTransformation()
        # >>> ms_problem = transformation.transform_problem(problem)
        # >>> # ms_problem uses resources R1, R2 directly (no fake employees)

    """

    def transform_problem(self, source_problem: RcpspProblem) -> MultiskillRcpspProblem:
        """Convert RCPSP to Multiskill RCPSP.

        Args:
            source_problem: Original RCPSP problem

        Returns:
            Equivalent MultiskillRcpspProblem

        """
        rcpsp = source_problem

        # Use resources directly (no skills/employees needed)
        resources_set = set(rcpsp.resources_list)
        skills_set = set()  # Empty skills
        employees = {}  # No employees

        # Resources availability (convert int to list if needed)
        resources_availability = {}
        for r in rcpsp.resources_list:
            if isinstance(rcpsp.resources[r], int):
                resources_availability[r] = [rcpsp.resources[r]] * rcpsp.horizon
            else:
                resources_availability[r] = list(rcpsp.resources[r])

        return MultiskillRcpspProblem(
            skills_set=skills_set,  # Empty
            resources_set=resources_set,
            non_renewable_resources=set(rcpsp.non_renewable_resources),
            resources_availability=resources_availability,
            employees=employees,  # Empty
            mode_details=rcpsp.mode_details,  # Same modes
            successors=rcpsp.successors,  # Same precedence
            horizon=rcpsp.horizon,
            source_task=rcpsp.source_task,
            sink_task=rcpsp.sink_task,
        )

    def back_transform_solution(
        self, solution: MultiskillRcpspSolution, source_problem: RcpspProblem
    ) -> RcpspSolution:
        """Convert Multiskill solution back to RCPSP solution.

        We only need the schedule (start times) and modes.
        Employee assignments are not relevant (empty in this transformation).

        Args:
            solution: Solution in multiskill problem space
            source_problem: Original RCPSP problem

        Returns:
            Corresponding RCPSP solution

        """
        # Extract schedule (convert to RCPSP format if needed)
        rcpsp_schedule = {}
        for task, details in solution.schedule.items():
            if isinstance(details, dict):
                # Already in correct format
                rcpsp_schedule[task] = {
                    "start_time": details["start_time"],
                    "end_time": details["end_time"],
                }
            else:
                # Handle other formats if necessary
                rcpsp_schedule[task] = details

        # Extract modes (convert dict to list in correct order)
        rcpsp_modes = [
            solution.modes[task] for task in source_problem.tasks_list_non_dummy
        ]

        return RcpspSolution(
            problem=source_problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=rcpsp_modes,
        )

    def forward_transform_solution(
        self, solution: RcpspSolution, target_problem: MultiskillRcpspProblem
    ) -> MultiskillRcpspSolution:
        """Convert RCPSP solution to Multiskill solution.

        Since we don't use employees, this is straightforward.

        Args:
            solution: Solution in RCPSP problem space
            target_problem: Transformed multiskill problem

        Returns:
            Corresponding multiskill solution (without employee assignments)

        """
        # Extract schedule and modes from RCPSP solution
        schedule = solution.rcpsp_schedule
        modes = {
            task: solution.rcpsp_modes[target_problem.index_task_non_dummy[task]]
            for task in target_problem.tasks_list_non_dummy
        }

        # Add source and sink with their modes
        modes[target_problem.source_task] = 1
        modes[target_problem.sink_task] = 1

        # No employee assignments needed
        employee_usage = {}

        return MultiskillRcpspSolution(
            problem=target_problem,
            modes=modes,
            schedule=schedule,
            employee_usage=employee_usage,
        )
