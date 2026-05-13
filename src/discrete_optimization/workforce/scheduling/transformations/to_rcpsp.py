#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Workforce Scheduling to RCPSP."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    transform_alloc_solution_to_rcpsp_solution,
    transform_rcpsp_solution_to_alloc_solution,
    transform_to_multimode_rcpsp,
)


class WorkforceSchedulingToRcpspTransformation(
    ProblemTransformation[
        AllocSchedulingProblem,
        AllocSchedulingSolution,
        RcpspProblem,
        RcpspSolution,
    ]
):
    """Transform Workforce Scheduling to RCPSP (Resource-Constrained Project Scheduling).

    Mapping:
    - Tasks → RCPSP tasks
    - Teams → Modes for each task
    - Team availability calendars → Resource calendars
    - Task duration → Task duration in each mode
    - Precedence constraints → RCPSP successors
    - Time windows → Start/end time windows

    This transformation is EXACT:
    - All workforce scheduling constraints are preserved in RCPSP formulation
    - Team assignment becomes mode selection in RCPSP
    """

    def __init__(
        self,
        build_calendar: bool = True,
        add_window_time_constraint: bool = True,
        add_additional_constraint: bool = True,
    ):
        """Initialize transformation.

        Args:
            build_calendar: Build resource calendars from team availability
            add_window_time_constraint: Add time window constraints to RCPSP
            add_additional_constraint: Add special constraints (pair mode constraints)

        """
        self.build_calendar = build_calendar
        self.add_window_time_constraint = add_window_time_constraint
        self.add_additional_constraint = add_additional_constraint
        # Will store the mode-to-team mapping after transformation
        self.ac_mode_to_team: dict = {}

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (WorkforceScheduling → RCPSP).

        This direction is EXACT: all workforce constraints map to RCPSP constraints.
        """
        return exact_transformation(
            use_cases=[
                "Use RCPSP solvers to solve workforce scheduling problems",
                "Team assignment becomes mode selection in RCPSP",
                "All precedence and time window constraints preserved",
            ]
        )

    def transform_problem(self, source_problem: AllocSchedulingProblem) -> RcpspProblem:
        """Transform Workforce Scheduling to RCPSP.

        Args:
            source_problem: AllocSchedulingProblem instance

        Returns:
            Equivalent RCPSP problem

        """
        # Use the existing transformation function
        rcpsp_problem, ac_mode_to_team = transform_to_multimode_rcpsp(
            problem=source_problem,
            build_calendar=self.build_calendar,
            add_window_time_constraint=self.add_window_time_constraint,
            add_additional_constraint=self.add_additional_constraint,
        )

        # Store the mapping for solution back-transformation
        self.ac_mode_to_team = ac_mode_to_team

        return rcpsp_problem

    def back_transform_solution(
        self, solution: RcpspSolution, source_problem: AllocSchedulingProblem
    ) -> AllocSchedulingSolution:
        """Transform RCPSP solution back to Workforce Scheduling solution.

        Args:
            solution: RCPSP solution
            source_problem: Original AllocSchedulingProblem

        Returns:
            Equivalent AllocSchedulingSolution

        """
        # Use the existing solution transformation function
        return transform_rcpsp_solution_to_alloc_solution(
            rcpsp_solution=solution,
            rcpsp_problem=solution.problem,
            ac_mode_to_team=self.ac_mode_to_team,
            alloc_scheduling_problem=source_problem,
        )

    def forward_transform_solution(
        self, solution: AllocSchedulingSolution, target_problem: RcpspProblem
    ) -> Optional[RcpspSolution]:
        """Transform Workforce Scheduling solution to RCPSP solution (for warmstart).

        Args:
            solution: AllocSchedulingSolution
            target_problem: Target RCPSP problem

        Returns:
            Equivalent RCPSP solution for warmstart

        """
        # Use the existing solution transformation function
        return transform_alloc_solution_to_rcpsp_solution(
            alloc_solution=solution,
            rcpsp_problem=target_problem,
            ac_mode_to_team=self.ac_mode_to_team,
            alloc_scheduling_problem=solution.problem,
        )
