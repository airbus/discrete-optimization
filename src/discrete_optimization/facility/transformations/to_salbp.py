#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Facility Location to SALBP.

Note: This transformation DISCARDS facility setup costs and distances!
Assumes uniform facility costs and no precedence.
"""

from typing import Optional

from discrete_optimization.alb.base.problem import TaskData
from discrete_optimization.alb.salbp.problem import SalbpProblem, SalbpSolution
from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
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


class FacilityToSalbpTransformation(
    ProblemTransformation[
        FacilityProblem, FacilitySolution, SalbpProblem, SalbpSolution
    ]
):
    """Transform Facility Location to SALBP.

    Mapping:
    - Customers (with demands) → Tasks (with processing times)
    - Facility capacity → Cycle time
    - Facilities → Stations
    - Minimize facilities → Minimize stations
    - **Setup costs and distances DISCARDED**
    - **No precedence constraints** (Facility has none)

    This transformation is ASYMMETRIC:
    - Forward (problem): LOSSY - setup costs and assignment costs lost
    - Backward (solution): EXACT - station assignments map to facility assignments

    Assumptions:
        - All facilities have SAME capacity (uses first facility's capacity)
        - Setup costs IGNORED
        - Distances/allocation costs IGNORED

    Use case:
        - Facility location viewed as line balancing problem
        - Capacitated facility location simplified to SALBP

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Facility → SALBP).

        This direction is LOSSY: setup costs and assignment costs lost.
        """
        losses = [
            InformationLoss(
                name="setup_costs",
                loss_type=LossType.OBJECTIVE,
                description="Facility setup costs (opening costs)",
                reason="SALBP has no concept of setup/opening costs - all stations have implicit uniform cost",
                impact=LossImpact.MAJOR,
                workaround="Cannot be modeled in SALBP. Use Facility solvers directly if setup costs matter.",
            ),
            InformationLoss(
                name="assignment_costs",
                loss_type=LossType.OBJECTIVE,
                description="Customer-to-facility assignment costs (distances)",
                reason="SALBP has no concept of assignment costs - tasks have no location/distance",
                impact=LossImpact.MAJOR,
                workaround="Cannot be modeled in SALBP. Use Facility solvers directly if distances matter.",
            ),
            InformationLoss(
                name="heterogeneous_capacities",
                loss_type=LossType.PARAMETER,
                description="Different facility capacities (if facilities have different capacities)",
                reason="SALBP assumes uniform cycle time. Transformation uses first facility's capacity.",
                impact=LossImpact.MODERATE,
                workaround="Verify all facilities have same capacity.",
            ),
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "Setup costs are uniform or negligible",
                "Assignment costs (distances) are zero or negligible",
                "All facilities have same capacity",
            ],
            use_cases=[
                "Facility location problem without distance costs",
                "When only capacity allocation matters",
            ],
            warnings=[
                "Solutions may be suboptimal in original Facility problem due to ignored costs",
                "Verify that setup costs and assignment costs are truly negligible",
            ],
        )

    def transform_problem(self, source_problem: FacilityProblem) -> SalbpProblem:
        """Transform Facility Location to SALBP.

        Args:
            source_problem: Facility Location problem instance

        Returns:
            Equivalent SALBP problem (without distances/costs)

        """
        # Map customers to tasks with TaskData
        tasks_data = [
            TaskData(task_id=customer.index, processing_time=int(customer.demand))
            for customer in source_problem.customers
        ]

        # Use first facility's capacity as cycle time
        cycle_time = int(source_problem.facilities[0].capacity)

        # No precedence constraints (Facility Location has none)
        precedences = []

        return SalbpProblem(
            tasks_data=tasks_data,
            cycle_time=cycle_time,
            precedences=precedences,
        )

    def back_transform_solution(
        self, solution: SalbpSolution, source_problem: FacilityProblem
    ) -> FacilitySolution:
        """Transform SALBP solution back to Facility Location solution.

        Args:
            solution: SALBP solution
            source_problem: Original Facility Location problem

        Returns:
            Equivalent Facility Location solution

        """
        # Direct mapping: station allocation → facility allocation
        facility_for_customers = list(solution.allocation_to_station)

        return FacilitySolution(
            problem=source_problem,
            facility_for_customers=facility_for_customers,
        )

    def forward_transform_solution(
        self, solution: FacilitySolution, target_problem: SalbpProblem
    ) -> Optional[SalbpSolution]:
        """Transform Facility solution to SALBP solution (for warmstart).

        Args:
            solution: Facility Location solution
            target_problem: Target SALBP problem

        Returns:
            Equivalent SALBP solution for warmstart

        """
        # Direct mapping: facility allocation → station allocation
        allocation_to_station = list(solution.facility_for_customers)

        return SalbpSolution(
            problem=target_problem,
            allocation_to_station=allocation_to_station,
        )
