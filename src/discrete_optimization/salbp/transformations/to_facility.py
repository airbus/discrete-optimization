#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from SALBP to Facility Location.

Creates a facility problem with uniform facilities and zero distances.
"""

from typing import Optional

from discrete_optimization.facility.problem import (
    Customer,
    Facility,
    FacilityProblem,
    FacilitySolution,
    Point,
)
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
from discrete_optimization.salbp.problem import SalbpProblem, SalbpSolution


class SalbpToFacilityTransformation(
    ProblemTransformation[
        SalbpProblem, SalbpSolution, FacilityProblem, FacilitySolution
    ]
):
    """Transform SALBP to Facility Location.

    Mapping:
    - Tasks → Customers (demand = task_time)
    - Stations → Facilities (capacity = cycle_time)
    - Minimize stations → Minimize facilities
    - Precedence DISCARDED (Facility has no precedence concept)

    This transformation is ASYMMETRIC:
    - Forward (problem): LOSSY - precedence constraints lost
    - Backward (solution): EXACT - facility assignments map to station assignments

    Creates:
        - Uniform facilities (all same capacity = cycle_time, zero setup cost)
        - Zero distances between customers and facilities
        - This makes it equivalent to pure capacity allocation

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (SALBP → Facility).

        This direction is LOSSY: precedence constraints cannot be represented.
        """
        losses = [
            InformationLoss(
                name="precedence_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Task precedence constraints (task i must come before task j)",
                reason="Facility Location has no concept of ordering or precedence between customers",
                impact=LossImpact.MAJOR,
                workaround="Use SALBP→RCPSP transformation which preserves precedence constraints",
            )
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "No task precedence constraints (or safe to ignore)",
                "Pure capacity allocation problem",
            ],
            use_cases=[
                "SALBP without precedence constraints (equivalent to bin packing / facility location)",
                "Benchmarking Facility solvers on balancing problems without precedence",
            ],
            warnings=[
                "Solutions may violate precedence constraints if present in source problem",
                "Verify that source problem has no precedence before using this transformation",
            ],
        )

    def transform_problem(self, source_problem: SalbpProblem) -> FacilityProblem:
        """Transform SALBP to Facility Location.

        Args:
            source_problem: SALBP problem instance

        Returns:
            Equivalent Facility Location problem

        """
        # Create customers from tasks
        customers = [
            Customer(
                index=task,
                demand=float(source_problem.task_times[task]),
                location=Point(x=0.0, y=0.0),  # Dummy location
            )
            for task in source_problem.tasks
        ]

        # Create facilities (one per possible station)
        # Maximum stations = number of tasks (worst case)
        max_facilities = source_problem.number_of_tasks
        facilities = [
            Facility(
                index=i,
                setup_cost=0.0,  # Uniform cost (minimize count)
                capacity=float(source_problem.cycle_time),
                location=Point(x=0.0, y=0.0),  # Dummy location
            )
            for i in range(max_facilities)
        ]

        # Create a concrete FacilityProblem subclass
        # We need to implement evaluate_customer_facility
        class SalbpDerivedFacilityProblem(FacilityProblem):
            def evaluate_customer_facility(
                self, facility: Facility, customer: Customer
            ) -> float:
                # Zero distance (pure capacity allocation)
                return 0.0

        return SalbpDerivedFacilityProblem(
            facility_count=len(facilities),
            customer_count=len(customers),
            facilities=facilities,
            customers=customers,
        )

    def back_transform_solution(
        self, solution: FacilitySolution, source_problem: SalbpProblem
    ) -> SalbpSolution:
        """Transform Facility Location solution back to SALBP solution.

        Args:
            solution: Facility Location solution
            source_problem: Original SALBP problem

        Returns:
            Equivalent SALBP solution

        """
        # Direct mapping: facility allocation → station allocation
        allocation_to_station = list(solution.facility_for_customers)

        return SalbpSolution(
            problem=source_problem,
            allocation_to_station=allocation_to_station,
        )

    def forward_transform_solution(
        self, solution: SalbpSolution, target_problem: FacilityProblem
    ) -> Optional[FacilitySolution]:
        """Transform SALBP solution to Facility Location solution (for warmstart).

        Args:
            solution: SALBP solution
            target_problem: Target Facility Location problem

        Returns:
            Equivalent Facility Location solution for warmstart

        """
        # Direct mapping: station allocation → facility allocation
        facility_for_customers = list(solution.allocation_to_station)

        return FacilitySolution(
            problem=target_problem,
            facility_for_customers=facility_for_customers,
        )
