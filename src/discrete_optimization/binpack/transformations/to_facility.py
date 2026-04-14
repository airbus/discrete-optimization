#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from BinPack to Facility Location."""

from typing import Optional

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
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


class BinpackToFacilityTransformation(
    ProblemTransformation[
        BinPackProblem, BinPackSolution, FacilityProblem, FacilitySolution
    ]
):
    """Transform BinPack problem to Facility Location problem.

    Mapping:
    - Items (with weights) → Customers (with demands)
    - Bins (with capacity) → Facilities (with capacity)
    - Opening a bin → Setup cost for facility (cost = 1)
    - Minimize bins → Minimize setup costs
    - No spatial component (all locations at origin)
    - No assignment costs (distance-based cost = 0)

    This transformation is ASYMMETRIC:
    - Forward (problem): LOSSY - incompatibility constraints lost
    - Backward (solution): EXACT - facility assignments map to bins
    """

    def __init__(self, setup_cost_per_bin: float = 1.0):
        """Initialize transformation.

        Args:
            setup_cost_per_bin: Cost for opening each bin/facility (default: 1.0)

        """
        self.setup_cost_per_bin = setup_cost_per_bin

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → Facility).

        This direction is LOSSY: incompatibility constraints cannot be represented.
        """
        losses = [
            InformationLoss(
                name="incompatibility_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Item incompatibility constraints (items that cannot be in same bin)",
                reason="Facility Location has no concept of customer incompatibility or conflict constraints",
                impact=LossImpact.MAJOR,
                workaround="Use BinPack→RCPSP transformation which models incompatibility with virtual resources",
            )
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "No item incompatibility constraints (or safe to ignore)",
                "Pure capacity allocation problem",
            ],
            use_cases=[
                "Bin packing without incompatibility constraints",
                "Benchmarking Facility solvers on packing problems",
            ],
            warnings=[
                "Solutions may violate incompatibility constraints if present in source problem",
                "Verify solution feasibility in original BinPack problem after solving",
            ],
        )

    def transform_problem(self, source_problem: BinPackProblem) -> FacilityProblem:
        """Transform BinPack to Facility Location.

        Args:
            source_problem: BinPack problem instance

        Returns:
            Equivalent Facility Location problem

        """
        # Create customers from items
        customers = [
            Customer(
                index=item.index,
                demand=float(item.weight),
                location=Point(x=0.0, y=0.0),  # No spatial component
            )
            for item in source_problem.list_items
        ]

        # Create potential facilities (one per potential bin)
        # Upper bound: one facility per item (worst case)
        max_facilities = source_problem.nb_items
        facilities = [
            Facility(
                index=i,
                setup_cost=self.setup_cost_per_bin,
                capacity=float(source_problem.capacity_bin),
                location=Point(x=0.0, y=0.0),  # All at origin
            )
            for i in range(max_facilities)
        ]

        # Create a concrete subclass since FacilityProblem is abstract
        class BinPackAsFacilityProblem(FacilityProblem):
            """Facility problem for bin packing transformation."""

            def evaluate_customer_facility(
                self, facility: Facility, customer: Customer
            ) -> float:
                """No assignment cost (distance = 0 for bin packing).

                Args:
                    facility: Facility
                    customer: Customer

                Returns:
                    Cost of 0 (no spatial component in bin packing)

                """
                return 0.0  # No distance cost in bin packing

        return BinPackAsFacilityProblem(
            facility_count=len(facilities),
            customer_count=len(customers),
            facilities=facilities,
            customers=customers,
        )

    def back_transform_solution(
        self, solution: FacilitySolution, source_problem: BinPackProblem
    ) -> BinPackSolution:
        """Transform Facility solution back to BinPack solution.

        Args:
            solution: Facility Location solution
            source_problem: Original BinPack problem

        Returns:
            Equivalent BinPack solution

        """
        # Direct mapping: facility_for_customers → allocation
        # solution.facility_for_customers[i] gives facility index for customer i
        allocation = list(solution.facility_for_customers)

        return BinPackSolution(problem=source_problem, allocation=allocation)

    def forward_transform_solution(
        self, solution: BinPackSolution, target_problem: FacilityProblem
    ) -> Optional[FacilitySolution]:
        """Transform BinPack solution to Facility solution (for warmstart).

        Args:
            solution: BinPack solution
            target_problem: Target Facility problem

        Returns:
            Equivalent Facility solution for warmstart

        """
        # Direct mapping: allocation → facility_for_customers
        facility_for_customers = list(solution.allocation)

        return FacilitySolution(
            problem=target_problem, facility_for_customers=facility_for_customers
        )
