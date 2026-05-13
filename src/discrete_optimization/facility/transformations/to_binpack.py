#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Facility Location to BinPacking (reverse direction).

Note: This transformation DISCARDS facility setup costs and distances!
Assumes uniform facility costs (all equal).
"""

from typing import Optional

from discrete_optimization.binpack.problem import (
    BinPackProblem,
    BinPackSolution,
    ItemBinPack,
)
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


class FacilityToBinpackTransformation(
    ProblemTransformation[
        FacilityProblem, FacilitySolution, BinPackProblem, BinPackSolution
    ]
):
    """Transform Facility Location to BinPacking (reverse direction).

    Mapping:
    - Customers (with demands) → Items (with weights)
    - Facility capacity → Bin capacity
    - Minimize facilities → Minimize bins
    - **Setup costs and distances DISCARDED**

    This transformation is ASYMMETRIC:
    - Forward (problem): LOSSY - setup costs, assignment costs, and heterogeneous capacities lost
    - Backward (solution): EXACT - bin assignments map to facility assignments

    Assumptions:
        - All facilities have SAME capacity (uses first facility's capacity)
        - Setup costs IGNORED (assumed uniform)
        - Distances/allocation costs IGNORED

    Use case:
        - Facility location without distance costs = bin packing
        - Capacitated facility location simplified to bin packing
        - Testing different solvers

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Facility → BinPack).

        This direction is LOSSY: setup costs, assignment costs, and heterogeneous capacities lost.
        """
        losses = [
            InformationLoss(
                name="setup_costs",
                loss_type=LossType.OBJECTIVE,
                description="Facility setup costs (opening costs)",
                reason="BinPacking has no concept of setup/opening costs - all bins have implicit uniform cost",
                impact=LossImpact.MAJOR,
                workaround="Cannot be modeled in BinPacking. Use Facility solvers directly if setup costs matter.",
            ),
            InformationLoss(
                name="assignment_costs",
                loss_type=LossType.OBJECTIVE,
                description="Customer-to-facility assignment costs (distances)",
                reason="BinPacking has no concept of assignment costs - items have no location/distance",
                impact=LossImpact.MAJOR,
                workaround="Cannot be modeled in BinPacking. Use Facility solvers directly if distances matter.",
            ),
            InformationLoss(
                name="heterogeneous_capacities",
                loss_type=LossType.PARAMETER,
                description="Different facility capacities (if facilities have different capacities)",
                reason="BinPacking assumes uniform bin capacity. Transformation uses first facility's capacity.",
                impact=LossImpact.MODERATE,
                workaround="Verify all facilities have same capacity, or use multi-capacity bin packing variant.",
            ),
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "Setup costs are uniform or negligible",
                "Assignment costs (distances) are zero or negligible",
                "All facilities have same capacity (or only first facility capacity matters)",
            ],
            use_cases=[
                "Facility location problem without distance costs (pure capacity allocation)",
                "Benchmarking BinPack solvers on facility-like problems",
                "When only capacity constraints matter, not costs",
            ],
            warnings=[
                "Solutions may be suboptimal in original Facility problem due to ignored costs",
                "Verify that setup costs and assignment costs are truly negligible before using",
                "Check that all facilities have same capacity",
            ],
        )

    def transform_problem(self, source_problem: FacilityProblem) -> BinPackProblem:
        """Transform Facility Location to BinPacking.

        Args:
            source_problem: Facility Location problem instance

        Returns:
            Equivalent BinPacking problem (ignoring distances/costs)

        """
        # Map customers to items
        list_items = [
            ItemBinPack(index=customer.index, weight=customer.demand)
            for customer in source_problem.customers
        ]

        # Use first facility's capacity (assume all facilities have same capacity)
        # This is a simplification - facility location can have heterogeneous capacities
        capacity_bin = int(source_problem.facilities[0].capacity)

        return BinPackProblem(
            list_items=list_items,
            capacity_bin=capacity_bin,
            incompatible_items=set(),
        )

    def back_transform_solution(
        self, solution: BinPackSolution, source_problem: FacilityProblem
    ) -> FacilitySolution:
        """Transform BinPacking solution back to Facility Location solution.

        Args:
            solution: BinPacking solution
            source_problem: Original Facility Location problem

        Returns:
            Equivalent Facility Location solution

        """
        # Direct mapping: bin allocation → facility allocation
        facility_for_customers = list(solution.allocation)

        return FacilitySolution(
            problem=source_problem,
            facility_for_customers=facility_for_customers,
        )

    def forward_transform_solution(
        self, solution: FacilitySolution, target_problem: BinPackProblem
    ) -> Optional[BinPackSolution]:
        """Transform Facility solution to BinPacking solution (for warmstart).

        Args:
            solution: Facility Location solution
            target_problem: Target BinPacking problem

        Returns:
            Equivalent BinPacking solution for warmstart

        """
        # Direct mapping: facility allocation → bin allocation
        allocation = list(solution.facility_for_customers)

        return BinPackSolution(problem=target_problem, allocation=allocation)
