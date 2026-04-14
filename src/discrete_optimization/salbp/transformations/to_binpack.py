#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from SALBP to BinPacking (reverse direction).

Note: This transformation DISCARDS precedence constraints!
Only use when precedence can be safely ignored or doesn't exist.
"""

from typing import Optional

from discrete_optimization.binpack.problem import (
    BinPackProblem,
    BinPackSolution,
    ItemBinPack,
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


class SalbpToBinpackTransformation(
    ProblemTransformation[SalbpProblem, SalbpSolution, BinPackProblem, BinPackSolution]
):
    """Transform SALBP to BinPacking (reverse direction).

    Mapping:
    - Tasks (with processing times) → Items (with weights)
    - Station cycle time → Bin capacity
    - Minimize stations → Minimize bins
    - **Precedence constraints DISCARDED**

    This transformation is INCOMPLETE IN BOTH DIRECTIONS:
    - Forward (problem): LOSSY - precedence constraints cannot be represented in BinPack
    - Backward (solution): LOSSY - incompatibility constraints from BinPack cannot be verified in SALBP

    Only use when:
        - SALBP has NO precedence constraints AND
        - BinPack has NO incompatibility constraints
        - In this case, both problems are equivalent (pure capacity allocation)

    Warning:
        This transformation LOSES precedence information!
        Solutions from BinPack solvers may violate precedence if present in SALBP.

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (SALBP → BinPack).

        This direction is LOSSY: precedence constraints cannot be represented.
        """
        losses = [
            InformationLoss(
                name="precedence_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Task precedence constraints (task i must come before task j)",
                reason="BinPacking has no concept of ordering or precedence between items",
                impact=LossImpact.MAJOR,
                workaround="Use SALBP→RCPSP transformation which preserves precedence constraints, "
                "or verify that precedence list is empty before transformation",
            )
        ]

        return lossy_transformation(
            losses=losses,
            assumptions=[
                "No task precedence constraints (or safe to ignore)",
                "Pure capacity allocation problem",
            ],
            use_cases=[
                "SALBP without precedence constraints (equivalent to bin packing)",
                "Testing BinPack solvers on SALBP instances without precedence",
            ],
            warnings=[
                "Solutions may violate precedence constraints if present in source problem",
                "Verify that source problem has no precedence before using this transformation",
            ],
        )

    def transform_problem(self, source_problem: SalbpProblem) -> BinPackProblem:
        """Transform SALBP to BinPacking.

        Args:
            source_problem: SALBP problem instance

        Returns:
            Equivalent BinPacking problem (without precedence)

        """
        # Map tasks to items
        list_items = [
            ItemBinPack(index=task, weight=float(source_problem.task_times[task]))
            for task in source_problem.tasks
        ]

        # Capacity = cycle time
        capacity_bin = int(source_problem.cycle_time)

        # Precedence constraints are LOST (documented in forward_metadata)
        # BinPacking has no precedence concept

        return BinPackProblem(
            list_items=list_items,
            capacity_bin=capacity_bin,
            incompatible_items=set(),  # No incompatibility in base SALBP
        )

    def back_transform_solution(
        self, solution: BinPackSolution, source_problem: SalbpProblem
    ) -> SalbpSolution:
        """Transform BinPacking solution back to SALBP solution.

        Args:
            solution: BinPacking solution
            source_problem: Original SALBP problem

        Returns:
            Equivalent SALBP solution

        """
        # Direct mapping: bin allocation → station allocation (EXACT)
        allocation_to_station = list(solution.allocation)

        return SalbpSolution(
            problem=source_problem,
            allocation_to_station=allocation_to_station,
        )

    def forward_transform_solution(
        self, solution: SalbpSolution, target_problem: BinPackProblem
    ) -> Optional[BinPackSolution]:
        """Transform SALBP solution to BinPacking solution (for warmstart).

        Args:
            solution: SALBP solution
            target_problem: Target BinPacking problem

        Returns:
            Equivalent BinPacking solution for warmstart

        """
        # Direct mapping: station allocation → bin allocation (EXACT)
        allocation = list(solution.allocation_to_station)

        return BinPackSolution(problem=target_problem, allocation=allocation)
