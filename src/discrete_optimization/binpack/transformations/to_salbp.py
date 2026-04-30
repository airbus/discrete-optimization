#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from BinPack to SALBP (Assembly Line Balancing)."""

from typing import Optional

from discrete_optimization.alb.base.problem import TaskData
from discrete_optimization.alb.salbp import SalbpProblem, SalbpSolution
from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
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


class BinpackToSalbpTransformation(
    ProblemTransformation[BinPackProblem, BinPackSolution, SalbpProblem, SalbpSolution]
):
    """Transform BinPack problem to SALBP (Assembly Line Balancing) problem.

    Mapping:
    - Items (with weights) → Tasks (with processing times)
    - Bin capacity → Station cycle time
    - Minimize bins → Minimize stations
    - No precedence constraints (empty list)

    This transformation is INCOMPLETE IN BOTH DIRECTIONS:
    - Forward (problem): LOSSY - incompatibility constraints cannot be represented in SALBP
    - Backward (solution): LOSSY - precedence constraints from SALBP cannot be verified in BinPack

    Only use when:
        - BinPack has NO incompatibility constraints AND
        - SALBP has NO precedence constraints
        - In this case, both problems are equivalent (pure capacity allocation)

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → SALBP).

        This direction is LOSSY: incompatibility constraints cannot be represented.
        """
        losses = [
            InformationLoss(
                name="incompatibility_constraints",
                loss_type=LossType.CONSTRAINT,
                description="Item incompatibility constraints (items that cannot be in same bin)",
                reason="SALBP has no concept of task incompatibility or conflict constraints",
                impact=LossImpact.MAJOR,
                workaround="Use BinPack→RCPSP transformation which models incompatibility with virtual resources, "
                "or pre-filter incompatible items before transformation",
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
                "Benchmarking SALBP solvers on packing problems",
                "Exploring assembly line balancing formulation",
            ],
            warnings=[
                "Solutions may violate incompatibility constraints if present in source problem",
                "Verify solution feasibility in original BinPack problem after solving",
            ],
        )

    def transform_problem(self, source_problem: BinPackProblem) -> SalbpProblem:
        """Transform BinPack to SALBP.

        Args:
            source_problem: BinPack problem instance

        Returns:
            Equivalent SALBP problem

        """
        # Map items to tasks with processing times = weights
        cycle_time = int(source_problem.capacity_bin)

        # Create TaskData for each item
        tasks_data = [
            TaskData(task_id=item.index, processing_time=int(item.weight))
            for item in source_problem.list_items
        ]

        # No precedence constraints (bin packing has no ordering constraints)
        # Incompatibility constraints are LOST (documented in forward_metadata)
        precedences = []

        return SalbpProblem(
            tasks_data=tasks_data,
            cycle_time=cycle_time,
            precedences=precedences,
        )

    def back_transform_solution(
        self, solution: SalbpSolution, source_problem: BinPackProblem
    ) -> BinPackSolution:
        """Transform SALBP solution back to BinPack solution.

        Args:
            solution: SALBP solution (allocation_to_station)
            source_problem: Original BinPack problem

        Returns:
            Equivalent BinPack solution

        """
        # Direct mapping: station allocation → bin allocation (EXACT)
        # allocation_to_station[i] gives the station for task i
        # This directly maps to allocation[i] = bin for item i
        allocation = list(solution.allocation_to_station)

        return BinPackSolution(problem=source_problem, allocation=allocation)

    def forward_transform_solution(
        self, solution: BinPackSolution, target_problem: SalbpProblem
    ) -> Optional[SalbpSolution]:
        """Transform BinPack solution to SALBP solution (for warmstart).

        Args:
            solution: BinPack solution
            target_problem: Target SALBP problem

        Returns:
            Equivalent SALBP solution for warmstart

        """
        # Direct mapping: bin allocation → station allocation (EXACT)
        allocation_to_station = list(solution.allocation)

        return SalbpSolution(
            problem=target_problem, allocation_to_station=allocation_to_station
        )
