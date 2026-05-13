#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from Workforce Allocation to Graph Coloring (composed transformation).

This module implements WorkforceAllocationToColoringTransformation as a composition of:
1. WorkforceAllocationToListColoringTransformation (direct encoding)
2. ListColoringToColoringTransformation (dummy nodes encoding)

This avoids code duplication and leverages both transformation implementations.
"""

from typing import Optional

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.transformations import (
    ListColoringToColoringTransformation,
)
from discrete_optimization.generic_tools.transformation.composite import (
    chain_transformations,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
)
from discrete_optimization.workforce.allocation.problem import (
    TeamAllocationProblem,
    TeamAllocationSolution,
)
from discrete_optimization.workforce.allocation.transformations.to_list_coloring import (
    WorkforceAllocationToListColoringTransformation,
)


class WorkforceAllocationToColoringTransformation(
    ProblemTransformation[
        TeamAllocationProblem,
        TeamAllocationSolution,
        ColoringProblem,
        ColoringSolution,
    ]
):
    """Transform Workforce Allocation to Graph Coloring (composed transformation).

    This transformation is a COMPOSITION of:
    1. WorkforceAllocationToListColoringTransformation (direct encoding)
    2. ListColoringToColoringTransformation (dummy nodes encoding)

    Mapping (via composition):
    - Allocation → ListColoring: Tasks → Nodes, Teams → Colors, allowed teams → allowed_colors
    - ListColoring → Coloring: Add dummy team nodes, edges for forbidden colors

    Final encoding:
    - Tasks → Original nodes in coloring graph
    - Teams → Dummy nodes (one per team), forming a complete clique
    - Dummy node k fixed to color k
    - Forbidden allocations → Edges to dummy team nodes
    - subset_nodes focuses optimization on tasks only

    This is LOSSY (calendar & same_allocation constraints lost) but EXACT for supported constraints.
    """

    def __init__(self):
        """Initialize composed transformation."""
        # Create composite transformation: Allocation → ListColoring → Coloring
        self._composite = chain_transformations(
            WorkforceAllocationToListColoringTransformation(),
            ListColoringToColoringTransformation(),
        )

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Allocation → Coloring).

        This is the composition of two transformations, inheriting losses from step1.
        """
        # Get metadata from first transformation (the lossy part)
        return self._composite.transformations[0].get_forward_metadata()

    def transform_problem(
        self, source_problem: TeamAllocationProblem
    ) -> ColoringProblem:
        """Transform Workforce Allocation to Coloring via composition.

        Args:
            source_problem: TeamAllocationProblem instance

        Returns:
            ColoringProblem with dummy nodes

        """
        return self._composite.transform_problem(source_problem)

    def back_transform_solution(
        self, solution: ColoringSolution, source_problem: TeamAllocationProblem
    ) -> TeamAllocationSolution:
        """Transform Coloring solution back to Workforce Allocation solution.

        Args:
            solution: Coloring solution (with dummy nodes)
            source_problem: Original TeamAllocationProblem

        Returns:
            Equivalent TeamAllocationSolution

        """
        return self._composite.back_transform_solution(solution, source_problem)

    def forward_transform_solution(
        self, solution: TeamAllocationSolution, target_problem: ColoringProblem
    ) -> Optional[ColoringSolution]:
        """Transform Allocation solution to Coloring solution (for warmstart).

        Args:
            solution: TeamAllocationSolution
            target_problem: Target Coloring problem (with dummy nodes)

        Returns:
            Equivalent ColoringSolution

        """
        return self._composite.forward_transform_solution(solution, target_problem)
