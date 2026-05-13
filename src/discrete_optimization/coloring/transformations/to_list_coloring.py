#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from ColoringProblem to ListColoringProblem."""

from typing import Optional

from discrete_optimization.coloring.list_coloring_problem import ListColoringProblem
from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)


class ColoringToListColoringTransformation(
    ProblemTransformation[
        ColoringProblem,
        ColoringSolution,
        ListColoringProblem,
        ColoringSolution,
    ]
):
    """Transform ColoringProblem to ListColoringProblem.

    Mapping:
    - Standard coloring → List coloring with all colors allowed for all nodes
    - Graph structure preserved
    - Solution space identical (all colors available everywhere)

    This transformation is EXACT in both directions.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (Coloring → ListColoring).

        This direction is EXACT: standard coloring is list coloring with unrestricted lists.
        """
        return exact_transformation(
            use_cases=[
                "Use ListColoring solvers that may have better constraint propagation",
                "Prepare for adding color restrictions later",
                "Standard coloring is list coloring with all colors allowed",
            ]
        )

    def transform_problem(self, source_problem: ColoringProblem) -> ListColoringProblem:
        """Transform ColoringProblem to ListColoringProblem.

        Args:
            source_problem: ColoringProblem instance

        Returns:
            Equivalent ListColoringProblem with all colors allowed

        """
        # All nodes can use any color (unrestricted)
        allowed_colors = {
            node: set(range(source_problem.number_of_nodes))
            for node in source_problem.nodes_name
        }

        return ListColoringProblem(
            graph=source_problem.graph,
            allowed_colors=allowed_colors,
            subset_nodes=source_problem.subset_nodes,
            constraints_coloring=source_problem.constraints_coloring,
        )

    def back_transform_solution(
        self, solution: ColoringSolution, source_problem: ColoringProblem
    ) -> ColoringSolution:
        """Transform ListColoring solution back to Coloring solution.

        Args:
            solution: ListColoring solution
            source_problem: Original ColoringProblem

        Returns:
            Equivalent Coloring solution (same representation)

        """
        # Solutions are identical (both use ColoringSolution)
        # Just change the problem reference
        return ColoringSolution(
            problem=source_problem,
            colors=solution.colors,
            nb_color=solution.nb_color,
            nb_violations=solution.nb_violations,
        )

    def forward_transform_solution(
        self, solution: ColoringSolution, target_problem: ListColoringProblem
    ) -> Optional[ColoringSolution]:
        """Transform Coloring solution to ListColoring solution (for warmstart).

        Args:
            solution: Coloring solution
            target_problem: Target ListColoringProblem

        Returns:
            Equivalent ListColoring solution

        """
        # Solutions are identical (both use ColoringSolution)
        # Just change the problem reference
        return ColoringSolution(
            problem=target_problem,
            colors=solution.colors,
            nb_color=solution.nb_color,
            nb_violations=solution.nb_violations,
        )
