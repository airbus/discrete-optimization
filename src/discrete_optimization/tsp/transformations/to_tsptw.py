#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from TSP to TSPTW.

TSP is a special case of TSPTW with relaxed (infinite) time windows.
"""

from typing import Optional

import numpy as np

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class TspToTsptwTransformation(
    ProblemTransformation[TspProblem, TspSolution, TSPTWProblem, TSPTWSolution]
):
    """Transform TSP to TSPTW.

    Mapping:
    - Tour → Tour with relaxed time windows
    - No time constraints → Wide time windows [0, horizon]
    - Depot → Depot with wide time window
    - Distance matrix → Distance matrix (preserved)

    This transformation is EXACT:
    - TSP is exactly TSPTW with relaxed (unconstrained) time windows
    - All TSP constraints are preserved
    - Solution quality is identical

    """

    def __init__(self, horizon: Optional[int] = None):
        """Initialize transformation.

        Args:
            horizon: Maximum time for time windows.
                    If None, computed as sum of all distances (very conservative).

        """
        self.horizon = horizon

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (TSP → TSPTW).

        This direction is EXACT: TSP is exactly TSPTW with wide time windows.
        """
        return exact_transformation(
            use_cases=[
                "Use TSPTW solvers to solve TSP problems",
                "TSP is exactly TSPTW with relaxed time windows [0, horizon]",
                "All TSP constraints preserved in TSPTW",
                "No time window constraints in TSP → wide windows in TSPTW",
            ]
        )

    def transform_problem(self, source_problem: TspProblem) -> TSPTWProblem:
        """Transform TSP to TSPTW.

        Args:
            source_problem: TSP problem instance

        Returns:
            Equivalent TSPTW problem with relaxed time windows

        """
        # Build distance matrix (TSP uses evaluate_function_indexes)
        nb_nodes = source_problem.node_count
        distance_matrix = np.zeros((nb_nodes, nb_nodes))

        for i in range(nb_nodes):
            for j in range(nb_nodes):
                if i != j:
                    distance_matrix[i, j] = source_problem.evaluate_function_indexes(
                        i, j
                    )

        # Determine horizon if not provided
        if self.horizon is None:
            # Very conservative: sum of all distances
            horizon = int(np.sum(distance_matrix) + 1000)
        else:
            horizon = self.horizon

        # Wide time windows for all nodes (no constraints)
        time_windows = [(0, horizon) for _ in range(nb_nodes)]

        # Depot from TSP
        depot_node = source_problem.start_index

        return TSPTWProblem(
            nb_nodes=nb_nodes,
            distance_matrix=distance_matrix,
            time_windows=time_windows,
            depot_node=depot_node,
        )

    def back_transform_solution(
        self, solution: TSPTWSolution, source_problem: TspProblem
    ) -> TspSolution:
        """Transform TSPTW solution back to TSP solution.

        Args:
            solution: TSPTW solution
            source_problem: Original TSP problem

        Returns:
            Equivalent TSP solution

        """
        return TspSolution(
            problem=source_problem,
            permutation=list(solution.permutation),
            start_index=source_problem.start_index,
            end_index=source_problem.end_index,
        )

    def forward_transform_solution(
        self, solution: TspSolution, target_problem: TSPTWProblem
    ) -> Optional[TSPTWSolution]:
        """Transform TSP solution to TSPTW solution (for warmstart).

        Args:
            solution: TSP solution
            target_problem: Target TSPTW problem

        Returns:
            Equivalent TSPTW solution

        """
        return TSPTWSolution(
            problem=target_problem,
            permutation=list(solution.permutation),
        )
