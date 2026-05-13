#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from TSP to GPDP (General Pickup and Delivery Problem).

TSP can be modeled as a GPDP with:
- Single vehicle
- No pickup/delivery pairs
- No time windows
- No capacity constraints
"""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.gpdp.problem import GpdpProblem, GpdpSolution
from discrete_optimization.tsp.problem import TspProblem, TspSolution


class TspToGpdpTransformation(
    ProblemTransformation[TspProblem, TspSolution, GpdpProblem, GpdpSolution]
):
    """Transform TSP to GPDP.

    TSP is a special case of GPDP with:
    - Single vehicle
    - No pickup/delivery pairs
    - No time windows
    - No capacity constraints
    - All nodes must be visited

    This transformation is EXACT:
    - All TSP constraints are preserved in GPDP
    - Solution quality is preserved in both directions

    """

    def __init__(self, compute_graph: bool = False):
        """Initialize transformation.

        Args:
            compute_graph: Whether to compute the GPDP graph structure (default: False).
                          Set to True if using graph-based GPDP solvers.

        """
        self.compute_graph = compute_graph

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (TSP → GPDP).

        This direction is EXACT: TSP is a special case of GPDP.
        """
        return exact_transformation(
            use_cases=[
                "Use GPDP solvers to solve TSP problems",
                "TSP is exactly GPDP with 1 vehicle and no constraints",
                "All TSP constraints preserved in GPDP formulation",
                "No time windows, pickup/delivery, or capacity constraints",
            ]
        )

    def transform_problem(self, source_problem: TspProblem) -> GpdpProblem:
        """Transform TSP to GPDP using the existing ProxyClass implementation.

        Args:
            source_problem: TSP problem instance

        Returns:
            Equivalent GPDP problem with 1 vehicle and no constraints

        """
        from discrete_optimization.gpdp.problem import ProxyClass

        return ProxyClass.from_tsp_to_gpdp(
            source_problem, compute_graph=self.compute_graph
        )

    def back_transform_solution(
        self, solution: GpdpSolution, source_problem: TspProblem
    ) -> TspSolution:
        """Transform GPDP solution back to TSP solution.

        Args:
            solution: GPDP solution (should have 1 vehicle)
            source_problem: Original TSP problem

        Returns:
            Equivalent TSP solution

        """
        # Extract the single vehicle trajectory (vehicle 0)
        if 0 not in solution.trajectories:
            # Empty solution - use default permutation
            permutation = list(source_problem.ind_in_permutation)
        else:
            trajectory = solution.trajectories[0]

            # The trajectory includes: [origin_node, customer1, customer2, ..., target_node]
            # We need to extract only the customer nodes (skip first and last)
            # In ProxyClass.from_tsp_to_gpdp, customers are mapped to indices 0..nb_customers-1
            permutation = trajectory[1:-1]  # Skip origin and target nodes

            # Convert from GPDP indices to TSP indices
            # In the transformation, GPDP indices match TSP indices for customers
            permutation = [source_problem.ind_in_permutation[i] for i in permutation]

        return TspSolution(
            problem=source_problem,
            permutation=permutation,
            start_index=source_problem.start_index,
            end_index=source_problem.end_index,
        )

    def forward_transform_solution(
        self, solution: TspSolution, target_problem: GpdpProblem
    ) -> Optional[GpdpSolution]:
        """Transform TSP solution to GPDP solution (for warmstart).

        Args:
            solution: TSP solution
            target_problem: Target GPDP problem

        Returns:
            Equivalent GPDP solution with single vehicle trajectory

        """
        # Convert TSP permutation to GPDP customer indices
        # TSP permutation contains original node indices
        # GPDP uses sequential indices 0..nb_customers-1
        gpdp_customer_indices = []
        for tsp_node in solution.permutation:
            # Find index in ind_in_permutation
            gpdp_index = solution.problem.ind_in_permutation.index(tsp_node)
            gpdp_customer_indices.append(gpdp_index)

        # Build complete trajectory: origin -> customers -> target
        origin_node = target_problem.origin_vehicle[0]
        target_node = target_problem.target_vehicle[0]

        trajectory = [origin_node] + gpdp_customer_indices + [target_node]
        trajectories = {0: trajectory}

        # Compute times along the trajectory
        times = {}
        current_time = 0.0

        for i, node in enumerate(trajectory):
            times[node] = current_time
            if i < len(trajectory) - 1:
                next_node = trajectory[i + 1]
                # Add travel time to next node
                if node in target_problem.time_delta:
                    if next_node in target_problem.time_delta[node]:
                        current_time += target_problem.time_delta[node][next_node]

        return GpdpSolution(
            problem=target_problem,
            trajectories=trajectories,
            times=times,
            resource_evolution={},
        )
