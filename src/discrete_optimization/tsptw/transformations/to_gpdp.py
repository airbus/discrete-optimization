#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from TSPTW to GPDP (General Pickup and Delivery Problem).

TSPTW can be modeled as a GPDP with:
- Single vehicle
- No pickup/delivery pairs
- Time windows for all nodes
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
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class TsptwToGpdpTransformation(
    ProblemTransformation[TSPTWProblem, TSPTWSolution, GpdpProblem, GpdpSolution]
):
    """Transform TSPTW to GPDP.

    TSPTW is a special case of GPDP with:
    - Single vehicle
    - Time windows for each node
    - No pickup/delivery pairs
    - No capacity constraints
    - All nodes must be visited

    This transformation is EXACT:
    - All TSPTW constraints are preserved in GPDP
    - Time windows are directly mapped
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
        """Metadata for forward problem transformation (TSPTW → GPDP).

        This direction is EXACT: TSPTW is a special case of GPDP.
        """
        return exact_transformation(
            use_cases=[
                "Use GPDP solvers to solve TSPTW problems",
                "TSPTW is exactly GPDP with 1 vehicle and time windows",
                "All TSPTW constraints preserved in GPDP formulation",
                "Time windows constrain node visit times",
            ]
        )

    def transform_problem(self, source_problem: TSPTWProblem) -> GpdpProblem:
        """Transform TSPTW to GPDP using the existing ProxyClass implementation.

        Args:
            source_problem: TSPTW problem instance

        Returns:
            Equivalent GPDP problem with 1 vehicle and time windows

        """
        from discrete_optimization.gpdp.problem import ProxyClass

        return ProxyClass.from_tsptw_to_gpdp(
            source_problem, compute_graph=self.compute_graph
        )

    def back_transform_solution(
        self, solution: GpdpSolution, source_problem: TSPTWProblem
    ) -> TSPTWSolution:
        """Transform GPDP solution back to TSPTW solution.

        Args:
            solution: GPDP solution (should have 1 vehicle)
            source_problem: Original TSPTW problem

        Returns:
            Equivalent TSPTW solution

        """
        # Extract the single vehicle trajectory (vehicle 0)
        if 0 not in solution.trajectories:
            # Empty solution - use customer order
            permutation = list(source_problem.customers)
        else:
            trajectory = solution.trajectories[0]

            # The trajectory includes: [origin_node, customer1, customer2, ..., target_node]
            # We need to extract only the customer nodes and map them back
            # In the ProxyClass.from_tsptw_to_gpdp transformation:
            # - GPDP customers are indexed 0..nb_customers-1
            # - These correspond to TSPTW customers (which are source_problem.customers)
            # - Origin node = nb_customers
            # - Target node = nb_customers + 1
            #
            # Simply skip first and last nodes (origin/target) and map the rest
            permutation = []
            for node in trajectory[1:-1]:  # Skip origin and target
                # Map GPDP customer index to TSPTW customer index
                permutation.append(source_problem.customers[node])

        return TSPTWSolution(
            problem=source_problem,
            permutation=permutation,
        )

    def forward_transform_solution(
        self, solution: TSPTWSolution, target_problem: GpdpProblem
    ) -> Optional[GpdpSolution]:
        """Transform TSPTW solution to GPDP solution (for warmstart).

        Args:
            solution: TSPTW solution
            target_problem: Target GPDP problem

        Returns:
            Equivalent GPDP solution with single vehicle trajectory

        """
        # Build trajectory for vehicle 0
        # GPDP uses virtual nodes: customers are 0..nb_customers-1,
        # origin is nb_customers, target is nb_customers+1
        nb_customers = len(solution.problem.customers)

        # Map TSPTW customer indices to GPDP customer indices
        # TSPTW permutation contains original customer indices
        # GPDP uses sequential indices 0..nb_customers-1
        gpdp_customer_indices = []
        for tsptw_customer in solution.permutation:
            # Find the position of this customer in the customers list
            gpdp_index = solution.problem.customers.index(tsptw_customer)
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
                    else:
                        # No direct edge - this shouldn't happen in valid solutions
                        current_time += 0.0
                # Add service time at current node
                if node in target_problem.time_delta_node:
                    current_time += target_problem.time_delta_node[node]

        return GpdpSolution(
            problem=target_problem,
            trajectories=trajectories,
            times=times,
            resource_evolution={},
        )
