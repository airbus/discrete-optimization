#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from TSPTW to VRPTW.

TSP with Time Windows is a special case of VRP with Time Windows:
- Single vehicle
- No capacity constraints (zero demands)
- Time windows for each customer
- All customers must be visited
"""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution


class TsptwToVrptwTransformation(
    ProblemTransformation[TSPTWProblem, TSPTWSolution, VRPTWProblem, VRPTWSolution]
):
    """Transform TSPTW to VRPTW.

    Mapping:
    - Single tour → Single vehicle route
    - Time windows → Customer time windows
    - All customers must be visited → All customers served
    - No capacity constraints → Zero demands + infinite capacity
    - Depot with time window → VRPTW depot with time window

    This transformation is EXACT in both directions:
    - TSPTW is exactly VRPTW with 1 vehicle and no capacity constraints
    - All TSPTW constraints (time windows) are preserved in VRPTW
    - Solution mapping is exact both ways

    """

    def __init__(self, default_service_time: float = 0.0):
        """Initialize transformation.

        Args:
            default_service_time: Service time for each customer (default: 0.0).
                                 Note: In TSPTW, service time is often included in the
                                 distance matrix. Set to 0.0 if already included in distances.

        """
        self.default_service_time = default_service_time

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (TSPTW → VRPTW).

        This direction is EXACT: TSPTW is exactly VRPTW with 1 vehicle, no capacity,
        and time windows.
        """
        return exact_transformation(
            use_cases=[
                "Use VRPTW solvers to solve TSPTW problems",
                "TSPTW is exactly VRPTW with 1 vehicle and infinite capacity",
                "All TSPTW constraints (time windows) preserved in VRPTW",
                "Time windows constrain arrival/service times at each customer",
            ]
        )

    def transform_problem(self, source_problem: TSPTWProblem) -> VRPTWProblem:
        """Transform TSPTW to VRPTW.

        Args:
            source_problem: TSPTW problem instance

        Returns:
            Equivalent VRPTW problem with 1 vehicle and no capacity constraints

        """
        nb_nodes = source_problem.nb_nodes

        # Use existing distance matrix from TSPTW
        distance_matrix = source_problem.distance_matrix.copy()

        # Time windows from TSPTW
        time_windows = list(source_problem.time_windows)

        # Service times: zero or default (TSPTW often includes service in distance)
        service_times = [self.default_service_time for _ in range(nb_nodes)]

        # Demands: zero (no capacity constraints in TSPTW)
        demands = [0.0 for _ in range(nb_nodes)]

        # Single vehicle with infinite capacity
        nb_vehicles = 1
        vehicle_capacity = float("inf")

        # Depot node from TSPTW
        depot_node = source_problem.depot_node

        return VRPTWProblem(
            nb_vehicles=nb_vehicles,
            vehicle_capacity=vehicle_capacity,
            nb_nodes=nb_nodes,
            distance_matrix=distance_matrix,
            time_windows=time_windows,
            service_times=service_times,
            demands=demands,
            depot_node=depot_node,
        )

    def back_transform_solution(
        self, solution: VRPTWSolution, source_problem: TSPTWProblem
    ) -> TSPTWSolution:
        """Transform VRPTW solution back to TSPTW solution.

        Args:
            solution: VRPTW solution (should have 1 vehicle)
            source_problem: Original TSPTW problem

        Returns:
            Equivalent TSPTW solution

        """
        # Extract the single vehicle route
        if not solution.routes or len(solution.routes) == 0:
            # Empty solution - use customer order
            permutation = list(source_problem.customers)
        else:
            # Get first route (single vehicle)
            route = solution.routes[0]
            # Filter out depot if present (should not be in TSPTW permutation)
            permutation = [node for node in route if node != source_problem.depot_node]

        return TSPTWSolution(
            problem=source_problem,
            permutation=permutation,
        )

    def forward_transform_solution(
        self, solution: TSPTWSolution, target_problem: VRPTWProblem
    ) -> Optional[VRPTWSolution]:
        """Transform TSPTW solution to VRPTW solution (for warmstart).

        Args:
            solution: TSPTW solution
            target_problem: Target VRPTW problem

        Returns:
            Equivalent VRPTW solution with single vehicle route

        """
        # TSPTW permutation becomes the single vehicle route
        routes = [list(solution.permutation)]

        return VRPTWSolution(
            problem=target_problem,
            routes=routes,
        )
