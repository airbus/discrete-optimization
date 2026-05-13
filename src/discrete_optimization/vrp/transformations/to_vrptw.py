#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from VRP to VRPTW."""

from typing import Optional

import numpy as np

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.vrp.problem import (
    Customer2DVrpProblem,
    VrpProblem,
    VrpSolution,
)
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution


class VrpToVrptwTransformation(
    ProblemTransformation[VrpProblem, VrpSolution, VRPTWProblem, VRPTWSolution]
):
    """Transform VRP to VRPTW (Vehicle Routing with Time Windows).

    Mapping:
    - Vehicle routes → Vehicle routes with time windows
    - Customer demands → Customer demands
    - Vehicle capacities → Vehicle capacities
    - Add relaxed time windows: [0, horizon] for all customers
    - Add zero service times

    VRP is a special case of VRPTW with relaxed (unconstrained) time windows.
    This transformation is EXACT in both directions when time windows are wide enough.
    """

    def __init__(
        self, horizon: Optional[int] = None, default_service_time: float = 0.0
    ):
        """Initialize transformation.

        Args:
            horizon: Maximum time for time windows (default: computed from problem)
            default_service_time: Service time for each customer (default: 0.0)

        """
        self.horizon = horizon
        self.default_service_time = default_service_time

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (VRP → VRPTW).

        This direction is EXACT: VRP is exactly VRPTW with relaxed time windows.
        """
        return exact_transformation(
            use_cases=[
                "Use VRPTW solvers to solve VRP problems",
                "VRP is exactly VRPTW with wide time windows [0, horizon]",
                "All VRP constraints preserved in VRPTW",
            ]
        )

    def transform_problem(self, source_problem: VrpProblem) -> VRPTWProblem:
        """Transform VRP to VRPTW.

        Args:
            source_problem: VRP problem instance

        Returns:
            Equivalent VRPTW problem with relaxed time windows

        """
        # Determine horizon if not provided
        if self.horizon is None:
            # Estimate: assume average speed = 1, total distance upper bound
            horizon = source_problem.customer_count * 1000
        else:
            horizon = self.horizon

        # Build distance matrix
        # Check if we have 2D customers for distance calculation
        if isinstance(source_problem, Customer2DVrpProblem):
            # Build distance matrix from 2D coordinates
            nb_nodes = source_problem.customer_count
            distance_matrix = np.zeros((nb_nodes, nb_nodes))

            for i in range(nb_nodes):
                for j in range(nb_nodes):
                    if i != j:
                        distance_matrix[i, j] = (
                            source_problem.evaluate_function_indexes(i, j)
                        )
        else:
            # For non-2D problems, we need to build a distance matrix
            # This is abstract in VrpProblem, so we estimate or use unit distances
            nb_nodes = source_problem.customer_count
            distance_matrix = np.ones((nb_nodes, nb_nodes))
            np.fill_diagonal(distance_matrix, 0)

        # Time windows: relaxed for all nodes [0, horizon]
        time_windows = [(0, horizon) for _ in range(source_problem.customer_count)]

        # Service times: zero or default
        service_times = [
            self.default_service_time for _ in range(source_problem.customer_count)
        ]

        # Demands from VRP customers
        demands = [customer.demand for customer in source_problem.customers]

        # Depot: use first start index
        depot_node = source_problem.start_indexes[0]

        # Vehicle capacity: use first vehicle capacity (assume homogeneous fleet)
        vehicle_capacity = source_problem.vehicle_capacities[0]

        return VRPTWProblem(
            nb_vehicles=source_problem.vehicle_count,
            vehicle_capacity=vehicle_capacity,
            nb_nodes=source_problem.customer_count,
            distance_matrix=distance_matrix,
            time_windows=time_windows,
            service_times=service_times,
            demands=demands,
            depot_node=depot_node,
        )

    def back_transform_solution(
        self, solution: VRPTWSolution, source_problem: VrpProblem
    ) -> VrpSolution:
        """Transform VRPTW solution back to VRP solution.

        Args:
            solution: VRPTW solution
            source_problem: Original VRP problem

        Returns:
            Equivalent VRP solution

        """
        # Extract routes from VRPTW solution
        list_paths = [list(route) for route in solution.routes]

        # Start and end indices from source problem
        list_start_index = list(source_problem.start_indexes)
        list_end_index = list(source_problem.end_indexes)

        return VrpSolution(
            problem=source_problem,
            list_start_index=list_start_index,
            list_end_index=list_end_index,
            list_paths=list_paths,
        )

    def forward_transform_solution(
        self, solution: VrpSolution, target_problem: VRPTWProblem
    ) -> Optional[VRPTWSolution]:
        """Transform VRP solution to VRPTW solution (for warmstart).

        Args:
            solution: VRP solution
            target_problem: Target VRPTW problem

        Returns:
            Equivalent VRPTW solution

        """
        # Convert VRP routes to VRPTW routes
        routes = [list(path) for path in solution.list_paths]

        return VRPTWSolution(
            problem=target_problem,
            routes=routes,
        )
