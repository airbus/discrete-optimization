#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from VRPTW to VRP (lossy - drops time windows)."""

from typing import Optional

import numpy as np

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
from discrete_optimization.vrp.problem import (
    Customer2D,
    Customer2DVrpProblem,
    VrpSolution,
)
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution


class VrptwToVrpTransformation(
    ProblemTransformation[
        VRPTWProblem, VRPTWSolution, Customer2DVrpProblem, VrpSolution
    ]
):
    """Transform VRPTW to VRP (LOSSY transformation).

    Mapping:
    - Vehicle routes with time windows → Vehicle routes (no time constraints)
    - Customer demands → Customer demands
    - Vehicle capacities → Vehicle capacities
    - Time windows → LOST
    - Service times → LOST

    This is a LOSSY transformation because:
    - Time window constraints are dropped
    - Service times are ignored
    - Solutions from VRP may violate time windows in original VRPTW

    Use cases:
    - Get initial solutions from VRP solvers (faster, simpler)
    - Benchmark VRP solvers on VRPTW instances (ignoring time)
    - Analyze impact of time windows by comparing with unconstrained VRP
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward transformation (VRPTW → VRP)."""
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="time_windows",
                    loss_type=LossType.CONSTRAINT,
                    description="Customer time window constraints [earliest, latest]",
                    reason="VRP has no concept of time or temporal constraints",
                    impact=LossImpact.MAJOR,
                    workaround="Post-process VRP solution to check time window feasibility",
                ),
                InformationLoss(
                    name="service_times",
                    loss_type=LossType.PARAMETER,
                    description="Service time at each customer location",
                    reason="VRP doesn't model service times",
                    impact=LossImpact.MODERATE,
                    workaround="Add service times to route evaluation post-hoc",
                ),
            ],
            use_cases=[
                "Get quick initial solutions from VRP solvers",
                "Benchmark VRP algorithms on VRPTW instances",
                "Analyze impact of time windows on routing decisions",
            ],
            warnings=[
                "VRP solutions may violate time windows",
                "Service times are ignored in VRP optimization",
                "Route feasibility must be verified against time constraints",
            ],
        )

    def transform_problem(self, source_problem: VRPTWProblem) -> Customer2DVrpProblem:
        """Transform VRPTW to VRP.

        Args:
            source_problem: VRPTW problem instance

        Returns:
            VRP problem (time windows and service times dropped)

        """
        # Extract 2D coordinates from distance matrix
        # VRPTW uses distance_matrix, we need to create Customer2D objects
        nb_nodes = source_problem.nb_nodes

        # Reconstruct 2D coordinates using MDS (Multi-Dimensional Scaling) approximation
        # For simplicity, we'll use a heuristic: place nodes on a grid
        # In practice, you might need the original coordinates if available
        customers = []
        grid_size = int(np.ceil(np.sqrt(nb_nodes)))

        for i in range(nb_nodes):
            # Simple grid layout (can be improved with MDS)
            x = (i % grid_size) * 100.0
            y = (i // grid_size) * 100.0
            demand = source_problem.demands[i]

            customers.append(
                Customer2D(
                    name=i,
                    demand=demand,
                    x=x,
                    y=y,
                )
            )

        # Note: This is a heuristic layout. In practice, VRPTW problems
        # often come from files that include coordinates. Users should
        # provide coordinates if available.

        # Vehicle parameters
        vehicle_count = source_problem.nb_vehicles
        vehicle_capacities = [source_problem.vehicle_capacity] * vehicle_count

        # Depot: VRPTW depot_node becomes start/end index for all vehicles
        depot = source_problem.depot_node
        start_indexes = [depot] * vehicle_count
        end_indexes = [depot] * vehicle_count

        return Customer2DVrpProblem(
            vehicle_count=vehicle_count,
            vehicle_capacities=vehicle_capacities,
            customer_count=nb_nodes,
            customers=customers,
            start_indexes=start_indexes,
            end_indexes=end_indexes,
        )

    def back_transform_solution(
        self, solution: VrpSolution, source_problem: VRPTWProblem
    ) -> VRPTWSolution:
        """Transform VRP solution back to VRPTW solution.

        Args:
            solution: VRP solution
            source_problem: Original VRPTW problem

        Returns:
            VRPTW solution (may violate time windows!)

        Warning:
            The returned solution may not satisfy time window constraints.
            Use problem.satisfy() to check feasibility.

        """
        # Extract routes from VRP solution
        routes = [list(path) for path in solution.list_paths]

        return VRPTWSolution(
            problem=source_problem,
            routes=routes,
        )

    def forward_transform_solution(
        self, solution: VRPTWSolution, target_problem: Customer2DVrpProblem
    ) -> Optional[VrpSolution]:
        """Transform VRPTW solution to VRP solution (for warmstart).

        Args:
            solution: VRPTW solution
            target_problem: Target VRP problem

        Returns:
            Equivalent VRP solution

        """
        # Convert VRPTW routes to VRP routes
        list_paths = [list(route) for route in solution.routes]

        # Start and end indices from target problem
        list_start_index = list(target_problem.start_indexes)
        list_end_index = list(target_problem.end_indexes)

        return VrpSolution(
            problem=target_problem,
            list_start_index=list_start_index,
            list_end_index=list_end_index,
            list_paths=list_paths,
        )
