#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from VRPTW to GPDP (General Pickup and Delivery Problem)."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.gpdp.problem import GpdpProblem, GpdpSolution
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution


class VrptwToGpdpTransformation(
    ProblemTransformation[VRPTWProblem, VRPTWSolution, GpdpProblem, GpdpSolution]
):
    """Transform VRPTW to GPDP.

    VRPTW is a special case of GPDP with time windows and demand constraints.
    """

    def __init__(self, compute_graph: bool = False):
        self.compute_graph = compute_graph

    def get_forward_metadata(self) -> TransformationMetadata:
        return exact_transformation(
            use_cases=[
                "Use GPDP solvers for VRPTW problems",
                "VRPTW constraints preserved in GPDP",
            ]
        )

    def transform_problem(self, source_problem: VRPTWProblem) -> GpdpProblem:
        """Transform VRPTW to GPDP using the existing ProxyClass implementation."""
        from discrete_optimization.gpdp.problem import ProxyClass

        return ProxyClass.from_vrptw_to_gpdp(
            source_problem, compute_graph=self.compute_graph
        )

    def back_transform_solution(
        self, solution: GpdpSolution, source_problem: VRPTWProblem
    ) -> VRPTWSolution:
        """Transform GPDP solution back to VRPTW solution."""
        routes = []
        for v in range(source_problem.nb_vehicles):
            if v in solution.trajectories:
                # Filter depot nodes from trajectory
                route = [
                    node
                    for node in solution.trajectories[v]
                    if node != source_problem.depot_node
                ]
                routes.append(route)
            else:
                routes.append([])

        return VRPTWSolution(problem=source_problem, routes=routes)

    def forward_transform_solution(
        self, solution: VRPTWSolution, target_problem: GpdpProblem
    ) -> Optional[GpdpSolution]:
        """Transform VRPTW solution to GPDP solution."""
        trajectories = {}
        for v, route in enumerate(solution.routes):
            trajectories[v] = (
                [target_problem.origin_vehicle[v]]
                + list(route)
                + [target_problem.target_vehicle[v]]
            )

        # Compute times
        times = {}
        for v, traj in trajectories.items():
            current_time = 0.0
            for i, node in enumerate(traj):
                times[node] = current_time
                if i < len(traj) - 1:
                    next_node = traj[i + 1]
                    current_time += target_problem.time_delta[node][next_node]

        return GpdpSolution(
            problem=target_problem,
            trajectories=trajectories,
            times=times,
            resource_evolution={},
        )
