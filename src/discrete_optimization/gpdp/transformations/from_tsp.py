#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from TSP to GPDP (General Pickup and Delivery Problem)."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    subset_transformation,
)
from discrete_optimization.gpdp.problem import GpdpProblem, GpdpSolution
from discrete_optimization.tsp.problem import Point2DTspProblem, TspSolution


class TspToGpdpTransformation(
    ProblemTransformation[Point2DTspProblem, TspSolution, GpdpProblem, GpdpSolution]
):
    """Transform TSP to GPDP.

    TSP is a special case of GPDP with a single vehicle and no capacity constraints.
    """

    def __init__(self, compute_graph: bool = False):
        self.compute_graph = compute_graph

    def get_forward_metadata(self) -> TransformationMetadata:
        return subset_transformation(
            use_cases=["Use GPDP solvers for TSP problems"],
            assumptions=["Single vehicle", "No capacity constraints"],
        )

    def transform_problem(self, source_problem: Point2DTspProblem) -> GpdpProblem:
        """Transform TSP to GPDP using the existing ProxyClass implementation."""
        # Reuse the existing ProxyClass method (cleaned up version)
        from discrete_optimization.gpdp.problem import ProxyClass

        return ProxyClass.from_tsp_to_gpdp(
            source_problem, compute_graph=self.compute_graph
        )

    def back_transform_solution(
        self, solution: GpdpSolution, source_problem: Point2DTspProblem
    ) -> TspSolution:
        """Transform GPDP solution back to TSP solution."""
        # Extract the single vehicle's trajectory
        trajectory = solution.trajectories[0] if 0 in solution.trajectories else []

        # Filter to get only customer nodes (exclude origin/target)
        permutation = [
            node
            for node in trajectory
            if node not in {0, len(source_problem.list_points) - 1}
        ]

        return TspSolution(problem=source_problem, permutation=permutation)

    def forward_transform_solution(
        self, solution: TspSolution, target_problem: GpdpProblem
    ) -> Optional[GpdpSolution]:
        """Transform TSP solution to GPDP solution."""
        # Build trajectory from TSP permutation
        trajectories = {
            0: [target_problem.origin_vehicle[0]]
            + list(solution.permutation)
            + [target_problem.target_vehicle[0]]
        }

        # Compute times
        times = {}
        current_time = 0.0
        traj = trajectories[0]
        for i, node in enumerate(traj):
            times[node] = current_time
            if i < len(traj) - 1:
                next_node = traj[i + 1]
                current_time += target_problem.distance_delta[node][next_node]

        return GpdpSolution(
            problem=target_problem,
            trajectories=trajectories,
            times=times,
            resource_evolution={},
        )
