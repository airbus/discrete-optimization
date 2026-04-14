#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from VRP to Team Orienteering Problem (TOP)."""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.top.problem import (
    CustomerTop2D,
    TeamOrienteeringProblem2D,
)
from discrete_optimization.vrp.problem import (
    Customer2DVrpProblem,
    VrpSolution,
)


class VrpToTopTransformation(
    ProblemTransformation[
        Customer2DVrpProblem, VrpSolution, TeamOrienteeringProblem2D, VrpSolution
    ]
):
    """Transform VRP (2D) to Team Orienteering Problem (TOP).

    Mapping:
    - VRP customers → TOP customers with rewards
    - VRP vehicle routes → TOP tours
    - VRP capacity constraints → Dropped (TOP has no capacity)
    - Add max_length_tours constraint
    - Objective: minimize total distance → maximize total reward

    Note: This transformation assigns uniform rewards (reward=1) to all customers.
    Users can modify rewards after transformation if needed.
    """

    def __init__(self, max_length_tours: Optional[float] = None, reward_function=None):
        """Initialize transformation.

        Args:
            max_length_tours: Maximum length for each tour (default: computed from problem)
            reward_function: Function to compute reward from Customer2D (default: uniform reward=1)

        """
        self.max_length_tours = max_length_tours
        self.reward_function = reward_function or (lambda customer: 1.0)

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward transformation (VRP → TOP).

        This is EXACT when:
        - VRP capacity constraints are not binding (infinite capacity)
        - VRP objective is compatible with reward maximization

        In practice, this transformation is useful for:
        - Using TOP solvers for VRP problems
        - Exploring reward-based routing strategies
        """
        return exact_transformation(
            use_cases=[
                "Use TOP solvers for VRP problems",
                "VRP can be modeled as TOP with uniform rewards",
                "Reframe VRP as reward maximization problem",
            ]
        )

    def transform_problem(
        self, source_problem: Customer2DVrpProblem
    ) -> TeamOrienteeringProblem2D:
        """Transform VRP to TOP.

        Args:
            source_problem: VRP problem instance (must be 2D)

        Returns:
            Equivalent TOP problem with rewards

        """
        # Determine max_length_tours if not provided
        if self.max_length_tours is None:
            # Estimate: sum of all pairwise distances / number of vehicles
            total_distance_estimate = 0.0
            nb_customers = source_problem.customer_count
            for i in range(nb_customers):
                for j in range(i + 1, nb_customers):
                    total_distance_estimate += source_problem.evaluate_function_indexes(
                        i, j
                    )
            max_length = (
                total_distance_estimate / source_problem.vehicle_count
                if source_problem.vehicle_count > 0
                else total_distance_estimate
            )
        else:
            max_length = self.max_length_tours

        # Create TOP customers with rewards
        top_customers = [
            CustomerTop2D(
                name=customer.name,
                reward=self.reward_function(customer),
                x=customer.x,
                y=customer.y,
            )
            for customer in source_problem.customers
        ]

        return TeamOrienteeringProblem2D(
            vehicle_count=source_problem.vehicle_count,
            max_length_tours=max_length,
            customer_count=source_problem.customer_count,
            customers=top_customers,
            start_indexes=list(source_problem.start_indexes),
            end_indexes=list(source_problem.end_indexes),
        )

    def back_transform_solution(
        self, solution: VrpSolution, source_problem: Customer2DVrpProblem
    ) -> VrpSolution:
        """Transform TOP solution back to VRP solution.

        Args:
            solution: TOP solution (uses VrpSolution representation)
            source_problem: Original VRP problem

        Returns:
            Equivalent VRP solution

        """
        # TOP and VRP use the same solution representation (VrpSolution)
        # Just need to change the problem reference
        return VrpSolution(
            problem=source_problem,
            list_start_index=solution.list_start_index,
            list_end_index=solution.list_end_index,
            list_paths=solution.list_paths,
            lengths=solution.lengths,
            length=solution.length,
            capacities=solution.capacities,
        )

    def forward_transform_solution(
        self, solution: VrpSolution, target_problem: TeamOrienteeringProblem2D
    ) -> Optional[VrpSolution]:
        """Transform VRP solution to TOP solution (for warmstart).

        Args:
            solution: VRP solution
            target_problem: Target TOP problem

        Returns:
            Equivalent TOP solution

        """
        # Both use VrpSolution representation, just change problem reference
        return VrpSolution(
            problem=target_problem,
            list_start_index=solution.list_start_index,
            list_end_index=solution.list_end_index,
            list_paths=solution.list_paths,
            lengths=solution.lengths,
            length=solution.length,
            capacities=solution.capacities,
        )
