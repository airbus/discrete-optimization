#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from TSP to VRP.

This transformation is EXACT: TSP is a special case of VRP with 1 vehicle,
infinite capacity (or zero demands), and all customers must be visited.
"""

from typing import Optional

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.vrp.problem import BasicCustomer, VrpProblem, VrpSolution


class TspToVrpTransformation(
    ProblemTransformation[TspProblem, TspSolution, VrpProblem, VrpSolution]
):
    """Transform TSP to VRP.

    Mapping:
    - Single tour → Single vehicle route
    - All nodes must be visited → All customers served
    - No capacity constraints → Infinite capacity (or zero demands)
    - Start/end depot → VRP start/end depot

    This transformation is EXACT:
    - TSP is exactly VRP with 1 vehicle and no capacity constraints
    - All TSP constraints are preserved in VRP formulation

    Solution mapping is exact in both directions:
    - TSP tour → VRP route for vehicle 0
    - VRP route (single vehicle) → TSP tour

    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (TSP → VRP).

        This direction is EXACT: TSP is a perfect subset of VRP.
        """
        return exact_transformation(
            use_cases=[
                "Use VRP solvers to solve TSP problems",
                "TSP is exactly VRP with 1 vehicle and infinite capacity",
                "All TSP constraints preserved in VRP",
            ]
        )

    def transform_problem(self, source_problem: TspProblem) -> VrpProblem:
        """Transform TSP to VRP.

        Args:
            source_problem: TSP problem instance

        Returns:
            Equivalent VRP problem with 1 vehicle

        """
        # Create customers from TSP nodes (excluding start/end if they're the same as depot)
        # TSP nodes that need to be visited (permutation indices)
        customers = [
            BasicCustomer(name=node, demand=0.0)  # Zero demand = infinite capacity
            for node in source_problem.ind_in_permutation
        ]

        # Create concrete VRP problem class
        # We need to implement the abstract methods
        class TspDerivedVrpProblem(VrpProblem):
            def __init__(self, tsp_problem: TspProblem, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tsp_problem = tsp_problem

            def evaluate_function(self, var_vrp: VrpSolution):
                # Convert VRP solution to TSP solution and evaluate
                if not var_vrp.list_paths or len(var_vrp.list_paths[0]) == 0:
                    return [[]], [0.0], float("inf"), [0.0]

                # Extract the single route
                route = var_vrp.list_paths[0]

                # Build TSP permutation from route
                tsp_solution = TspSolution(
                    problem=self.tsp_problem,
                    permutation=route,
                    start_index=var_vrp.list_start_index[0],
                    end_index=var_vrp.list_end_index[0],
                )

                # Evaluate in TSP
                self.tsp_problem.evaluate(tsp_solution)

                # Return in VRP format
                lengths = [tsp_solution.lengths if tsp_solution.lengths else []]
                obj = tsp_solution.length if tsp_solution.length else 0.0
                return [lengths], [obj], obj, [0.0]  # Zero capacity used

            def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
                return self.tsp_problem.evaluate_function_indexes(index_1, index_2)

        return TspDerivedVrpProblem(
            tsp_problem=source_problem,
            customers=customers,
            vehicle_count=1,
            vehicle_capacities=[float("inf")],  # Infinite capacity
            customer_count=len(customers),
            start_indexes=[source_problem.start_index],
            end_indexes=[source_problem.end_index],
        )

    def back_transform_solution(
        self, solution: VrpSolution, source_problem: TspProblem
    ) -> TspSolution:
        """Transform VRP solution back to TSP solution.

        Args:
            solution: VRP solution (should have 1 vehicle)
            source_problem: Original TSP problem

        Returns:
            Equivalent TSP solution

        """
        # Extract the single vehicle route
        if not solution.list_paths or len(solution.list_paths) == 0:
            # Empty solution
            permutation = list(source_problem.ind_in_permutation)
        else:
            permutation = solution.list_paths[0]

        return TspSolution(
            problem=source_problem,
            permutation=permutation,
            start_index=source_problem.start_index,
            end_index=source_problem.end_index,
        )

    def forward_transform_solution(
        self, solution: TspSolution, target_problem: VrpProblem
    ) -> Optional[VrpSolution]:
        """Transform TSP solution to VRP solution (for warmstart).

        Args:
            solution: TSP solution
            target_problem: Target VRP problem

        Returns:
            Equivalent VRP solution with single vehicle route

        """
        return VrpSolution(
            problem=target_problem,
            list_start_index=[solution.start_index],
            list_end_index=[solution.end_index],
            list_paths=[solution.permutation],
        )
