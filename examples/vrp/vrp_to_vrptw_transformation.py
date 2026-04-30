#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Solve VRP via VRPTW  transformation.

This example demonstrates:
1. Loading a VRP problem
2. Transforming to VRPTW
3. Solving with a VRPTW solver
4. Back-transforming to VRP solution
"""

import logging

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.transformations import VrpToVrptwTransformation
from discrete_optimization.vrptw.solvers.dp import DpVrptwSolver

logging.basicConfig(level=logging.DEBUG)


def example_vrp_to_vrptw():
    """Example: Solve VRP via TOP transformation."""
    print("=" * 80)
    print("Example: Solve VRP via TOP Transformation")
    print("=" * 80)

    # Load VRP problem
    print("\nLoading VRP problem from benchmark...")
    files = get_data_available()
    file_path = [f for f in files if "vrp_76_8_1" in f][0]
    vrp_problem = parse_file(file_path)
    print(f"\nOriginal VRP problem:")
    print(f"  - Customers: {vrp_problem.customer_count}")
    print(f"  - Vehicles: {vrp_problem.vehicle_count}")
    print(f"  - Vehicle capacities: {vrp_problem.vehicle_capacities}")
    print(f"  - Objective: Minimize total distance + capacity violation")

    # Create transformation with custom reward function
    # Example: reward = 1 / (demand + 1) to prioritize low-demand customers
    def reward_fn(customer):
        return 1.0 / (customer.demand + 1.0)

    print("\nCreating transformation: VRP → TOP...")
    transformation = VrpToVrptwTransformation()

    # Show metadata
    metadata = transformation.get_forward_metadata()
    print(f"\nTransformation properties:")
    print(f"  - Exactness: {metadata.completeness.value}")
    print(f"  - Use cases: {metadata.use_cases}")

    print("\nSolving with DP VRPTW solver (time_limit=10s)...")
    solver = TransformationSolver(
        transformation=transformation,
        source_problem=vrp_problem,
        solver_brick=SubBrick(
            cls=DpVrptwSolver, kwargs={"time_limit": 30, "solver": "CABS"}
        ),
    )
    result = solver.solve()

    # Analyze results
    print(f"\nResults:")
    print(f"  - Solutions found: {len(result)}")

    if len(result) > 0:
        best_solution = result.get_best_solution()
        best_fit = result.get_best_solution_fit()[1]

        print(f"\nBest solution (back-transformed to VRP):")
        print(f"  - Fitness: {best_fit}")
        print(f"  - Solution type: {type(best_solution).__name__}")
        print(f"  - Feasible: {vrp_problem.satisfy(best_solution)}")

        # Evaluate in both spaces
        vrp_eval = vrp_problem.evaluate(best_solution)
        print(f"\nVRP evaluation:")
        print(f"  - Total length: {vrp_eval['length']:.2f}")
        print(f"  - Max length: {vrp_eval['max_length']:.2f}")
        print(f"  - Capacity violation: {vrp_eval['capacity_violation']:.2f}")
        print(f"  - Vehicles used: {vrp_eval['nb_vehicles']}")

        # Show routes
        print(f"\nRoutes (first 3 vehicles):")
        for v in range(min(3, len(best_solution.list_paths))):
            path = best_solution.list_paths[v]
            print(f"  - Vehicle {v}: {len(path)} customers - {path[:10]}...")

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    example_vrp_to_vrptw()
