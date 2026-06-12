#!/usr/bin/env python3
#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example 02: Composite Transformation

Demonstrates:
- Chaining multiple transformations (TSP → VRP → VRPTW)
- Using chain_transformations() for automatic composition
- Automatic reverse-order back-transformation
- Round-trip transformations (transform and transform back)
"""

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.composite import (
    chain_transformations,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.tsp.transformations.to_vrp import TspToVrpTransformation
from discrete_optimization.vrp.transformations.to_vrptw import VrpToVrptwTransformation
from discrete_optimization.vrptw.solvers.cpsat import CpSatVRPTWSolver


def main():
    """Chain transformations: TSP → VRP → VRPTW."""
    print("=" * 70)
    print("Example 02: Composite Transformation (TSP → VRP → VRPTW)")
    print("=" * 70)

    # Create a small TSP instance for quick solving
    # Simple 10-node TSP problem
    import random

    from discrete_optimization.tsp.problem import Point2D, Point2DTspProblem

    random.seed(42)
    list_points = [
        Point2D(x=random.random() * 100, y=random.random() * 100) for _ in range(10)
    ]

    tsp_problem = Point2DTspProblem(
        list_points=list_points,
        node_count=10,
        start_index=0,
        end_index=0,
    )
    print("Created 10-node TSP problem")

    print(f"Nodes: {tsp_problem.node_count}")

    # Create individual transformations
    trans1 = TspToVrpTransformation()
    trans2 = VrpToVrptwTransformation(horizon=10000)

    # Chain them: TSP → VRP → VRPTW
    composite = chain_transformations(trans1, trans2)

    print("\nTransformation chain:")
    print("  TSP → VRP → VRPTW")

    # Demonstrate round-trip transformation
    print("\n--- Round-trip test ---")

    # Forward: TSP → VRPTW (through VRP)
    vrptw_problem = composite.transform_problem(tsp_problem)
    print(
        f"✓ Transformed to {type(vrptw_problem).__name__}: "
        f"{vrptw_problem.nb_nodes} nodes"
    )

    # Use with solver
    print("\n--- Solving with composite transformation ---")
    solver = TransformationSolver(
        transformation=composite,
        source_problem=tsp_problem,
        solver_brick=SubBrick(cls=CpSatVRPTWSolver, kwargs={"time_limit": 5}),
    )

    print("Solving in VRPTW space...")
    result = solver.solve()

    # Solution is automatically back-transformed through the chain:
    # VRPTW → VRP → TSP
    if len(result) > 0:
        solution = result.get_best_solution()
        print(f"\nSolution type: {type(solution).__name__}")
        print(f"Tour length: {tsp_problem.evaluate(solution)['length']:.2f}")

        # Verify solution validity
        assert tsp_problem.satisfy(solution)
        print("✓ Solution is valid in original TSP problem space")
    else:
        print("\nNo solution found (may need more time for this instance)")

    print("\nKey takeaway:")
    print("- chain_transformations() handles composition automatically")
    print("- Back-transformation happens in reverse order")
    print("- Solution is correctly mapped back to original problem")


if __name__ == "__main__":
    main()
