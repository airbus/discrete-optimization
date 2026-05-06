#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.multibatching.solvers.netx import NetxMultibatchingSolver
from discrete_optimization.multibatching.solvers.packing_subproblem import (
    GreedyPackingForMultibatching,
)
from discrete_optimization.multibatching.solvers.two_steps import (
    TwoStepMultibatchingSolver,
)
from discrete_optimization.multibatching.utils import generate_multibatching_problem

logging.basicConfig(level=logging.INFO)


def main():
    # Generate a random problem instance
    print("Generating random multibatching problem...")
    problem = generate_multibatching_problem(
        num_locations=20,
        num_transport_types=5,
        num_products=5,
        seed=42,
    )

    print(f"Problem has:")
    print(f"  - {problem.nb_locations} locations")
    print(f"  - {problem.nb_transport_types} transport types")
    print(f"  - {problem.nb_products} products")
    print(f"  - {problem.nb_transport_links} transport links")

    # Initialize two-step solver
    print("\nSolving with Two-Step approach...")
    print("  Step 1: Minizinc flow solver")
    print("  Step 2: Greedy packing solver")

    solver = TwoStepMultibatchingSolver(problem)
    # Configure the two steps
    flow_solver_config = SubBrick(
        cls=NetxMultibatchingSolver,
        kwargs={
            "restrict_to_shortest_paths": True,
        },
    )

    packing_solver_config = SubBrick(
        cls=GreedyPackingForMultibatching,
        kwargs={},
    )

    # Solve
    result_storage = solver.solve(
        flow_solver=flow_solver_config,
        packing_solver=packing_solver_config,
        best_n_flow_solution=3,  # Try packing the best 3 flow solutions
    )

    # Get best solution
    if len(result_storage) > 0:
        solution, fitness = result_storage.get_best_solution_fit()
        print(f"\nSolution found!")
        print(f"  - Objective value: {fitness}")
        print(f"  - Number of flows: {len(solution.list_flows)}")

        # Evaluate solution
        evaluation = problem.evaluate(solution)
        print(f"  - Transport cost: {evaluation['transport']}")
        print(f"  - Emission cost: {evaluation['emission']}")

        # Validate solution
        print("\nValidating solution...")
        is_feasible = problem.satisfy(solution)
        if is_feasible:
            print("✓ Solution is FEASIBLE")
        else:
            print("✗ Solution is NOT feasible")

        return solution, is_feasible
    else:
        print("No solution found")
        return None, False


if __name__ == "__main__":
    main()
