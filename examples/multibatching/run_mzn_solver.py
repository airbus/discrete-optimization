#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.multibatching.solvers.cp_mzn import (
    CpMultibatchingSolver,
    CpSolverName,
)
from discrete_optimization.multibatching.utils import generate_multibatching_problem

logging.basicConfig(level=logging.INFO)


def main():
    # Generate a random problem instance
    print("Generating random multibatching problem...")
    problem = generate_multibatching_problem(
        num_locations=10,
        num_transport_types=5,
        num_products=5,
        seed=42,
    )

    print(f"Problem has:")
    print(f"  - {problem.nb_locations} locations")
    print(f"  - {problem.nb_transport_types} transport types")
    print(f"  - {problem.nb_products} products")
    print(f"  - {problem.nb_transport_links} transport links")

    # Initialize and solve with CPSat
    print("\nSolving with minizinc (Flow modeling)...")
    solver = CpMultibatchingSolver(problem)
    solver.init_model(cp_solver_name=CpSolverName.GECODE)
    params_cp = ParametersCp.default_cpsat()
    params_cp.free_search = True
    # Solve with time limit
    result_storage = solver.solve(time_limit=200, parameters_cp=params_cp)

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
