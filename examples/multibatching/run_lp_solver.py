#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.lp_tools import mathopt
from discrete_optimization.multibatching.solvers.lp import (
    GurobiMultibatchingSolver,
    MathOptMultibatchingSolver,
)
from discrete_optimization.multibatching.utils import generate_multibatching_problem


def main_gurobi():
    # Generate a random problem instance
    print("Generating random multibatching problem...")
    problem = generate_multibatching_problem(
        num_locations=4,
        num_transport_types=2,
        num_products=2,
        seed=42,
    )

    # Initialize and solve with Gurobi
    print("\nSolving with Gurobi MILP (Flow modeling)...")
    solver = GurobiMultibatchingSolver(problem)
    solver.init_model(single_batching=False)

    # Solve with time limit
    from discrete_optimization.generic_tools.lp_tools import ParametersMilp

    params = ParametersMilp.default()
    params.time_limit = 30

    result_storage = solver.solve(parameters_milp=params)

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


def main_mathopt():
    # Generate a random problem instance
    print("Generating random multibatching problem...")
    problem = generate_multibatching_problem(
        num_locations=4,
        num_transport_types=2,
        num_products=2,
        seed=42,
    )

    # Initialize and solve with Gurobi
    print("\nSolving with Gurobi MILP (Flow modeling)...")
    solver = MathOptMultibatchingSolver(problem)
    solver.init_model(single_batching=False)

    # Solve with time limit
    from discrete_optimization.generic_tools.lp_tools import ParametersMilp

    params = ParametersMilp.default()
    params.time_limit = 30

    result_storage = solver.solve(
        parameters_milp=params,
        mathopt_solver_type=mathopt.SolverType.GSCIP,
        mathopt_additional_solve_parameters=mathopt.SolveParameters(threads=10),
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
    main_mathopt()
