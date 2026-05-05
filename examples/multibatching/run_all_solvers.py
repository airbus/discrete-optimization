"""Example demonstrating multiple solver configurations for multibatching.

This example runs different solver configurations (1-step, 2-step, 3-step approaches)
on a generated multibatching problem and validates the solutions.

Key insights:
- Direct solvers (CPSat FLOW, Gurobi FLOW) may not produce feasible solutions
  because they optimize flows without considering packing constraints
- 2-step solvers (flow + packing) produce feasible solutions by first computing
  flows and then solving a packing subproblem
- 3-step solvers refine the solution further with detailed trip modeling

This serves as both an example and a validation test for the solver implementations.
"""

import logging

from discrete_optimization.multibatching.utils import generate_multibatching_problem
from examples.multibatching.solvers_config import all_configs

try:
    import gurobipy

    gurobi_available = True
except ImportError:
    gurobi_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Generate a test problem
    print("=" * 80)
    print("Generating random multibatching problem...")
    print("=" * 80)
    problem = generate_multibatching_problem(
        num_locations=5,
        num_transport_types=2,
        num_products=2,
        max_demand_abs=20,
        seed=42,
    )

    print(f"\nProblem characteristics:")
    print(f"  - {problem.nb_locations} locations")
    print(f"  - {problem.nb_transport_types} transport types")
    print(f"  - {problem.nb_products} products")
    print(f"  - {problem.nb_transport_links} transport links")

    # Get all solver configs
    timeout = 30
    solver_configs = all_configs(
        timeout=timeout,
        add_1step=False,  # Direct solvers may not give feasible solutions
        add_2step=True,  # 2-step solvers should give feasible solutions
        add_3steps=False,  # Skip 3-step for faster example
        add_asp=False,  # Skip ASP solver (requires clingo)
        verbose=False,
    )

    print(f"\nRunning {len(solver_configs)} solver configurations...")
    print("=" * 80)

    results = {}
    for config_name, solver_config in solver_configs.items():
        print(f"\n### Running: {config_name} ###")

        # Skip Gurobi configs if not available
        if not gurobi_available and (
            "milp" in str(config_name).lower()
            or "gurobi" in str(solver_config.cls.__name__).lower()
        ):
            print("  Skipping (Gurobi not available)")
            continue

        try:
            # Initialize solver
            if gurobi_available and (
                "milp" in str(config_name).lower()
                or "gurobi" in str(solver_config.cls.__name__).lower()
            ):
                with gurobipy.Env() as env:
                    solver = solver_config.cls(problem, **solver_config.kwargs)
                    solver.init_model(**solver_config.kwargs)
                    result_storage = solver.solve(**solver_config.kwargs)
            else:
                solver = solver_config.cls(problem, **solver_config.kwargs)
                solver.init_model(**solver_config.kwargs)
                result_storage = solver.solve(**solver_config.kwargs)

            # Process results
            if len(result_storage) > 0:
                solution, fitness = result_storage.get_best_solution_fit()
                is_feasible = problem.satisfy(solution)
                evaluation = problem.evaluate(solution)

                results[config_name] = {
                    "feasible": is_feasible,
                    "objective": fitness,
                    "evaluation": evaluation,
                    "num_flows": len(solution.list_flows),
                }

                print(f"  ✓ Solution found")
                print(f"    - Objective: {fitness:.2f}")
                print(f"    - Feasible: {is_feasible}")
                print(f"    - Transport cost: {evaluation['transport']}")
                print(f"    - Emission cost: {evaluation['emission']}")
                print(f"    - Number of flows: {len(solution.list_flows)}")
            else:
                results[config_name] = {"feasible": False, "objective": None}
                print(f"  ✗ No solution found")

        except Exception as e:
            logger.error(f"  ✗ Error running {config_name}: {e}")
            results[config_name] = {
                "feasible": False,
                "objective": None,
                "error": str(e),
            }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    feasible_count = sum(1 for r in results.values() if r.get("feasible", False))
    total_count = len(results)
    print(f"\nFeasible solutions: {feasible_count}/{total_count}")

    if feasible_count > 0:
        print("\nFeasible configurations:")
        for config_name, result in results.items():
            if result.get("feasible", False):
                print(f"  - {config_name}: objective = {result['objective']:.2f}")

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    main()
