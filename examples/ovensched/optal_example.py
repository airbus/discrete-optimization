#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example of using the OptalCP solver for the Oven Scheduling Problem."""

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
from discrete_optimization.ovensched.problem import OvenSchedulingSolution
from discrete_optimization.ovensched.solvers.optal import OvenSchedulingOptalSolver
from discrete_optimization.ovensched.utils import (
    plot_attribute_distribution,
    plot_machine_utilization,
    plot_solution,
)

logging.basicConfig(level=logging.INFO)


def run_optal_example():
    """Run a simple example with the OptalCP solver."""

    # Get available data files
    files = get_data_available()

    if not files:
        print("No data files found. Please download instances first.")
        return

    # Use a small instance for demonstration
    file_path = [
        f
        for f in files
        if "65NewRandomOvenSchedulingInstance-n100-k2-a2--2904-17.08.59.dat" in f
    ]
    if file_path:
        file_path = file_path[0]
    else:
        file_path = files[0]

    print(f"Loading instance: {file_path}")

    # Parse the problem
    problem = parse_dat_file(file_path)
    print(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    if hasattr(problem, "additional_data"):
        print(f"Additional data: {problem.additional_data}")

    # Create the OptalCP solver
    print("\nCreating OptalCP solver...")
    solver = OvenSchedulingOptalSolver(problem=problem, setup_modeling="baseline")

    # Initialize the model
    print("Initializing OptalCP model...")
    solver.init_model()

    # Set up solve parameters
    # Optimized configuration based on parameter tuning:
    # - 12 LNS workers with moderate propagation
    # - Enable precedence energy propagation (key improvement!)
    params = ParametersCp.default_cpsat()
    params.nb_process = 12
    import optalcp as cp

    # Moderate propagation works best for this problem size
    lns_worker = cp.WorkerParameters(
        searchType="LNS",
        noOverlapPropagationLevel=2,
        cumulPropagationLevel=2,
    )

    # Solve the problem
    print(f"\nSolving with OptalCP (200s time limit)...")
    result = solver.solve(
        time_limit=200,
        parameters_cp=params,
        workers=[lns_worker] * 12,
        usePrecedenceEnergy=1,  # Enable precedence energy propagation
        # Alternative configurations:
        # - For quick results: preset="Large"
        # - For mixed strategy: workers=[lns]*10 + [settimes]*2
        # - For better bounds: workers=[lns]*8 + [fdsdual]*4
    )

    print(f"Status: {solver.status_solver}")

    # Display results
    if len(result) > 0:
        best_solution: OvenSchedulingSolution = result.get_best_solution()
        print(f"\nFound {len(result)} solutions")
        print(f"Best solution fitness: {result.get_best_solution_fit()}")
        # Use the built-in summary method
        best_solution.print_summary()
        # Demonstrate warm start
        print("\n" + "=" * 80)
        print("Testing warm start feature...")
        print("=" * 80)

        # solver2 = OvenSchedulingOptalSolver(problem=problem)
        # solver2.init_model()
        # solver2.set_warm_start(best_solution)
        #
        # print("Solving again with warm start (30s)...")
        # result2 = solver2.solve(time_limit=30, parameters_cp=params)
        #
        # if len(result2) > 0:
        #     solution2 = result2.get_best_solution()
        #     print(f"Warm start solution fitness: {result2.get_best_solution_fit()}")
        #     comparison = result2.get_best_solution_fit() / result.get_best_solution_fit()
        #     print(f"Improvement ratio: {comparison:.4f}")

        # Visualize the solution
        print("\n" + "=" * 80)
        print("Generating visualizations...")
        print("=" * 80)

        try:
            # Main Gantt chart with batches and setup times
            plot_solution(
                best_solution,
                show_task_ids=True,
                show_setup_times=True,
                title=f"OptalCP Solution - {problem.n_jobs} jobs, {problem.n_machines} machines",
            )

            # Machine utilization analysis
            plot_machine_utilization(best_solution)

            # Attribute distribution
            plot_attribute_distribution(best_solution)

        except ImportError as e:
            print(f"Visualization skipped: {e}")
            print("Install matplotlib to see visualizations: pip install matplotlib")

    else:
        print("No solution found within time limit.")


if __name__ == "__main__":
    run_optal_example()
