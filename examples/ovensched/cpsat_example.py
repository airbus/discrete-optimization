#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example of using the CP-SAT solver for the Oven Scheduling Problem."""

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
from discrete_optimization.ovensched.problem import OvenSchedulingSolution
from discrete_optimization.ovensched.solvers.cpsat import OvenSchedulingCpSatSolver
from discrete_optimization.ovensched.utils import (
    plot_attribute_distribution,
    plot_machine_utilization,
    plot_solution,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat_example():
    """Run a simple example with the CP-SAT solver."""

    # Get available data files
    files = get_data_available()

    if not files:
        print("No data files found. Please download instances first.")
        return
    # Use the first available instance
    file_path = [
        f
        for f in files
        if "64NewRandomOvenSchedulingInstance-n100-k2-a2--2904-17.04.52.dat" in f
    ][0]
    # file_path = files[0]
    print(f"Loading instance: {file_path}")
    # Parse the problem
    problem = parse_dat_file(file_path)
    print(problem.additional_data)
    print(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    # Create the solver (params_objective_function is automatically constructed from problem)
    solver = OvenSchedulingCpSatSolver(problem=problem)
    # Initialize the model (objective is set automatically using params_objective_function)
    print("Initializing CP-SAT model...")
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    # Solve the problem
    print("Solving with CP-SAT (60s time limit)...")
    result = solver.solve(
        time_limit=100,
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    print("Status : ", solver.status_solver)
    # Display results
    if len(result) > 0:
        best_solution: OvenSchedulingSolution = result.get_best_solution()
        print(f"\nFound {len(result)} solutions")
        print(f"Best solution fitness: {result.get_best_solution_fit()}")
        # Use the built-in summary method
        best_solution.print_summary()

        # Visualize the solution
        print("\nGenerating visualizations...")
        # Main Gantt chart with batches and setup times
        plot_solution(
            best_solution,
            show_task_ids=True,
            show_setup_times=True,
            title=f"Oven Scheduling - {problem.n_jobs} jobs, {problem.n_machines} machines",
        )

        # Machine utilization analysis
        plot_machine_utilization(best_solution)

        # Attribute distribution
        plot_attribute_distribution(best_solution)

    else:
        print("No solution found within time limit.")


if __name__ == "__main__":
    run_cpsat_example()
