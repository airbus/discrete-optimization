#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example of using the CP-SAT solver for the Oven Scheduling Problem."""

import logging

from example_utils import (
    load_problem,
    print_comparison,
    solve_with_ls,
    visualize_solution,
)

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.ovensched.problem import OvenSchedulingSolution
from discrete_optimization.ovensched.solution_vector import VectorOvenSchedulingSolution
from discrete_optimization.ovensched.solvers.cpsat import OvenSchedulingCpSatSolver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def solve_with_cpsat(
    problem, time_limit: int = 100, solution: OvenSchedulingSolution = None
):
    """Solve with CP-SAT solver."""
    solver = OvenSchedulingCpSatSolver(problem=problem)

    max_nb_batch = None
    if solution:
        if isinstance(solution, VectorOvenSchedulingSolution):
            solution = solution.to_oven_scheduling_solution()
        max_batches_in_solution = solution.get_max_nb_batch_per_machine()
        max_nb_batch = max(1, int(max_batches_in_solution * 1.2))
        print(
            f"Warm-start has max {max_batches_in_solution} batches/machine, using {max_nb_batch} for model"
        )

    solver.init_model(max_nb_batch_per_machine=max_nb_batch)

    params = ParametersCp.default_cpsat()
    params.nb_process = 12

    if solution:
        solver.set_warm_start(solution)

    result = solver.solve(
        time_limit=time_limit,
        parameters_cp=params,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    return result, solver


def run_cpsat_example():
    """Run a simple example with the CP-SAT solver."""
    instance_name = "50NewRandomOvenSchedulingInstance-n50-k2-a5--2904-15.05.46.dat"

    print(f"Loading instance: {instance_name}")
    problem = load_problem(instance_name)
    print(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    print(f"Additional data: {problem.additional_data}")

    result, solver = solve_with_cpsat(problem=problem, time_limit=100, solution=None)
    print(f"Status: {solver.status_solver}")

    if len(result) > 0:
        best_solution: OvenSchedulingSolution = result.get_best_solution()
        print(f"\nFound {len(result)} solutions")
        print(f"Best solution fitness: {result.get_best_solution_fit()}")
        best_solution.print_summary()
        visualize_solution(best_solution)
    else:
        print("No solution found within time limit.")


def run_cpsat_warmstart():
    """Run CP-SAT with warm start from local search."""
    instance_name = "50NewRandomOvenSchedulingInstance-n50-k2-a5--2904-15.05.46.dat"

    print(f"Loading instance: {instance_name}")
    problem = load_problem(instance_name)
    print(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    print(f"Additional data: {problem.additional_data}")

    # Step 1: Local search
    print("\n" + "=" * 80)
    print("STEP 1: Local Search (Simulated Annealing)")
    print("=" * 80)
    result_ls = solve_with_ls(problem=problem, ls_solver="sa", time_limit=20)
    solution_ls = result_ls.get_best_solution()

    if isinstance(solution_ls, VectorOvenSchedulingSolution):
        solution_ls = solution_ls.to_oven_scheduling_solution()

    print(f"\nLocal search evaluation: {problem.evaluate(solution_ls)}")
    print(f"Feasible: {problem.satisfy(solution_ls)}")

    # Step 2: CP-SAT with warm start
    print("\n" + "=" * 80)
    print("STEP 2: CP-SAT with Warm Start")
    print("=" * 80)
    result, solver = solve_with_cpsat(
        problem=problem, time_limit=500, solution=solution_ls
    )
    print(f"Status: {solver.status_solver}")

    if len(result) > 0:
        best_solution: OvenSchedulingSolution = result.get_best_solution()
        print(f"\nFound {len(result)} solutions")
        print(f"Best solution fitness: {result.get_best_solution_fit()}")
        best_solution.print_summary()

        print_comparison(problem, solution_ls, best_solution, "Local Search", "CP-SAT")
        visualize_solution(best_solution, "CP-SAT Solution (warm-started)")
    else:
        print("No solution found within time limit.")


if __name__ == "__main__":
    # run_cpsat_example()
    run_cpsat_warmstart()
