#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example of using the OptalCP solver for the Oven Scheduling Problem."""

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
from discrete_optimization.ovensched.solvers.optal import OvenSchedulingOptalSolver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def solve_with_optal(
    problem,
    time_limit: int = 200,
    solution: OvenSchedulingSolution = None,
    setup_modeling: str = "baseline",
):
    """Solve with OptalCP solver."""
    solver = OvenSchedulingOptalSolver(problem=problem, setup_modeling=setup_modeling)
    solver.init_model()

    params = ParametersCp.default_cpsat()
    params.nb_process = 12

    import optalcp as cp

    lns_worker = cp.WorkerParameters(
        searchType="LNS",
        noOverlapPropagationLevel=2,
        cumulPropagationLevel=2,
    )
    fds_worker = cp.WorkerParameters(
        searchType="FDS",
        noOverlapPropagationLevel=2,
        cumulPropagationLevel=2,
    )

    fds_dual_worker = cp.WorkerParameters(
        searchType="FDSDual",
        noOverlapPropagationLevel=2,
        cumulPropagationLevel=2,
    )

    if solution:
        if isinstance(solution, VectorOvenSchedulingSolution):
            solution = solution.to_oven_scheduling_solution()

        print(f"Warm-start evaluation: {problem.evaluate(solution)}")
        print(f"Warm-start feasible: {problem.satisfy(solution)}")
        solver.set_warm_start(solution)

    result = solver.solve(
        time_limit=time_limit,
        parameters_cp=params,
        # preset="Default",
        workers=[lns_worker] * 8 + [fds_worker] * 2 + [fds_dual_worker] * 2,
        # usePrecedenceEnergy=1,
    )

    return result, solver


def run_optal_example():
    """Run a simple example with the OptalCP solver."""
    instance_name = "65NewRandomOvenSchedulingInstance-n100-k2-a2--2904-17.08.59.dat"

    print(f"Loading instance: {instance_name}")
    problem = load_problem(instance_name)
    print(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines")
    print(f"Additional data: {problem.additional_data}")

    result, solver = solve_with_optal(
        problem=problem, time_limit=200, setup_modeling="baseline"
    )

    print(f"\nStatus: {solver.status_solver}")

    if len(result) > 0:
        best_solution: OvenSchedulingSolution = result.get_best_solution()
        print(f"\nFound {len(result)} solutions")
        print(f"Best solution fitness: {result.get_best_solution_fit()}")
        best_solution.print_summary()
        visualize_solution(best_solution, "OptalCP Solution")
    else:
        print("No solution found within time limit.")


def run_optal_warmstart():
    """Run OptalCP with warm start from local search."""
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

    # Step 2: OptalCP with warm start
    print("\n" + "=" * 80)
    print("STEP 2: OptalCP with Warm Start")
    print("=" * 80)
    result_optal, solver = solve_with_optal(
        problem=problem, time_limit=200, solution=solution_ls, setup_modeling="baseline"
    )

    print(f"\nStatus: {solver.status_solver}")

    if len(result_optal) > 0:
        best_solution: OvenSchedulingSolution = result_optal.get_best_solution()
        print(f"\nFound {len(result_optal)} solutions")
        print(f"Best solution fitness: {result_optal.get_best_solution_fit()}")
        best_solution.print_summary()

        print_comparison(problem, solution_ls, best_solution, "Local Search", "OptalCP")
        visualize_solution(best_solution, "OptalCP Solution (warm-started)")
    else:
        print("No solution found within time limit.")


if __name__ == "__main__":
    # run_optal_example()
    run_optal_warmstart()
