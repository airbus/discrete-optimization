#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Common utilities for oven scheduling examples."""

import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    PortfolioMutation,
)
from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
from discrete_optimization.ovensched.problem import (
    OvenSchedulingProblem,
    OvenSchedulingSolution,
)
from discrete_optimization.ovensched.solution_vector import (
    VectorOvenSchedulingSolution,
    generate_random_permutation,
)
from discrete_optimization.ovensched.solvers.mutations import (
    OvenPermutationMixedMutation,
    ScheduleAwareMixedMutation,
)
from discrete_optimization.ovensched.utils import (
    plot_attribute_distribution,
    plot_machine_utilization,
    plot_solution,
)

logger = logging.getLogger(__name__)


def load_problem(instance_name: str) -> OvenSchedulingProblem:
    """Load a problem instance by name.

    Args:
        instance_name: Name or partial name of the instance file

    Returns:
        OvenSchedulingProblem instance
    """
    files = get_data_available()
    file_path = [f for f in files if instance_name in f]
    if file_path:
        return parse_dat_file(file_path[0])
    else:
        raise FileNotFoundError(f"Instance {instance_name} not found")


def solve_with_ls(
    problem: OvenSchedulingProblem, ls_solver: str = "sa", time_limit: int = 20
):
    """Solve with local search (Hill Climber or Simulated Annealing).

    Args:
        problem: Problem instance
        ls_solver: "hc" for Hill Climber or "sa" for Simulated Annealing
        time_limit: Time limit in seconds

    Returns:
        ResultStorage with solutions
    """
    perm = generate_random_permutation(problem)
    initial_solution = VectorOvenSchedulingSolution(problem=problem, permutation=perm)

    # Create mutation portfolio with tuned weights
    mutation = PortfolioMutation(
        problem=problem,
        list_mutations=[
            ScheduleAwareMixedMutation(problem),
            OvenPermutationMixedMutation(problem),
        ],
        weight_mutations=[0.8083973481164611, 1 - 0.8083973481164611],
    )

    restart_handler = RestartHandlerLimit(nb_iteration_no_improvement=485)

    solver = None
    if ls_solver == "hc":
        solver = HillClimber(
            problem=problem,
            mutator=mutation,
            restart_handler=restart_handler,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=False,
        )
    elif ls_solver == "sa":
        temperature_handler = TemperatureSchedulingFactor(
            temperature=0.15673095467235415,
            restart_handler=restart_handler,
            coefficient=0.9999999997,
        )
        solver = SimulatedAnnealing(
            problem=problem,
            mutator=mutation,
            restart_handler=restart_handler,
            temperature_handler=temperature_handler,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=False,
        )

    logger.info(f"Initial cost: {solver.aggreg_from_sol(initial_solution):.0f}")
    solver.set_warm_start(initial_solution)
    logger.info(f"Running {ls_solver.upper()} for {time_limit}s...")

    result = solver.solve(
        nb_iteration_max=10000000,
        callbacks=[TimerStopper(total_seconds=time_limit)],
    )

    best_sol = result.get_best_solution()
    logger.info(f"Final cost: {solver.aggreg_from_sol(best_sol):.0f}")
    return result


def visualize_solution(solution: OvenSchedulingSolution, title: str = None):
    """Generate visualizations for a solution.

    Args:
        solution: Solution to visualize
        title: Optional title for the main plot
    """
    if title is None:
        n_jobs = solution.problem.n_jobs
        n_machines = solution.problem.n_machines
        title = f"Oven Scheduling - {n_jobs} jobs, {n_machines} machines"

    plot_solution(solution, show_task_ids=True, show_setup_times=True, title=title)
    plot_machine_utilization(solution)
    plot_attribute_distribution(solution)


def print_comparison(
    problem: OvenSchedulingProblem,
    warmstart_solution: OvenSchedulingSolution,
    final_solution: OvenSchedulingSolution,
    warmstart_name: str = "Warm-start",
    final_name: str = "Final",
):
    """Print comparison between two solutions.

    Args:
        problem: Problem instance
        warmstart_solution: Initial solution
        final_solution: Final solution
        warmstart_name: Name for the warmstart solution
        final_name: Name for the final solution
    """
    # Evaluate solutions using the problem's evaluation method
    warmstart_eval = problem.evaluate(warmstart_solution)
    final_eval = problem.evaluate(final_solution)

    # Compute aggregated cost manually from objectives and weights
    if hasattr(problem, "additional_data"):
        weight_tard = problem.additional_data.get("weight_tardiness", 1)
        weight_proc = problem.additional_data.get("weight_processing", 1)
        weight_setup = problem.additional_data.get("weight_setup_cost", 1)

        warmstart_cost = (
            warmstart_eval.get("nb_late_jobs", 0) * weight_tard
            + warmstart_eval.get("processing_time", 0) * weight_proc
            + warmstart_eval.get("setup_cost", 0) * weight_setup
        )
        final_cost = (
            final_eval.get("nb_late_jobs", 0) * weight_tard
            + final_eval.get("processing_time", 0) * weight_proc
            + final_eval.get("setup_cost", 0) * weight_setup
        )
    else:
        # Fallback to sum of all objectives
        warmstart_cost = sum(warmstart_eval.values())
        final_cost = sum(final_eval.values())

    improvement = ((warmstart_cost - final_cost) / warmstart_cost) * 100

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{warmstart_name}: {warmstart_eval}")
    print(f"{warmstart_name} cost: {warmstart_cost:.0f}")
    print(f"{final_name}: {final_eval}")
    print(f"{final_name} cost: {final_cost:.0f}")
    print(f"Improvement: {improvement:.2f}%")
