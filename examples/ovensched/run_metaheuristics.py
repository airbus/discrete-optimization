#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Run metaheuristics (Hill Climber and Simulated Annealing) on oven scheduling.

This script demonstrates using the discrete-optimization library's built-in
metaheuristics with the vector-based solution encoding.
"""

import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
from discrete_optimization.ovensched.solution_vector import (
    VectorOvenSchedulingSolution,
    generate_random_permutation,
)
from discrete_optimization.ovensched.solvers.mutations import (
    OvenPermutationMixedMutation,
    OvenPermutationSwap,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_hill_climber(problem, initial_solution, time_limit=30):
    """
    Run Hill Climber metaheuristic.

    Hill Climber only accepts improving moves (greedy local search).
    """
    logger.info("=" * 80)
    logger.info("HILL CLIMBER")
    logger.info("=" * 80)

    # Create mutation operator
    mutation = OvenPermutationMixedMutation(
        problem,
        swap_prob=0.4,
        insert_prob=0.3,
        two_opt_prob=0.2,
        shuffle_prob=0.1,
    )

    # Create restart handler (restart after 100 iterations without improvement)
    restart_handler = RestartHandlerLimit(
        nb_iteration_no_improvement=100,
    )

    # Create Hill Climber solver
    solver = HillClimber(
        problem=problem,
        mutator=mutation,
        restart_handler=restart_handler,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=True,  # Store all improving solutions
    )

    # Set initial solution
    solver.set_warm_start(initial_solution)

    # Create callback for time limit
    callbacks = [TimerStopper(total_seconds=time_limit)]

    # Solve
    logger.info(f"Running Hill Climber for {time_limit}s...")
    initial_eval = problem.evaluate(initial_solution)
    logger.info(f"Initial solution: {sum(initial_eval.values()):.0f}")
    logger.info(f"  Processing time: {initial_eval['processing_time']}")
    logger.info(f"  Setup cost: {initial_eval['setup_cost']}")
    logger.info(f"  Late jobs: {initial_eval['nb_late_jobs']}")

    result = solver.solve(nb_iteration_max=100000, callbacks=callbacks)

    # Get best solution
    best_solution = result.get_best_solution()
    best_eval = problem.evaluate(best_solution)

    logger.info(f"\nBest solution found: {sum(best_eval.values()):.0f}")
    logger.info(f"  Processing time: {best_eval['processing_time']}")
    logger.info(f"  Setup cost: {best_eval['setup_cost']}")
    logger.info(f"  Late jobs: {best_eval['nb_late_jobs']}")
    logger.info(f"  Feasible: {problem.satisfy(best_solution)}")
    logger.info(f"  Solutions explored: {len(result)}")

    improvement = sum(initial_eval.values()) - sum(best_eval.values())
    logger.info(
        f"  Improvement: {improvement:.0f} ({100 * improvement / sum(initial_eval.values()):.1f}%)"
    )

    return best_solution, result


def run_simulated_annealing(problem, initial_solution, time_limit=30):
    """
    Run Simulated Annealing metaheuristic.

    Simulated Annealing accepts worsening moves with decreasing probability
    (allows escaping local optima).
    """
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATED ANNEALING")
    logger.info("=" * 80)

    # Create mutation operator
    mutation = OvenPermutationMixedMutation(
        problem,
        swap_prob=0.4,
        insert_prob=0.3,
        two_opt_prob=0.2,
        shuffle_prob=0.1,
    )

    # Create restart handler (restart after 200 iterations without improvement)
    restart_handler = RestartHandlerLimit(
        nb_iteration_no_improvement=200,
    )

    # Create temperature schedule (exponential cooling)
    temperature_handler = TemperatureSchedulingFactor(
        temperature=100.0,  # Initial temperature
        restart_handler=restart_handler,
        coefficient=0.9999,  # Cooling factor (applied each iteration)
    )

    # Create Simulated Annealing solver
    solver = SimulatedAnnealing(
        problem=problem,
        mutator=mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_handler,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=True,
    )

    # Set initial solution
    solver.set_warm_start(initial_solution)

    # Create callback for time limit
    callbacks = [TimerStopper(total_seconds=time_limit)]

    # Solve
    logger.info(f"Running Simulated Annealing for {time_limit}s...")
    initial_eval = problem.evaluate(initial_solution)
    logger.info(f"Initial solution: {sum(initial_eval.values()):.0f}")
    logger.info(f"  Processing time: {initial_eval['processing_time']}")
    logger.info(f"  Setup cost: {initial_eval['setup_cost']}")
    logger.info(f"  Late jobs: {initial_eval['nb_late_jobs']}")

    result = solver.solve(nb_iteration_max=100000, callbacks=callbacks)

    # Get best solution
    best_solution = result.get_best_solution()
    best_eval = problem.evaluate(best_solution)

    logger.info(f"\nBest solution found: {sum(best_eval.values()):.0f}")
    logger.info(f"  Processing time: {best_eval['processing_time']}")
    logger.info(f"  Setup cost: {best_eval['setup_cost']}")
    logger.info(f"  Late jobs: {best_eval['nb_late_jobs']}")
    logger.info(f"  Feasible: {problem.satisfy(best_solution)}")
    logger.info(f"  Solutions explored: {len(result)}")

    improvement = sum(initial_eval.values()) - sum(best_eval.values())
    logger.info(
        f"  Improvement: {improvement:.0f} ({100 * improvement / sum(initial_eval.values()):.1f}%)"
    )

    return best_solution, result


def run_ga(problem, initial_solution, time_limit=30):
    # Create mutation operator
    mutation = OvenPermutationMixedMutation(
        problem,
        swap_prob=0.4,
        insert_prob=0.3,
        two_opt_prob=0.2,
        shuffle_prob=0.1,
    )

    # Create Simulated Annealing solver
    solver = Ga(
        problem=problem,
        # mutation=mutation,
        crossover=DeapCrossover.CX_ONE_POINT,
        selection=DeapSelection.SEL_BEST,
        encoding="permutation",
        pop_size=12,
        deap_verbose=True,
        max_evals=100000,
    )
    # Set initial solution
    solver.set_warm_start(initial_solution)

    # Create callback for time limit
    callbacks = [TimerStopper(total_seconds=time_limit)]
    # Solve
    logger.info(f"Running GA for {time_limit}s...")
    initial_eval = problem.evaluate(initial_solution)
    logger.info(f"Initial solution: {solver.aggreg_from_sol(initial_solution):.0f}")
    logger.info(f"  Processing time: {initial_eval['processing_time']}")
    logger.info(f"  Setup cost: {initial_eval['setup_cost']}")
    logger.info(f"  Late jobs: {initial_eval['nb_late_jobs']}")

    result = solver.solve(callbacks=callbacks)

    # Get best solution
    best_solution = result.get_best_solution()
    best_eval = problem.evaluate(best_solution)

    logger.info(f"\nBest solution found: {solver.aggreg_from_sol(best_solution):.0f}")
    logger.info(f"  Processing time: {best_eval['processing_time']}")
    logger.info(f"  Setup cost: {best_eval['setup_cost']}")
    logger.info(f"  Late jobs: {best_eval['nb_late_jobs']}")
    logger.info(f"  Feasible: {problem.satisfy(best_solution)}")
    logger.info(f"  Solutions explored: {len(result)}")

    improvement = sum(initial_eval.values()) - sum(best_eval.values())
    logger.info(
        f"  Improvement: {improvement:.0f} ({100 * improvement / solver.aggreg_from_sol(best_solution):.1f}%)"
    )
    logger.info(
        f"{solver.aggreg_from_sol(best_solution) / problem.additional_data['ub']} ratio on UB"
    )

    return best_solution, result


def compare_metaheuristics(problem, time_limit=60):
    """
    Compare Hill Climber and Simulated Annealing.
    """
    logger.info("=" * 80)
    logger.info("COMPARING METAHEURISTICS")
    logger.info("=" * 80)

    # Generate initial solution
    logger.info("Generating initial random solution...")
    initial_perm = generate_random_permutation(problem)
    initial_solution = VectorOvenSchedulingSolution(
        problem=problem, permutation=initial_perm
    )

    # Run Hill Climber
    hc_solution, hc_result = run_hill_climber(
        problem, initial_solution.copy(), time_limit=time_limit
    )

    # Run Simulated Annealing
    sa_solution, sa_result = run_simulated_annealing(
        problem, initial_solution.copy(), time_limit=time_limit
    )

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)

    hc_eval = problem.evaluate(hc_solution)
    sa_eval = problem.evaluate(sa_solution)

    logger.info(f"\nHill Climber:")
    logger.info(f"  Total cost: {sum(hc_eval.values()):.0f}")
    logger.info(f"  Solutions explored: {len(hc_result)}")

    logger.info(f"\nSimulated Annealing:")
    logger.info(f"  Total cost: {sum(sa_eval.values()):.0f}")
    logger.info(f"  Solutions explored: {len(sa_result)}")

    if sum(hc_eval.values()) < sum(sa_eval.values()):
        logger.info(f"\nWinner: Hill Climber")
        diff = sum(sa_eval.values()) - sum(hc_eval.values())
        logger.info(
            f"  Better by: {diff:.0f} ({100 * diff / sum(sa_eval.values()):.1f}%)"
        )
    else:
        logger.info(f"\nWinner: Simulated Annealing")
        diff = sum(hc_eval.values()) - sum(sa_eval.values())
        logger.info(
            f"  Better by: {diff:.0f} ({100 * diff / sum(hc_eval.values()):.1f}%)"
        )


def run_simple_hill_climber(problem, time_limit=30):
    """
    Run a simple Hill Climber with just swap mutation.

    This is a minimal example showing the essentials.
    """
    logger.info("=" * 80)
    logger.info("SIMPLE HILL CLIMBER (Swap mutation only)")
    logger.info("=" * 80)

    # Generate initial solution
    initial_perm = generate_random_permutation(problem)
    initial_solution = VectorOvenSchedulingSolution(
        problem=problem, permutation=initial_perm
    )

    # Simple swap mutation
    from discrete_optimization.generic_tools.mutations.mutation_permutation import (
        SwapMutation,
    )

    mutation = SwapMutation(problem=problem, attribute="permutation")
    mutation = OvenPermutationSwap(problem)

    # Restart handler
    restart_handler = RestartHandlerLimit(
        nb_iteration_no_improvement=50,
    )

    # Create solver
    solver = HillClimber(
        problem=problem,
        mutator=mutation,
        restart_handler=restart_handler,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=False,  # Don't store intermediate solutions
    )

    solver.set_warm_start(initial_solution)

    # Solve
    logger.info(f"Running for {time_limit}s...")
    initial_eval = solver.aggreg_from_sol(initial_solution)
    logger.info(f"Initial: {initial_eval:.0f}")

    result = solver.solve(
        nb_iteration_max=100000, callbacks=[TimerStopper(total_seconds=time_limit)]
    )

    best_solution = result.get_best_solution()
    best_eval = problem.evaluate(best_solution)

    logger.info(f"Best: {solver.aggreg_from_sol(best_solution):.0f}")
    logger.info(f"Safisfy: {problem.satisfy(best_solution)}")
    improvement = initial_eval - solver.aggreg_from_sol(best_solution)
    logger.info(f"Improvement: {improvement:.0f}")


def main():
    """Run examples."""
    # Get available data files
    files = get_data_available()

    if not files:
        logger.error("No data files found")
        return

    # Use a small instance for testing
    test_file = None
    for f in files:
        if "87RandomOvenSchedulingInstance-n250-k2-a5--2212-22.47.15.dat" in f:
            test_file = f
            break

    if test_file is None:
        logger.warning("Preferred instance not found, using first available")
        test_file = files[0]

    logger.info(f"Loading instance: {test_file.split('/')[-1]}")
    problem = parse_dat_file(test_file)
    logger.info(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines\n")
    logger.info(problem.additional_data)
    # Run examples
    # run_simulated_annealing(problem, initial_solution=problem.get_dummy_solution(), time_limit=10)
    # run_ga(problem, initial_solution=problem.get_dummy_solution(), time_limit=10)
    run_simple_hill_climber(problem, time_limit=10)
    # compare_metaheuristics(problem, time_limit=30)

    logger.info("\n" + "=" * 80)
    logger.info("ALL EXAMPLES COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
