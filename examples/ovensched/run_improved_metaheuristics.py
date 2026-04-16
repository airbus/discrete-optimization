#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Test metaheuristics with the improved decoder (multiple open batches).

This demonstrates the impact of max_open_batches parameter on solution quality.
"""

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
from discrete_optimization.ovensched.parser import get_data_available, parse_dat_file
from discrete_optimization.ovensched.solution_vector import (
    VectorOvenSchedulingSolution,
    generate_random_permutation,
)
from discrete_optimization.ovensched.solvers.mutations import (
    OvenPermutationMixedMutation,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_decoder_impact_on_metaheuristic(problem, ls_solver: str = "hc", time_limit=20):
    """
    Compare Hill Climber performance with different max_open_batches values.
    """
    logger.info("=" * 80)
    logger.info("METAHEURISTIC WITH DIFFERENT DECODER CONFIGURATIONS")
    logger.info("=" * 80)

    configs = [1, 2, 3, 5, 10]

    for max_open in configs:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing max_open_batches = {max_open}")
        logger.info(f"{'=' * 80}")

        # Set the class-level default
        VectorOvenSchedulingSolution.DEFAULT_MAX_OPEN_BATCHES = max_open

        # Generate initial solution
        perm = generate_random_permutation(problem)
        initial_solution = VectorOvenSchedulingSolution(
            problem=problem, permutation=perm
        )

        initial_eval = initial_solution.evaluate()

        # Create mutation and solver
        mutation = OvenPermutationMixedMutation(problem)
        restart_handler = RestartHandlerLimit(nb_iteration_no_improvement=50)
        solver = None
        if ls_solver == "hc":
            solver = HillClimber(
                problem=problem,
                mutator=mutation,
                restart_handler=restart_handler,
                mode_mutation=ModeMutation.MUTATE,
                store_solution=False,
            )
        if ls_solver == "sa":
            temperature_handler = TemperatureSchedulingFactor(
                temperature=100.0,  # Initial temperature
                restart_handler=restart_handler,
                coefficient=0.9999,  # Cooling factor (applied each iteration)
            )
            solver = SimulatedAnnealing(
                problem=problem,
                mutator=mutation,
                restart_handler=restart_handler,
                temperature_handler=temperature_handler,
                mode_mutation=ModeMutation.MUTATE,
                store_solution=False,
            )
        initial_cost = solver.aggreg_from_sol(initial_solution)
        logger.info(f"Initial solution cost: {initial_cost:.0f}")
        logger.info(f"  Processing time: {initial_eval['processing_time']}")
        logger.info(f"  Setup cost: {initial_eval['setup_cost']}")
        logger.info(f"  Late jobs: {initial_eval['nb_late_jobs']}")
        solver.set_warm_start(initial_solution)

        # Solve
        logger.info(f"Running Hill Climber for {time_limit}s...")
        result = solver.solve(
            nb_iteration_max=100000, callbacks=[TimerStopper(total_seconds=time_limit)]
        )

        best_solution = result.get_best_solution()
        best_eval = problem.evaluate(best_solution)
        best_cost = solver.aggreg_from_sol(best_solution)
        logger.info(f"\nBest solution cost: {best_cost:.0f}")
        logger.info(f"  Processing time: {best_eval['processing_time']}")
        logger.info(f"  Setup cost: {best_eval['setup_cost']}")
        logger.info(f"  Late jobs: {best_eval['nb_late_jobs']}")
        logger.info(f"  Feasible: {problem.satisfy(best_solution)}")

        improvement = initial_cost - best_cost
        logger.info(
            f"\nImprovement: {improvement:.0f} ({100 * improvement / initial_cost:.1f}%)"
        )


def compare_initial_solutions(problem, n_trials=20):
    """
    Compare initial solution quality with different decoder configs.
    """
    logger.info("\n" + "=" * 80)
    logger.info("INITIAL SOLUTION QUALITY COMPARISON")
    logger.info("=" * 80)

    configs = [1, 2, 3, 5, 10]

    results = {}
    for max_open in configs:
        VectorOvenSchedulingSolution.DEFAULT_MAX_OPEN_BATCHES = max_open

        costs = []
        for _ in range(n_trials):
            perm = generate_random_permutation(problem)
            solution = VectorOvenSchedulingSolution(problem=problem, permutation=perm)
            evaluation = solution.evaluate()
            costs.append(sum(evaluation.values()))

        import numpy as np

        results[max_open] = {
            "mean": np.mean(costs),
            "std": np.std(costs),
            "min": np.min(costs),
        }

        logger.info(
            f"max_open={max_open:2d}: mean={results[max_open]['mean']:7.0f}, "
            f"std={results[max_open]['std']:6.0f}, "
            f"min={results[max_open]['min']:7.0f}"
        )

    best_config = min(configs, key=lambda x: results[x]["mean"])
    logger.info(f"\nBest: max_open={best_config}")
    logger.info(
        f"Improvement vs max_open=1: "
        f"{results[1]['mean'] - results[best_config]['mean']:.0f} "
        f"({100 * (results[1]['mean'] - results[best_config]['mean']) / results[1]['mean']:.1f}%)"
    )


def main():
    """Run tests."""
    # Get available data files
    files = get_data_available()

    if not files:
        logger.error("No data files found")
        return

    # Use a small instance for testing
    test_file = None
    for f in files:
        if "64NewRandomOvenSchedulingInstance-n100-k2-a2" in f:
            test_file = f
            break

    if test_file is None:
        logger.warning("Preferred instance not found, using first available")
        test_file = files[0]

    logger.info(f"Loading instance: {test_file.split('/')[-1]}\n")
    problem = parse_dat_file(test_file)
    logger.info(f"Problem: {problem.n_jobs} jobs, {problem.n_machines} machines\n")

    # Run tests
    compare_initial_solutions(problem, n_trials=20)
    run_decoder_impact_on_metaheuristic(problem, ls_solver="sa", time_limit=15)

    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
