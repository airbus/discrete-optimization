"""Generic SA solver with threshold-based cooling and GPI mutations.

This demonstrates the new TemperatureSchedulingThresholdBased scheduler
combined with GPI mutations, matching the custom SA's behavior but with
the flexibility of the generic framework.

Example usage:
    python run_sa_generic.py
"""

import logging
import os

import numpy as np

from discrete_optimization.generic_tools.callbacks.loggers import (
    ProblemEvaluateLogger,
)
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingThresholdBased,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    PortfolioMutation,
)
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
from discrete_optimization.lotsizing.solvers.greedy import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)
from discrete_optimization.lotsizing.solvers.mutation import GPIMixedMutation

logging.basicConfig(level=logging.INFO)


def run_threshold_based(instance_name="PSP_100_1", max_iterations=100000):
    """Run generic SA with threshold-based cooling on a hard instance.

    Args:
        instance_name: Instance pattern to match (e.g., "PSP_100_1", "PSP_100_4")
        max_iterations: Maximum iterations for SA

    This configuration matches the paper's custom SA:
    - T0 = 37.0
    - alpha = 0.99
    - n_s = 60,240
    - n_a = 12,049
    - GPI mutations (70% INSERT, 30% SWAP)
    """
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1], reverse=True)

    # Select instance
    instance_file = [inst for inst in instances if instance_name in inst][0]

    print("=" * 70)
    print("GENERIC SA - THRESHOLD-BASED COOLING + GPI MUTATIONS")
    print("=" * 70)
    print(f"Instance: {instance_file}")
    print(f"File size: {os.path.getsize(instance_file):,} bytes")

    # Parse problem
    problem = parse_file(instance_file)
    print(f"\nProblem:")
    print(f"  Items: {problem.nb_items_type}")
    print(f"  Horizon: {problem.horizon}")
    print(f"  Total demands: {sum(problem.total_demands_per_item.values())}")
    if problem.known_bound is not None:
        print(f"  Known bound: {problem.known_bound}")

    # Generate initial solution with greedy
    print(f"\nGenerating initial solution with greedy...")
    greedy_solver = GreedyLotSizingSolver(problem)
    greedy_result = greedy_solver.solve(strategy=GreedyStrategy.BALANCED)
    init_sol = greedy_result[0][0]
    init_fitness = greedy_result[0][1]

    print(f"  Greedy fitness: {init_fitness:.2f}")
    print(f"  Greedy valid: {problem.satisfy(init_sol)}")

    # Create GPI mutation (matches paper's INSERT 70%, SWAP 30%)
    print(f"\nConfiguring GPI mutations...")
    gpi_mutation = GPIMixedMutation.build(
        problem=problem,
        attribute="list_item_per_time",
        beta=0.7,  # 70% INSERT, 30% SWAP (paper's value)
    )
    mutation = PortfolioMutation(
        problem=problem,
        list_mutations=[gpi_mutation],
        weight_mutations=np.array([1.0]),
    )
    print(f"  Mutation: GPIMixedMutation (70% INSERT, 30% SWAP)")

    # Create threshold-based temperature scheduler (paper's parameters)
    print(f"\nConfiguring temperature scheduler...")
    restart_handler = RestartHandlerLimit(2000)
    temperature_scheduler = TemperatureSchedulingThresholdBased(
        initial_temperature=37.0,  # T0 from paper
        restart_handler=restart_handler,
        cooling_factor=0.99,  # alpha from paper
        n_moves_sampled_before_cooling=60240,  # n_s from paper
        n_moves_accepted_before_cooling=12049,  # n_a from paper
    )
    print(f"  Initial temperature (T0): 37.0")
    print(f"  Cooling factor (alpha): 0.99")
    print(f"  Moves sampled threshold (n_s): 60,240")
    print(f"  Moves accepted threshold (n_a): 12,049")

    # Create SA solver
    print(f"\nConfiguring SA solver...")
    solver = SimulatedAnnealing(
        problem=problem,
        mutator=mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_scheduler,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=False,  # Don't store all solutions (performance)
    )
    print(f"  Max iterations: {max_iterations:,}")

    # Solve
    print(f"\nSolving...")
    result = solver.solve(
        initial_variable=init_sol,
        nb_iteration_max=max_iterations,
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
    )

    # Results
    final_sol = result[-1][0]
    final_fitness = result[-1][1]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nSolution quality:")
    print(f"  Final fitness: {final_fitness:.2f}")
    print(f"  Greedy fitness: {init_fitness:.2f}")
    print(
        f"  Improvement: {init_fitness - final_fitness:.2f} ({(init_fitness - final_fitness) / init_fitness * 100:.1f}%)"
    )

    print(f"\nConstraint satisfaction:")
    print(f"  Valid: {problem.satisfy(final_sol)}")
    eval_dict = problem.evaluate(final_sol)
    print(f"  Evaluation: {eval_dict}")

    if problem.known_bound is not None:
        gap = (final_fitness - problem.known_bound) / problem.known_bound * 100
        print(f"\nGap to known bound:")
        print(f"  Known bound: {problem.known_bound}")
        print(f"  Gap: {gap:.2f}%")

    print(f"\nSolutions found: {len(result)}")

    # Temperature scheduler statistics
    stats = temperature_scheduler.get_statistics()
    print(f"\nCooling statistics:")
    print(f"  Initial temperature: {stats['initial_temperature']:.2f}")
    print(f"  Final temperature: {stats['current_temperature']:.2f}")
    print(f"  Cooling events: {stats['n_cooling_events']}")
    print(
        f"  Avg iterations per cooling: {max_iterations / max(1, stats['n_cooling_events']):.0f}"
    )

    return final_sol, final_fitness


def run_comparison(instance_name="PSP_100_1", max_iterations=50000):
    """Compare threshold-based vs per-iteration cooling on same instance.

    Args:
        instance_name: Instance to test
        max_iterations: Iterations for both approaches
    """
    from discrete_optimization.generic_tools.ls.simulated_annealing import (
        TemperatureSchedulingFactor,
    )

    instances = get_data_available()
    instance_file = [inst for inst in instances if instance_name in inst][0]
    problem = parse_file(instance_file)

    print("=" * 70)
    print("COMPARISON: THRESHOLD-BASED vs PER-ITERATION COOLING")
    print("=" * 70)
    print(f"Instance: {instance_file}")
    print(f"Iterations: {max_iterations:,}\n")

    # Generate initial solution
    greedy_solver = GreedyLotSizingSolver(problem)
    init_sol = greedy_solver.solve(strategy=GreedyStrategy.BALANCED)[0][0]

    # Create GPI mutation
    gpi_mutation = GPIMixedMutation.build(
        problem=problem, attribute="list_item_per_time", beta=0.7
    )
    mutation = PortfolioMutation(
        problem=problem, list_mutations=[gpi_mutation], weight_mutations=np.array([1.0])
    )

    results = {}

    # Test 1: Threshold-based (paper's approach)
    print("1. Threshold-based cooling (paper's approach)...")
    rs1 = RestartHandlerLimit(100)
    scheduler1 = TemperatureSchedulingThresholdBased(
        initial_temperature=37.0,
        restart_handler=rs1,
        cooling_factor=0.99,
        n_moves_sampled_before_cooling=60240,
        n_moves_accepted_before_cooling=12049,
    )
    solver1 = SimulatedAnnealing(
        problem=problem,
        mutator=mutation,
        restart_handler=rs1,
        temperature_handler=scheduler1,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=False,
    )
    result1 = solver1.solve(initial_variable=init_sol, nb_iteration_max=max_iterations)
    fitness1 = result1[-1][1]
    stats1 = scheduler1.get_statistics()
    results["Threshold-based"] = {
        "fitness": fitness1,
        "valid": problem.satisfy(result1[-1][0]),
        "final_temp": stats1["current_temperature"],
        "cooling_events": stats1["n_cooling_events"],
    }
    print(f"   Fitness: {fitness1:.2f}, Cooling events: {stats1['n_cooling_events']}")

    # Test 2: Per-iteration (old generic approach)
    print("\n2. Per-iteration cooling (old generic approach)...")
    rs2 = RestartHandlerLimit(100)
    scheduler2 = TemperatureSchedulingFactor(
        initial_temperature=37.0, restart_handler=rs2, cooling_factor=0.99999
    )
    solver2 = SimulatedAnnealing(
        problem=problem,
        mutator=mutation,
        restart_handler=rs2,
        temperature_handler=scheduler2,
        mode_mutation=ModeMutation.MUTATE,
        store_solution=False,
    )
    result2 = solver2.solve(initial_variable=init_sol, nb_iteration_max=max_iterations)
    fitness2 = result2[-1][1]
    results["Per-iteration"] = {
        "fitness": fitness2,
        "valid": problem.satisfy(result2[-1][0]),
        "final_temp": scheduler2.temperature,
        "cooling_events": max_iterations,  # Every iteration
    }
    print(f"   Fitness: {fitness2:.2f}, Final temp: {scheduler2.temperature:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Approach':<25} {'Fitness':>12} {'Valid':>8} {'Final T':>10} {'Cool Events':>12}"
    )
    print("-" * 70)
    for name, res in results.items():
        valid_str = "✓" if res["valid"] else "✗"
        print(
            f"{name:<25} {res['fitness']:>12.2f} {valid_str:>8} {res['final_temp']:>10.2f} {res['cooling_events']:>12}"
        )

    print(
        f"\nThreshold-based stays warmer: {stats1['current_temperature']:.2f}° vs {scheduler2.temperature:.2f}°"
    )
    print(f"Quality difference: {abs(fitness1 - fitness2):.2f} points")


if __name__ == "__main__":
    # Run on hard instance with threshold-based cooling
    run_threshold_based(instance_name="PSP_100_1", max_iterations=1000000)

    # Uncomment to compare cooling approaches
    # run_comparison(instance_name="PSP_100_1", max_iterations=50000)
