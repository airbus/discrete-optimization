import logging
import os

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
from discrete_optimization.lotsizing.solvers.cpsat import (
    ChangeoverModel,
    CpSatLotSizingSolver,
    CpSatSchedLotSizingSolver,
)
from discrete_optimization.lotsizing.solvers.greedy import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)
from discrete_optimization.lotsizing.solvers.sa_fast import (
    SimulatedAnnealingLotSizingSolverFast,
)

logging.basicConfig(level=logging.INFO)


def run():
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_instance = instance_sizes[0][0]
    largest_size = instance_sizes[0][1]

    print(f"Available instances: {len(instances)}")
    print(f"\nLargest instance: {largest_instance}")
    print(f"File size: {largest_size:,} bytes")
    largest_instance = [ist for ist in instances if "PSP_100_4" in ist][0]
    problem = parse_file(largest_instance)
    sol = GreedyLotSizingSolver(problem).solve(
        strategy=GreedyStrategy.EARLIEST_DEMAND_FIRST
    )[-1][0]
    solver = CpSatLotSizingSolver(problem)
    solver.init_model(changeover_model=ChangeoverModel.STATE_BASED)
    # solver.set_warm_start_from_previous_run(s)
    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 16
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        parameters_cp=params_cp,
        time_limit=1000,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_scheduling():
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_instance = instance_sizes[0][0]
    largest_size = instance_sizes[0][1]
    largest_instance = [inst for inst in instances if "PSP_100_1" in inst][0]
    print(f"Available instances: {len(instances)}")
    print(f"\nLargest instance: {largest_instance}")
    print(f"File size: {largest_size:,} bytes")
    # largest_instance = [ist for ist in instances
    #                    if "PSP_100_4" in ist][0]
    problem = parse_file(largest_instance)
    solver = CpSatSchedLotSizingSolver(problem)
    solver.init_model()
    # solver.set_warm_start_from_previous_run(s)
    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 10
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        parameters_cp=params_cp,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
        time_limit=1000,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_cpsat():
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_instance = instance_sizes[0][0]
    largest_size = instance_sizes[0][1]
    largest_instance = [inst for inst in instances if "PSP_100_1" in inst][0]
    print(f"Available instances: {len(instances)}")
    print(f"\nLargest instance: {largest_instance}")
    print(f"File size: {largest_size:,} bytes")
    # largest_instance = [ist for ist in instances
    #                    if "PSP_100_4" in ist][0]
    problem = parse_file(largest_instance)
    greedy = GreedyLotSizingSolver(problem)
    greedy_result = greedy.solve(strategy=GreedyStrategy.BALANCED)
    greedy_fitness = greedy_result[0][1]
    print(f"  Greedy fitness: {greedy_fitness:.2f}")

    # Create fast solver
    print("\nCreating fast SA solver (numpy/numba)...")
    solver = SimulatedAnnealingLotSizingSolverFast(
        problem,
        T0=37.0,  # Initial temperature
        alpha=0.99,  # Cooling rate
        beta=0.7,  # Insert move probability
        n_a=12049,  # Moves accepted at each temperature
        n_s=60240,  # Moves sampled at each temperature
        max_iterations=4 * 10**7,
        restart_after_no_improvement=0,
    )

    # Run with live logging (shows progress every 10k iterations)
    print("\nRunning SA with live progress logging...")
    print(
        "Algorithm: GPI mutations (70% INSERT, 30% SWAP) - matches Ceschia et al. 2017"
    )
    print("Cooling: Threshold-based (T × 0.99 when sampled≥60,240 OR accepted≥12,049)")
    print()

    result = solver.solve(
        log_interval=max(10000, 3 * 10**7 // 10000)
    )  # Log every 10k or 1% of total

    solver = CpSatSchedLotSizingSolver(problem)
    solver.init_model()
    solver.set_warm_start(result[-1][0])
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        parameters_cp=p,
        time_limit=1000,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )


if __name__ == "__main__":
    run_cpsat()
