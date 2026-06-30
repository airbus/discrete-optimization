import logging
import os

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
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


def run_scheduling_warm_start():
    """Run MILP solver on a lot sizing instance."""
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1])

    print(f"Available instances: {len(instances)}")

    # Use a medium-hard instance (not the largest to keep it manageable)
    # instance_file = [inst for inst in instances if "PSP_100_1" in inst][0]
    instance_file = [inst for inst in instances if "PSP_200_3" in inst][0]
    print(f"\nInstance: {instance_file}")
    problem = parse_file(instance_file)
    params = ParametersCp.default_cpsat()
    params.nb_process = 12
    seq_solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[
            SubBrick(
                SimulatedAnnealingLotSizingSolverFast,
                kwargs=dict(
                    T0=37.0,  # Initial temperature
                    alpha=0.999,  # Cooling rate
                    beta=0.7,  # Insert move probability
                    n_a=12049,  # Moves accepted at each temperature
                    n_s=60240,  # Moves sampled at each temperature
                    max_iterations=5 * 10**7,
                    restart_after_no_improvement=0,
                ),
            ),
            SubBrick(
                CpSatSchedLotSizingSolver,
                kwargs=dict(
                    callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
                    parameters_cp=params,
                    time_limit=1000,
                    ortools_cpsat_solver_kwargs={"log_search_progress": True},
                ),
            ),
        ],
    )
    result = seq_solver.solve()
    # Results
    sol = result[-1][0]
    fitness = result[-1][1]

    print(f"\nSolution found:")
    print(f"  - Fitness: {fitness}")
    print(f"  - Evaluation: {problem.evaluate(sol)}")
    print(f"  - Satisfies constraints: {problem.satisfy(sol)}")

    if problem.known_bound is not None:
        gap = (fitness - problem.known_bound) / problem.known_bound * 100
        print(f"  - Gap to known bound: {gap:.2f}%")

    return sol


if __name__ == "__main__":
    run_scheduling_warm_start()
