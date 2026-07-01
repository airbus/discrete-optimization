import logging
import os

from didppy import BeamParallelizationMethod

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
from discrete_optimization.lotsizing.solvers import (
    SimulatedAnnealingLotSizingSolverFast,
)
from discrete_optimization.lotsizing.solvers.dp import (
    DpLotSizingSolver,
    DpSchedLotSizingSolver,
)
from discrete_optimization.lotsizing.solvers.greedy import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)

logging.basicConfig(level=logging.DEBUG)


def run():
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_instance = instance_sizes[0][0]
    largest_size = instance_sizes[0][1]

    print(f"Available instances: {len(instances)}")
    print(f"\nLargest instance: {largest_instance}")
    print(f"File size: {largest_size:,} bytes")

    problem = parse_file(largest_instance)
    greedy = GreedyLotSizingSolver(problem)
    sol = greedy.solve(strategy=GreedyStrategy.JUST_IN_TIME)[-1][0]
    solver = DpLotSizingSolver(
        problem,
        ParamsObjectiveFunction(
            ObjectiveHandling.AGGREGATE,
            objectives=["delays", "stock", "changeover"],
            weights=[1, 1, 1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model(
        relax_delays=True,
        use_lookahead_constraints=False,
        use_flexibility_delays=False,
        flexibility_delta=1,
    )
    solver.set_warm_start(sol)
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        threads=4,
        time_limit=500,
        solver="LNBS",
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_sched():
    instances = get_data_available()
    instance_sizes = [(inst, os.path.getsize(inst)) for inst in instances]
    instance_sizes.sort(key=lambda x: x[1], reverse=True)

    largest_instance = instance_sizes[5][0]
    largest_size = instance_sizes[5][1]
    largest_instance = [inst for inst in instances if "ps-400-10-80" in inst][0]

    print(f"Available instances: {len(instances)}")
    print(f"\nLargest instance: {largest_instance}")
    print(f"File size: {largest_size:,} bytes")

    problem = parse_file(largest_instance)
    # greedy = GreedyLotSizingSolver(problem)
    # sol = greedy.solve(strategy=GreedyStrategy.JUST_IN_TIME)[-1][0]
    solver = DpSchedLotSizingSolver(
        problem,
        ParamsObjectiveFunction(
            ObjectiveHandling.AGGREGATE,
            objectives=["delays", "stock", "changeover"],
            weights=[1, 1, 1],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model(use_precedence_bound=False, use_refined_changeover_bound=False)
    # solver.set_warm_start(sol)
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        threads=4,
        # use_cost_weight=False,
        # keep_all_layers=True,
        # no_bandit=False,
        # no_transition_mutex=False,
        # primal_bound=11000,
        parallelization_method=BeamParallelizationMethod.Hdbs1,
        time_limit=500,
        solver="LNBS",
    )
    sol = res[-1][0]
    print(solver.status_solver)
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_sched_warmstart():
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
    seq_solver = SequentialMetasolver(
        problem=problem,
        list_subbricks=[
            SubBrick(
                SimulatedAnnealingLotSizingSolverFast,
                kwargs=dict(
                    T0=37.0,  # Initial temperature
                    alpha=0.99,  # Cooling rate
                    beta=0.7,  # Insert move probability
                    n_a=12049,  # Moves accepted at each temperature
                    n_s=60240,  # Moves sampled at each temperature
                    max_iterations=5 * 10**6,
                    restart_after_no_improvement=0,
                ),
            ),
            SubBrick(
                DpSchedLotSizingSolver,
                kwargs=dict(
                    callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
                    threads=8,
                    # parallelization_method=BeamParallelizationMethod.Hdbs1,
                    time_limit=500,
                    solver="LNBS",
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
    run_sched()
