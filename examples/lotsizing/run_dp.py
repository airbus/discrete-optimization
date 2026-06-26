import logging
import os

from didppy import BeamParallelizationMethod

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
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
    largest_instance = [inst for inst in instances if "pigment30b" in inst][0]

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


if __name__ == "__main__":
    run_sched()
