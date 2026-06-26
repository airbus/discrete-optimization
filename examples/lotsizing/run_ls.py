import logging
import os

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
from discrete_optimization.lotsizing.solvers.ls import (
    LocalSearchAlgo,
    LSLotSizingSolver,
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

    problem = parse_file(largest_instance)
    solver = LSLotSizingSolver(problem)
    res = solver.solve(
        nb_iteration_max=100000,
        solver=LocalSearchAlgo.SA,
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run()
