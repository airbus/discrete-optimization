import logging
import os

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.lotsizing.parser import get_data_available, parse_file
from discrete_optimization.lotsizing.solvers.optal import OptalSchedLotSizingSolver

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
    solver = OptalSchedLotSizingSolver(problem)
    solver.init_model()
    # solver.set_warm_start_from_previous_run(s)
    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 4
    res = solver.solve(
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        parameters_cp=params_cp,
        time_limit=1000,
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run()
