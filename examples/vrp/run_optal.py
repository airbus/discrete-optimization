#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ModeOptim, ObjectiveHandling
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.plot import plot_vrp_solution
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers.optal import (
    OptalVrpSolver,
    ParamsObjectiveFunction,
)

logging.basicConfig(level=logging.INFO)


def run_optal_vrp():
    file = [f for f in get_data_available() if "vrp_262_25_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = OptalVrpSolver(
        problem=problem,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["nb_vehicles", "max_length", "length"],
            weights=[100, 1, 100],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model(scale=1)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=20,
    )
    print(solver.status_solver)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    plot_vrp_solution(vrp_problem=problem, solution=sol)
    plt.show()


if __name__ == "__main__":
    run_optal_vrp()
