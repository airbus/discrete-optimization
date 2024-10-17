#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random

import numpy as np

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import (
    CpSatCumulativeResourceRcpspSolver,
    CpSatRcpspSolver,
    CpSatResourceRcpspSolver,
)

logging.basicConfig(level=logging.DEBUG)


def run_ortools_resource_optim(objectives):
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    subsolver = CpSatCumulativeResourceRcpspSolver(problem=rcpsp_problem)

    solver = LexicoSolver(
        problem=rcpsp_problem,
        subsolver=subsolver,
    )
    solver.init_model()
    parameters_cp = ParametersCp.default_cpsat()

    result_storage = solver.solve(
        subsolver_callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=parameters_cp,
        time_limit=5,
        objectives=objectives,
    )
    print([sol._internal_objectives for sol, fit in result_storage.list_solution_fits])
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "tab:blue"
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("KPI 1", color=color1)
    obj_array_0 = np.array(
        [
            sol._internal_objectives[objectives[0]]
            for sol, fit in result_storage.list_solution_fits
        ]
    )
    obj_array_1 = np.array(
        [
            sol._internal_objectives[objectives[1]]
            for sol, fit in result_storage.list_solution_fits
        ]
    )

    ax1.plot(obj_array_0, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create the second y-axis and plot the second KPI
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("KPI 2", color=color2)
    ax2.plot(obj_array_1, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    print(len(obj_array_1))
    print(len(result_storage.list_solution_fits))
    plt.show()


if __name__ == "__main__":
    run_ortools_resource_optim(["makespan", "used_resource"])
    run_ortools_resource_optim(["used_resource", "makespan"])
