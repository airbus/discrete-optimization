#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import numpy as np

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.did_rcpsp_solver import (
    DidRCPSPModeling,
    DidRCPSPSolver,
    dp,
)

logging.basicConfig(level=logging.INFO)


def run_did_rcpsp():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    data_folder_rcp = f"{get_data_home()}/rcpsp/RG30/Set 1/"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "Pat8.rcp" in f][0]
    # rcpsp_problem.horizon = 55
    solver = DidRCPSPSolver(problem=rcpsp_problem)
    solver.init_model_multimode()
    res = solver.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=30,
        solver=dp.LNBS,
        use_callback=True,
        threads=10,
    )
    sol, fit = res.get_best_solution_fit()
    print(rcpsp_problem.evaluate(sol))
    print(rcpsp_problem.satisfy(sol))


def run_did_rcpsp_calendar():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    rcpsp_problem.horizon = 180
    for resource in rcpsp_problem.resources:
        rcpsp_problem.resources[resource] = np.array(
            rcpsp_problem.get_resource_availability_array(resource)
        )
        rcpsp_problem.resources[resource][10:15] = 5
        rcpsp_problem.resources[resource][30:35] = 5
        rcpsp_problem.resources[resource][45:55] = 3
        rcpsp_problem.resources[resource][65:80] = 2
    rcpsp_problem.is_calendar = True
    rcpsp_problem.update_functions()
    solver = DidRCPSPSolver(problem=rcpsp_problem)
    solver.init_model(modeling=DidRCPSPModeling.TASK_AND_TIME)
    # solver.init_model(modeling=DidRCPSPModeling.TASK_MULTIMODE, dual_bound=False)
    res = solver.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=10,
        solver=dp.CABS,
        use_callback=False,
        threads=5,
    )
    sol, fit = res.get_best_solution_fit()
    print(rcpsp_problem.evaluate(sol))
    print(rcpsp_problem.satisfy(sol))
    from discrete_optimization.rcpsp.rcpsp_utils import (
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_ressource_view(rcpsp_model=rcpsp_problem, rcpsp_sol=sol)
    plot_task_gantt(rcpsp_model=rcpsp_problem, rcpsp_sol=sol)
    plt.show()


def run_did_mrcpsp():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_10.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = DidRCPSPSolver(problem=rcpsp_problem)
    solver.init_model_multimode()
    res = solver.solve(time_limit=30, solver=dp.LNBS, threads=10)
    sol, fit = res.get_best_solution_fit()
    print(rcpsp_problem.evaluate(sol))
    print(rcpsp_problem.satisfy(sol))


if __name__ == "__main__":
    run_did_rcpsp_calendar()
