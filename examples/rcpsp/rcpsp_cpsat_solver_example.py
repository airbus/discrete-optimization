#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ObjectiveLogger,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_ressource_view,
    plot_task_gantt,
    plt,
)
from discrete_optimization.rcpsp.solver.cpsat_solver import (
    CPSatRCPSPSolver,
    CPSatRCPSPSolverCumulativeResource,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def solve_makespan_with_cp_sat(problem: RCPSPModel):
    solver = CPSatRCPSPSolver(problem)
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    parameters_cp.nr_solutions = 1
    parameters_cp.nb_process = 8
    result_storage = solver.solve(
        callbacks=[
            ObjectiveLogger(step_verbosity_level=logging.INFO),
            NbIterationTracker(step_verbosity_level=logging.INFO),
        ],
        parameters_cp=parameters_cp,
    )
    solution, fit = result_storage.get_best_solution_fit()
    plot_task_gantt(rcpsp_model=problem, rcpsp_sol=solution, title="Makespan optim")
    plot_ressource_view(
        rcpsp_model=problem, rcpsp_sol=solution, title_figure="Makespan optim"
    )


def solve_resource_with_cp_sat(problem: RCPSPModel):
    solver = CPSatRCPSPSolverCumulativeResource(problem)
    solver.init_model(weight_on_used_resource=100, weight_on_makespan=1)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    parameters_cp.nr_solutions = 1
    parameters_cp.nb_process = 8
    result_storage = solver.solve(
        callbacks=[
            ObjectiveLogger(step_verbosity_level=logging.INFO),
            NbIterationTracker(step_verbosity_level=logging.INFO),
        ],
        parameters_cp=parameters_cp,
    )
    # solution, fit = result_storage.get_best_solution_fit()
    solution, fit = result_storage.list_solution_fits[-1]
    plot_task_gantt(rcpsp_model=problem, rcpsp_sol=solution, title="Resource optim")
    plot_ressource_view(
        rcpsp_model=problem, rcpsp_sol=solution, title_figure="Resource optim"
    )


def cpsat_single_mode_makespan_optimization():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solve_makespan_with_cp_sat(rcpsp_problem)


def cpsat_single_mode_resource_optimization():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solve_resource_with_cp_sat(rcpsp_problem)


def cpsat_single_mode_makespan_optimization_rcp():
    data_folder_rcp = f"{get_data_home()}/rcpsp/RG30/Set 1/"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "Pat8.rcp" in f][0]
    rcpsp_problem = parse_file(file)
    solve_makespan_with_cp_sat(rcpsp_problem)


def cpsat_single_mode_resource_optimization_rcp():
    data_folder_rcp = f"{get_data_home()}/rcpsp/RG30/Set 1/"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "Pat8.rcp" in f][0]
    rcpsp_problem = parse_file(file)
    solve_resource_with_cp_sat(rcpsp_problem)


def cpsat_single_mode_resource_optimization_rcp_sd():
    data_folder_rcp = f"{get_data_home()}/rcpsp/sD"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "sD1.rcp" in f][0]
    rcpsp_problem = parse_file(file)
    solve_resource_with_cp_sat(rcpsp_problem)


if __name__ == "__main__":
    # cpsat_single_mode_resource_optimization_rcp_sd()
    cpsat_single_mode_makespan_optimization()
    cpsat_single_mode_resource_optimization()
    cpsat_single_mode_makespan_optimization_rcp()
    cpsat_single_mode_resource_optimization_rcp()
    plt.show()
