#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os

from discrete_optimization.datasets import get_data_home

os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
    plt,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import LP_RCPSP_CPLEX


def run_rcpsp_sm_lp_cplex():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    solver = LP_RCPSP_CPLEX(problem=rcpsp_problem)
    solver.init_model(greedy_start=True)
    parameters_milp = ParametersMilp.default()
    parameters_milp.time_limit = 10000
    results_storage: ResultStorage = solver.solve(parameters_milp=parameters_milp)
    solution, fit = results_storage.get_best_solution_fit()
    print(results_storage.list_solution_fits)
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


def run_rcpsp_lp_cplex():
    data_folder_rcp = f"{get_data_home()}/rcpsp/RG300/"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "RG300_190.rcp" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    solver = LP_RCPSP_CPLEX(problem=rcpsp_problem)
    solver.init_model(greedy_start=False, max_horizon=370)
    parameters_milp = ParametersMilp.default()
    parameters_milp.time_limit = 10000
    results_storage: ResultStorage = solver.solve(parameters_milp=parameters_milp)
    solution, fit = results_storage.get_best_solution_fit()
    print(results_storage.list_solution_fits)
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


if __name__ == "__main__":
    run_rcpsp_lp_cplex()
