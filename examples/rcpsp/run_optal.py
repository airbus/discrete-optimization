#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solvers.optal import OptalRcpspSolver
from discrete_optimization.rcpsp.utils import plot_ressource_view, plot_task_gantt, plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def solve_makespan_with_optal(problem: RcpspProblem):
    import optalcp as cp

    solver = OptalRcpspSolver(problem)
    solver.init_model()
    parameters_cp = ParametersCp.default()
    parameters_cp.nb_process = 8
    fds = cp.WorkerParameters(
        searchType="FDS", noOverlapPropagationLevel=4, cumulPropagationLevel=2
    )
    fds_dual = cp.WorkerParameters(
        searchType="FDSDual", noOverlapPropagationLevel=4, cumulPropagationLevel=2
    )
    result_storage = solver.solve(
        parameters_cp=parameters_cp,
        time_limit=2,
        workers=[fds, fds_dual],
    )
    solution, fit = result_storage.get_best_solution_fit()
    assert problem.satisfy(solution)
    plot_task_gantt(rcpsp_problem=problem, rcpsp_sol=solution, title="Makespan optim")
    plot_ressource_view(
        rcpsp_problem=problem, rcpsp_sol=solution, title_figure="Makespan optim"
    )


def optal_single_mode_makespan_optimization():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_7.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solve_makespan_with_optal(rcpsp_problem)
    plt.show()


def optal_multi_mode_makespan_optimization():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_2.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solve_makespan_with_optal(rcpsp_problem)
    plt.show()


if __name__ == "__main__":
    optal_single_mode_makespan_optimization()
    optal_multi_mode_makespan_optimization()
