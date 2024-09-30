#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

from discrete_optimization.generic_rcpsp_tools.ls_solver import LS_RCPSP_Solver
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt,
    plt,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPSolution
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_mslib_parser import (
    get_data_available,
    parse_file_mslib,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat_msrcpsp_solver import (
    CPSatMSRCPSPSolver,
)

logger = logging.getLogger(__name__)
this_folder = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO)


def example_parsing_and_local_search():
    files_dict = get_data_available()
    file = files_dict["MSLIB1"][0]
    logger.info("file = ", file)
    model = parse_file_mslib(file).to_variant_model()
    logging.basicConfig(level=logging.DEBUG)
    solver = LS_RCPSP_Solver(problem=model)
    result = solver.solve(nb_iteration_max=5000)
    sol, fit = result.get_best_solution_fit()
    plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=sol)
    plt.show()


def example_mslib_cpsat():
    files_dict = get_data_available()
    file = [f for f in files_dict["MSLIB4"] if "MSLIB_Set4_1003.msrcp" in f][0]
    model = parse_file_mslib(file, skill_level_version=False)
    solver = CPSatMSRCPSPSolver(
        problem=model,
    )
    solver.init_model(
        one_worker_per_task=False, slack_skill=False, one_skill_per_task=True
    )
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=20,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    print(solver.status_solver)
    from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
        plot_resource_individual_gantt,
    )

    solution = res.get_best_solution_fit()[0]
    solution: MS_RCPSPSolution
    plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=solution)
    plt.show()


if __name__ == "__main__":
    example_mslib_cpsat()
