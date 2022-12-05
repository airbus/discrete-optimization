#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

from discrete_optimization.rcpsp.solver.ls_solver import LS_RCPSP_Solver
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt,
    plt,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_mslib_parser import (
    get_data_available,
    parse_file_mslib,
)

logger = logging.getLogger(__name__)
this_folder = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.DEBUG)


def example_parsing_and_local_search():
    files_dict = get_data_available()
    file = files_dict["MSLIB1"][0]
    logger.info("file = ", file)
    model = parse_file_mslib(file).to_variant_model()
    logging.basicConfig(level=logging.DEBUG)
    solver = LS_RCPSP_Solver(model=model)
    result = solver.solve(nb_iteration_max=5000)
    sol, fit = result.get_best_solution_fit()
    plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=sol)
    plt.show()


if __name__ == "__main__":
    example_parsing_and_local_search()
