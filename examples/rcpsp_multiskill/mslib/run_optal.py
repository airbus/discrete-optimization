#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

from discrete_optimization.generic_rcpsp_tools.solvers.ls import LsGenericRcpspSolver
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_multiskill.parser_mslib import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt,
    plt,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspSolution
from discrete_optimization.rcpsp_multiskill.solvers.optal import OptalMSRcpspSolver

logger = logging.getLogger(__name__)
this_folder = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.DEBUG)


def example_mslib_optal():
    files_dict = get_data_available()
    file = [f for f in files_dict["MSLIB4"] if "MSLIB_Set4_1003.msrcp" in f][0]
    model = parse_file(file, skill_level_version=False)
    solver = OptalMSRcpspSolver(
        problem=model,
    )
    solver.init_model(one_skill_used_per_worker=False, one_worker_per_task=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=20,
        **{
            "worker0-1.searchType": "fds",
            "worker0-1.noOverlapPropagationLevel": 4,
            "worker0-1.cumulPropagationLevel": 3,
        },
    )
    print(solver.status_solver)
    from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
        plot_resource_individual_gantt,
    )

    solution = res.get_best_solution_fit()[0]
    print(model.satisfy(solution), model.evaluate(solution))
    solution: MultiskillRcpspSolution
    plot_resource_individual_gantt(rcpsp_problem=model, rcpsp_sol=solution)
    plt.show()


if __name__ == "__main__":
    example_mslib_optal()
