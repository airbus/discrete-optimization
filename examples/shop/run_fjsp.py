#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os

os.environ["HX_LICENSE_PATH"] = "/Users/poveda_g/Documents/hexaly/licence.dat"

import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.shop.fjsp.parser import get_data_available, parse_file
from discrete_optimization.shop.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.shop.solvers.cpmpy import CpmpyShopSolver
from discrete_optimization.shop.solvers.greedy import GreedyShopSolver
from discrete_optimization.shop.utils import plot_shop_solution, plt

logging.basicConfig(level=logging.INFO)


def run_cpsat_fjsp():
    file_path = get_data_available()[4]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    print(
        "Problem with ",
        problem.n_jobs,
        " jobs, ",
        problem.n_all_jobs,
        " subjobs, and ",
        problem.n_machines,
        " machines",
    )
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=10,
    )
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


def run_cpsat_fjsp_warm_start():
    file_path = get_data_available()[4]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    greedy_solver = GreedyShopSolver(problem=problem)
    res = greedy_solver.solve()
    sol = res[-1][0]
    print(
        "Satisfy greedy", problem.satisfy(sol), "Evaluate greedy", problem.evaluate(sol)
    )
    solver = CpSatFjspSolver(problem=problem)
    solver.init_model()
    solver.set_warm_start(sol)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=10,
    )
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


def run_cpmpy_fjsp():
    file_path = get_data_available()[4]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    print(
        "Problem with ",
        problem.n_jobs,
        " jobs, ",
        problem.n_all_jobs,
        " subjobs, and ",
        problem.n_machines,
        " machines",
    )
    solver = CpmpyShopSolver(problem=problem, solver_name="ortools")
    solver.init_model(link_mode_to_duration=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=10,
    )
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))
    plot_shop_solution(sol)
    plt.show()


if __name__ == "__main__":
    run_cpsat_fjsp()
