#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.utils import plot_ressource_view, plot_task_gantt, plt
from discrete_optimization.rcpsp_alternative.problem import get_optional_tasks_done
from discrete_optimization.rcpsp_alternative.solvers.cpsat import (
    CpsatRcpspWithAlternativePathSolver,
)
from discrete_optimization.rcpsp_alternative.utils import create_problem_rcpsp


def run_cpsat():
    problem = parse_file([f for f in get_data_available() if "j601_1.sm" in f][0])
    problem = create_problem_rcpsp(
        problem,
        nb_alternative_paths=5,
        range_nb_subpath=(1, 4),
        range_len_subpath=(3, 5),
    )
    solver = CpsatRcpspWithAlternativePathSolver(problem)
    solver.init_model(strict_alternative_path=True)
    res = solver.solve(parameters_cp=ParametersCp.default_cpsat(), time_limit=30)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    print(get_optional_tasks_done(sol, problem))
    plot_task_gantt(problem, sol)
    plot_ressource_view(problem, sol)
    plt.show()


if __name__ == "__main__":
    run_cpsat()
