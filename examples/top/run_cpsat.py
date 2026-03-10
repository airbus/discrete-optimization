#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.top.parser import get_data_available, parse_file
from discrete_optimization.top.solvers.cpsat import CpsatTopSolver


def run():
    files, files_dict = get_data_available()
    print(files[0])
    problem = parse_file(files[2])
    solver = CpsatTopSolver(problem)
    solver.init_model(scaling=100)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=30,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run()
