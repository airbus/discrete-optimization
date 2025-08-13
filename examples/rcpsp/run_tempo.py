#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt

from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.tempo import TempoRcpspSolver
from discrete_optimization.rcpsp.utils import plot_ressource_view, plot_task_gantt


def run_tempo():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = TempoRcpspSolver(problem=rcpsp_problem)
    solver.init_model()
    res = solver.solve(time_limit=10)
    sol = res.get_best_solution()
    print(solver._raw_logs)
    print(solver)
    plot_ressource_view(rcpsp_problem=rcpsp_problem, rcpsp_sol=sol)
    print(rcpsp_problem.satisfy(sol), rcpsp_problem.evaluate(sol))
    plt.show()


def run_tempo_multimode():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = TempoRcpspSolver(problem=rcpsp_problem)
    solver.init_model()
    res = solver.solve(time_limit=10)
    print(solver._raw_logs)
    print(solver)
    sol = res.get_best_solution()
    plot_ressource_view(rcpsp_problem=rcpsp_problem, rcpsp_sol=sol)
    print(rcpsp_problem.satisfy(sol), rcpsp_problem.evaluate(sol))
    plt.show()


if __name__ == "__main__":
    run_tempo()
