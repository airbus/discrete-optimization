#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from matplotlib import pyplot as plt

from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.tempo import TempoJspScheduler
from discrete_optimization.jsp.utils import plot_jobshop_solution

logging.basicConfig(level=logging.DEBUG)


def run_tempo():
    file = get_data_available()[0]
    print(file)
    problem = parse_file(file)
    solver = TempoJspScheduler(problem=problem)
    solver.init_model()
    res = solver.solve(time_limit=5)
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))
    plot_jobshop_solution(sol)
    plt.show()


if __name__ == "__main__":
    run_tempo()
