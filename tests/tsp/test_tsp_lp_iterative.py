#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from tempfile import TemporaryDirectory

from discrete_optimization.tsp.solver.solver_lp_iterative import (
    LP_TSP_Iterative,
    MILPSolver,
    build_graph_complete,
)
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_lp():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    solver = LP_TSP_Iterative(model, build_graph_complete)
    solver.init_model(method=MILPSolver.CBC)
    with TemporaryDirectory() as plot_folder:
        sol = solver.solve(plot=False, plot_folder=plot_folder).get_best_solution()
        assert len(os.listdir(plot_folder)) == 5
    assert model.satisfy(sol)


if __name__ == "__main__":
    test_lp()
