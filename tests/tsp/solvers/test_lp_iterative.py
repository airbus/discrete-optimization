#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from tempfile import TemporaryDirectory

from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.solvers.lp_iterative import (
    LPIterativeTspSolver,
    MILPSolver,
    build_graph_complete,
)


def test_lp():
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    solver = LPIterativeTspSolver(model, build_graph_complete)
    solver.init_model(method=MILPSolver.CBC)
    with TemporaryDirectory() as plot_folder:
        sol = solver.solve(plot=False, plot_folder=plot_folder).get_best_solution()
        assert len(os.listdir(plot_folder)) == 5
    assert model.satisfy(sol)


if __name__ == "__main__":
    test_lp()
