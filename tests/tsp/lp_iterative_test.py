import os
import time
from tempfile import TemporaryDirectory

import numpy as np
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.tsp.solver.solver_lp_iterative import (
    LP_TSP_Iterative,
    MILPSolver,
    build_graph_complete,
    build_graph_pruned,
)
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel, compute_length
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def run_lp():
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
    run_lp()
