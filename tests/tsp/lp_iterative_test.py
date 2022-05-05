import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
import time

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
    solver.solve()


if __name__ == "__main__":
    run_lp()
