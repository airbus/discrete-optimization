import os
import sys

from tsp.solver.solver_ortools import TSP_ORtools
from tsp.tsp_parser import get_data_available, parse_file


def run_ortools():
    files = get_data_available()
    files = [f for f in files if "tsp_200_2" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    solution = model.get_dummy_solution()
    solver = TSP_ORtools(model)
    solver.init_model()
    sol, fitness = solver.solve()
    print(sol, fitness)


if __name__ == "__main__":
    run_ortools()
