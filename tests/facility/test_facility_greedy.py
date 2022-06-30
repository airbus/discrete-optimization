from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.solvers.greedy_solvers import GreedySolverFacility


def test_greedy_facility():
    file = [f for f in get_data_available() if "fl_50_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedySolverFacility(color_problem)
    solution, fit = solver.solve().get_best_solution_fit()
    assert color_problem.satisfy(solution)


if __name__ == "__main__":
    test_greedy_facility()
