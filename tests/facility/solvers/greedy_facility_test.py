from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.facility_parser import files_available, parse_file
from discrete_optimization.facility.solvers.greedy_solvers import GreedySolverFacility


def test_greedy_facility():
    file = [f for f in files_available if "fl_50_1" in f][0]
    color_problem = parse_file(file)
    solver = GreedySolverFacility(color_problem)
    solution, fit = solver.solve()
    print(solution)
    print(fit)
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    test_greedy_facility()
