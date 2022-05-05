import os

from discrete_optimization.facility.facility_parser import files_available, parse_file
from discrete_optimization.facility.solvers.facility_cp_solvers import (
    FacilityCP,
    FacilityCPModel,
)


def test_facility_cp():
    file = [f for f in files_available if os.path.basename(f) == "fl_50_6"][0]
    print(file)
    facility_problem = parse_file(file)
    solver = FacilityCP(facility_problem)
    solver.init_model(cp_model=FacilityCPModel.DEFAULT_INT)
    solution, fit = solver.solve(limit_time_s=20, verbose=True)
    print(solution)
    print("Satisfy : ", facility_problem.satisfy(solution))


def test_facility_cp_lns():
    file = [f for f in files_available if os.path.basename(f) == "fl_50_6"][0]
    print(file)
    facility_problem = parse_file(file)
    solver = FacilityCP(facility_problem)
    solution, fit = solver.solve_lns(
        fraction_to_fix=0.8,
        nb_iteration=1000,
        limit_time_s=20,
        greedy_start=True,
        cp_model=FacilityCPModel.DEFAULT_INT_LNS,
        verbose=True,
    )
    print(solution)
    print("Satisfy : ", facility_problem.satisfy(solution))


if __name__ == "__main__":
    test_facility_cp_lns()
