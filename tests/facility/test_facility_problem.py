from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)


def test_model_satisfy():
    file = [f for f in get_data_available() if "fl_50_1" in f][0]
    facility_problem: FacilityProblem = parse_file(file)
    dummy_solution = facility_problem.get_dummy_solution()
    print("Dummy ", dummy_solution)
    print("Dummy satisfy ", facility_problem.satisfy(dummy_solution))
    print(facility_problem.evaluate(dummy_solution))

    # bad_solution = ColoringSolution(color_problem, [1]*color_problem.number_of_nodes)
    # color_problem.evaluate(bad_solution)
    # print("Bad solution satisfy ", color_problem.satisfy(bad_solution))


if __name__ == "__main__":
    test_model_satisfy()
