import os

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.facility_parser import files_available, parse_file
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveHandling,
    TypeAttribute,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)


def test_ga_facility_1():
    file = [f for f in files_available if os.path.basename(f) == "fl_50_6"][0]
    facility_problem: FacilityProblem = parse_file(file)
    obj_handling, objs, objs_weight = get_default_objective_setup(facility_problem)
    ga_solver = Ga(
        facility_problem,
        encoding="facility_for_customers",
        objective_handling=obj_handling,
        objectives=objs,
        objective_weights=objs_weight,
        mutation=DeapMutation.MUT_UNIFORM_INT,
        max_evals=1000000,
    )
    facility_solution = ga_solver.solve()
    print("facility_solution: ", facility_solution)
    print("color_evaluate: ", facility_problem.evaluate(facility_solution))
    print("color_satisfy: ", facility_problem.satisfy(facility_solution))


if __name__ == "__main__":
    test_ga_facility_1()
