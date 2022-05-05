from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.facility.solvers.greedy_solvers import GreedySolverFacility, GreedySolverDistanceBased
from discrete_optimization.facility.solvers.facility_lp_solver import LP_Facility_Solver, \
    LP_Facility_Solver_CBC, LP_Facility_Solver_PyMip, ParametersMilp, MilpSolverName
from discrete_optimization.facility.solvers.facility_cp_solvers import FacilityCP, ParametersCP, CPSolverName, FacilityCPModel


solvers = {"lp": [(LP_Facility_Solver, {"parameters_milp": ParametersMilp.default(),
                                        "use_matrix_indicator_heuristic": True,
                                        "n_shortest": 10, "n_cheapest": 10}),
                  (LP_Facility_Solver_CBC, {"parameters_milp": ParametersMilp.default(),
                                            "use_matrix_indicator_heuristic": True,
                                            "n_shortest": 10, "n_cheapest": 10}),
                  (LP_Facility_Solver_PyMip, {"parameters_milp": ParametersMilp.default(),
                                              "use_matrix_indicator_heuristic": True,
                                              "milp_solver_name": MilpSolverName.CBC,
                                              "n_shortest": 10, "n_cheapest": 10})],
           "cp": [(FacilityCP, {"cp_solver_name": CPSolverName.CHUFFED,
                                "object_output": True,
                                "cp_model": FacilityCPModel.DEFAULT_INT,
                                "parameters_cp": ParametersCP.default()})],
           "greedy": [(GreedySolverFacility, {}), (GreedySolverDistanceBased, {})]}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [FacilityProblem]


def look_for_solver(domain):
    class_domain = domain.__class__
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    print("You have ", len(available), " solvers for your domain ")
    print([solvers_map[a] for a in available])
    return available


def look_for_solver_class(class_domain):
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    print("You have ", len(available), " solvers for your domain ")
    print([solvers_map[a] for a in available])
    return available


def solve(method,
          facility_problem: FacilityProblem,
          **args) -> ResultStorage:
    solver = method(facility_problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method,
                  coloring_model: FacilityProblem,
                  **args) -> ResultStorage:
    solver = method(coloring_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
