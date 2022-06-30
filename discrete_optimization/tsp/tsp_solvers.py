from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.tsp.solver.solver_lp_iterative import (
    LP_TSP_Iterative,
    MILPSolver,
)
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.tsp.solver.tsp_cp_solver import (
    CPSolverName,
    TSP_CP_Solver,
    TSP_CPModel,
)
from discrete_optimization.tsp.tsp_model import (
    TSPModel,
    TSPModel2D,
    TSPModelDistanceMatrix,
)

solvers = {
    "lp": [
        (
            LP_TSP_Iterative,
            {"method": MILPSolver.CBC, "nb_iteration_max": 20, "plot": False},
        )
    ],
    "ortools": [(TSP_ORtools, {})],
    "cp": [
        (
            TSP_CP_Solver,
            {
                "model_type": TSP_CPModel.INT_VERSION,
                "cp_solver_name": CPSolverName.CHUFFED,
            },
        )
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [TSPModel, TSPModel2D, TSPModelDistanceMatrix]


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


def solve(method, tsp_problem: TSPModel, **args) -> ResultStorage:
    solver = method(tsp_problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method, tsp_problem: TSPModel, **args) -> ResultStorage:
    solver = method(tsp_problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
