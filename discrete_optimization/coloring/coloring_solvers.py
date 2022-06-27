from discrete_optimization.coloring.solvers.coloring_cp_solvers import (
    ColoringCP,
    ColoringCPModel,
    CPSolver,
    CPSolverName,
    ParametersCP,
)
from discrete_optimization.coloring.solvers.coloring_lp_solvers import (
    ColoringLP,
    ColoringLP_MIP,
    MilpSolverName,
)
from discrete_optimization.coloring.solvers.greedy_coloring import (
    ColoringProblem,
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

# from discrete_optimization.coloring.solvers.coloring_cp_lns_solvers import InitialColoring, InitialColoringMethod, \
#     PostProcessSolutionColoring, ConstraintHandlerFixColorsCP
# from discrete_optimization.generic_tools.lns_cp import LNS_CP
# from discrete_optimization.generic_tools.lns_mip import LNS_MILP
# import discrete_optimization.coloring.solvers.coloring_lp_lns_solvers as coloring_lp_lns

solvers = {
    "lp": [
        (
            ColoringLP,
            {
                "greedy_start": True,
                "use_cliques": False,
                "parameters_milp": ParametersMilp.default(),
            },
        ),
        (
            ColoringLP_MIP,
            {
                "milp_solver_name": MilpSolverName.CBC,
                "greedy_start": True,
                "parameters_milp": ParametersMilp.default(),
                "use_cliques": False,
            },
        ),
    ],
    "cp": [
        (
            ColoringCP,
            {
                "cp_solver_name": CPSolverName.CHUFFED,
                "cp_model": ColoringCPModel.DEFAULT,
                "parameters_cp": ParametersCP.default(),
                "object_output": True,
            },
        )
    ],
    "greedy": [(GreedyColoring, {"strategy": NXGreedyColoringMethod.best})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [ColoringProblem]


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


def solve(method, coloring_model: ColoringProblem, **args) -> ResultStorage:
    solver = method(coloring_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method, coloring_model: ColoringProblem, **args) -> ResultStorage:
    solver = method(coloring_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
