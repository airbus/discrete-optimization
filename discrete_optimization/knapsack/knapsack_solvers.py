from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest, GreedyDummy
from discrete_optimization.knapsack.solvers.lp_solvers import KnapsackORTools, LPKnapsackCBC,\
    LPKnapsackGurobi, LPKnapsack, CBC, MilpSolverName
from discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN, CPKnapsackMZN2, CPSolverName
from discrete_optimization.knapsack.solvers.dyn_prog_knapsack import KnapsackDynProg
from discrete_optimization.knapsack.knapsack_model import KnapsackModel
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
solvers = {"lp": [(KnapsackORTools, {}),
                  (LPKnapsackCBC, {}),
                  (LPKnapsackGurobi, {"parameter_gurobi": ParametersMilp.default()}),
                  (LPKnapsack, {"milp_solver_name": MilpSolverName.CBC, "parameters_milp": ParametersMilp.default()})],
           "greedy": [(GreedyBest,  {})], 
           "cp": [(CPKnapsackMZN, {"cp_solver_name": CPSolverName.CHUFFED}),
                  (CPKnapsackMZN2, {"cp_solver_name": CPSolverName.CHUFFED})],
           "dyn_prog": [(KnapsackDynProg,  {'greedy_start': True, 
                                            'stop_after_n_item': True, 
                                            'max_items': 100, 
                                            'max_time_seconds': 100,
                                            'verbose': False})]}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [KnapsackModel]


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
          knapsack_model: KnapsackModel,
          **args):
    solver = method(knapsack_model)
    solver.init_model(**args)
    return solver.solve(**args)
