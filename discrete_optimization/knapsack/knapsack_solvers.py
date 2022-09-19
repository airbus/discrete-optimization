#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.knapsack.knapsack_model import KnapsackModel
from discrete_optimization.knapsack.solvers.cp_solvers import (
    CPKnapsackMZN,
    CPKnapsackMZN2,
    CPSolverName,
)
from discrete_optimization.knapsack.solvers.dyn_prog_knapsack import KnapsackDynProg
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.lp_solvers import (
    KnapsackORTools,
    LPKnapsack,
    LPKnapsackCBC,
    LPKnapsackGurobi,
    MilpSolverName,
)

solvers = {
    "lp": [
        (KnapsackORTools, {}),
        (LPKnapsackCBC, {}),
        (LPKnapsackGurobi, {"parameter_gurobi": ParametersMilp.default()}),
        (
            LPKnapsack,
            {
                "milp_solver_name": MilpSolverName.CBC,
                "parameters_milp": ParametersMilp.default(),
            },
        ),
    ],
    "greedy": [(GreedyBest, {})],
    "cp": [
        (CPKnapsackMZN, {"cp_solver_name": CPSolverName.CHUFFED}),
        (CPKnapsackMZN2, {"cp_solver_name": CPSolverName.CHUFFED}),
    ],
    "dyn_prog": [
        (
            KnapsackDynProg,
            {
                "greedy_start": True,
                "stop_after_n_item": True,
                "max_items": 100,
                "max_time_seconds": 100,
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
        solvers_compatibility[y[0]] = [KnapsackModel]


def look_for_solver(domain):
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain):
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(method, knapsack_model: KnapsackModel, **args):
    solver = method(knapsack_model)
    solver.init_model(**args)
    return solver.solve(**args)
