#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Example of https://www.csplib.org/Problems/prob058/
import logging

from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)
from discrete_optimization.lotsizing.solvers.cpsat import CpSatLotSizingSolver
from discrete_optimization.lotsizing.solvers.dp import DpLotSizingSolver
from discrete_optimization.lotsizing.solvers.lp import (
    GurobiLotSizingSolver,
    MathOptLotSizingSolver,
)

logging.basicConfig(level=logging.DEBUG)


def create_toy_model():
    # Example : Consider the problem with the following input data: number of items type nbItems=2
    # ; number of periods nbPeriods=5
    # ; stocking cost h=2
    # ; demand times for items of type 1 d1t∈1,…,5=(0,1,0,0,1)
    #  and for items of type 2 d2t∈1,…,5=(1,0,0,0,1)
    # ; q1,2=5
    # , q2,1=3
    # . A feasible solution of this problem is productionPlan=(2,1,2,0,1)
    #  which means that item 2
    #  will be produced in period 1
    # ; item 1
    #  in period 2
    # ; item 2
    #  in period 3
    #  and item 1
    #  in period 5
    # . Note that there is no production in period 4
    # , it is an idle period. The cost associated to this solution is q2,1+q1,2+q2,1+2∗h=15
    #  but it is not the optimal cost. The optimal solution is productionPlan=(2,1,0,1,2)
    #  with the cost q2,1+q1,2+h=10
    #
    problem = LotSizingProblem(
        nb_items_type=2,
        capacity_machine=1,
        changeover_costs=[[0, 1], [1, 0]],
        demands=[[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]],
        stock_capacity=10,
        stock_cost_per_type_per_time_per_unit=[1, 1],
        delay_cost_per_type_per_time_per_unit=[1, 1],
        allow_delays=False,
    )
    solution = LotSizingSolution(
        problem=problem,
        productions=[
            ProductionItem(item_type=1, quantity=1, time=0),
            ProductionItem(item_type=0, quantity=1, time=1),
            ProductionItem(item_type=1, quantity=1, time=2),
            ProductionItem(item_type=0, quantity=1, time=4),
        ],
    )

    solution_2 = LotSizingSolution(problem=problem, list_item_per_time=[1, 0, 1, -1, 0])
    print(problem.satisfy(solution), problem.satisfy(solution_2))
    print(problem.evaluate(solution), problem.evaluate(solution_2))
    print(solution.productions, solution_2.productions)
    print(solution.deliveries, solution_2.deliveries)

    solution_3 = LotSizingSolution(problem=problem, list_item_per_time=[1, 0, -1, 0, 1])

    print(problem.satisfy(solution_3))
    print(problem.evaluate(solution_3))
    return problem


def run_cpsat():
    problem = create_toy_model()
    solver = CpSatLotSizingSolver(problem)
    res = solver.solve(time_limit=10)
    sol: LotSizingSolution = res[-1][0]
    print(sol.productions, sol.deliveries)
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_dp():
    problem = create_toy_model()
    solver = DpLotSizingSolver(problem)
    res = solver.solve(time_limit=10, solver="LNBS")
    sol: LotSizingSolution = res[-1][0]
    print(sol.productions, sol.deliveries)
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_mathopt():
    problem = create_toy_model()
    solver = MathOptLotSizingSolver(problem)
    res = solver.solve(time_limit=10)
    sol: LotSizingSolution = res[-1][0]
    print("MathOpt Solution:")
    print(sol.productions, sol.deliveries)
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_gurobi():
    problem = create_toy_model()
    solver = GurobiLotSizingSolver(problem)
    res = solver.solve(time_limit=10)
    sol: LotSizingSolution = res[-1][0]
    print("Gurobi Solution:")
    print(sol.productions, sol.deliveries)
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    # run_dp()
    # run_cpsat()
    run_mathopt()
    run_gurobi()  # Uncomment if you have Gurobi installed
