#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from ortools.constraint_solver import routing_enums_pb2

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.gpdp.problem import GpdpProblem, GpdpSolution, ProxyClass
from discrete_optimization.gpdp.solvers.ortools_routing import (
    OrtoolsGpdpSolver,
    ParametersCost,
)
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution
from discrete_optimization.tsptw.solvers.cpsat import CpSatTSPTWSolver

logging.basicConfig(level=logging.INFO)


def run_ortools():
    problem = parse_tsptw_file(get_data_available()[8])
    gpdp_problem = ProxyClass.from_tsptw_to_gpdp(problem, True)
    solver = OrtoolsGpdpSolver(problem=gpdp_problem, factor_multiplier_time=100)
    solver.init_model(
        one_visit_per_node=True,
        include_time_windows=True,
        include_time_dimension=True,
        local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        time_limit=10,
        parameters_cost=[ParametersCost(dimension_name="Time", global_span=True)],
    )
    res = solver.solve()
    sol: GpdpSolution = res[-1][0]
    print(sol)
    print(len(sol.trajectories[0]), problem.nb_nodes, problem.nb_customers)
    tsptwsol = TSPTWSolution(
        problem=problem, permutation=[i + 1 for i in sol.trajectories[0][1:-1]]
    )
    print(gpdp_problem.satisfy(sol), gpdp_problem.evaluate(sol))
    print(problem.satisfy(tsptwsol), problem.evaluate(tsptwsol))


if __name__ == "__main__":
    run_ortools()
