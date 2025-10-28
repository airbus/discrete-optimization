#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution
from discrete_optimization.vrptw.solvers.cpsat import CpSatVRPTWSolver
from discrete_optimization.vrptw.solvers.dp import DpVrptwSolver

logging.basicConfig(level=logging.INFO)


def run_dp():
    file = [f for f in get_data_available() if "R1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    solver = DpVrptwSolver(problem=problem)
    solver.init_model(
        scaling=100,
        cost_per_vehicle=100000,
        resource_var_load=False,
        resource_var_time=False,
    )
    res = solver.solve(callbacks=[], solver="CABS", time_limit=100, threads=2)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_dp_ws():
    file = [f for f in get_data_available() if "R1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    subs = CpSatVRPTWSolver(problem)
    subs.init_model(scaling=100, cost_per_vehicle=100000)
    res = subs.solve(
        parameters_cp=ParametersCp.default_cpsat(), callbacks=[], time_limit=10
    )
    sol = res[-1][0]
    print(subs.status_solver)
    print(problem.evaluate(sol))
    solver = DpVrptwSolver(problem=problem)
    solver.init_model(
        scaling=100,
        cost_per_vehicle=100000,
        resource_var_load=False,
        resource_var_time=False,
    )
    solver.set_warm_start(sol)
    res = solver.solve(callbacks=[], solver="CABS", time_limit=100, threads=4)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_dp_ws()
