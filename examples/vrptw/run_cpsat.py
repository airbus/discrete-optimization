#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrptw.parser import parse_solomon
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution
from discrete_optimization.vrptw.solvers.cpsat import CpSatVRPTWSolver


def run_cpsat():
    problem = parse_solomon("C1_2_1.TXT")
    solver = CpSatVRPTWSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        callbacks=[],
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_cpsat()
