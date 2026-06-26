#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.callbacks.loggers import (
    ObjectiveBoundLogger,
    ProblemEvaluateLogger,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lp_tools import mathopt
from discrete_optimization.lotsizing.parser import (
    LotSizingProblem,
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.solvers import (
    CpSatLotSizingSolver,
    DpLotSizingSolver,
    GurobiLotSizingSolver,
    MathOptLotSizingSolver,
)

logging.basicConfig(level=logging.DEBUG)


def run_cpsat(problem: LotSizingProblem):
    print("Known ub", problem.known_bound)
    solver = CpSatLotSizingSolver(problem)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        callbacks=[ObjectiveBoundLogger(logging.INFO, logging.INFO)],
        time_limit=10,
    )
    sol = res[-1][0]
    print(solver.status_solver, solver.get_current_best_internal_objective_bound())
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_dp(problem: LotSizingProblem):
    print("Known ub", problem.known_bound)
    solver = DpLotSizingSolver(problem)
    res = solver.solve(
        solver="LNBS",
        threads=1,
        retrieve_intermediate_solutions=False,
        callbacks=[],  # [ProblemEvaluateLogger(logging.INFO,
        #                       logging.INFO)],
        time_limit=10,
    )
    sol = res[-1][0]
    print(solver.status_solver, solver.early_stopping_exception)
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_lp(problem: LotSizingProblem):
    print("Known ub", problem.known_bound)
    solver = MathOptLotSizingSolver(problem)
    res = solver.solve(
        mathopt_solver_type=mathopt.SolverType.GSCIP,
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        time_limit=10,
    )
    sol = res[-1][0]
    print(solver.status_solver)
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_gurobi(problem: LotSizingProblem):
    print("Known ub", problem.known_bound)
    solver = GurobiLotSizingSolver(problem)
    res = solver.solve(
        gurobi_solver_kwargs={},
        callbacks=[ProblemEvaluateLogger(logging.INFO, logging.INFO)],
        time_limit=10,
    )
    sol = res[-1][0]
    print(solver.status_solver)
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    instance_files = get_data_available()
    print(instance_files[0])
    problem = parse_file(instance_files[0])
    # run_gurobi(problem)
    # run_lp(problem)
    # run_cpsat(problem)
    run_dp(problem)
    # First try to get data from the datasets repository
