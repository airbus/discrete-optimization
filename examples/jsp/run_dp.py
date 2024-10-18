#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import didppy as dp
from didppy import BeamParallelizationMethod

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver
from discrete_optimization.jsp.solvers.dp import DpJspSolver
from discrete_optimization.jsp.utils import transform_jsp_to_rcpsp
from discrete_optimization.rcpsp.solvers.dp import DpRcpspSolver

logging.basicConfig(level=logging.INFO)


def debug():
    model = dp.Model()
    cur_time_per_machine = [model.add_int_var(target=0) for m in range(4)]
    cur_time_per_job = [model.add_int_var(target=1) for m in range(2)]
    max_total = model.add_int_var(target=0)
    t = dp.Transition(
        name="t",
        cost=dp.IntExpr.state_cost() + 1,
        preconditions=[],
        effects=[
            (
                cur_time_per_machine[0],
                dp.max(cur_time_per_machine[0] + 2, cur_time_per_job[0] + 2),
            ),
            (
                cur_time_per_job[0],
                dp.max(cur_time_per_machine[0] + 2, cur_time_per_job[0] + 2),
            ),
            (
                max_total,
                dp.max(
                    max_total,
                    dp.max(cur_time_per_machine[0] + 2, cur_time_per_job[0] + 2),
                ),
            ),
        ],
    )
    state = model.target_state
    # preconditions = t.preconditions
    # preconditions[0].eval(state, model)
    print("machine", t[cur_time_per_machine[0]].eval(state, model))
    print("cur time job", t[cur_time_per_job[0]].eval(state, model))
    print("Max totoal", t[max_total].eval(state, model))
    t[max_total] = max_total + 1
    print(t[max_total].eval(state, model))


def run_dp_jsp():
    # file_path = get_data_available()[1]
    file_path = [f for f in get_data_available() if "ta68" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    solver = DpJspSolver(problem=problem)
    res = solver.solve(
        solver=dp.LNBS,
        time_limit=100,
        max_beam_size=2048,
        keep_all_layers=False,
        parallelization_method=BeamParallelizationMethod.Hdbs2,
    )
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


def run_dp_jsp_ws():
    # file_path = get_data_available()[1]
    file_path = [f for f in get_data_available() if "ta68" in f][0]
    problem = parse_file(file_path)
    solver_ws = CpSatJspSolver(problem)
    sol_ws = solver_ws.solve(time_limit=2)[0][0]
    print("File path ", file_path)
    solver = DpJspSolver(problem=problem)
    solver.init_model()
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        solver=dp.LNBS,
        time_limit=100,
    )
    sol = res.get_best_solution_fit()[0]
    print(sol_ws.schedule)
    print(sol.schedule)
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


def run_dp_of_rcpsp():
    file_path = [f for f in get_data_available() if "ta68" in f][0]
    problem = parse_file(file_path)
    rcpsp_problem = transform_jsp_to_rcpsp(problem)
    solver = DpRcpspSolver(rcpsp_problem)
    solver.init_model()
    res = solver.solve(
        solver=dp.LNBS,
        time_limit=100,
        max_beam_size=2048,
        keep_all_layers=False,
        parallelization_method=BeamParallelizationMethod.Hdbs2,
    )
    sol = res.get_best_solution_fit()[0]
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


if __name__ == "__main__":
    run_dp_of_rcpsp()
