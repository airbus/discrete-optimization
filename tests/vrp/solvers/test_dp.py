#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers.dp import DpVrpSolver, dp


def test_dp_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    solver = DpVrpSolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    assert problem.satisfy(sol)


def test_dp_vrp_ws():
    from discrete_optimization.vrp.solvers.ortools_routing import OrtoolsVrpSolver

    file = [f for f in get_data_available() if "vrp_135_7_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    solver = OrtoolsVrpSolver(problem)
    sol_ws = solver.solve(time_limit=10)[0][0]
    solver = DpVrpSolver(problem=problem)
    solver.init_model()
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        threads=6,
        time_limit=20,
    )
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    assert sol_ws.list_paths == sol.list_paths
    assert problem.satisfy(sol)
