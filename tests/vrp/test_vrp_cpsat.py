#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from itertools import product

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.vrp.solver.vrp_cpsat_lns import (
    ConstraintHandlerSubpathVRP,
    ConstraintHandlerVRP,
)
from discrete_optimization.vrp.solver.vrp_cpsat_solver import CpSatVrpSolver
from discrete_optimization.vrp.vrp_model import VrpSolution
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file


@pytest.mark.parametrize(
    "optional_node,cut_transition", list(product([True, False], repeat=2))
)
def test_cpsat_vrp(optional_node, cut_transition):
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=optional_node, cut_transition=cut_transition)
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    p.time_limit = 10
    res = solver.solve(parameters_cp=p)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)


def test_cpsat_lns_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    solver_lns = LNS_OrtoolsCPSat(
        problem=problem,
        subsolver=solver,
        initial_solution_provider=None,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                ConstraintHandlerVRP(problem, 0.5),
                ConstraintHandlerSubpathVRP(problem, 0.5),
            ],
            list_proba=[0.5, 0.5],
        ),
    )
    p = ParametersCP.default_cpsat()
    p.time_limit = 10
    p.time_limit_iter0 = 10
    res = solver_lns.solve(
        skip_initial_solution_provider=True, nb_iteration_lns=30, parameters_cp=p
    )
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
