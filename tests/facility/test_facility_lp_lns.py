#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import platform

import pytest

from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.solvers.facility_lp_lns_solver import (
    ConstraintHandlerFacility,
    InitialFacilityMethod,
    InitialFacilitySolution,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver_PyMip,
    MilpSolverName,
    ParametersMilp,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_mip import LNS_MILP

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
@pytest.mark.skipif(
    platform.machine() == "arm64",
    reason=(
        "Python-mip has issues with cbclib on macos arm64. "
        "See https://github.com/coin-or/python-mip/issues/167"
    ),
)
def test_facility_lns():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_16_1"][0]
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    params_milp = ParametersMilp(
        time_limit=20,
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
        n_solutions_max=1000,
    )
    solver = LP_Facility_Solver_PyMip(
        facility_problem,
        milp_solver_name=MilpSolverName.CBC,
        params_objective_function=params_objective_function,
    )
    solver.init_model(use_matrix_indicator_heuristic=False)
    initial_solution_provider = InitialFacilitySolution(
        problem=facility_problem,
        initial_method=InitialFacilityMethod.GREEDY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerFacility(
        problem=facility_problem, fraction_to_fix=0.5, skip_first_iter=True
    )
    lns_solver = LNS_MILP(
        problem=facility_problem,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )

    result_store = lns_solver.solve_lns(
        parameters_milp=params_milp,
        nb_iteration_lns=100,
        callbacks=[TimerStopper(total_seconds=100)],
    )
    solution = result_store.get_best_solution_fit()[0]
    assert facility_problem.satisfy(solution)
    facility_problem.evaluate(solution)


if __name__ == "__main__":
    test_facility_lns()
