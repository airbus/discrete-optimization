#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.problem import FacilityProblem
from discrete_optimization.facility.solvers.dp import (
    DpFacilityModeling,
    DpFacilitySolver,
    dp,
)


@pytest.mark.parametrize(
    "modeling", [DpFacilityModeling.CUSTOMER, DpFacilityModeling.FACILITY]
)
def test_facility_example(modeling):
    file = [f for f in get_data_available() if "fl_25_2" in f][0]
    problem: FacilityProblem = parse_file(file)
    print("customer : ", problem.customer_count, "facility : ", problem.facility_count)
    solver = DpFacilitySolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, modeling=modeling, time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
