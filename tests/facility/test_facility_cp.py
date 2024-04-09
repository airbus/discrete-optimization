#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import sys

import pytest

from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.solvers.facility_cp_solvers import (
    FacilityCP,
    FacilityCPModel,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCP


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_facility_cp():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_16_1"][0]
    facility_problem = parse_file(file)
    solver = FacilityCP(facility_problem)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    solver.init_model(cp_model=FacilityCPModel.DEFAULT_INT, object_output=True)
    solution, fit = solver.solve(parameters_cp=parameters_cp).get_best_solution_fit()
    assert facility_problem.satisfy(solution)


if __name__ == "__main__":
    test_facility_cp()
