#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.facility.facility_model import FacilityProblem
from discrete_optimization.generic_tools.do_solver import SolverDO


class SolverFacility(SolverDO):
    def __init__(self, facility_problem: FacilityProblem, **kwargs: Any):
        self.facility_problem = facility_problem

    def init_model(self, **kwargs: Any) -> None:
        pass
