#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp_cal_preemptive.solvers.cpsat import (
    CpSatCalendarPreemptiveSolver,
)
from discrete_optimization.rcpsp_cal_preemptive.solvers.cpsat_auto import (
    CpSatAutoCalendarPreemptiveSolver,
)
from discrete_optimization.rcpsp_cal_preemptive.utils import (
    load_calendar_preemptive_rcpsp_problem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cpsat():
    problem = parse_file([f for f in get_data_available() if "j301_1.sm" in f][0])
    problem = load_calendar_preemptive_rcpsp_problem(problem)
    solver = CpSatCalendarPreemptiveSolver(problem)
    solver.init_model()
    res = solver.solve(parameters_cp=ParametersCp.default_cpsat(), time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_cpsat_auto():
    problem = parse_file([f for f in get_data_available() if "j301_1.sm" in f][0])
    problem = load_calendar_preemptive_rcpsp_problem(problem)
    solver = CpSatAutoCalendarPreemptiveSolver(problem)
    solver.init_model()
    res = solver.solve(parameters_cp=ParametersCp.default_cpsat(), time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)
