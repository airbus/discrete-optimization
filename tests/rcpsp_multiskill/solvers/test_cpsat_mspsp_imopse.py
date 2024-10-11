#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  In this example script we're parsing dzn files that are retrieved from https://github.com/youngkd/MSPSP-InstLib
#  And run CP solver with different mzn models.

import logging

import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def test_imopse_cpsat():
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    model, _ = parse_file(file, max_horizon=1000)
    cp_model = CpSatMultiskillRcpspSolver(
        problem=model,
    )
    cp_model.init_model(
        one_worker_per_task=True,
    )
    cp_model.cp_model.Minimize(cp_model.variables["makespan"])
    p = ParametersCp.default_cpsat()
    res = cp_model.solve(parameters_cp=p, time_limit=20)
    solution = res.get_best_solution_fit()[0]
    assert model.satisfy(solution)


def test_imopse_cpsat_with_calendar():
    """
    To test some part of the cp model relative to calendar handling
    """
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    model, _ = parse_file(file, max_horizon=1000)
    for emp in model.employees:
        model.employees[emp].calendar_employee = np.array(
            model.employees[emp].calendar_employee
        )
        model.employees[emp].calendar_employee[5:10] = 0
    model.update_functions()
    cp_model = CpSatMultiskillRcpspSolver(
        problem=model,
    )
    cp_model.init_model(
        one_worker_per_task=True,
    )
    cp_model.cp_model.Minimize(cp_model.variables["makespan"])
    p = ParametersCp.default_cpsat()
    res = cp_model.solve(parameters_cp=p, time_limit=20)
    solution = res.get_best_solution_fit()[0]
    assert model.satisfy(solution)
