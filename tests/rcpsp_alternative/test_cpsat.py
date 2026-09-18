#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp_alternative.solvers.cpsat import (
    CpsatRcpspWithAlternativePathSolver,
)
from discrete_optimization.rcpsp_alternative.solvers.cpsat_auto import (
    CpsatAutoRcpspWithAlternativePathSolver,
)
from discrete_optimization.rcpsp_alternative.utils import create_problem_rcpsp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cpsat():
    problem = parse_file([f for f in get_data_available() if "j301_1.sm" in f][0])
    problem = create_problem_rcpsp(
        problem,
        nb_alternative_paths=2,
        range_nb_subpath=(1, 4),
        range_len_subpath=(3, 5),
    )
    solver = CpsatRcpspWithAlternativePathSolver(problem)
    solver.init_model(strict_alternative_path=True)
    res = solver.solve(parameters_cp=ParametersCp.default_cpsat(), time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_cpsat_auto():
    problem = parse_file([f for f in get_data_available() if "j301_1.sm" in f][0])
    problem = create_problem_rcpsp(
        problem,
        nb_alternative_paths=2,
        range_nb_subpath=(1, 4),
        range_len_subpath=(3, 5),
    )
    solver = CpsatAutoRcpspWithAlternativePathSolver(problem)
    solver.init_model()
    res = solver.solve(parameters_cp=ParametersCp.default_cpsat(), time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)
