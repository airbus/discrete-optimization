#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Define the usecases on which we want non-regression.

(We will store previous results and compare newer versions of solvers to them)

"""

import os.path

from pytest_cases import parametrize

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.shop.jsp.parser import get_data_available, parse_file
from discrete_optimization.shop.jsp.problem import JobShopSolution
from discrete_optimization.shop.jsp.solvers.cpsat import (
    CpSatJspSolver,
)


# Problems
@parametrize("filename", ["la01", "orb10", "swv17", "ta66", "ta80"])
def problem_jsp(filename):
    filepaths = get_data_available()
    filepath = [
        filepath for filepath in filepaths if os.path.basename(filepath) == filename
    ][0]
    return parse_file(filepath)


# Solvers + objectives
@parametrize("time_limit", [30])
def solver_cpsat(time_limit):
    parameters_cp = (
        ParametersCp.default()
    )  # only 1 process to avoid discrepancy with github runners
    kwargs = dict(time_limit=time_limit, parameters_cp=parameters_cp)

    def objective_fn(solution: JobShopSolution) -> int:
        return solution.get_max_end_time()

    mode_optim = ModeOptim.MINIMIZATION
    return CpSatJspSolver, kwargs, objective_fn, mode_optim
