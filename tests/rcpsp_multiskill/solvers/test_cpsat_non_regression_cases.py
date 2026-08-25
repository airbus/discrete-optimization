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
from discrete_optimization.rcpsp_multiskill.parser_mslib import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspSolution
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


# Problems
@parametrize(
    "subdir, filename",
    [
        ("MSLIB1", "MSLIB_Set1_42"),
        ("MSLIB2", "MSLIB_Set2_3360"),
        ("MSLIB2", "MSLIB_Set2_6363"),
        ("MSLIB2", "MSLIB_Set2_42"),
        ("MSLIB3", "MSLIB_Set3_42"),
        ("MSLIB3", "MSLIB_Set3_7127"),
        ("MSLIB4", "MSLIB_Set4_3860"),
        ("MSLIB4", "MSLIB_Set4_42"),
    ],
)
def problem_mslib(subdir, filename):
    filepaths = get_data_available()
    filepath = [
        filepath
        for filepath in filepaths[subdir]
        if os.path.splitext(os.path.basename(filepath))[0] == filename
    ][0]
    return parse_file(filepath)


# Solvers + objectives
@parametrize("time_limit", [30])
def solver_cpsat(time_limit):
    parameters_cp = (
        ParametersCp.default()
    )  # only 1 process to avoid discrepancy with github runners
    kwargs = dict(time_limit=time_limit, parameters_cp=parameters_cp)

    def objective_fn(solution: MultiskillRcpspSolution) -> int:
        return solution.get_max_end_time()

    mode_optim = ModeOptim.MINIMIZATION
    return CpSatMultiskillRcpspSolver, kwargs, objective_fn, mode_optim
