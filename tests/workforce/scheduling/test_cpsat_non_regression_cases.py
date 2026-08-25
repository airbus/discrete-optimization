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
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingSolution
from discrete_optimization.workforce.scheduling.solvers import ObjectivesEnum
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
)


# Problems
@parametrize(
    "filename",
    [
        "instance_42",
        "instance_30",
        "instance_170",
        "instance_252",
    ],
)
def problem_wf_sched(filename):
    filepaths = get_data_available()
    filepath = [
        filepath
        for filepath in filepaths
        if os.path.splitext(os.path.basename(filepath))[0] == filename
    ][0]
    return parse_json_to_problem(filepath)


# Solvers + objectives
@parametrize("time_limit", [30])
@parametrize("objective", ["nbteams", "nbteams+dispersion"])
def solver_cpsat(time_limit, objective):
    parameters_cp = (
        ParametersCp.default()
    )  # only 1 process to avoid discrepancy with github runners
    if objective == "nbteams":
        objectives = [ObjectivesEnum.NB_TEAMS]
    else:
        objectives = [ObjectivesEnum.NB_TEAMS, ObjectivesEnum.DISPERSION]
    kwargs = dict(
        time_limit=time_limit, parameters_cp=parameters_cp, objectives=objectives
    )

    if objective == "nbteams":

        def objective_fn(solution: AllocSchedulingSolution) -> int:
            return solution.compute_nb_unary_resources_used()
    else:

        def objective_fn(solution: AllocSchedulingSolution) -> tuple[int, int]:
            kpis = solution.problem.evaluate(solution)
            return kpis["nb_teams"], kpis["workload_dispersion"]

    mode_optim = ModeOptim.MINIMIZATION
    return CPSatAllocSchedulingSolver, kwargs, objective_fn, mode_optim
