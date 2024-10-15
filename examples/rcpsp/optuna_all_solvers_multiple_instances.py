#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters on multiple rcpsp instances.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""

import logging
import re
from collections import defaultdict
from os.path import basename
from typing import Any

from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp import (
    LnsCpMznGenericRcpspSolver,
)
from discrete_optimization.generic_tools.cp_tools import (
    CpSolverName,
    MinizincCpSolver,
    ParametersCp,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    gurobi_available,
)
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_multiproblem,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.lp import (
    MathOptMultimodeRcpspSolver,
    MathOptRcpspSolver,
)
from discrete_optimization.rcpsp.solvers_map import look_for_solver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

seed = 42  # set this to an integer to get reproducible results, else to None
n_trials = 100  # number of trials to launch
gurobi_full_license_available = False  # is the installed gurobi having a full license? (contrary to the license installed by `pip install gurobipy`)
create_another_study = True  # True: generate a study name with timestamp to avoid overwriting previous study, False: keep same study name
max_time_per_solver = 20  # max duration per solver (seconds)

problem_pattern = "j301_.*\.sm"
nb_max_problems = 10
study_basename = f"rcpsp_multiple_instances-{problem_pattern}-{nb_max_problems}"


problems_files = [
    f for f in get_data_available() if re.match(problem_pattern, basename(f))
]
if nb_max_problems > 0:
    problems_files = problems_files[:nb_max_problems]
problems = [parse_file(f) for f in problems_files]


solvers_to_test = look_for_solver(problems[0])

# Fixed parameters
parameters_cp = ParametersCp.default_cpsat()
parameters_cp.nb_process = 6
kwargs_fixed_by_solver: dict[type[SolverDO], dict[str, Any]] = defaultdict(
    dict,  # default kwargs for unspecified solvers
    {
        LnsCpMznGenericRcpspSolver: dict(
            nb_iteration_lns=10,
            parameters_cp=parameters_cp,
            time_limit=max_time_per_solver,
        ),
        MathOptRcpspSolver: dict(time_limit=max_time_per_solver),
        MathOptMultimodeRcpspSolver: dict(time_limit=max_time_per_solver),
    },
)

# Restrict some hyperparameters choices, for some solvers (making use of `kwargs_by_name` of `suggest_with_optuna`)
suggest_optuna_kwargs_by_name_by_solver: dict[
    type[SolverDO], dict[str, dict[str, Any]]
] = defaultdict(
    dict,  # default kwargs_by_name for unspecified solvers
    {},
)
if not gurobi_available or not gurobi_full_license_available:
    # Remove possibility for gurobi if not available
    solvers_to_test = [
        s for s in solvers_to_test if not isinstance(s, GurobiMilpSolver)
    ]
    for s in solvers_to_test:
        if isinstance(s, MinizincCpSolver):
            suggest_optuna_kwargs_by_name_by_solver[s].update(
                {
                    "cp_solver_name": dict(
                        choices=[x for x in CpSolverName if x != CpSolverName.GUROBI]
                    )
                }
            )
    suggest_optuna_kwargs_by_name_by_solver[LnsCpMznGenericRcpspSolver].update(
        {
            "cp_solver_name": dict(
                choices=[x for x in CpSolverName if x != CpSolverName.GUROBI]
            )
        }
    )

# Generate and launch the optuna study
generic_optuna_experiment_multiproblem(
    problems=problems,
    solvers_to_test=solvers_to_test,
    kwargs_fixed_by_solver=kwargs_fixed_by_solver,
    suggest_optuna_kwargs_by_name_by_solver=suggest_optuna_kwargs_by_name_by_solver,
    n_trials=n_trials,
    study_basename=study_basename,
    create_another_study=create_another_study,
    seed=seed,
)
