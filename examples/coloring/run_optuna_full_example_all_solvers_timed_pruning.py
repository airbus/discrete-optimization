#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters for coloring.

This example show different features of optuna integration with discrete-optimization:
- use of `suggest_hyperparameters_with_optuna()` to get hyperparameters values
- use of a dedicated callback to report intermediate results with corresponding time to optuna
  and potentially prune the trial
- time-based pruner
- how to fix some parameters/hyperparameters

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""
import logging
from collections import defaultdict
from typing import Any

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers.cp_mzn import CpColoringSolver
from discrete_optimization.coloring.solvers.cpsat import CpSatColoringSolver
from discrete_optimization.coloring.solvers.greedy import NxGreedyColoringMethod
from discrete_optimization.coloring.solvers.toulbar import ToulbarColoringSolver
from discrete_optimization.coloring.solvers_map import (
    AspColoringSolver,
    GurobiColoringSolver,
    solvers_map,
    toulbar2_available,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import gurobi_available
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


seed = 42  # set this to an integer to get reproducible results, else to None
n_trials = 100  # number of trials to launch
gurobi_full_license_available = False  # is the installed gurobi having a full license? (contrary to the license installed by `pip install gurobipy`)
create_another_study = True  # True: generate a study name with timestamp to avoid overwriting previous study, False: keep same study name
max_time_per_solver = 20  # max duration per solver (seconds)
min_time_per_solver = 5  # min duration before pruning a solver (seconds)

modelfilename = "gc_70_9"  # filename of the model used

study_basename = f"coloring_all_solvers-auto-pruning-{modelfilename}"

# solvers to test
solvers_to_remove = {CpColoringSolver}
if not gurobi_available or not gurobi_full_license_available:
    solvers_to_remove.add(GurobiColoringSolver)
if not toulbar2_available:
    solvers_to_remove.add(ToulbarColoringSolver)
solvers_to_test: list[type[SolverDO]] = [
    s for s in solvers_map if s not in solvers_to_remove
]
# fixed kwargs per solver: either hyperparameters we do not want to search, or other parameters like time limits
p = ParametersCp.default_cpsat()
p.nb_process = 6
kwargs_fixed_by_solver: dict[type[SolverDO], dict[str, Any]] = defaultdict(
    dict,  # default kwargs for unspecified solvers
    {
        CpSatColoringSolver: dict(parameters_cp=p, time_limit=max_time_per_solver),
        CpColoringSolver: dict(parameters_cp=p, time_limit=max_time_per_solver),
        GurobiColoringSolver: dict(time_limit=max_time_per_solver),
        AspColoringSolver: dict(time_limit=max_time_per_solver),
        ToulbarColoringSolver: dict(time_limit=max_time_per_solver),
    },
)

# restrict some hyperparameters choices, for some solvers (making use of `kwargs_by_name` of `suggest_with_optuna`)
suggest_optuna_kwargs_by_name_by_solver: dict[
    type[SolverDO], dict[str, dict[str, Any]]
] = defaultdict(
    dict,  # default kwargs_by_name for unspecified solvers
    {
        ToulbarColoringSolver: {  # options for ToulbarColoringSolver hyperparameters
            "tolerance_delta_max": dict(low=1, high=2),  # we restrict to [1, 2]
            "greedy_method": dict(  # we restrict the available choices for greedy_method
                choices=[
                    NxGreedyColoringMethod.best,
                    NxGreedyColoringMethod.largest_first,
                    NxGreedyColoringMethod.random_sequential,
                ]
            ),
        }
    },
)


# problem definition
file = [f for f in get_data_available() if "gc_70_9" in f][0]
problem = parse_file(file)

# generate and launch the optuna study
generic_optuna_experiment_monoproblem(
    problem=problem,
    solvers_to_test=solvers_to_test,
    kwargs_fixed_by_solver=kwargs_fixed_by_solver,
    suggest_optuna_kwargs_by_name_by_solver=suggest_optuna_kwargs_by_name_by_solver,
    n_trials=n_trials,
    computation_time_in_study=True,
    study_basename=study_basename,
    create_another_study=create_another_study,
    seed=seed,
    min_time_per_solver=min_time_per_solver,
)
