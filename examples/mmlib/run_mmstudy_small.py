#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import re

import pandas as pd

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ProblemEvaluateLogger,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study import (
    SolverConfig,
)
from discrete_optimization.generic_tools.study.database import (
    is_empty_metrics,
)
from discrete_optimization.generic_tools.study.study import Study
from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp.parser import parse_file_mmlib
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.cpsat_auto import CpSatAutoRcpspSolver
from discrete_optimization.rcpsp.solvers.optal import OptalRcpspSolver

mmlib_home = os.path.join(get_data_home(), "mmlib")

n_instances = 40
time_limit = 120
study_name = f"mmlibtest-{n_instances}-largest-instance-{time_limit}s"

overwrite = False  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
max_retry = 0  # retry failed xps (and how many times) ?


def get_data(prefix="", pattern=""):
    files = []
    for subfolder in os.listdir(mmlib_home):
        if os.path.isdir(os.path.join(mmlib_home, subfolder)):
            for file in os.listdir(os.path.join(mmlib_home, subfolder)):
                if file.endswith(".mm"):
                    filename_in_mmlib_home = os.path.join(subfolder, file)
                    if filename_in_mmlib_home.startswith(prefix) and re.match(
                        pattern, filename_in_mmlib_home
                    ):
                        files.append(os.path.join(mmlib_home, subfolder, file))
    return files


# keep only larger instances
dataset_filepaths = get_data()
dataset_sizes = {filepath: os.path.getsize(filepath) for filepath in dataset_filepaths}
dataset_sizes_ser = pd.Series(dataset_sizes).sort_values(ascending=False)
instances = dataset_sizes_ser[:n_instances].index.tolist()
# remove prefix mmlib_home
instances = [instance[len(mmlib_home) + 1 :] for instance in instances]


def problem_factory(instance: str) -> RcpspProblem:
    return parse_file_mmlib(f"{mmlib_home}/{instance}")


p = ParametersCp.default_cpsat()
p.nb_process = 12
solver_configs = {
    "cpsat-multiproc": SolverConfig(
        cls=CpSatRcpspSolver,
        kwargs=dict(
            time_limit=time_limit,
            parameters_cp=p,
        ),
    ),
    "cpsat-auto+CPM": SolverConfig(
        cls=CpSatAutoRcpspSolver,
        kwargs=dict(
            time_limit=time_limit,
            parameters_cp=p,
            use_cpm_for_task_bounds=True,
            avoid_interval_optional=False,
        ),
    ),
    "cpsat-auto+CPM+energy": SolverConfig(
        cls=CpSatAutoRcpspSolver,
        kwargs=dict(
            time_limit=time_limit,
            parameters_cp=p,
            add_energy_constraints=True,
            use_cpm_for_task_bounds=True,
            avoid_interval_optional=False,
        ),
    ),
    "cpsat-auto+no_opt_interval": SolverConfig(
        cls=CpSatAutoRcpspSolver,
        kwargs=dict(
            time_limit=time_limit, parameters_cp=p, avoid_interval_optional=True
        ),
    ),
    "cpsat-auto+no_opt_interval+CPM+energy": SolverConfig(
        cls=CpSatAutoRcpspSolver,
        kwargs=dict(
            time_limit=time_limit,
            parameters_cp=p,
            add_energy_constraints=True,
            use_cpm_for_task_bounds=True,
            avoid_interval_optional=True,
        ),
    ),
    "optal": SolverConfig(
        cls=OptalRcpspSolver,
        kwargs=dict(
            time_limit=time_limit, parameters_cp=p, do_not_retrieve_solutions=True
        ),
    ),
}


def run_study():
    study = Study(
        name=study_name,
        instances=instances,
        solver_configs=solver_configs,
        overwrite=overwrite,
        max_retry=max_retry,
        problem_factory=problem_factory,
    )

    for problem, solver, solver_kwargs in study:
        try:
            stats_cb = StatsWithBoundsCallback()
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ProblemEvaluateLogger(logging.INFO, logging.INFO),
                ],
                **solver_kwargs,
            )
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            logging.error(e)
            reason = f"{type(e)}: {str(e)}"
            success = False
        else:
            # get metrics and solver status
            status = solver.status_solver
            metrics = stats_cb.get_df_metrics()
            success = not is_empty_metrics(metrics)
            if success:
                logging.info("experiment successful")
            else:
                logging.info("experiment unsuccessful (no metrics found)")
            reason = ""
            logging.info(
                f"Instance {study.get_current_instance()},"
                f"Solver config {study.get_current_config_name()},"
                f"value={solver.get_current_best_internal_objective_value()},"
                f"bound={solver.get_current_best_internal_objective_bound()}"
            )
        # store corresponding experiment
        study.store_current_xp(
            metrics=metrics, status=status, reason=reason, success=success
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_study()
