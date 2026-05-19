#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import pandas as pd

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
    Experiment,
    Hdf5Database,
    SolverConfig,
)
from discrete_optimization.rcpsp.parser import parse_file_mmlib
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.optal import OptalRcpspSolver

logging.basicConfig(level=logging.INFO)
import os

from discrete_optimization.datasets import get_data_home

mmlib_home = os.path.join(get_data_home(), "mmlib")


def get_data():
    files = []
    for subfolder in os.listdir(mmlib_home):
        if os.path.isdir(os.path.join(mmlib_home, subfolder)):
            for file in os.listdir(os.path.join(mmlib_home, subfolder)):
                if file[-2:] == "mm":
                    files.append(os.path.join(mmlib_home, subfolder, file))
    return files


def run_study():
    study_name = "mmlibtest"
    overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
    instances = get_data()
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    solver_configs = {
        "cpsat-multiproc": SolverConfig(
            cls=CpSatRcpspSolver,
            kwargs=dict(
                time_limit=30,
                parameters_cp=p,
            ),
        ),
        "optal": SolverConfig(
            cls=OptalRcpspSolver,
            kwargs=dict(time_limit=30, parameters_cp=p),
        ),
    }

    database_filepath = f"{study_name}.h5"
    if overwrite:
        try:
            os.remove(database_filepath)
        except FileNotFoundError:
            pass

    # loop over instances x configs
    for instance in instances:
        for config_name, solver_config in solver_configs.items():
            logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")
            try:
                # init problem
                problem = parse_file_mmlib(instance)
                # init solver
                stats_cb = StatsWithBoundsCallback()
                solver = solver_config.cls(problem, **solver_config.kwargs)
                solver.init_model(**solver_config.kwargs)
                # solve
                result_store = solver.solve(
                    callbacks=[
                        stats_cb,
                        NbIterationTracker(step_verbosity_level=logging.INFO),
                        ProblemEvaluateLogger(logging.INFO, logging.INFO),
                    ],
                    **solver_config.kwargs,
                )
            except Exception as e:
                # failed experiment
                metrics = pd.DataFrame([])
                status = StatusSolver.ERROR
                print(e)
                reason = f"{type(e)}: {str(e)}"
            else:
                # get metrics and solver status
                status = solver.status_solver
                metrics = stats_cb.get_df_metrics()
                reason = ""
                print(
                    "Instance",
                    os.path.basename(instance),
                    "solver",
                    solver_config.cls,
                    "value",
                    result_store[-1][1],
                    "bound",
                    solver.get_current_best_internal_objective_bound(),
                )
            # store corresponding experiment
            with (
                Hdf5Database(database_filepath) as database
            ):  # ensure closing the database at the end of computation (even if error)
                xp_id = database.get_new_experiment_id()
                xp = Experiment.from_solver_config(
                    xp_id=xp_id,
                    instance=os.path.basename(instance),
                    config_name=config_name,
                    solver_config=solver_config,
                    metrics=metrics,
                    status=status,
                    reason=reason,
                )
                database.store(xp)


if __name__ == "__main__":
    run_study()
