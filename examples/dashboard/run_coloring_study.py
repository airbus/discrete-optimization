import logging
import os

import pandas as pd

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.coloring.solvers.lp import (
    GurobiColoringSolver,
    MathOptColoringSolver,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
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

logging.basicConfig(level=logging.INFO)

study_name = "Coloring-Study-0"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = ["gc_50_3", "gc_50_1"]
solver_configs = {
    "cpsat-integer": SolverConfig(
        cls=CpSatColoringSolver,
        kwargs=dict(
            parameters_cp=ParametersCp.default_cpsat(),
            modeling=ModelingCpSat.INTEGER,
            do_warmstart=False,
            value_sequence_chain=False,
            used_variable=True,
            symmetry_on_used=True,
        ),
    ),
    "cpsat-binary": SolverConfig(
        cls=CpSatColoringSolver,
        kwargs=dict(
            parameters_cp=ParametersCp.default_cpsat(),
            modeling=ModelingCpSat.BINARY,
            do_warmstart=False,
            value_sequence_chain=False,
            used_variable=True,
            symmetry_on_used=True,
        ),
    ),
    "gurobi": SolverConfig(cls=GurobiColoringSolver, kwargs=dict()),
    "mathopt": SolverConfig(cls=MathOptColoringSolver, kwargs=dict()),
}

database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass
with Hdf5Database(
    database_filepath
) as database:  # ensure closing the database at the end of computation (even if error)

    # loop over instances x configs
    for instance in instances:
        for config_name, solver_config in solver_configs.items():

            logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")

            try:
                # init problem
                file = [f for f in get_data_available() if instance in f][0]
                color_problem = parse_file(file)
                # init solver
                stats_cb = StatsWithBoundsCallback()
                solver = solver_config.cls(color_problem, **solver_config.kwargs)
                solver.init_model(**solver_config.kwargs)
                # solve
                result_store = solver.solve(
                    callbacks=[
                        stats_cb,
                        NbIterationTracker(step_verbosity_level=logging.INFO),
                    ],
                    **solver_config.kwargs,
                )
            except Exception as e:
                # failed experiment
                metrics = pd.DataFrame([])
                status = StatusSolver.ERROR
                reason = f"{type(e).__name__}: {str(e)}"
            else:
                # get metrics and solver status
                status = solver.status_solver
                metrics = stats_cb.get_df_metrics()
                reason = ""

            # store corresponding experiment
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=instance,
                config_name=config_name,
                solver_config=solver_config,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
