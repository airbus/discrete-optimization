import logging
import os

import pandas as pd

from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.hub_solver.tempo.tempo_tools import (
    TempoLogsCallback,
)
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.dp import DpRcpspModeling, DpRcpspSolver
from discrete_optimization.rcpsp.solvers.tempo import TempoRcpspSolver

logging.basicConfig(level=logging.INFO)

study_name = "rcpsp-study-with-dp"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = [os.path.basename(p) for p in get_data_available() if "sm" in p]
p = ParametersCp.default_cpsat()
p.nb_process = 32
solver_configs = {
    "cpsat-multiproc": SolverConfig(
        cls=CpSatRcpspSolver,
        kwargs=dict(
            time_limit=30,
            parameters_cp=p,
        ),
    ),
    "dp-task-no-domin": SolverConfig(
        cls=DpRcpspSolver,
        kwargs=dict(
            modeling=DpRcpspModeling.TASK_ORIGINAL,
            add_dominated_transition=False,
            solver="CABS",
            threads=32,
            time_limit=30,
        ),
    ),
    "dp-task-domin": SolverConfig(
        cls=DpRcpspSolver,
        kwargs=dict(
            modeling=DpRcpspModeling.TASK_ORIGINAL,
            add_dominated_transition=True,
            solver="CABS",
            threads=32,
            time_limit=30,
        ),
    ),
    "dp-task-time": SolverConfig(
        cls=DpRcpspSolver,
        kwargs=dict(
            modeling=DpRcpspModeling.TASK_AND_TIME,
            solver="CABS",
            threads=32,
            time_limit=30,
        ),
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
            file = [f for f in get_data_available() if instance in f][0]
            problem = parse_file(file)
            # init solver
            stats_cb = StatsWithBoundsCallback()
            if config_name in {"sa", "hc"} or solver_config.cls == DpRcpspSolver:
                stats_cb = BasicStatsCallback()
            if solver_config.cls == TempoRcpspSolver:
                stats_cb = TempoLogsCallback()
            solver = solver_config.cls(problem, **solver_config.kwargs)
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
            reason = f"{type(e)}: {str(e)}"
        else:
            # get metrics and solver status
            status = solver.status_solver
            if isinstance(stats_cb, TempoLogsCallback):
                metrics = stats_cb.get_df_metrics(phase=0)
                metrics["fit"] = -metrics["fit"]
            else:
                metrics = stats_cb.get_df_metrics()
            reason = ""

        # store corresponding experiment
        with (
            Hdf5Database(database_filepath) as database
        ):  # ensure closing the database at the end of computation (even if error)
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
