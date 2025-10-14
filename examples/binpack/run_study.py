import logging
import os

import pandas as pd

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
)
from discrete_optimization.binpack.solvers.greedy import (
    GreedyBinPackOpenEvolve,
    GreedyBinPackSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
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

study_name = "bppc-study-0"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = [os.path.basename(p) for p in get_data_available_bppc()][:100]
p = ParametersCp.default_cpsat()
p.nb_process = 10
solver_configs = {}
for proc in [16]:
    p = ParametersCp.default_cpsat()
    p.nb_process = proc
    for modeling in [ModelingBinPack.SCHEDULING]:  # , ModelingBinPack.BINARY]:
        solver_configs[f"cpsat-proc-{proc}-{modeling.name}"] = SolverConfig(
            cls=CpSatBinPackSolver,
            kwargs={
                "time_limit": 10,
                "upper_bound": 250,
                "modeling": modeling,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
                "parameters_cp": p,
            },
        )
solver_configs = {}
solver_configs["greedy-1"] = SolverConfig(cls=GreedyBinPackSolver, kwargs={})
solver_configs["greedy-open-evolve"] = SolverConfig(
    cls=GreedyBinPackOpenEvolve, kwargs={}
)
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
            file = [f for f in get_data_available_bppc() if instance in f][0]
            color_problem = parse_bin_packing_constraint_file(file)
            # init solver
            if solver_config.cls == CpSatBinPackSolver:
                stats_cb = StatsWithBoundsCallback()
            else:
                stats_cb = BasicStatsCallback()
            solver = solver_config.cls(color_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
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
