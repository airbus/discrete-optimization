#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

import pandas as pd

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat_relaxed import (
    CPSatAllocSchedulingSolverCumulative,
)
from discrete_optimization.workforce.scheduling.solvers.tempo import (
    TempoLogsCallback,
    TempoScheduler,
)

study_name = "scheduling-study-tempo-0"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = [os.path.basename(p) for p in get_data_available()][:20]
p = ParametersCp.default_cpsat()
p.nb_process = 10
p_mono_worker = ParametersCp.default_cpsat()
p_mono_worker.nb_process = 1
solver_configs = {}
for p_cp, label in zip([p, p_mono_worker], ["multipro-10", "monopro"]):
    for add_lower_bound in [True, False]:
        solver_configs[("cpsat", label, add_lower_bound)] = SolverConfig(
            cls=CPSatAllocSchedulingSolver,
            kwargs={
                "parameters_cp": p_cp,
                "time_limit": 10,
                "add_lower_bound": add_lower_bound,
            },
        )

solver_configs["tempo"] = SolverConfig(
    cls=TempoScheduler,
    kwargs={"time_limit": 10, "path_to_tempo_scheduler": os.environ["TEMPO_PATH"]},
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
            file = [f for f in get_data_available() if instance in f][0]
            problem = parse_json_to_problem(file)
            # init solver
            if solver_config.cls == TempoScheduler:
                stats_cb = TempoLogsCallback()
                solver = solver_config.cls(
                    problem,
                    params_objective_function=ParamsObjectiveFunction(
                        objective_handling=ObjectiveHandling.SINGLE,
                        objectives=["nb_teams"],
                        weights=[1],
                        sense_function=ModeOptim.MINIMIZATION,
                    ),
                    **solver_config.kwargs,
                )
                solver.init_model(**solver_config.kwargs)
                # solve
                result_store = solver.solve(
                    callbacks=[stats_cb],
                    **solver_config.kwargs,
                )
            if solver_config.cls == CPSatAllocSchedulingSolver:
                stats_cb = StatsWithBoundsCallback()
                solver = solver_config.cls(
                    problem,
                    params_objective_function=ParamsObjectiveFunction(
                        objective_handling=ObjectiveHandling.SINGLE,
                        objectives=["nb_teams"],
                        weights=[1],
                        sense_function=ModeOptim.MINIMIZATION,
                    ),
                    **solver_config.kwargs,
                )
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
            print(e)
        else:
            # get metrics and solver status
            status = solver.status_solver
            bs, fit = result_store.get_best_solution_fit()
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
                config_name=str(config_name),
                solver_config=solver_config,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
