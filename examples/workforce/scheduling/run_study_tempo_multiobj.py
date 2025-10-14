#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from time import perf_counter
from typing import Callable, Optional

import pandas as pd

from discrete_optimization.generic_tools.callbacks.callback import Callback
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
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
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
    ObjectivesEnum,
)
from discrete_optimization.workforce.scheduling.solvers.tempo import (
    TempoLogsCallback,
    TempoScheduler,
)


class LexicoCpsatPrevStartCallback(Callback):
    def on_step_end(
        self, step: int, res: ResultStorage, solver: LexicoSolver
    ) -> Optional[bool]:
        subsolver: CPSatAllocSchedulingSolver = solver.subsolver
        subsolver.set_warm_start_from_previous_run()


class StatsCallbackForLexico(Callback):
    def __init__(self, solver: LexicoSolver, callback_factory: Callable):
        self.solver = solver
        self.starting_time: list[float] = []
        self.end_time: list[float] = []
        self.stats: list[dict] = []
        self.callback_factory = callback_factory
        self.callbacks = []
        self.status = []

    def on_solve_start(self, solver: SolverDO):
        self.starting_time.append(perf_counter())
        self.callbacks.append(self.callback_factory())
        self.callbacks[-1].on_solve_start(solver=solver)

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        return self.callbacks[-1].on_step_end(step, res, solver)

    def on_solve_end(self, res: ResultStorage, solver: SolverDO):
        self.end_time.append(perf_counter())
        self.status.append(solver.status_solver)


study_name = "scheduling-study-final-"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = [os.path.basename(p) for p in get_data_available()]
print(len(instances))
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

solver_configs["tempo-lns-greedy"] = SolverConfig(
    cls=TempoScheduler,
    kwargs={
        "time_limit": 10,
        "path_to_tempo_scheduler": os.environ["TEMPO_PATH"],
        "use_lns": True,
        "greedy_runs": 1,
    },
)
solver_configs["tempo-nolns-greedy"] = SolverConfig(
    cls=TempoScheduler,
    kwargs={
        "time_limit": 10,
        "path_to_tempo_scheduler": os.environ["TEMPO_PATH"],
        "use_lns": False,
        "greedy_runs": 1,
    },
)
solver_configs["tempo-lns-0greedy"] = SolverConfig(
    cls=TempoScheduler,
    kwargs={
        "time_limit": 10,
        "path_to_tempo_scheduler": os.environ["TEMPO_PATH"],
        "use_lns": True,
        "greedy_runs": 0,
    },
)
solver_configs["tempo-nolns-0greedy"] = SolverConfig(
    cls=TempoScheduler,
    kwargs={
        "time_limit": 10,
        "path_to_tempo_scheduler": os.environ["TEMPO_PATH"],
        "use_lns": False,
        "greedy_runs": 0,
    },
)


database_nb_teams_filepath = f"{study_name}_teams.h5"
database_workload_filepath = f"{study_name}_workload.h5"

if overwrite:
    if os.path.exists(database_workload_filepath):
        os.remove(database_workload_filepath)
    if os.path.exists(database_nb_teams_filepath):
        os.remove(database_nb_teams_filepath)

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
                subsolver = CPSatAllocSchedulingSolver(
                    problem,
                    **solver_config.kwargs,
                )
                subsolver.init_model(
                    objectives=[ObjectivesEnum.NB_TEAMS, ObjectivesEnum.DISPERSION],
                    **solver_config.kwargs,
                )
                solver = LexicoSolver(subsolver=subsolver, problem=problem)
                stats_cp_for_lexico = StatsCallbackForLexico(
                    solver=solver, callback_factory=lambda: StatsWithBoundsCallback()
                )
                callbacks = [
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    LexicoCpsatPrevStartCallback(),
                ]
                # solve
                result_store = solver.solve(
                    callbacks=callbacks,
                    objectives=[ObjectivesEnum.NB_TEAMS, ObjectivesEnum.DISPERSION],
                    subsolver_callbacks=[
                        stats_cp_for_lexico,
                        ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
                    ],
                    **solver_config.kwargs,
                )
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            metrics_nb_teams = pd.DataFrame([])
            metrics_workforce_dispersion = pd.DataFrame([])
            status = StatusSolver.ERROR
            reason = f"{type(e)}: {str(e)}"
            print(e)
        else:
            # get metrics and solver status
            status_nb_teams = StatusSolver.UNKNOWN
            status_dispersion = StatusSolver.UNKNOWN
            bs, fit = result_store.get_best_solution_fit()
            if solver_config.cls == CPSatAllocSchedulingSolver:
                call: list[StatsWithBoundsCallback] = stats_cp_for_lexico.callbacks
                print(len(call))
                try:
                    metrics_nb_teams = call[0].get_df_metrics()
                    metrics_nb_teams.index += (
                        subsolver.time_bounds
                    )  # Add time for the init of bounds.
                    metrics_workforce_dispersion = call[1].get_df_metrics()
                    status_nb_teams = stats_cp_for_lexico.status[0]
                    status_dispersion = stats_cp_for_lexico.status[1]
                except Exception as e:
                    status = StatusSolver.ERROR
                    reason = f"{type(e)}: {str(e)}"
                    print(e)
            elif solver_config.cls == TempoScheduler:
                try:
                    stats_cb: TempoLogsCallback
                    metrics_nb_teams = stats_cb.get_df_metrics(phase=0)
                    metrics_workforce_dispersion = stats_cb.get_df_metrics(phase=1)
                except Exception as e:
                    status = StatusSolver.ERROR
                    reason = f"{type(e)}: {str(e)}"
                    print(e)

            reason = ""

        # store corresponding experiment
        with (
            Hdf5Database(database_nb_teams_filepath) as database
        ):  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=instance,
                config_name=str(config_name),
                solver_config=solver_config,
                metrics=metrics_nb_teams,
                status=status_nb_teams,
                reason=reason,
            )
            database.store(xp)

            # store corresponding experiment
        with (
            Hdf5Database(database_workload_filepath) as database
        ):  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=instance,
                config_name=str(config_name),
                solver_config=solver_config,
                metrics=metrics_workforce_dispersion,
                status=status_dispersion,
                reason=reason,
            )
            database.store(xp)
