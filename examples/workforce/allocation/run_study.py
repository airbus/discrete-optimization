#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from copy import copy, deepcopy

import didppy as dp
import pandas as pd

from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
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
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpmpy import (
    CPMpyTeamAllocationSolver,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    CpsatTeamAllocationSolver,
    ModelisationAllocationOrtools,
)
from discrete_optimization.workforce.allocation.solvers.dp import DpAllocationSolver
from discrete_optimization.workforce.allocation.solvers.optal import (
    OptalTeamAllocationSolver,
)

study_name = "allocation-study-0"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates

if __name__ == "__main__":
    instances = [os.path.basename(p) for p in get_data_available()]
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    solver_configs = {
        "cpmpy-cpsat-1proc": SolverConfig(
            cls=CPMpyTeamAllocationSolver,
            kwargs={"time_limit": 5, "solver_name": "ortools", "num_search_workers": 1},
        ),
        "optal": SolverConfig(
            cls=OptalTeamAllocationSolver, kwargs={"time_limit": 5, "parameters_cp": p}
        ),
        "cpmpy-cpsat-10proc": SolverConfig(
            cls=CPMpyTeamAllocationSolver,
            kwargs={
                "time_limit": 5,
                "solver_name": "ortools",
                "num_search_workers": 10,
            },
        ),
        "cpmpy-exact": SolverConfig(
            cls=CPMpyTeamAllocationSolver,
            kwargs={
                "time_limit": 5,
                "display": lambda: None,
                "solver_name": "exact",
            },
        ),
        "cpsat-10": SolverConfig(
            cls=CpsatTeamAllocationSolver,
            kwargs={
                "parameters_cp": p,
                "time_limit": 5,
                "add_lower_bound_nb_teams": True,
            },
        ),
        "cpsat-10-integer": SolverConfig(
            cls=CpsatTeamAllocationSolver,
            kwargs={
                "parameters_cp": p,
                "time_limit": 5,
                "modelisation_allocation": ModelisationAllocationOrtools.INTEGER,
                "add_lower_bound_nb_teams": True,
            },
        ),
    }
    for solver in ["CABS", "LNBS"]:
        solver_configs[f"dp-{solver}-0"] = SolverConfig(
            cls=DpAllocationSolver,
            kwargs=dict(
                solver=solver, force_allocation_when_possible=False, time_limit=5
            ),
        )
        solver_configs[f"dp-{solver}-1"] = SolverConfig(
            cls=DpAllocationSolver,
            kwargs=dict(
                solver=solver, force_allocation_when_possible=True, time_limit=5
            ),
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
                problem = parse_to_allocation_problem(file, multiobjective=False)
                # init solver
                # stats_cb = StatsWithBoundsCallback()
                stats_cb = BasicStatsCallback()
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
                if solver_config.cls == CPMpyTeamAllocationSolver:
                    continue
                    tr_kwargs = copy(solver_config.kwargs)
                    tr_kwargs.pop("solver_name")
                    solver.init_model(**tr_kwargs)
                else:
                    solver.init_model(**solver_config.kwargs)
                kwargs_modif = deepcopy(solver_config.kwargs)
                if isinstance(solver, DpAllocationSolver):
                    kwargs_modif["solver"] = {"LNBS": dp.LNBS, "CABS": dp.CABS}[
                        kwargs_modif["solver"]
                    ]
                if solver_config.cls == CPMpyTeamAllocationSolver:
                    kwargs_modif = tr_kwargs
                # solve
                result_store = solver.solve(
                    callbacks=[
                        stats_cb,
                        NbIterationTracker(step_verbosity_level=logging.INFO),
                        # ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
                    ],
                    **kwargs_modif,
                )
            except Exception as e:
                # failed experiment
                metrics = pd.DataFrame([])
                status = StatusSolver.ERROR
                reason = f"{type(e)}: {str(e)}"
                print(e)
            else:
                # get metrics and solver status
                print(solver)
                status = solver.status_solver
                print(status)
                print(result_store)
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
                    config_name=config_name,
                    solver_config=solver_config,
                    metrics=metrics,
                    status=status,
                    reason=reason,
                )
                database.store(xp)
