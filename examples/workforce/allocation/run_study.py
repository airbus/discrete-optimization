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
from discrete_optimization.workforce.allocation.parser import (
    get_data_available,
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.solvers.cpmpy import (
    CPMpyTeamAllocationSolver,
)
from discrete_optimization.workforce.allocation.solvers.cpsat import (
    OrtoolsTeamAllocationSolver,
)

study_name = "allocation-study-0"
overwrite = False  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = [os.path.basename(p) for p in get_data_available()]
p = ParametersCp.default_cpsat()
solver_configs = {
    "cpmpy-cpsat-1proc": SolverConfig(
        cls=CPMpyTeamAllocationSolver,
        kwargs={"time_limit": 5, "solver": "ortools", "num_search_workers": 1},
    ),
    "cpmpy-cpsat-10proc": SolverConfig(
        cls=CPMpyTeamAllocationSolver,
        kwargs={"time_limit": 5, "solver": "ortools", "num_search_workers": 10},
    ),
    "cpmpy-gurobi": SolverConfig(
        cls=CPMpyTeamAllocationSolver,
        kwargs={"time_limit": 5, "solver": "gurobi"},
    ),
    "cpmpy-exact": SolverConfig(
        cls=CPMpyTeamAllocationSolver,
        kwargs={
            "time_limit": 5,
            # "display": lambda: None,
            "solver": "exact",
        },
    ),
}
solver_configs = {
    "cpmpy-gurobi": SolverConfig(
        cls=CPMpyTeamAllocationSolver,
        kwargs={"time_limit": 5, "solver": "gurobi"},
    )
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
            problem = parse_to_allocation_problem(file)
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
            solver.init_model(**solver_config.kwargs)
            # solve
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    # ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
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
            print(solver)
            status = solver.status_solver
            print(status)
            print(result_store)
            bs, fit = result_store.get_best_solution_fit()
            print(problem.evaluate(bs))
            print(problem.satisfy(bs))
            metrics = stats_cb.get_df_metrics()
            reason = ""

        # store corresponding experiment
        with Hdf5Database(
            database_filepath
        ) as database:  # ensure closing the database at the end of computation (even if error)
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
