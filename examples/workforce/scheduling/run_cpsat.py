#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

import plotly.io as pio

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.workforce.generators.resource_scenario import (
    ParamsRandomness,
    create_scheduling_problem_several_resource_dropping,
    generate_scheduling_disruption,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
    ObjectivesEnum,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat_relaxed import (
    CPSatAllocSchedulingSolverCumulative,
)
from discrete_optimization.workforce.scheduling.utils import plotly_schedule_comparison

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


def run_cpsat_delta():
    from discrete_optimization.workforce.scheduling.utils import (
        compute_changes_between_solution,
    )

    instances = [p for p in get_data_available()]
    problem = parse_json_to_problem(instances[1])
    solver = CPSatAllocSchedulingSolver(problem)
    solver.init_model(
        objectives=[ObjectivesEnum.NB_TEAMS], adding_redundant_cumulative=True
    )
    sol, _ = solver.solve(time_limit=2).get_best_solution_fit()
    d = generate_scheduling_disruption(
        original_scheduling_problem=problem,
        original_solution=sol,
        list_drop_resource=None,
        params_randomness=ParamsRandomness(
            lower_nb_disruption=1,
            upper_nb_disruption=2,
            lower_nb_teams=1,
            upper_nb_teams=1,
        ),
    )
    solver_relaxed = CPSatAllocSchedulingSolver(d["scheduling_problem"])
    solver_relaxed.init_model(
        objectives=[
            ObjectivesEnum.NB_DONE_AC,
            ObjectivesEnum.DELTA_TO_EXISTING_SOLUTION,
            ObjectivesEnum.NB_TEAMS,
        ],
        additional_constraints=d["additional_constraint_scheduling"],
        optional_activities=True,
        base_solution=sol,
    )
    res = solver_relaxed.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
    )
    plotly_schedule_comparison(
        base_solution=sol,
        updated_solution=res[-1][0],
        problem=d["scheduling_problem"],
        use_color_map_per_task=False,
        color_map_per_task={},
        plot_team_breaks=True,
        display=True,
    )
    print(d)


if __name__ == "__main__":
    run_cpsat_delta()
