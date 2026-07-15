#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import plotly.io as pio

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.cpsat_auto import (
    CPSatAutoAllocSchedulingSolver,
    ObjectivesEnum,
)
from discrete_optimization.workforce.scheduling.utils import (
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    instance = [p for p in get_data_available() if "instance_191.json" in p][0]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = CPSatAutoAllocSchedulingSolver(problem)
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
    plotly_schedule_comparison(sol, sol, problem, display=True)


if __name__ == "__main__":
    run_cpsat()
