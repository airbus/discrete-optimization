#  Copyright (c) 2025 AIRBUS and its affiliates.
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
from discrete_optimization.workforce.scheduling.solvers.optal import (
    OptalAllocSchedulingSolver,
)
from discrete_optimization.workforce.scheduling.utils import (
    compute_changes_between_solution,
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_optal():
    instance = get_data_available()[1]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = OptalAllocSchedulingSolver(problem)
    solver.init_model(model_dispersion=True, run_lexico=True)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=30,
        **{
            "worker0-3.searchType": "fdslb",
            "worker0-1.noOverlapPropagationLevel": 4,
            "worker0-10.cumulPropagationLevel": 3,
            "worker0-10.reservoirPropagationLevel": 2,
        },
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)


if __name__ == "__main__":
    run_optal()
