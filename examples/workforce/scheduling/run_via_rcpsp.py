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
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.rcpsp.solvers.cpsat_auto import (
    CpSatAutoCumulativeResourceRcpspSolver,
)
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.transformations.to_rcpsp import (
    WorkforceSchedulingToRcpspTransformation,
)
from discrete_optimization.workforce.scheduling.utils import (
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    instance = [p for p in get_data_available() if "instance_191.json" in p][0]
    problem = parse_json_to_problem(instance)
    problem.same_allocation = []
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver = TransformationSolver(
        transformation=WorkforceSchedulingToRcpspTransformation(True, True, True),
        solver_brick=SubBrick(
            CpSatAutoCumulativeResourceRcpspSolver,
            {
                "time_limit": 100,
                "parameters_cp": p,
                "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
            },
        ),
        source_problem=problem,
    )
    res = solver.solve(
        callbacks=[ObjectiveGapStopper(0, 0), BasicStatsCallback()],
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)


if __name__ == "__main__":
    run_cpsat()
