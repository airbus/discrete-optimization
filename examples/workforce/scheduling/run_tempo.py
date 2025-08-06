import logging
import os

import pandas as pd
import plotly.io as pio

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
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
from discrete_optimization.workforce.scheduling.solvers.tempo import (
    TempoLogsCallback,
    TempoScheduler,
)
from discrete_optimization.workforce.scheduling.utils import plotly_schedule_comparison

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_tempo():
    instance = [p for p in get_data_available() if "instance_0.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = TempoScheduler(problem, path_to_tempo_scheduler=os.environ["TEMPO_PATH"])
    callback = TempoLogsCallback()
    solver.init_model()
    res = solver.solve(
        callbacks=[callback],
        time_limit=10,
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    metrics = callback.get_df_metrics()
    plotly_schedule_comparison(
        base_solution=sol, updated_solution=sol, problem=problem, display=True
    )
    print(callback.logs)


if __name__ == "__main__":
    run_tempo()
