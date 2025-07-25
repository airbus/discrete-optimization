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
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingProblem
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


def test_problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    assert isinstance(problem, AllocSchedulingProblem)
