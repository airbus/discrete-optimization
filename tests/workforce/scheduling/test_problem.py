#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import plotly.io as pio

from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingProblem

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def test_problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    assert isinstance(problem, AllocSchedulingProblem)
