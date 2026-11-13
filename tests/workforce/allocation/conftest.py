#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.workforce.allocation.parser import (
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.scheduling.parser import get_data_available


@pytest.fixture()
def problem():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    return parse_to_allocation_problem(instance, multiobjective=True)
