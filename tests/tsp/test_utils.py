#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.utils import closest_greedy


def test_common_tools():
    file_location = [f for f in get_data_available() if f.endswith("tsp_574_1")][0]
    model = parse_file(file_location)
    sol, length_circuit, opt = closest_greedy(model.node_count, model.list_points)
    assert len(sol) == 574
