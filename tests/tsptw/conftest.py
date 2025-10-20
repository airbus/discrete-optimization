#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file


@pytest.fixture
def problem():
    filename = "rc_201.1.txt"
    filepath = [f for f in get_data_available() if f.endswith(filename)][0]
    problem = parse_tsptw_file(filepath)
    return problem
