#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.singlemachine.parser import get_data_available, parse_file


@pytest.fixture()
def problem():
    filename = "wt40.txt"
    filepath = [f for f in get_data_available() if f.endswith(filename)][0]
    problems = parse_file(filepath)
    problem = problems[0]
    return problem
