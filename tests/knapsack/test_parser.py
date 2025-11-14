#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.knapsack.parser import get_data_available, parse_file


def test_parser():
    file_location = [f for f in get_data_available() if f.endswith("ks_4_0")][0]
    knapsack_problem = parse_file(file_location)
    assert knapsack_problem.nb_items == 4
    assert knapsack_problem.max_capacity == 11


def test_no_dataset(fake_data_home):
    with pytest.raises(
        FileNotFoundError, match="python -m discrete_optimization.datasets"
    ):
        get_data_available()


if __name__ == "__main__":
    test_parser()
