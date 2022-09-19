#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)


def test_parser():
    file_location = [f for f in get_data_available() if f.endswith("ks_4_0")][0]
    knapsack_model = parse_file(file_location)
    assert knapsack_model.nb_items == 4
    assert knapsack_model.max_capacity == 11


if __name__ == "__main__":
    test_parser()
