#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)


def run_parser():
    f = [ff for ff in get_data_available_bppc() if "BPPC_1_1_5.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    print(problem.list_items)
    print(problem.incompatible_items)


if __name__ == "__main__":
    run_parser()
