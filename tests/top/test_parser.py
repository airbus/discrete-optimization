#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.top.parser import get_data_available, parse_file


def test_parser():
    files, files_dict = get_data_available()
    print(files[0])
    for f in files:
        problem = parse_file(f)
        print(problem)
