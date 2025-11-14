#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.tsp.parser import get_data_available


def test_no_dataset(fake_data_home):
    with pytest.raises(
        FileNotFoundError, match="python -m discrete_optimization.datasets"
    ):
        get_data_available()
