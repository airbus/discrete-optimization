#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.rcpsp_multiskill import (
    parser_imopse,
    parser_mslib,
    parser_mspsp,
)


@pytest.mark.parametrize("module", [parser_mspsp, parser_mslib, parser_imopse])
def test_no_dataset(fake_data_home, module):
    with pytest.raises(
        FileNotFoundError, match="python -m discrete_optimization.datasets"
    ):
        module.get_data_available()
