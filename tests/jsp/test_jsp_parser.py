#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.jsp.job_shop_parser import get_data_available, parse_file
from discrete_optimization.jsp.job_shop_problem import JobShopProblem


@pytest.mark.parametrize("jsp_file", get_data_available())
def test_parser(jsp_file: str):
    model: JobShopProblem = parse_file(file_path=jsp_file)
    assert isinstance(model, JobShopProblem)
    assert model.n_jobs > 0
