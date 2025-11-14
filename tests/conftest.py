#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest import fixture

from discrete_optimization.datasets import DO_DEFAULT_DATAHOME_ENVVARNAME


@fixture
def fake_data_home(monkeypatch):
    data_home = "~/discrete_optimization_data_not_existing"
    monkeypatch.setenv(DO_DEFAULT_DATAHOME_ENVVARNAME, data_home)
