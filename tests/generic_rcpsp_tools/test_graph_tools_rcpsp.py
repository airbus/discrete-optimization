#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import pytest

from discrete_optimization.generic_rcpsp_tools.graph_tools_rcpsp import (
    build_graph_rcpsp_object,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file

files_rcpsp = get_data_available()
single_modes_files = [f for f in files_rcpsp if "sm" in f]
multi_modes_files = [f for f in files_rcpsp if "mm" in f]


@pytest.mark.parametrize("rcpsp_model_file", single_modes_files[:1])
def test_graph_tools(rcpsp_model_file):
    rcpsp_model: RCPSPModel = parse_file(rcpsp_model_file)
    assert rcpsp_model.is_rcpsp_multimode() is False
    assert rcpsp_model.is_varying_resource() is False
    graph_rcpsp = build_graph_rcpsp_object(rcpsp_problem=rcpsp_model)
    assert graph_rcpsp.check_loop() is None
    rcpsp_model_copy = rcpsp_model.copy()
    # We choose a task that has successors
    some_task = random.choice(
        [t for t in rcpsp_model.successors if len(rcpsp_model.successors[t]) > 0]
    )
    some_successor_task = random.choice(rcpsp_model.successors[some_task])
    rcpsp_model_copy.successors[some_successor_task].append(some_task)
    graph_rcpsp_with_loop = build_graph_rcpsp_object(rcpsp_model_copy)
    cycles = graph_rcpsp_with_loop.check_loop()
    # With this precedence graph, the rcpsp is no longer feasible.
    assert cycles is not None
