#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_resource_dependent.problem import (
    RcpspResourceDependentProblem,
)


@pytest.fixture
def toy_problem():
    """Fixture providing a toy RCPSP resource-dependent problem instance."""
    resources = {"R1": 5, "R2": 8, "R3": 10, "N1": 20, "N2": 20}
    mode_details = {
        "source": {1: {"duration": 0}},
        "1": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "2": {
            1: {
                "R1": {frozenset([("1", 1)]): 1, frozenset([("1", 2)]): 5},
                "N1": {frozenset([("1", 1)]): 20, frozenset([("1", 2)]): 10},
                "duration": 2,
            },
            2: {
                "R2": {frozenset([("1", 1)]): 1, frozenset([("1", 2)]): 5},
                "N2": 5,
                "duration": 1,
            },
        },
        "3": {
            1: {"R1": 1, "N1": 5, "duration": 3},
            2: {"R2": 4, "N2": 5, "duration": 2},
        },
        "4": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 3},
        },
        "5": {
            1: {"R1": 1, "N1": 5, "duration": 4},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "6": {
            1: {"R1": 1, "N1": 5, "duration": 2},
            2: {"R2": 4, "N2": 5, "duration": 1},
        },
        "sink": {1: {"duration": 0}},
    }
    successors = {
        "source": ["1", "2"],
        "1": ["3", "4"],
        "2": ["5"],
        "3": ["4"],
        "4": ["6"],
        "5": ["sink"],
        "6": ["sink"],
        "sink": [],
    }
    return RcpspResourceDependentProblem(
        resources=resources,
        non_renewable_resources=["N1", "N2"],
        mode_details=mode_details,
        successors=successors,
        horizon=30,
        source_task="source",
        sink_task="sink",
    )


@pytest.fixture
def base_rcpsp_problem():
    """Fixture providing a base RCPSP problem for transformation."""
    resources = {"R1": 5, "R2": 8, "N1": 20}
    mode_details = {
        "source": {1: {"duration": 0}},
        "1": {
            1: {"R1": 2, "N1": 3, "duration": 3},
            2: {"R2": 3, "N1": 5, "duration": 2},
        },
        "2": {1: {"R1": 3, "R2": 1, "N1": 4, "duration": 4}},
        "3": {
            1: {"R1": 1, "N1": 2, "duration": 2},
            2: {"R2": 2, "N1": 3, "duration": 3},
        },
        "4": {1: {"R2": 2, "N1": 5, "duration": 2}},
        "sink": {1: {"duration": 0}},
    }
    successors = {
        "source": ["1", "2"],
        "1": ["3"],
        "2": ["4"],
        "3": ["sink"],
        "4": ["sink"],
        "sink": [],
    }
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=["N1"],
        mode_details=mode_details,
        successors=successors,
        horizon=20,
        source_task="source",
        sink_task="sink",
    )
