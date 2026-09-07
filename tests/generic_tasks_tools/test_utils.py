#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from pytest_cases import parametrize

from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.utils import (
    get_mandatory_methods_to_implement,
    get_optional_methods_to_override,
)


@parametrize("include_subclass_overrides", [True, False])
def test_get_optional_methods_to_override(include_subclass_overrides):
    methods = get_optional_methods_to_override(
        GenericSchedulingProblem, include_subclass_overrides=include_subclass_overrides
    )
    assert "get_end_to_end_max_time_lags" in methods
    assert ("get_last_tasks" in methods) == include_subclass_overrides


def test_get_mandatory_methods_to_implement():
    methods = get_mandatory_methods_to_implement(GenericSchedulingProblem)
    assert "get_task_mode_duration" in methods
