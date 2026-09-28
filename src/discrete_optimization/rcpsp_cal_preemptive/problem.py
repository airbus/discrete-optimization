#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any, Optional, Union

from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp.solution import (
    Task,
)
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


class CalendarPreemptiveRcpspProblem(RcpspProblem):
    def __init__(
        self,
        resources: dict[str, Union[int, list[int]]],
        non_renewable_resources: list[str],
        mode_details: dict[Hashable, dict[int, dict[str, int]]],
        successors: dict[Hashable, list[Hashable]],
        horizon: int,
        tasks_list: Optional[list[Hashable]] = None,
        source_task: Optional[Hashable] = None,
        sink_task: Optional[Hashable] = None,
        name_task: Optional[dict[Hashable, str]] = None,
        calendar_details: Optional[dict[str, list[list[int]]]] = None,
        special_constraints: Optional[SpecialConstraintsDescription] = None,
        fixed_permutation: Optional[list[int]] = None,
        fixed_modes: Optional[list[int]] = None,
        calendar_preemptive_tasks: set[Hashable] = None,
        **kwargs: Any,
    ):
        super().__init__(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            tasks_list=tasks_list,
            source_task=source_task,
            sink_task=sink_task,
            name_task=name_task,
            calendar_details=calendar_details,
            special_constraints=special_constraints,
            fixed_permutation=fixed_permutation,
            fixed_modes=fixed_modes,
            **kwargs,
        )
        self.calendar_preemptive_tasks = calendar_preemptive_tasks

    def is_task_calendar_preempted(self, task: Task) -> bool:
        return task in self.calendar_preemptive_tasks
