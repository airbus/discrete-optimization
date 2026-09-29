#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any, Optional, Union

from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.utils import optional_override_implem
from discrete_optimization.rcpsp import RcpspProblem
from discrete_optimization.rcpsp.solution import (
    Task,
)
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


class RcpspProblemWithExclusion(RcpspProblem):
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
        exclusion_resource_capacity: dict[ExclusionResource, int] = None,
        exclusion_resource_consumptions: dict[
            Task, dict[int, dict[ExclusionResource, int]]
        ] = None,
        exclusion_resource_boolean: dict[
            Task, dict[int, dict[ExclusionResource, bool]]
        ] = None,
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
        self.exclusion_resource_capacity = exclusion_resource_capacity
        self.exclusion_resource_consumptions = exclusion_resource_consumptions
        self.exclusion_resource_boolean = exclusion_resource_boolean

    @optional_override_implem
    def is_task_mode_excluding_others(
        self, task: Task, mode: int, resource: ExclusionResource
    ) -> bool:
        if task in self.exclusion_resource_boolean:
            if mode in self.exclusion_resource_boolean[task]:
                if resource in self.exclusion_resource_boolean[task][mode]:
                    return self.exclusion_resource_boolean[task][mode][resource]
        return False

    @optional_override_implem
    def get_task_consumption_exclusion_resource(
        self, resource: ExclusionResource, task: Task, mode: int
    ):
        if task in self.exclusion_resource_consumptions:
            if mode in self.exclusion_resource_consumptions[task]:
                if resource in self.exclusion_resource_consumptions[task][mode]:
                    return self.exclusion_resource_consumptions[task][mode][resource]
        return 0

    @property
    @optional_override_implem
    def exclusion_resources_list(self) -> list[ExclusionResource]:
        return list(self.exclusion_resource_capacity.keys())

    @optional_override_implem
    def get_capacity_exclusion_resource(self, resource: ExclusionResource):
        return self.exclusion_resource_capacity[resource]
