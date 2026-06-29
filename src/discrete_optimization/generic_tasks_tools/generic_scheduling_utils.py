#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.skill import Skill


@dataclass
class TaskVariable(Generic[UnaryResource, Skill]):
    """Task characteristics found in a generic scheduling solution."""

    start: int  # start time of the task
    end: int  # end time of the task
    mode: int  # chosen mode for the task
    allocated: dict[UnaryResource, set[Skill]] = field(
        default_factory=dict
    )  # resources allocated to the task
    info: dict[str, Any] = field(
        default_factory=dict
    )  # additional information if needed

    def get_start_or_end(self, start_or_end: StartOrEnd) -> int:
        if start_or_end == StartOrEnd.START:
            return self.start
        else:
            return self.end


@dataclass
class RawSolution(Generic[Task, UnaryResource, Skill]):
    """Raw format for a generic scheduling solution

    Does not inherit from d-o `Solution` class.

    You can do `raw_sol_1 | raw_sol_2`, it will return another raw solution merging both `task_variables` dictionaries,
    but dropping metadata.

    ."""

    task_variables: dict[Task, TaskVariable[UnaryResource, Skill]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __or__(
        self, other: RawSolution[Task, UnaryResource, Skill]
    ) -> RawSolution[Task, UnaryResource, Skill]:
        return RawSolution(
            task_variables=self.task_variables | other.task_variables,
        )


class Objective(Enum):
    """Objective for a generic scheduling problem."""

    MAKESPAN = "makespan"
    """Global makespan of the schedule, to minimize."""
    NB_TASKS_DONE = "nb_tasks_done"
    """Number of tasks with at least one resource allocated, to maximize."""
    NB_UNARY_RESOURCES_USED = "nb_unary_resources_used"
    """Number of allocated unary resources, to minimize."""
    NB_RESOURCES_USED = "nb_resources_used"
    """Weighted sum of resources used, to minimize.

    Include non-renewable, cumulative, and unary resources.
    The weigths are to be defined in `solver.objective_resource_weights`.

    """
    RESOURCES_LEVELS = "resources_levels"
    """Weighted sum of resources levels (i.e. needed capacities), to minimize.

    Include non-renewable, cumulative, and unary resources.
    The weigths are to be defined in `solver.objective_resource_weights`.

    """
    COST = "cost"
    """Cost of the solution taking into account mode choice and resources consumptions."""
    CUSTOM = "custom_objective"


OBJECTIVE_DEFAULT_WEIGHTS: dict[Objective, int] = {
    Objective.MAKESPAN: -1,
    Objective.NB_TASKS_DONE: 1,
    Objective.NB_UNARY_RESOURCES_USED: -1,
    Objective.NB_RESOURCES_USED: -1,
    Objective.RESOURCES_LEVELS: -1,
    Objective.COST: -1,
    Objective.CUSTOM: 1,
}
"""Default weight applied to a given objective so that it will be *maximized*."""


class Penalty(Enum):
    "Penalties for a generic scheduling problem."

    TIME = "time_penalty"


PENALTY_DEFAULT_WEIGHTS: dict[Penalty, int] = {
    Penalty.TIME: -100,
}
"""Default weight applied to a given penalty to be added to the objective so that it will be *maximized*."""
