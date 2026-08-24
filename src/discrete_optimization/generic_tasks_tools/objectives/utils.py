#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Type

from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.allocated_tasks import (
    AllocatedTasksObjective,
)
from discrete_optimization.generic_tasks_tools.objectives.allocation_changes import (
    AllocationSwitchObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.allocation_cost import (
    AllocationCostComputer,
    AllocationCostComputerMultimode,
)
from discrete_optimization.generic_tasks_tools.objectives.earliness_tardiness import (
    EarlinessTardinessComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.makespan import (
    MakespanObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.mode_cost import (
    ModeCostComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.resource_levels import (
    CalendarRenewableResourceLevelObjectiveComputer,
    NonRenewableResourceLevelObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.schedule_changes import (
    ScheduleChangesComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.soft_time_penalty import (
    SoftTimePenaltyComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)


def get_mapping():
    list_available = [
        AllocatedTasksObjective,
        AllocationSwitchObjectiveComputer,
        AllocationCostComputer,
        AllocationCostComputerMultimode,
        EarlinessTardinessComputer,
        MakespanObjectiveComputer,
        ModeCostComputer,
        NonRenewableResourceLevelObjectiveComputer,
        CalendarRenewableResourceLevelObjectiveComputer,
        ScheduleChangesComputer,
        SoftTimePenaltyComputer,
        UnaryResourcesUsedComputer,
    ]
    mapping = {}
    for objective_computer in list_available:
        objective_computer: ObjectiveComputer
        key = objective_computer.get_objective_name()
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(objective_computer)
    return mapping


def get_objective_computer_class(objective: Objective) -> Type[ObjectiveComputer]:
    mapping = get_mapping()
    return mapping.get(objective, None)


if __name__ == "__main__":
    print(get_mapping())
    for k in get_mapping().keys():
        print(len(get_mapping()[k]))
        print(get_mapping()[k])
