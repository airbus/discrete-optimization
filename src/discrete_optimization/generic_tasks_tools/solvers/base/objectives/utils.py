#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
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
from discrete_optimization.generic_tasks_tools.objectives.calendar_preempted import (
    CalendarPreemptedComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.cumul_cost import (
    CumulCostComputer,
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
from discrete_optimization.generic_tasks_tools.objectives.resource_levels import (
    CalendarRenewableResourceLevelObjectiveComputer,
    NonRenewableResourceLevelObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.schedule_changes import (
    ScheduleChangesComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.scheduled_tasks import (
    ScheduledTasksComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.soft_time_penalty import (
    SoftTimePenaltyComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.allocated_tasks import (
    AllocatedTasksObjectiveCpBaseModeler,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.allocation_changes import (
    AllocationSwitchModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.allocation_cost import (
    AllocationCostModelerCpBase,
    AllocationCostMultimodeModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.calendar_preempted import (
    CalendarPreemptedModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.cumul_objective import (
    CumulCostModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.earliness_tardiness import (
    EarlinessTardinessCpBaseModeler,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.makespan import (
    MakespanObjectiveModelCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.mode_cost import (
    ModeCostModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.resource_levels import (
    CalendarRenewableResourceLevelModelerCpBase,
    NonRenewableResourceLevelModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.schedule_changes import (
    ScheduleChangesModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.scheduled_tasks import (
    ScheduledTasksCpBaseModeler,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.soft_time_penalty import (
    SoftTimePenaltyModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.unary_resource_used import (
    UnaryResourcesUsedModelerCpBase,
)

mapping_computer_to_modeler = {
    AllocationCostComputerMultimode: AllocationCostMultimodeModelerCpBase,
    AllocationCostComputer: AllocationCostModelerCpBase,
    AllocatedTasksObjective: AllocatedTasksObjectiveCpBaseModeler,
    AllocationSwitchObjectiveComputer: AllocationSwitchModelerCpBase,
    EarlinessTardinessComputer: EarlinessTardinessCpBaseModeler,
    MakespanObjectiveComputer: MakespanObjectiveModelCpBase,
    ModeCostComputer: ModeCostModelerCpBase,
    NonRenewableResourceLevelObjectiveComputer: NonRenewableResourceLevelModelerCpBase,
    CalendarRenewableResourceLevelObjectiveComputer: CalendarRenewableResourceLevelModelerCpBase,
    ScheduleChangesComputer: ScheduleChangesModelerCpBase,
    SoftTimePenaltyComputer: SoftTimePenaltyModelerCpBase,
    UnaryResourcesUsedComputer: UnaryResourcesUsedModelerCpBase,
    CumulCostComputer: CumulCostModelerCpBase,
    ScheduledTasksComputer: ScheduledTasksCpBaseModeler,
    CalendarPreemptedComputer: CalendarPreemptedModelerCpBase,
}
