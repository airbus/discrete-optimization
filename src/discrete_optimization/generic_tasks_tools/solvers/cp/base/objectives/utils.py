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
from discrete_optimization.generic_tasks_tools.objectives.custom import CustomComputer
from discrete_optimization.generic_tasks_tools.objectives.earliness_tardiness import (
    EarlinessTardinessComputer,
    OnlyEarlinessComputer,
    OnlyTardinessComputer,
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
from discrete_optimization.generic_tasks_tools.objectives.routing_costs import (
    RoutingCostComputer,
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
from discrete_optimization.generic_tasks_tools.objectives.weighted_number_late_tasks import (
    WeightedNumberLateComputer,
    WeightedNumberLateTasksInBatchComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.weighted_sum_start_or_end import (
    WeightedSumStartEndObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.allocated_tasks import (
    AllocatedTasksModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.allocation_changes import (
    AllocationSwitchModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.allocation_cost import (
    AllocationCostModelerCp,
    AllocationCostMultimodeModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.calendar_preempted import (
    CalendarPreemptedModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.cumul_objective import (
    CumulCostModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.custom import (
    CustomModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.earliness_tardiness import (
    EarlinessTardinessModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.makespan import (
    MakespanModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.mode_cost import (
    ModeCostModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.resource_levels import (
    CalendarRenewableResourceLevelModelerCp,
    NonRenewableResourceLevelModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.routing_costs import (
    RoutingCostsModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.schedule_changes import (
    ScheduleChangesModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.scheduled_tasks import (
    ScheduledTasksModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.soft_time_penalty import (
    SoftTimePenaltyModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.unary_resource_used import (
    UnaryResourcesUsedModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.weighted_number_late_tasks import (
    WeightedNumberLateInBatchModelerCp,
    WeightedNumberLateModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.weighted_sum_start_or_end import (
    WeightedSumStartEndModelerCp,
)

mapping_computer_to_modeler = {
    AllocationCostComputerMultimode: AllocationCostMultimodeModelerCp,
    AllocationCostComputer: AllocationCostModelerCp,
    AllocatedTasksObjective: AllocatedTasksModelerCp,
    AllocationSwitchObjectiveComputer: AllocationSwitchModelerCp,
    EarlinessTardinessComputer: EarlinessTardinessModelerCp,
    OnlyEarlinessComputer: EarlinessTardinessModelerCp,
    OnlyTardinessComputer: EarlinessTardinessModelerCp,
    MakespanObjectiveComputer: MakespanModelerCp,
    ModeCostComputer: ModeCostModelerCp,
    NonRenewableResourceLevelObjectiveComputer: NonRenewableResourceLevelModelerCp,
    CalendarRenewableResourceLevelObjectiveComputer: CalendarRenewableResourceLevelModelerCp,
    ScheduleChangesComputer: ScheduleChangesModelerCp,
    SoftTimePenaltyComputer: SoftTimePenaltyModelerCp,
    UnaryResourcesUsedComputer: UnaryResourcesUsedModelerCp,
    CumulCostComputer: CumulCostModelerCp,
    ScheduledTasksComputer: ScheduledTasksModelerCp,
    CalendarPreemptedComputer: CalendarPreemptedModelerCp,
    CustomComputer: CustomModelerCp,
    WeightedSumStartEndObjectiveComputer: WeightedSumStartEndModelerCp,
    WeightedNumberLateComputer: WeightedNumberLateModelerCp,
    WeightedNumberLateTasksInBatchComputer: WeightedNumberLateInBatchModelerCp,
    RoutingCostComputer: RoutingCostsModelerCp,
}
