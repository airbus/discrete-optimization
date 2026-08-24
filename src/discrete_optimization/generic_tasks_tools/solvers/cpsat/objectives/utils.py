#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


def create_computer_to_modeler_mapping():
    # TODO : automatize this. This is awful.
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
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.allocated_tasks import (
        AllocatedTasksObjectiveCpSatModeler,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.allocation_changes import (
        AllocationSwitchModelerCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.allocation_cost import (
        AllocationCostModelerCpSat,
        AllocationCostMultimodeModelerCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.earliness_tardiness import (
        EarlinessTardinessCpSatModeler,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.makespan import (
        MakespanObjectiveModelCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.mode_cost import (
        ModeCostModelerCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.resource_levels import (
        CalendarRenewableResourceLevelModelerCpSat,
        NonRenewableResourceLevelModelerCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.schedule_changes import (
        ScheduleChangesModelerCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.soft_time_penalty import (
        SoftTimePenaltyModelerCpSat,
    )
    from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.unary_resource_used import (
        UnaryResourcesUsedModelerCpSat,
    )

    return {
        AllocationCostComputerMultimode: AllocationCostMultimodeModelerCpSat,
        AllocationCostComputer: AllocationCostModelerCpSat,
        AllocatedTasksObjective: AllocatedTasksObjectiveCpSatModeler,
        AllocationSwitchObjectiveComputer: AllocationSwitchModelerCpSat,
        EarlinessTardinessComputer: EarlinessTardinessCpSatModeler,
        MakespanObjectiveComputer: MakespanObjectiveModelCpSat,
        ModeCostComputer: ModeCostModelerCpSat,
        NonRenewableResourceLevelObjectiveComputer: NonRenewableResourceLevelModelerCpSat,
        CalendarRenewableResourceLevelObjectiveComputer: CalendarRenewableResourceLevelModelerCpSat,
        ScheduleChangesComputer: ScheduleChangesModelerCpSat,
        SoftTimePenaltyComputer: SoftTimePenaltyModelerCpSat,
        UnaryResourcesUsedComputer: UnaryResourcesUsedModelerCpSat,
    }
