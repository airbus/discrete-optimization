#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from RCALBP To generic scheduling"""

from copy import deepcopy

from discrete_optimization.alb.rcalbp.problem import RCALBPProblem, RCALBPSolution
from discrete_optimization.generic_tasks_tools.entities import (
    CompositeEntity,
    MultiplyTaskEntity,
    TaskEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.earliness_tardiness import (
    EarlinessTardinessComputer,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
    SpanBlockingConstraint,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    InformationLoss,
    LossImpact,
    LossType,
    TransformationMetadata,
    lossy_transformation,
)


class RCALBPToGenericSchedulingTransformation(
    ProblemTransformation[
        RCALBPProblem,
        RCALBPSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    """
    Transform BinPacking to RCPSP.
    """

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (BinPack → RCPSP).

        This direction is EXACT: all constraints can be represented in RCPSP.
        """
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="Loss shared resource",
                    loss_type=LossType.CONSTRAINT,
                    description="shared resource constraint not yet taken into account, "
                    "only station specific ones",
                    reason="unfolded view without folding mechanims",
                    impact=LossImpact.CRITICAL,
                    workaround="if no shared resource, it's fine",
                )
            ],
            use_cases=["Use for problem with station specific resource only"],
        )

    def transform_problem(
        self, source_problem: RCALBPProblem
    ) -> GenericSchedulingImplProblem:
        """Transform RCALBPProblem to RCPSP.

        Args:
            source_problem: RCALBPProblem problem instance

        Returns:
            Equivalent GenericSchedulingImpl problem

        """
        tasks = list(source_problem.tasks_list)
        # We create two virtual task, one representing the cycle time
        # and one in the end of the horizon.
        tasks.append("cycle_time")  # Virtual task
        tasks.append("end")
        durations_per_mode = {
            task.task_id: {0: task.processing_time}
            for task in source_problem.tasks_data
        }
        durations_per_mode["cycle_time"] = {0: 0}  # dummy task
        durations_per_mode["end"] = {0: 0}

        # Storing the max capacity of each resource, over the station.
        max_capacity_station = {}
        for resource in source_problem.resources:
            max_cap = 0
            for station in source_problem.stations:
                cap = source_problem.get_station_capacity(station, resource)
                max_cap = max(max_cap, cap)
            max_capacity_station[resource] = max_cap
        resource_capa = deepcopy(max_capacity_station)
        resource_capa["virtual_end_resource"] = 1
        for t in source_problem.tasks_list:
            resource_capa[f"res_{t}"] = 1
        # Duplication of the task "cycle_time" to create such entity at a regular frequency.
        entities_cycle_time = [
            MultiplyTaskEntity(
                task="cycle_time", multiply_factor=i, start_or_end=StartOrEnd.START
            )
            for i in range(source_problem.nb_stations + 1)
        ]
        # The last ending slot from the end of the last cycle time until the end
        last_entity = CompositeEntity(
            frozenset({entities_cycle_time[-1], TaskEntity("end")})
        )
        # Slot of the i-th station.
        entities_cycle_slot = [
            CompositeEntity(
                frozenset({entities_cycle_time[i], entities_cycle_time[i + 1]})
            )
            for i in range(len(entities_cycle_time) - 1)
        ]
        calendar_span_blocking = []
        # To model variable resource capacity per station,
        # it's represented by consumption of max_capa-capa[station]
        # by the entity_cycle_slot of the station.
        for i, station in enumerate(source_problem.stations):
            d = {}
            for r in source_problem.resources:
                station_cap = source_problem.get_station_capacity(station, r)
                max_cap = max_capacity_station[r]
                blocking_demand = max_cap - station_cap
                if blocking_demand > 0:
                    d[r] = blocking_demand
            calendar_span_blocking.append(
                SpanBlockingConstraint(
                    metadata=BlockingConstraintMetadata(
                        mode=BlockingMode.RESERVATION,
                        description=f"resource drop on station {i}",
                    ),
                    default_resource_blocked=d,
                    entity=entities_cycle_slot[i],
                )
            )
        # Prevent task to be on the ending slot of the schedule :
        # TODO : create a declarative way of doing this via some new kind of BlockingConstraint?
        calendar_span_blocking.append(
            SpanBlockingConstraint(
                metadata=BlockingConstraintMetadata(
                    mode=BlockingMode.ACTIVE, description="finalblocking"
                ),
                default_resource_blocked=resource_capa,
                entity=last_entity,
            )
        )
        # Intermediate blocking, prevent task to cross the cycle time.
        for j in range(1, len(entities_cycle_time)):
            calendar_span_blocking.append(
                SpanBlockingConstraint(
                    metadata=BlockingConstraintMetadata(
                        mode=BlockingMode.ACTIVE, description="intermediateblocking"
                    ),
                    default_resource_blocked=resource_capa,
                    entity=entities_cycle_time[j],
                )
            )
        resource_consumptions = {
            t.task_id: {
                0: {
                    r: source_problem.get_task_demand(t.task_id, r)
                    for r in source_problem.resources
                }
            }
            for t in source_problem.tasks_data
        }
        for t in resource_consumptions:
            resource_consumptions[t][0][f"res_{t}"] = 1
            if all(
                resource_consumptions[t][0][r] == 0 for r in resource_consumptions[t][0]
            ):
                resource_consumptions[t][0]["virtual_end_resource"] = 1
        horizon = sum(source_problem.task_times.values())
        return GenericSchedulingImplProblem(
            horizon=horizon,
            durations_per_mode=durations_per_mode,
            resource_consumptions=resource_consumptions,
            time_windows={
                "cycle_time": (1, None, None, None),
                "end": (horizon, horizon, horizon, horizon),
            },
            span_blocking_constraints=calendar_span_blocking,
            non_skill_cumulative_resources=resource_capa,
            successors=source_problem.successors,
            # only minimize the cycle time value, can be done via earlinessTardiness computer.
            list_objective_computer=[
                EarlinessTardinessComputer(
                    problem=None,
                    weight_objective=1,
                    max_start_and_weight_for_tardiness={"cycle_time": (0, 1)},
                )
            ],
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: RCALBPProblem
    ) -> RCALBPSolution:
        """Transform GenericSchedulingImpl solution back to BinPacking solution.

        Returns:
            Equivalent BinPacking solution

        """
        cycle_time = solution.get_start_time("cycle_time")
        allocation_to_station = [None for _ in source_problem.tasks_list]
        print(cycle_time)
        schedule = {}
        for i, t in enumerate(source_problem.tasks_list):
            start = solution.get_start_time(t)
            end = solution.get_end_time(t)
            station_idx = start // cycle_time
            if start == end and station_idx >= len(source_problem.stations):
                station_idx = len(source_problem.stations) - 1
                start_modulo = cycle_time
            else:
                start_modulo = start % cycle_time
            allocation_to_station[i] = station_idx
            schedule[t] = start_modulo
        return RCALBPSolution(
            problem=source_problem,
            allocation_to_station=allocation_to_station,
            task_schedule=schedule,
        )
