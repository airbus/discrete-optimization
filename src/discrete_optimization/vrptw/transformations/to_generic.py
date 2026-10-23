#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from VrpTw to GenSchedulingImpl
"""

import numpy as np

from discrete_optimization.generic_tasks_tools.entities import (
    TaskEntity,
    TaskModeEntity,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.objectives.earliness_tardiness import (
    EarlinessTardinessComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.routing_costs import (
    RoutingCostComputer,
)
from discrete_optimization.generic_tasks_tools.routing import RoutingEntity
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution


class VrpTwToGenericSchedulingTransformation(
    ProblemTransformation[
        VRPTWProblem,
        VRPTWSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    def __init__(
        self,
        nb_closest: int = None,
        use_predecessors: bool = False,
        use_first_last: bool = False,
        factor_matrix: int = 100,
        relax_time_window: bool = False,
    ):
        self.nb_closest = nb_closest
        self.use_predecessors = use_predecessors
        self.use_first_last = use_first_last
        self.factor_matrix = factor_matrix
        self.relax_time_window = relax_time_window

    def transform_problem(
        self, source_problem: VRPTWProblem
    ) -> GenericSchedulingImplProblem:
        nodes_to_visit = [
            i
            for i in range(source_problem.nb_nodes)
            if i not in {source_problem.depot_node}
        ]
        matrix = np.array(
            self.factor_matrix * source_problem.distance_matrix, dtype=int
        )
        sorted_distance = np.argsort(matrix, axis=1)
        routing_entities = []
        resource_usage_per_mode = {}
        duration_per_mode = {}
        entity_precedence_constraints = {}
        for j in range(source_problem.nb_vehicles):
            start_depot = f"start_depot_{j}"
            end_depot = f"end_depot_{j}"
            task_start_entity = TaskEntity(start_depot)
            task_end_entity = TaskEntity(end_depot)
            task_entities = [TaskModeEntity(t, mode=j) for t in nodes_to_visit]
            duration_per_mode[start_depot] = {0: 0}
            duration_per_mode[end_depot] = {0: 0}
            for t in nodes_to_visit:
                if t not in resource_usage_per_mode:
                    resource_usage_per_mode[t] = {}
                    duration_per_mode[t] = {}
                # Mode corresponds to vehicle choice.
                duration_per_mode[t][j] = int(
                    self.factor_matrix * source_problem.service_times[t]
                )
                resource_usage_per_mode[t][j] = {
                    f"capa_{j}": int(source_problem.demands[t])
                }
            possible_attribute = dict()
            possible_attribute[task_start_entity] = {source_problem.depot_node}
            possible_attribute[task_end_entity] = {source_problem.depot_node}
            for t in task_entities:
                possible_attribute[t] = {t.task}
            graph = None
            if self.nb_closest is not None:
                entity_map = {}
                for t in task_entities:
                    entity_map[t.task] = t
                graph = {}
                for ent in [task_start_entity, task_end_entity] + task_entities:
                    index = list(possible_attribute[ent])[0]
                    closest = sorted_distance[index, : self.nb_closest]
                    graph[ent] = {
                        entity_map[int(ind)]
                        for ind in closest
                        if int(ind) in entity_map
                    }
                    graph[ent].add(task_end_entity)
                    for ind in closest:
                        if int(ind) not in entity_map:
                            if int(ind) == source_problem.depot_node:
                                graph[ent].add(task_start_entity)
                            if int(ind) == source_problem.depot_node:
                                graph[ent].add(task_end_entity)
            all_task_entities = [task_start_entity, task_end_entity] + task_entities
            entity_precedence_constraints[task_start_entity] = [
                task_end_entity
            ] + task_entities
            for t in task_entities:
                entity_precedence_constraints[t] = [task_end_entity]
            routing_entity = RoutingEntity(
                entities=frozenset(all_task_entities),
                attributes=list(range(source_problem.nb_nodes)),
                cumul_dimensions=["cost"],
                first_entity=all_task_entities[0] if self.use_first_last else None,
                last_entity=all_task_entities[1] if self.use_first_last else None,
                possible_attributes=possible_attribute,
                transition_times=matrix,
                graph=graph,
                cumul_matrix={"cost": matrix},
                has_to_be_visited={ent: False for ent in all_task_entities},
                nb_means=1,
            )
            routing_entities.append(routing_entity)
        upper_bound = 0
        for i in range(matrix.shape[0]):
            upper_bound += np.max(matrix[i, :])
        list_objective_computer = [
            RoutingCostComputer(
                problem=None,
                weight_objective=1,
                weight_per_routing_entities={
                    routing_entity: {"cost": 1} for routing_entity in routing_entities
                },
            )
        ]
        if self.relax_time_window:
            deadline_tasks = dict()
            for j in range(source_problem.nb_vehicles):
                deadline_tasks[f"start_depot_{j}"] = (0, 1)
                deadline_tasks[f"end_depot_{j}"] = (
                    self.factor_matrix
                    * source_problem.time_windows[source_problem.depot_node][1],
                    1,
                )
            for t in source_problem.customers:
                deadline_tasks[t] = (
                    self.factor_matrix * source_problem.time_windows[t][1],
                    1,
                )
            release_tasks = dict()
            for j in range(source_problem.nb_vehicles):
                release_tasks[f"start_depot_{j}"] = (0, 1)
                release_tasks[f"end_depot_{j}"] = (
                    self.factor_matrix
                    * source_problem.time_windows[source_problem.depot_node][0],
                    1,
                )
            for t in source_problem.customers:
                release_tasks[t] = (
                    self.factor_matrix * source_problem.time_windows[t][0],
                    1,
                )
            list_objective_computer.append(
                EarlinessTardinessComputer(
                    problem=None,
                    weight_objective=100,
                    max_start_and_weight_for_tardiness=deadline_tasks,
                    min_start_and_weight_for_earliness=release_tasks,
                )
            )
        max_time_window = max(
            [self.factor_matrix * x[1] for x in source_problem.time_windows]
        )
        return GenericSchedulingImplProblem(
            horizon=int(2 * max_time_window),
            durations_per_mode=duration_per_mode,
            list_routing_entities=routing_entities,
            resource_consumptions=resource_usage_per_mode,
            non_renewable_resources={
                f"capa_{j}": int(source_problem.vehicle_capacity)
                for j in range(source_problem.nb_vehicles)
            },
            list_objective_computer=list_objective_computer,
            entity_precedence_constraints=entity_precedence_constraints
            if self.use_predecessors
            else {},
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: VRPTWProblem
    ) -> VRPTWSolution:
        problem_scheduling = solution.problem
        list_paths = []
        for j, routing in enumerate(problem_scheduling.get_routing_entities()):
            path = solution.get_routing_path(routing_entity=routing)
            rebuild_path = []
            for _, _, ent in path:
                ent: TaskModeEntity
                task = ent.task
                if isinstance(task, int):
                    if task not in {source_problem.depot_node}:
                        rebuild_path.append(task)
            list_paths.append(rebuild_path)
        return VRPTWSolution(problem=source_problem, routes=list_paths)
