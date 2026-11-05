#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from Vrp to GenSchedulingImpl
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
from discrete_optimization.generic_tasks_tools.objectives.routing_costs import (
    RoutingCostComputer,
)
from discrete_optimization.generic_tasks_tools.routing import RoutingEntity
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.vrp.problem import VrpProblem, VrpSolution
from discrete_optimization.vrp.utils import compute_length_matrix


class VrpToGenericSchedulingTransformation(
    ProblemTransformation[
        VrpProblem,
        VrpSolution,
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
    ):
        self.nb_closest = nb_closest
        self.use_predecessors = use_predecessors
        self.use_first_last = use_first_last
        self.factor_matrix = factor_matrix

    def transform_problem(
        self, source_problem: VrpProblem
    ) -> GenericSchedulingImplProblem:
        nodes_to_visit = [
            i
            for i in range(source_problem.customer_count)
            if i not in source_problem.start_indexes
            and i not in source_problem.end_indexes
        ]
        sorted_distance, matrix = compute_length_matrix(source_problem)
        # matrix[source_problem.end_index, source_problem.start_index] = 0
        matrix = np.array(100 * matrix, dtype=int)
        routing_entities = []
        resource_usage_per_mode = {}
        duration_per_mode = {}
        entity_precedence_constraints = {}
        for j in range(source_problem.vehicle_count):
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
                duration_per_mode[t][j] = 0
                resource_usage_per_mode[t][j] = {
                    f"capa_{j}": source_problem.customers[t].demand
                }
            possible_attribute = dict()
            possible_attribute[task_start_entity] = {source_problem.start_indexes[j]}
            possible_attribute[task_end_entity] = {source_problem.end_indexes[j]}
            for t in task_entities:
                possible_attribute[t] = {t.task}
            ones = np.ones_like(matrix)
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
                            if int(ind) == source_problem.start_indexes[j]:
                                graph[ent].add(task_start_entity)
                            if int(ind) == source_problem.end_indexes[j]:
                                graph[ent].add(task_end_entity)
            all_task_entities = [task_start_entity, task_end_entity] + task_entities
            entity_precedence_constraints[task_start_entity] = [
                task_end_entity
            ] + task_entities
            for t in task_entities:
                entity_precedence_constraints[t] = [task_end_entity]
            routing_entity = RoutingEntity(
                entities=frozenset(all_task_entities),
                attributes=list(range(source_problem.customer_count)),
                cumul_dimensions=["cost"],
                first_entity=all_task_entities[0] if self.use_first_last else None,
                last_entity=all_task_entities[1] if self.use_first_last else None,
                possible_attributes=possible_attribute,
                transition_times=ones,
                graph=graph,
                cumul_matrix={"cost": matrix},
                has_to_be_visited={ent: False for ent in all_task_entities},
                nb_means=1,
            )
            routing_entities.append(routing_entity)
        upper_bound = 0
        for i in range(matrix.shape[0]):
            upper_bound += np.max(matrix[i, :])
        optional_tasks = set()
        # for j in range(source_problem.customer_count):
        #     optional_tasks.add(f"start_depot_{j}")
        #     optional_tasks.add(f"end_depot_{j}")
        return GenericSchedulingImplProblem(
            horizon=source_problem.customer_count + 2,
            durations_per_mode=duration_per_mode,
            list_routing_entities=routing_entities,
            optional_tasks=optional_tasks,
            resource_consumptions=resource_usage_per_mode,
            non_renewable_resources={
                f"capa_{j}": int(source_problem.vehicle_capacities[j])
                for j in range(source_problem.vehicle_count)
            },
            list_objective_computer=[
                RoutingCostComputer(
                    problem=None,
                    weight_objective=1,
                    weight_per_routing_entities={
                        routing_entity: {"cost": 1}
                        for routing_entity in routing_entities
                    },
                )
            ],
            entity_precedence_constraints=entity_precedence_constraints
            if self.use_predecessors
            else {},
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: VrpProblem
    ) -> VrpSolution:
        problem_scheduling = solution.problem
        list_paths = []
        for j, routing in enumerate(problem_scheduling.get_routing_entities()):
            path = solution.get_routing_path(routing_entity=routing)
            rebuild_path = []
            print(path)
            for _, _, ent in path:
                ent: TaskModeEntity
                task = ent.task
                if isinstance(task, int):
                    if task not in {
                        source_problem.start_indexes[j],
                        source_problem.end_indexes[j],
                    }:
                        rebuild_path.append(task)
            list_paths.append(rebuild_path)
        return VrpSolution(
            problem=source_problem,
            list_start_index=source_problem.start_indexes,
            list_end_index=source_problem.end_indexes,
            list_paths=list_paths,
        )
