#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from Tsp to GenSchedulingImpl
"""

from typing import Iterable

import numpy as np

from discrete_optimization.generic_tasks_tools.entities import (
    SchedulingEntity,
    TaskEntity,
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
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.utils import build_matrice_distance


class TspToGenericSchedulingTransformation(
    ProblemTransformation[
        TspProblem,
        TspSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    def __init__(
        self,
        nb_closest: int = None,
        use_predecessors: bool = False,
        use_first_last: bool = False,
    ):
        self.nb_closest = nb_closest
        self.use_predecessors = use_predecessors
        self.use_first_last = use_first_last

    def transform_problem(
        self, source_problem: TspProblem
    ) -> GenericSchedulingImplProblem:
        start_depot_task = "start_depot"
        end_depot_task = "end_depot"
        tasks_list = list(source_problem.tasks_list)
        new_tasks_list = [start_depot_task, end_depot_task] + tasks_list
        task_entities = [TaskEntity(start_depot_task), TaskEntity(end_depot_task)] + [
            TaskEntity(t) for t in tasks_list
        ]
        entity_map = {ent.task: ent for ent in task_entities}
        possible_attribute = {}
        possible_attribute[task_entities[0]] = {source_problem.start_index}
        possible_attribute[task_entities[1]] = {source_problem.end_index}
        for t in task_entities[2:]:
            possible_attribute[t] = {t.task}
        matrix = build_matrice_distance(
            source_problem.node_count,
            method=source_problem.evaluate_function_indexes,
        )
        graph = None
        if self.nb_closest is not None:
            sorted_distance = np.argsort(matrix, axis=1)
            graph = {}
            for ent in task_entities:
                index = list(possible_attribute[ent])[0]
                closest = sorted_distance[index, : self.nb_closest]
                graph[ent] = {
                    entity_map[int(ind)] for ind in closest if int(ind) in entity_map
                }
                graph[ent].add(task_entities[1])
                for ind in closest:
                    if int(ind) not in entity_map:
                        if int(ind) == source_problem.start_index:
                            graph[ent].add(task_entities[0])
                        if int(ind) == source_problem.end_index:
                            graph[ent].add(task_entities[1])

        # matrix[source_problem.end_index, source_problem.start_index] = 0
        matrix = np.array(100 * matrix, dtype=int)
        zero = np.ones_like(matrix)
        routing_entity = RoutingEntity(
            entities=frozenset(task_entities),
            attributes=list(range(source_problem.node_count)),
            cumul_dimensions=["cost"],
            first_entity=task_entities[0] if self.use_first_last else None,
            last_entity=task_entities[1] if self.use_first_last else None,
            possible_attributes=possible_attribute,
            transition_times=zero,
            graph=graph,
            cumul_matrix={"cost": matrix},
            has_to_be_visited={ent: True for ent in task_entities},
            nb_means=1,
        )
        upper_bound = 0
        for i in range(matrix.shape[0]):
            upper_bound += np.max(matrix[i, :])
        entity_precedence_constraints: dict[
            SchedulingEntity, Iterable[SchedulingEntity]
        ] = {}
        real_node_entities = task_entities[2:]
        source_entity = task_entities[0]
        sink_entity = task_entities[1]
        entity_precedence_constraints[source_entity] = [
            sink_entity
        ] + real_node_entities
        for t in real_node_entities:
            entity_precedence_constraints[t] = [sink_entity]
        # real_source_entity = entity_map[source_problem.start_index]
        # real_sink_entity = entity_map[source_problem.end_index]
        # entity_precedence_constraints[real_source_entity] = [sink_entity]+[rn for rn in real_node_entities
        #                                                                    if rn != real_source_entity]
        # if real_sink_entity != real_source_entity:
        #     for t in real_node_entities:
        #         if t == real_sink_entity:
        #             continue
        #         entity_precedence_constraints[t].append(real_sink_entity)

        return GenericSchedulingImplProblem(
            horizon=source_problem.node_count + 2,
            durations_per_mode={t: {0: 0} for t in new_tasks_list},
            list_routing_entities=[routing_entity],
            list_objective_computer=[
                RoutingCostComputer(
                    problem=None,
                    weight_objective=1,
                    weight_per_routing_entities={routing_entity: {"cost": 1}},
                )
            ],
            entity_precedence_constraints=entity_precedence_constraints
            if self.use_predecessors
            else {},
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: TspProblem
    ) -> TspSolution:
        problem_scheduling = solution.problem
        routing = problem_scheduling.get_routing_entities()[0]
        path = solution.get_routing_path(routing_entity=routing)
        rebuild_path = []
        print(path)
        for _, _, ent in path:
            ent: TaskEntity
            task = ent.task
            if isinstance(task, int):
                if task not in {source_problem.start_index, source_problem.end_index}:
                    rebuild_path.append(task)
        return TspSolution(
            problem=source_problem,
            start_index=source_problem.start_index,
            end_index=source_problem.end_index,
            permutation=rebuild_path,
        )
