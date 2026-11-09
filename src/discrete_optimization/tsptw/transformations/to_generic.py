#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Transformation from TspTw to GenSchedulingImpl
"""

from typing import Iterable

import numpy as np

from discrete_optimization.generic_tasks_tools.entities import (
    SchedulingEntity,
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
from discrete_optimization.generic_tasks_tools.objectives.weighted_sum_start_or_end import (
    WeightedSumStartEndObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.routing import RoutingEntity
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class TspTwToGenericSchedulingTransformation(
    ProblemTransformation[
        TSPTWProblem,
        TSPTWSolution,
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
        self, source_problem: TSPTWProblem
    ) -> GenericSchedulingImplProblem:
        start_depot_task = "start_depot"
        end_depot_task = "end_depot"
        tasks_list = source_problem.customers
        new_tasks_list = [start_depot_task, end_depot_task] + tasks_list
        task_entities = [TaskEntity(start_depot_task), TaskEntity(end_depot_task)] + [
            TaskEntity(t) for t in tasks_list
        ]
        entity_map = {ent.task: ent for ent in task_entities}
        possible_attribute = {}
        possible_attribute[task_entities[0]] = {source_problem.depot_node}
        possible_attribute[task_entities[1]] = {source_problem.depot_node}
        for t in task_entities[2:]:
            possible_attribute[t] = {t.task}
        matrix = source_problem.distance_matrix
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
                        if int(ind) == source_problem.depot_node:
                            graph[ent].add(task_entities[0])
                            graph[ent].add(task_entities[1])
        # matrix[source_problem.end_index, source_problem.start_index] = 0
        matrix = np.array(self.factor_matrix * matrix, dtype=int)
        routing_entity = RoutingEntity(
            entities=frozenset(task_entities),
            attributes=list(range(source_problem.nb_nodes)),
            cumul_dimensions=["cost"],
            first_entity=task_entities[0] if self.use_first_last else None,
            last_entity=task_entities[1] if self.use_first_last else None,
            possible_attributes=possible_attribute,
            transition_times=matrix,
            graph=graph,
            cumul_matrix={"cost": matrix},
            has_to_be_visited={ent: True for ent in task_entities},
            nb_means=1,
        )
        upper_bound = 0
        for i in range(matrix.shape[0]):
            upper_bound += np.max(matrix[i, :])
        time_window_constraint = {}
        for ent in task_entities:
            task = ent.task
            if isinstance(task, int):
                tw = (
                    self.factor_matrix * source_problem.time_windows[task][0],
                    self.factor_matrix * source_problem.time_windows[task][1],
                )
                time_window_constraint[(ent, StartOrEnd.START)] = tw
            if task == "end_depot":
                tw = (
                    self.factor_matrix
                    * source_problem.time_windows[source_problem.depot_node][0],
                    self.factor_matrix
                    * source_problem.time_windows[source_problem.depot_node][1],
                )
                time_window_constraint[(ent, StartOrEnd.START)] = tw
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
        list_objective_computer = [
            WeightedSumStartEndObjectiveComputer(
                problem=None,
                weight_objective=1,
                dict_weight={"end_depot": {StartOrEnd.END: 1}},
            )
        ]
        if self.relax_time_window:
            deadline_tasks = dict()
            deadline_tasks["start_depot"] = (0, 1)
            deadline_tasks["end_depot"] = (
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
            release_tasks["start_depot"] = (0, 1)
            release_tasks["end_depot"] = (
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
            horizon=int(1.1 * max_time_window),
            durations_per_mode={t: {0: 0} for t in new_tasks_list},
            list_routing_entities=[routing_entity],
            entity_time_window_bounds=time_window_constraint
            if not self.relax_time_window
            else {},
            list_objective_computer=list_objective_computer,
            entity_precedence_constraints=entity_precedence_constraints
            if self.use_predecessors
            else {},
        )

    def back_transform_solution(
        self, solution: GenericSchedulingImplSolution, source_problem: TSPTWProblem
    ) -> TSPTWSolution:
        problem_scheduling = solution.problem
        routing = problem_scheduling.get_routing_entities()[0]
        path = solution.get_routing_path(routing_entity=routing)
        rebuild_path = []
        print(path)
        for _, _, ent in path:
            ent: TaskEntity
            task = ent.task
            if isinstance(task, int):
                if task not in {source_problem.depot_node}:
                    rebuild_path.append(task)
        return TSPTWSolution(
            problem=source_problem,
            permutation=rebuild_path,
        )
