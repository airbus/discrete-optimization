#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.routing import (
    CumulDimensions,
    RoutingEntity,
    RoutingProblem,
    RoutingSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    Task,
)


class RoutingCostComputer(ObjectiveComputer[Task]):
    problem: RoutingProblem

    def __init__(
        self,
        problem: RoutingProblem[Task] = None,
        weight_objective: float = 1.0,
        weight_per_routing_entities: dict[
            RoutingEntity, dict[CumulDimensions, float]
        ] = None,
    ):
        super().__init__(problem, weight_objective)
        if weight_per_routing_entities is None:
            self.weight_per_routing_entities = {}
        else:
            self.weight_per_routing_entities = weight_per_routing_entities

    def compute_objective(self, solution: RoutingSolution) -> float:
        merged_val = 0
        for r in self.weight_per_routing_entities:
            path = solution.get_routing_path(routing_entity=r)
            if len(path) >= 1:
                prev_ent = path[0][2]
                prev_attr = solution.get_attribute_of_entity(path[0][2])
                for s, e, ent in path[1:]:
                    new_attr = solution.get_attribute_of_entity(ent)
                    for c in self.weight_per_routing_entities[r]:
                        val = (
                            self.problem.get_dynamic_cumul_dimensions_between_entities(
                                cumul_dimension=c,
                                entity_from=prev_ent,
                                attribute_from=prev_attr,
                                entity_to=ent,
                                attribute_to=new_attr,
                            )
                        )
                        merged_val += self.weight_per_routing_entities[r][c] * val
        return merged_val

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.ROUTING_COSTS
