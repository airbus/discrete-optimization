#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Class to specify routing kind of constraint between task
#  or entities.
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar

import numpy as np

from discrete_optimization.generic_tasks_tools.entities import (
    SchedulingEntity,
)
from discrete_optimization.generic_tasks_tools.entity_constraints import (
    EntityConstraintsProblem,
    EntityConstraintsSolution,
)
from discrete_optimization.generic_tasks_tools.enums import UndefinedData
from discrete_optimization.generic_tasks_tools.scheduling import (
    Task,
)
from discrete_optimization.generic_tasks_tools.utils import (
    optional_override,
    optional_override_implem,
)

logger = logging.getLogger(__name__)

AttributeEntities = TypeVar("AttributeEntities", bound=Hashable)
CumulDimensions = TypeVar("CumulDimensions", bound=Hashable)


@dataclass(frozen=False)
class RoutingEntity(Generic[Task, AttributeEntities, CumulDimensions]):
    entities: frozenset[SchedulingEntity[Task]]
    attributes: list[AttributeEntities]
    cumul_dimensions: list[CumulDimensions]
    possible_attributes: dict[SchedulingEntity[Task], set[AttributeEntities]]
    transition_times: np.ndarray
    cumul_matrix: dict[CumulDimensions, np.ndarray]
    cumul_bounds: dict[CumulDimensions, tuple[int | None, int | None]] = None
    has_to_be_visited: dict[SchedulingEntity[Task], bool] = None
    has_to_be_visited_if_present: dict[SchedulingEntity[Task], bool] = None
    first_entity: SchedulingEntity[Task] = field(default=None)
    last_entity: SchedulingEntity[Task] = field(default=None)
    graph: dict[SchedulingEntity[Task], set[SchedulingEntity[Task]]] = field(
        default=None
    )
    nb_means: int = 1

    def __hash__(self):
        return hash(self.entities)

    @property
    def index_attributes(self):
        return {self.attributes[i]: i for i in range(len(self.attributes))}

    def get_possible_attributes(
        self, scheduling_entity: SchedulingEntity
    ) -> set[AttributeEntities]:
        return self.possible_attributes.get(scheduling_entity, set())

    def get_active_entities(
        self, solution: EntityConstraintsSolution
    ) -> set[SchedulingEntity]:
        return set([ent for ent in self.entities if ent.is_active(solution)])


@dataclass(frozen=False)
class OrderedRoutingEntity(RoutingEntity[Task, AttributeEntities, CumulDimensions]):
    list_entities: list[SchedulingEntity] = None

    def __hash__(self):
        return hash(tuple(self.list_entities))

    @optional_override_implem
    def unfold_entities(self):
        for entity in self.list_entities:
            yield entity
            entity.unfold_entities()


# @dataclass(frozen=False)
class RoutingProblem(EntityConstraintsProblem[Task]):
    # list_routing_entities: list[RoutingEntity] = field(default_factory=list)
    # list_ordered_routing_entities: list[OrderedRoutingEntity] = field(default_factory=list)
    from_entity_to_routing_entity: dict[SchedulingEntity, list[RoutingEntity]] = None
    from_entity_to_ordered_routing_entity: dict[
        SchedulingEntity, list[OrderedRoutingEntity]
    ] = None

    @optional_override
    def get_routing_entities(self) -> list[RoutingEntity]:
        return []

    @optional_override
    def get_ordered_routing_entities(self) -> list[OrderedRoutingEntity]:
        return []

    def get_bounds_cumul_dimension(
        self, routing_entity: RoutingEntity, cumul_dimension: CumulDimensions
    ) -> tuple[int | None, int | None]:
        if routing_entity.cumul_bounds is None:
            return None, None
        if cumul_dimension not in routing_entity.cumul_bounds:
            return None, None
        return routing_entity.cumul_bounds[cumul_dimension]

    def get_routing_entity(self, entity: SchedulingEntity) -> RoutingEntity:
        if self.from_entity_to_ordered_routing_entity is None:
            self.from_entity_to_ordered_routing_entity = defaultdict(list)
            for o in self.get_ordered_routing_entities():
                for e in o.list_entities:
                    self.from_entity_to_ordered_routing_entity[e].append(o)
        if self.from_entity_to_routing_entity is None:
            self.from_entity_to_routing_entity = defaultdict(list)
            for o in self.get_routing_entities():
                for e in o.entities:
                    self.from_entity_to_routing_entity[e].append(o)
        if entity in self.from_entity_to_routing_entity:
            return self.from_entity_to_routing_entity[entity][0]
        if entity in self.from_entity_to_ordered_routing_entity:
            return self.from_entity_to_ordered_routing_entity[entity][0]
        return None

    def get_static_setup_time_between_entities(
        self, entity_from: SchedulingEntity, entity_to: SchedulingEntity
    ):
        routing_entity_from = self.get_routing_entity(entity_from)
        routing_entity_to = self.get_routing_entity(entity_to)
        if routing_entity_from != routing_entity_to:
            return UndefinedData.UNDEFINED
        if routing_entity_from is None or routing_entity_to is None:
            return UndefinedData.UNDEFINED
        possible_attribute_from = routing_entity_from.possible_attributes[entity_from]
        possible_attribute_to = routing_entity_to.possible_attributes[entity_to]
        if len(possible_attribute_to) == len(possible_attribute_from) == 1:
            att_from = list(possible_attribute_from)[0]
            att_to = list(possible_attribute_to)[0]
            index_from = routing_entity_from.index_attributes[att_from]
            index_to = routing_entity_to.index_attributes[att_to]
            return routing_entity_from.transition_times[index_from, index_to]
        return UndefinedData.UNDEFINED

    @optional_override
    def get_dynamic_setup_time_between_entities(
        self,
        entity_from: SchedulingEntity,
        attribute_from: int,
        entity_to: SchedulingEntity,
        attribute_to: int,
    ):
        routing_entity_from = self.get_routing_entity(entity_from)
        routing_entity_to = self.get_routing_entity(entity_to)
        if routing_entity_from != routing_entity_to:
            return UndefinedData.UNDEFINED
        if routing_entity_from is None or routing_entity_to is None:
            return UndefinedData.UNDEFINED
        possible_attribute_from = routing_entity_from.possible_attributes[entity_from]
        possible_attribute_to = routing_entity_to.possible_attributes[entity_to]
        if attribute_from not in possible_attribute_from:
            return UndefinedData.UNDEFINED
        if attribute_to not in possible_attribute_to:
            return UndefinedData.UNDEFINED
        index_from = routing_entity_from.index_attributes[attribute_from]
        index_to = routing_entity_to.index_attributes[attribute_to]
        if isinstance(routing_entity_from.transition_times, np.ndarray):
            return routing_entity_from.transition_times[index_from, index_to]
        if isinstance(routing_entity_from.transition_times, list):
            return routing_entity_from.transition_times[index_from][index_to]

    @optional_override
    def get_dynamic_cumul_dimensions_between_entities(
        self,
        cumul_dimension: CumulDimensions,
        entity_from: SchedulingEntity,
        attribute_from: int,
        entity_to: SchedulingEntity,
        attribute_to: int,
    ):
        routing_entity_from = self.get_routing_entity(entity_from)
        routing_entity_to = self.get_routing_entity(entity_to)
        if routing_entity_from != routing_entity_to:
            return UndefinedData.UNDEFINED
        if routing_entity_from is None or routing_entity_to is None:
            return UndefinedData.UNDEFINED
        possible_attribute_from = routing_entity_from.possible_attributes[entity_from]
        possible_attribute_to = routing_entity_to.possible_attributes[entity_to]
        if attribute_from not in possible_attribute_from:
            return UndefinedData.UNDEFINED
        if attribute_to not in possible_attribute_to:
            return UndefinedData.UNDEFINED
        index_from = routing_entity_from.index_attributes[attribute_from]
        index_to = routing_entity_to.index_attributes[attribute_to]
        if isinstance(routing_entity_from.cumul_matrix[cumul_dimension], np.ndarray):
            return routing_entity_from.cumul_matrix[cumul_dimension][
                index_from, index_to
            ]
        if isinstance(routing_entity_from.cumul_matrix[cumul_dimension], list):
            return routing_entity_from.cumul_matrix[cumul_dimension][index_from][
                index_to
            ]

    @optional_override
    def get_attribute_of_entity(
        self, entity: SchedulingEntity[Task]
    ) -> AttributeEntities:
        routing = self.get_routing_entity(entity)
        if routing is None:
            return UndefinedData.UNDEFINED
        attr = routing.possible_attributes[entity]
        if len(attr) == 1:
            return list(attr)[0]
        return UndefinedData.UNDEFINED


class RoutingSolution(EntityConstraintsSolution[Task]):
    problem: RoutingProblem

    def get_routing_path(
        self, routing_entity: RoutingEntity
    ) -> list[tuple[int, int, SchedulingEntity[Task]]]:
        """
        Works for mono vehicle only
        """
        active_entities = routing_entity.get_active_entities(solution=self)
        sched = [
            (ent.get_start_time(self), ent.get_end_time(self), ent)
            for ent in active_entities
        ]
        return sorted(sched, key=lambda x: (x[0], x[1]))

    @optional_override
    def get_attribute_of_entity(
        self, scheduling_entity: SchedulingEntity[Task]
    ) -> AttributeEntities:
        routing_entity_from = self.problem.get_routing_entity(scheduling_entity)
        if scheduling_entity in routing_entity_from.possible_attributes:
            possible_attribute_from = routing_entity_from.possible_attributes[
                scheduling_entity
            ]
            return list(possible_attribute_from)[0]
        else:
            return UndefinedData.UNDEFINED

    def check_transition_time_constraint(self):
        for r in self.problem.get_routing_entities():
            path = self.get_routing_path(routing_entity=r)
            for p0, p1 in zip(path[:-1], path[1:]):
                ent0: SchedulingEntity = p0[2]
                ent1: SchedulingEntity = p1[2]
                attr0 = self.get_attribute_of_entity(ent0)
                attr1 = self.get_attribute_of_entity(ent1)
                transition_time = self.problem.get_dynamic_setup_time_between_entities(
                    entity_from=ent0,
                    attribute_from=attr0,
                    entity_to=ent1,
                    attribute_to=attr1,
                )
                # st0 = p0[0]
                end0 = p0[1]
                st1 = p1[0]
                # end1 = p1[1]
                if not (st1 >= end0 + transition_time):
                    logger.info(f"Transition time not respected ")
                    return False
        return True
