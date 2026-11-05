#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Generic

from ortools.sat.python.cp_model import Domain, IntVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    SchedulingEntity,
    VariableBatchEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.routing import (
    CumulDimensions,
    OrderedRoutingEntity,
    RoutingEntity,
    RoutingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.entity_constraints import (
    EntityConstraintsCpSatSolver,
)


class RoutingCpSatProblem(EntityConstraintsCpSatSolver[Task], Generic[Task]):
    problem: RoutingProblem
    next_variable_routing: dict[
        int, list[tuple[SchedulingEntity, SchedulingEntity, IntVar]]
    ]

    attribute_variable_routing: dict[int, dict[SchedulingEntity, dict[int, IntVar]]]
    cumul_addition_variable_routing: dict[
        int, dict["CumulDimensions", dict[SchedulingEntity, IntVar]]
    ]

    def create_routing_vars_and_constraints(self):
        self.next_variable_routing = {}
        self.attribute_variable_routing = {}
        self.cumul_addition_variable_routing = {}
        for index_routing_entity, routing_entity in enumerate(
            self.problem.get_routing_entities()
            + self.problem.get_ordered_routing_entities()
        ):
            if isinstance(routing_entity, OrderedRoutingEntity):
                self.init_attribute_vars(index_routing_entity, routing_entity)
                self.add_constraint_ordered_routing(
                    index_routing_entity, routing_entity
                )
                self.cumul_addition_ordered_routing(
                    index_routing_entity, routing_entity
                )
            else:
                self.init_routing_vars(index_routing_entity, routing_entity)
                self.init_attribute_vars(index_routing_entity, routing_entity)
                self.add_constraint_routing(index_routing_entity, routing_entity)
                self.cumul_addition_routing(index_routing_entity, routing_entity)

    def init_routing_vars(
        self, index_routing_entity: int, routing_entity: RoutingEntity
    ):
        list_entities = list(routing_entity.entities)
        depot = "depot"
        list_nodes = ["depot"] + list_entities
        node_to_index = {list_nodes[i]: i for i in range(len(list_nodes))}
        arcs = []
        self.next_variable_routing[index_routing_entity] = []
        for node_from in list_nodes:
            is_last_entity = False
            if routing_entity.last_entity is not None:
                is_last_entity = node_from == routing_entity.last_entity
            index_from = node_to_index[node_from]
            if node_from == depot:
                lb_s_from, ub_s_from, lb_e_from, ub_e_from = 0, 0, 0, 0
            else:
                lb_s_from, ub_s_from, lb_e_from, ub_e_from = self.get_lb_ub_entity(
                    node_from
                )
            for node_to in list_nodes:
                is_first_entity = False
                if routing_entity.first_entity is not None:
                    if node_from == "depot":
                        if node_to != routing_entity.first_entity:
                            continue
                    is_first_entity = node_to == routing_entity.first_entity
                if is_first_entity:
                    if node_from != "depot":
                        continue
                if is_last_entity:
                    if node_to != "depot":
                        continue
                if routing_entity.graph is not None:
                    if node_to != "depot" and node_from != "depot":
                        if node_to not in routing_entity.graph[node_from]:
                            continue
                index_to = node_to_index[node_to]
                if node_to == depot:
                    lb_s_to, ub_s_to, lb_e_to, ub_e_to = 0, 0, 0, 0
                else:
                    lb_s_to, ub_s_to, lb_e_to, ub_e_to = self.get_lb_ub_entity(node_to)
                if lb_e_from > lb_s_to:
                    # No need of creating edge.
                    continue
                if node_from == node_to:
                    if node_from == depot:
                        # No self loop in depot
                        continue
                    if routing_entity.has_to_be_visited[node_from]:
                        # No self loop when visit is mandatory.
                        continue
                if is_first_entity or is_last_entity:
                    next_from_to = 1
                else:
                    next_from_to = self.cp_model.new_bool_var(
                        f"next_{index_routing_entity}_{index_from}_{index_to}"
                    )

                self.next_variable_routing[index_routing_entity].append(
                    (node_from, node_to, next_from_to)
                )
                arcs.append((index_from, index_to, next_from_to))
        if routing_entity.nb_means > 1:
            self.cp_model.add_multiple_circuit(arcs)
        if routing_entity.nb_means == 1:
            self.cp_model.add_circuit(arcs)

    def init_attribute_vars(
        self, index_routing_entity: int, routing_entity: RoutingEntity
    ):
        entities = list(routing_entity.entities)
        self.attribute_variable_routing[index_routing_entity] = {}
        for entity in entities:
            attributes = routing_entity.get_possible_attributes(
                scheduling_entity=entity
            )
            index_attrs = {
                attr: routing_entity.index_attributes[attr] for attr in attributes
            }
            self.attribute_variable_routing[index_routing_entity][entity] = {}
            if len(attributes) == 1:
                index_attr = list(attributes)[0]
                self.attribute_variable_routing[index_routing_entity][entity][
                    index_attr
                ] = 1
            else:
                for index_attr in index_attrs:
                    var = self.cp_model.new_bool_var(
                        f"attr_{index_routing_entity}_{entity.entity_id}_{index_attr}"
                    )
                    self.attribute_variable_routing[index_routing_entity][entity][
                        index_attr
                    ] = var
                self.cp_model.add_exactly_one(
                    list(
                        self.attribute_variable_routing[index_routing_entity][
                            entity
                        ].values()
                    )
                )
                if isinstance(entity, VariableBatchEntity):
                    batch_id = entity.batch_id
                    int_attr_from_batch = self.batch_attr_vars[batch_id]
                    for index_attr in index_attrs:
                        attr = routing_entity.attributes[index_attr]
                        self.cp_model.add(
                            int_attr_from_batch == self.attr_to_int[attr]
                        ).only_enforce_if(
                            self.attribute_variable_routing[index_routing_entity][
                                entity
                            ][index_attr]
                        )

    def add_constraint_routing(
        self, index_routing_entity: int, routing_entity: RoutingEntity
    ):
        incoming = defaultdict(list)
        outcoming = defaultdict(list)
        for node_from, node_to, next_from_to in self.next_variable_routing[
            index_routing_entity
        ]:
            if node_from == "depot":
                continue
            if node_to == "depot":
                continue
            if node_to == node_from:
                is_present_from = self.get_entity_is_active_variable(node_from)
                self.cp_model.add(is_present_from == ~next_from_to)
                continue
            attributes_from = list(
                self.attribute_variable_routing[index_routing_entity][node_from]
            )
            attributes_to = list(
                self.attribute_variable_routing[index_routing_entity][node_to]
            )
            time_node_from = self.get_entity_start_or_end_variable(
                entity=node_from, start_or_end=StartOrEnd.END
            )
            time_node_to = self.get_entity_start_or_end_variable(
                entity=node_to, start_or_end=StartOrEnd.START
            )
            is_active_from = self.get_entity_is_active_variable(entity=node_from)
            is_active_to = self.get_entity_is_active_variable(entity=node_to)
            for index_attr_from in attributes_from:
                attr_from = routing_entity.attributes[index_attr_from]
                var_attr_from = self.attribute_variable_routing[index_routing_entity][
                    node_from
                ][index_attr_from]
                for index_attr_to in attributes_to:
                    var_attr_to = self.attribute_variable_routing[index_routing_entity][
                        node_to
                    ][index_attr_to]
                    attr_to = routing_entity.attributes[index_attr_to]
                    travel_time = self.problem.get_dynamic_setup_time_between_entities(
                        entity_from=node_from,
                        entity_to=node_to,
                        attribute_from=attr_from,
                        attribute_to=attr_to,
                    )
                    self.cp_model.add(
                        time_node_to >= time_node_from + int(travel_time)
                    ).only_enforce_if(
                        is_active_from,
                        is_active_to,
                        var_attr_from,
                        var_attr_to,
                        next_from_to,
                    )
        for node in incoming:
            pass
            # self.cp_model.add_at_most_one(incoming[node])
            # self.cp_model.add_at_most_one(outcoming[node])
            # self.cp_model.add(sum(incoming[node])==1).only_enforce_if(self.get_entity_is_active_variable(node))

    def add_constraint_ordered_routing(
        self, index_routing_entity: int, routing_entity: OrderedRoutingEntity
    ):
        list_entities = routing_entity.list_entities
        for node_from, node_to in zip(list_entities[:-1], list_entities[1:]):
            attributes_from = list(
                self.attribute_variable_routing[index_routing_entity][node_from]
            )
            attributes_to = list(
                self.attribute_variable_routing[index_routing_entity][node_to]
            )
            time_node_from = self.get_entity_start_or_end_variable(
                entity=node_from, start_or_end=StartOrEnd.END
            )
            time_node_to = self.get_entity_start_or_end_variable(
                entity=node_to, start_or_end=StartOrEnd.START
            )
            is_active_from = self.get_entity_is_active_variable(entity=node_from)
            is_active_to = self.get_entity_is_active_variable(entity=node_to)
            for index_attr_from in attributes_from:
                attr_from = routing_entity.attributes[index_attr_from]
                var_attr_from = self.attribute_variable_routing[index_routing_entity][
                    node_from
                ][index_attr_from]
                for index_attr_to in attributes_to:
                    var_attr_to = self.attribute_variable_routing[index_routing_entity][
                        node_to
                    ][index_attr_to]
                    attr_to = routing_entity.attributes[index_attr_to]
                    travel_time = self.problem.get_dynamic_setup_time_between_entities(
                        entity_from=node_from,
                        entity_to=node_to,
                        attribute_from=attr_from,
                        attribute_to=attr_to,
                    )
                    self.cp_model.add(
                        time_node_to >= time_node_from + travel_time
                    ).only_enforce_if(
                        is_active_from, is_active_to, var_attr_from, var_attr_to
                    )

    def cumul_addition_routing(
        self, index_routing_entity: int, routing_entity: RoutingEntity
    ):
        cumul_dimensions = routing_entity.cumul_dimensions
        self.cumul_addition_variable_routing[index_routing_entity] = {}
        for c in cumul_dimensions:
            self.cumul_addition_variable_routing[index_routing_entity][c] = {}
            for s in routing_entity.entities:
                attrs = routing_entity.get_possible_attributes(s)
                index_attr = [routing_entity.index_attributes[attr] for attr in attrs]
                values = list(
                    set(routing_entity.cumul_matrix[c][:, index_attr].flatten())
                )
                if 0 not in values:
                    values.append(0)
                domain = Domain.from_values(values)
                self.cumul_addition_variable_routing[index_routing_entity][c][s] = (
                    self.cp_model.new_int_var_from_domain(
                        domain, name=f"cumul_{index_routing_entity}_{c}_{s.entity_id}"
                    )
                )

            for node_from, node_to, next_from_to in self.next_variable_routing[
                index_routing_entity
            ]:
                if node_to == "depot":
                    continue
                if node_from == "depot":
                    continue
                attributes_from = list(
                    self.attribute_variable_routing[index_routing_entity][node_from]
                )
                attributes_to = list(
                    self.attribute_variable_routing[index_routing_entity][node_to]
                )
                is_active_from = self.get_entity_is_active_variable(entity=node_from)
                is_active_to = self.get_entity_is_active_variable(entity=node_to)
                for index_attr_from in attributes_from:
                    attr_from = routing_entity.attributes[index_attr_from]
                    var_attr_from = self.attribute_variable_routing[
                        index_routing_entity
                    ][node_from][index_attr_from]
                    for index_attr_to in attributes_to:
                        var_attr_to = self.attribute_variable_routing[
                            index_routing_entity
                        ][node_to][index_attr_to]
                        attr_to = routing_entity.attributes[index_attr_to]
                        cumul_value = (
                            self.problem.get_dynamic_cumul_dimensions_between_entities(
                                cumul_dimension=c,
                                entity_from=node_from,
                                entity_to=node_to,
                                attribute_from=attr_from,
                                attribute_to=attr_to,
                            )
                        )
                        self.cp_model.add(
                            self.cumul_addition_variable_routing[index_routing_entity][
                                c
                            ][node_to]
                            == cumul_value
                        ).only_enforce_if(
                            is_active_from,
                            is_active_to,
                            var_attr_from,
                            var_attr_to,
                            next_from_to,
                        )
            bounds = self.problem.get_bounds_cumul_dimension(
                routing_entity, cumul_dimension=c
            )
            if bounds[0] or bounds[1]:
                val = sum(
                    self.cumul_addition_variable_routing[index_routing_entity][
                        c
                    ].values()
                )
                if bounds[0]:
                    self.cp_model.add(val >= bounds[0])
                if bounds[1]:
                    self.cp_model.add(val <= bounds[1])

    def cumul_addition_ordered_routing(
        self, index_routing_entity: int, routing_entity: OrderedRoutingEntity
    ):
        cumul_dimensions = routing_entity.cumul_dimensions
        self.cumul_addition_variable_routing[index_routing_entity] = {}
        for c in cumul_dimensions:
            self.cumul_addition_variable_routing[index_routing_entity][c] = {}
            for s in routing_entity.entities:
                attrs = routing_entity.get_possible_attributes(s)
                index_attr = [routing_entity.index_attributes[attr] for attr in attrs]
                values = list(
                    set(routing_entity.cumul_matrix[c][:, index_attr].flatten())
                )
                if 0 not in values:
                    values.append(0)
                domain = Domain.from_values(values)
                self.cumul_addition_variable_routing[index_routing_entity][c][s] = (
                    self.cp_model.new_int_var_from_domain(
                        domain, name=f"cumul_{index_routing_entity}_{c}_{s.entity_id}"
                    )
                )
            for entity_from, entity_to in zip(
                routing_entity.list_entities[:-1], routing_entity.list_entities[1:]
            ):
                attributes_from = list(
                    self.attribute_variable_routing[index_routing_entity][entity_from]
                )
                attributes_to = list(
                    self.attribute_variable_routing[index_routing_entity][entity_to]
                )
                is_active_from = self.get_entity_is_active_variable(entity=entity_from)
                is_active_to = self.get_entity_is_active_variable(entity=entity_to)
                for index_attr_from in attributes_from:
                    attr_from = routing_entity.attributes[index_attr_from]
                    var_attr_from = self.attribute_variable_routing[
                        index_routing_entity
                    ][entity_from][index_attr_from]
                    for index_attr_to in attributes_to:
                        var_attr_to = self.attribute_variable_routing[
                            index_routing_entity
                        ][entity_to][index_attr_to]
                        attr_to = routing_entity.attributes[index_attr_to]
                        cumul_value = (
                            self.problem.get_dynamic_cumul_dimensions_between_entities(
                                cumul_dimension=c,
                                entity_from=entity_from,
                                entity_to=entity_to,
                                attribute_from=attr_from,
                                attribute_to=attr_to,
                            )
                        )
                        self.cp_model.add(
                            self.cumul_addition_variable_routing[index_routing_entity][
                                c
                            ][entity_to]
                            == cumul_value
                        ).only_enforce_if(
                            is_active_from, is_active_to, var_attr_from, var_attr_to
                        )
