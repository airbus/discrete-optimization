#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Dynamic programming solvers for capacitated multi-item lot sizing problem."""

import logging
from copy import deepcopy
from typing import Any, Optional

import didppy as dp
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)
from discrete_optimization.lotsizing.production_solution import ProductionDecision

logger = logging.getLogger(__name__)


class DpCapacitatedLotSizingSolver(DpSolver, WarmstartMixin):
    hyperparameters = [
        CategoricalHyperparameter(
            name="relax_delays", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="use_lookahead_constraints", choices=[True, False], default=False
        ),
        IntegerHyperparameter(
            name="lookahead_window",
            low=1,
            high=20,
            default=5,
            depends_on=[("use_lookahead_constraints", True)],
        ),
        CategoricalHyperparameter(
            name="use_flexibility_delays", choices=[True, False], default=False
        ),
        IntegerHyperparameter(
            name="flexibility_delta",
            low=0,
            high=30,
            default=2,
            depends_on=[("use_flexibility_delays", True)],
        ),
    ]
    problem: CapacitatedMultiItemLSP
    transitions: dict  # Maps transition name -> (action_type, item, quantity)
    transition_objects: dict  # Maps transition name -> dp.Transition object
    variables: dict

    def __init__(
        self,
        problem: CapacitatedMultiItemLSP,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem = problem
        # To implement the lexico :)
        self.modified_params_objective_function = deepcopy(
            self.params_objective_function
        )

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        # relax_delays = kwargs["relax_delays"]
        use_lookahead = kwargs["use_lookahead_constraints"]
        self.transitions = {}
        self.transition_objects = {}
        self.variables = {}
        model = dp.Model()
        time_object = model.add_object_type(number=self.problem.horizon)
        item_type_object_with_dummy = model.add_object_type(
            number=self.problem.nb_items + 1
        )
        # Track cumulative production for each item
        cumulative_production = [
            model.add_int_var(target=0) for _ in range(self.problem.nb_items)
        ]
        current_time_element = model.add_element_var(object_type=time_object, target=0)
        current_time = model.add_int_var(target=0)
        current_item = model.add_element_var(
            object_type=item_type_object_with_dummy, target=self.problem.nb_items
        )

        # Build cumulative demands for each item
        cumulative_demands = []
        for item in range(self.problem.nb_items):
            demands_list = [
                self.problem.get_demand(item, t) for t in range(self.problem.horizon)
            ]
            cumsum_demands = [int(x) for x in np.cumsum(demands_list)]
            cumulative_demands.append(model.add_int_table(cumsum_demands))

        # Build changeover cost matrix
        change_over_cost = []
        for from_item in range(self.problem.nb_items):
            row = []
            for to_item in range(self.problem.nb_items):
                row.append(int(self.problem.get_changeover_cost(from_item, to_item)))
            row.append(0)  # changeover to dummy (idle) has 0 cost
            change_over_cost.append(row)
        # Add dummy row (from idle to any item has 0 cost)
        change_over_cost.append([0 for _ in range(self.problem.nb_items + 1)])

        transitions_cost = model.add_int_table(change_over_cost)
        produced_current_step = model.add_int_var(target=0)

        # Get total demands per item
        total_demands_per_item = [
            self.problem.get_total_demand(item) for item in range(self.problem.nb_items)
        ]

        # Production transitions - only produce, no separate delivery
        for item in range(self.problem.nb_items):
            for quantity in range(
                1,
                min(
                    total_demands_per_item[item] + 1,
                    self.problem.capacity_machine + 1,
                ),
            ):
                add_cost = 0
                if (
                    "changeover_cost"
                    in self.modified_params_objective_function.objectives
                ):
                    index = self.modified_params_objective_function.objectives.index(
                        "changeover_cost"
                    )
                    add_cost = (
                        int(self.modified_params_objective_function.weights[index])
                        * transitions_cost[current_item, item]
                    )
                tr = dp.Transition(
                    name=f"produce_{item}_{quantity}",
                    cost=add_cost + dp.IntExpr.state_cost(),
                    effects=[
                        (
                            cumulative_production[item],
                            cumulative_production[item] + quantity,
                        ),
                        (current_item, item),
                        (produced_current_step, 1),
                    ],
                    preconditions=[
                        produced_current_step == 0,
                        cumulative_production[item] + quantity
                        <= total_demands_per_item[item],
                    ],
                )
                trans_name = f"produce_{item}_{quantity}"
                self.transitions[trans_name] = ("prod", item, quantity)
                self.transition_objects[trans_name] = tr
                model.add_transition(tr)

        params = self.modified_params_objective_function
        objectives, weights = params.objectives, params.weights
        cost_expr = 0
        if "backlog_cost" in objectives or "inventory_cost" in objectives:
            cost_expr = dp.IntExpr(0)
            index_delays = None
            index_stock = None
            if "backlog_cost" in objectives:
                index_delays = objectives.index("backlog_cost")
            if "inventory_cost" in objectives:
                index_stock = objectives.index("inventory_cost")
            for item in range(self.problem.nb_items):
                # Inventory = max(0, cumulative_production - cumulative_demand)
                # Backlog = max(0, cumulative_demand - cumulative_production)
                if index_stock is not None:
                    inventory = dp.max(
                        0,
                        cumulative_production[item]
                        - cumulative_demands[item][current_time_element],
                    )
                    cost_expr += (
                        int(weights[index_stock])
                        * inventory
                        * int(self.problem.get_inventory_cost_per_unit(item, 0))
                    )
                if index_delays is not None:
                    backlog = dp.max(
                        0,
                        cumulative_demands[item][current_time_element]
                        - cumulative_production[item],
                    )
                    cost_expr += (
                        int(weights[index_delays])
                        * backlog
                        * int(self.problem.get_backlog_cost_per_unit(item, 0))
                    )

        precondition_advance_time = [current_time_element < self.problem.horizon - 1]
        advance_in_time = dp.Transition(
            name="advance_in_time",
            cost=cost_expr + dp.IntExpr.state_cost(),
            effects=[
                (current_time_element, current_time_element + 1),
                (produced_current_step, 0),
            ],
            preconditions=precondition_advance_time,
        )
        self.transition_objects["advance_in_time"] = advance_in_time
        model.add_transition(advance_in_time)

        finish = dp.Transition(
            name="finish",
            cost=cost_expr + dp.IntExpr.state_cost(),
            effects=[(current_time, self.problem.horizon)],
            preconditions=[current_time_element == self.problem.horizon - 1]
            + precondition_advance_time[1:],
        )
        self.transition_objects["finish"] = finish
        model.add_transition(finish)
        model.add_base_case([current_time == self.problem.horizon])

        # Add lookahead-based state constraints to prune infeasible states
        # These check if from current state, we can still satisfy remaining demands
        if use_lookahead:
            lookahead_window = kwargs.get("lookahead_window", 5)
            logger.info(
                f"Adding lookahead state constraints (window={lookahead_window})"
            )
            fun_ = {}
            # For each lookahead distance i, compute total production needed across ALL items
            # within the next i timesteps, and constrain it to be <= i (max capacity)
            for i in range(1, min(lookahead_window + 1, self.problem.horizon)):
                needed_production = 0
                for item in range(self.problem.nb_items):
                    fun_[(item, i)] = model.add_int_state_fun(
                        dp.max(
                            0,
                            cumulative_demands[item][
                                dp.min(
                                    current_time_element + i, self.problem.horizon - 1
                                )
                            ]
                            - cumulative_production[item],
                        )
                    )
                    needed_production += fun_[(item, i)]

                model.add_dual_bound(
                    sum(
                        [
                            (
                                fun_[(item, i)]
                                - (
                                    dp.min(current_time + i, self.problem.horizon - 1)
                                    - current_time
                                )
                                * self.problem.capacity_machine
                            )
                            * int(self.problem.get_backlog_cost_per_unit(item, 0))
                            for item in range(self.problem.nb_items)
                        ]
                    )
                )

        model.add_dual_bound(0)
        self.model = model
        self.variables = {"current_time": current_time_element}

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        productions = []
        current_time = 0
        for t in sol.transitions:
            t: dp.Transition
            name = t.name
            if name == "advance_in_time":
                current_time += 1
            elif name in self.transitions:
                key, item, quantity = self.transitions[name]
                if key == "prod":
                    prod_decision = ProductionDecision(
                        item=item, period=current_time, quantity=quantity
                    )
                    productions.append(prod_decision)

        solution = CapacitatedMultiItemSolution(
            problem=self.problem, productions=productions
        )

        known_bound = self.problem.infos.get("known_bound", None)
        if known_bound is not None:
            logger.info(f"{self.aggreg_from_sol(solution) / known_bound} relative perf")
        return solution

    def set_warm_start(self, solution: CapacitatedMultiItemSolution) -> None:
        """
        Convert a CapacitatedMultiItemSolution to a sequence of DP transitions for warmstart.

        The DP model expects transitions in a specific order:
        - For each timestep: production(s), delivery(ies), advance_in_time
        - Final timestep: production(s), delivery(ies), finish
        """
        if self.model is None:
            self.init_model()

        # Sort productions by time
        sorted_productions = sorted(
            solution.productions, key=lambda p: (p.period, p.item)
        )

        # Compute deliveries from productions (ProductionBasedSolution already does this)
        # For DP we need to reconstruct delivery transitions
        # This is tricky - we need to match the delivery pattern expected by DP
        # For now, skip warm start support - user said DP needs significant work
        logger.warning(
            "Warmstart not yet implemented for DpLotSizingSolver with new API"
        )
        self.initial_solution = None


class DpSchedCapacitatedLotSizingSolver(DpSolver, WarmstartMixin):
    """Scheduling-based DP solver for capacitated multi-item lot sizing.

    This solver models the problem as scheduling demand occurrences,
    where each demand is a task with a deadline.
    """

    hyperparameters = [
        CategoricalHyperparameter(
            name="relax_delays", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="use_lookahead_constraints", choices=[True, False], default=False
        ),
        IntegerHyperparameter(
            name="lookahead_window",
            low=1,
            high=20,
            default=5,
            depends_on=[("use_lookahead_constraints", True)],
        ),
        CategoricalHyperparameter(
            name="use_flexibility_delays", choices=[True, False], default=False
        ),
        IntegerHyperparameter(
            name="flexibility_delta",
            low=0,
            high=30,
            default=2,
            depends_on=[("use_flexibility_delays", True)],
        ),
    ]
    problem: CapacitatedMultiItemLSP
    transitions: dict  # Maps transition name -> (action_type, item, quantity)
    transition_objects: dict  # Maps transition name -> dp.Transition object
    variables: dict

    def __init__(
        self,
        problem: CapacitatedMultiItemLSP,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem = problem
        # To implement the lexico :)
        if params_objective_function is None:
            self.modified_params_objective_function = self.params_objective_function
        else:
            self.modified_params_objective_function = params_objective_function

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.deadlines = {}

        # Build deadlines: for each demand occurrence, record its due date
        for item in range(self.problem.nb_items):
            nb = 0
            for period in range(self.problem.horizon):
                demand = self.problem.get_demand(item, period)
                if demand > 0:
                    # We assume demand is 1 here (binary problem)
                    self.deadlines[(item, nb)] = period
                    nb += 1

        all_items = sorted(self.deadlines.keys(), key=lambda x: self.deadlines[x])
        item_to_index = {all_items[i]: i for i in range(len(all_items))}
        predecessor = [set() for i in range(len(all_items))]

        for item, nb in all_items:
            index = item_to_index[(item, nb)]
            if nb >= 1:
                index_pred = item_to_index[(item, nb - 1)]
                predecessor[index].add(index_pred)

        self.transitions = {}
        self.transition_objects = {}
        self.variables = {}
        model = dp.Model()
        all_items_object = model.add_object_type(number=len(all_items))
        item_type_object_with_dummy = model.add_object_type(
            number=self.problem.nb_items + 1
        )
        deadlines_table = model.add_int_table(
            [self.deadlines[all_items[i]] for i in range(len(all_items))]
        )
        current_time = model.add_int_var(target=0)
        current_item = model.add_element_var(
            object_type=item_type_object_with_dummy, target=self.problem.nb_items
        )

        # Build changeover cost matrix
        change_over_cost = []
        for from_item in range(self.problem.nb_items):
            row = []
            for to_item in range(self.problem.nb_items):
                row.append(int(self.problem.get_changeover_cost(from_item, to_item)))
            row.append(0)  # changeover to dummy (idle) has 0 cost
            change_over_cost.append(row)
        # Add dummy row (from idle to any item has 0 cost)
        change_over_cost.append([0 for _ in range(self.problem.nb_items + 1)])

        transitions_cost = model.add_int_table(change_over_cost)
        scheduled = model.add_set_var(object_type=all_items_object, target=set())
        nb_skip_allowed = self.problem.horizon - len(all_items)
        nb_skip = model.add_int_var(target=0)

        # State constraint to prune infeasible states
        # Allow some slack for backlog-allowed problems, but be strict otherwise
        slack = 10 if self.problem.allow_backlog else 1
        for i in range(len(all_items)):
            model.add_state_constr(
                scheduled.contains(i)
                | (current_time <= self.deadlines[all_items[i]] + slack)
            )

        for i in range(len(all_items)):
            item, nb = all_items[i]
            add_cost = 0
            if "changeover_cost" in self.modified_params_objective_function.objectives:
                index = self.modified_params_objective_function.objectives.index(
                    "changeover_cost"
                )
                add_cost = (
                    int(self.modified_params_objective_function.weights[index])
                    * transitions_cost[current_item, item]
                )
            # Calculate delay (periods late) and advance (periods early)
            delay_periods = dp.max(current_time - self.deadlines[all_items[i]], 0)
            advance_periods = dp.max(self.deadlines[all_items[i]] - current_time, 0)

            # Compute costs: weight * periods * per_unit_per_period_cost
            delay_cost = 0
            if "backlog_cost" in self.modified_params_objective_function.objectives:
                index = self.modified_params_objective_function.objectives.index(
                    "backlog_cost"
                )
                delay_cost = (
                    int(self.modified_params_objective_function.weights[index])
                    * delay_periods
                    * int(self.problem.get_backlog_cost_per_unit(item, 0))
                )

            advance_cost = 0
            if "inventory_cost" in self.modified_params_objective_function.objectives:
                index = self.modified_params_objective_function.objectives.index(
                    "inventory_cost"
                )
                advance_cost = (
                    int(self.modified_params_objective_function.weights[index])
                    * advance_periods
                    * int(self.problem.get_inventory_cost_per_unit(item, 0))
                )

            pred = item, nb - 1
            precond = []
            if nb - 1 >= 0:
                precond = [scheduled.contains(item_to_index[pred])]
            tr = dp.Transition(
                name=f"produce_{i}",
                cost=add_cost + delay_cost + advance_cost + dp.IntExpr.state_cost(),
                effects=[
                    (scheduled, scheduled.add(i)),
                    (current_time, current_time + 1),
                    (current_item, item),
                ],
                preconditions=[
                    current_time < self.problem.horizon,
                    ~scheduled.contains(i),
                ]
                + precond,
            )
            trans_name = f"produce_{i}"
            self.transitions[trans_name] = (
                "prod",
                item,
                nb,
                self.deadlines[(item, nb)],
            )
            self.transition_objects[trans_name] = tr
            model.add_transition(tr)

        model.add_state_constr(current_time <= self.problem.horizon)
        advance_in_time = dp.Transition(
            name="advance_in_time",
            cost=dp.IntExpr.state_cost(),
            effects=[(current_time, current_time + 1), (nb_skip, nb_skip + 1)],
            preconditions=[nb_skip < nb_skip_allowed],
        )
        self.transition_objects["advance_in_time"] = advance_in_time
        model.add_transition(advance_in_time)
        model.add_base_case([scheduled.len() == len(all_items)])

        # Dual Bound 1: Delay-based lower bound
        # For unscheduled items where current_time > deadline, we will incur delay costs
        delay_bound = dp.IntExpr(0)
        for i in range(len(all_items)):
            item, nb = all_items[i]
            # Calculate delay if we produce this item now
            delay = dp.max(current_time - deadlines_table[i], 0)
            item_delay_cost = delay * int(
                self.problem.get_backlog_cost_per_unit(item, 0)
            )
            # Only count if item is not yet scheduled
            delay_bound += (scheduled.contains(i)).if_then_else(0, item_delay_cost)

        # Dual Bound 2: Minimum changeover cost
        # If we have k different item types remaining, we need at least (k-1) changeovers
        unscheduled_item_types_count = dp.IntExpr(0)
        for item_type in range(self.problem.nb_items):
            # Check if this item type has any unscheduled items
            has_unscheduled = dp.IntExpr(0)
            for i in range(len(all_items)):
                if all_items[i][0] == item_type:
                    # Add 1 if this specific item is not scheduled
                    has_unscheduled += (scheduled.contains(i)).if_then_else(0, 1)
            # If has_unscheduled > 0, this type needs scheduling
            unscheduled_item_types_count += (has_unscheduled > 0).if_then_else(1, 0)

        # Find minimum changeover cost
        min_changeover_cost = 0
        if self.problem.nb_items > 1:
            min_changeover_cost = min(
                [
                    self.problem.get_changeover_cost(i, j)
                    for i in range(self.problem.nb_items)
                    for j in range(self.problem.nb_items)
                    if i != j
                ]
            )

        changeover_bound = dp.IntExpr(0)
        if (
            min_changeover_cost > 0
            and "changeover_cost" in self.modified_params_objective_function.objectives
        ):
            idx = self.modified_params_objective_function.objectives.index(
                "changeover_cost"
            )
            weight = int(self.modified_params_objective_function.weights[idx])
            # We need at least (k-1) changeovers for k different types
            num_changeovers_needed = dp.max(unscheduled_item_types_count - 1, 0)
            changeover_bound = (
                weight * int(min_changeover_cost) * num_changeovers_needed
            )

        # Add combined dual bound
        model.add_dual_bound(delay_bound + changeover_bound)
        model.add_dual_bound(0)
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        productions = []
        current_time = 0
        for t in sol.transitions:
            t: dp.Transition
            name = t.name
            if name == "advance_in_time":
                current_time += 1
            if name in self.transitions:
                key, item, nb, deadline = self.transitions[name]
                prod_decision = ProductionDecision(
                    item=item, period=current_time, quantity=1
                )
                productions.append(prod_decision)
                current_time += 1

        solution = CapacitatedMultiItemSolution(
            problem=self.problem, productions=productions
        )

        known_bound = self.problem.infos.get("known_bound", None)
        if known_bound is not None:
            logger.info(f"{self.aggreg_from_sol(solution) / known_bound} relative perf")
        return solution

    def set_warm_start(self, solution: CapacitatedMultiItemSolution) -> None:
        """
        Convert a CapacitatedMultiItemSolution to a sequence of DP transitions for warmstart.

        DpSchedLotSizingSolver uses a scheduling-based DP model where:
        - Each demand occurrence is a task with a deadline
        - Transitions are produce_{i} where i is the index into all_items list
        - advance_in_time is used for idle periods
        - No deliver or finish transitions
        """
        if self.model is None:
            self.init_model()

        # Rebuild the same all_items structure as in init_model
        # This maps each demand occurrence to (item_type, occurrence_index)
        all_items = sorted(self.deadlines.keys(), key=lambda x: self.deadlines[x])
        item_to_index = {all_items[i]: i for i in range(len(all_items))}

        # Sort productions by time
        sorted_productions = sorted(
            solution.productions, key=lambda p: (p.period, p.item)
        )

        # Map productions to all_items indices
        # We need to track how many of each item type we've seen
        item_occurrence_counter = {item: 0 for item in range(self.problem.nb_items)}

        production_schedule = []  # List of (time, item_index)

        for prod in sorted_productions:
            item_type = prod.item
            occurrence = item_occurrence_counter[item_type]

            # Find the index in all_items
            key = (item_type, occurrence)
            if key not in item_to_index:
                logger.warning(
                    f"Warmstart: production for {key} not found in all_items"
                )
                self.initial_solution = None
                return

            item_index = item_to_index[key]
            production_schedule.append((prod.period, item_index))
            item_occurrence_counter[item_type] += 1

        # Check if we have all items
        if len(production_schedule) != len(all_items):
            logger.warning(
                f"Warmstart: solution has {len(production_schedule)} productions but model expects {len(all_items)}"
            )
            self.initial_solution = None
            return

        # Build transition sequence respecting the production schedule
        # Sort by production time
        production_schedule.sort(key=lambda x: x[0])

        transition_names = []
        current_time = 0

        for prod_time, item_index in production_schedule:
            # Add idle periods (advance_in_time) if needed
            while current_time < prod_time:
                transition_names.append("advance_in_time")
                current_time += 1

            # Add production
            transition_names.append(f"produce_{item_index}")
            current_time += 1

        # Convert transition names to actual transition objects
        transitions_list = []
        for name in transition_names:
            if name not in self.transition_objects:
                logger.warning(
                    f"Warmstart: transition '{name}' not found in transition_objects"
                )
                self.initial_solution = None
                return
            transitions_list.append(self.transition_objects[name])

        self.initial_solution = transitions_list
        logger.info(f"Warmstart set with {len(transitions_list)} transitions")
