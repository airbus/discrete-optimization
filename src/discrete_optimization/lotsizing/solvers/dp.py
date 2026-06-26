#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

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
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)

logger = logging.getLogger(__name__)


class DpLotSizingSolver(DpSolver, WarmstartMixin):
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
    problem: LotSizingProblem
    transitions: dict  # Maps transition name -> (action_type, item, quantity)
    transition_objects: dict  # Maps transition name -> dp.Transition object
    variables: dict

    def __init__(
        self,
        problem: LotSizingProblem,
        params_objective_function: ParamsObjectiveFunction,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem = problem
        # To implement the lexico :)
        self.modified_params_objective_function = params_objective_function

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        relax_delays = kwargs["relax_delays"]
        use_lookahead = kwargs["use_lookahead_constraints"]
        use_flexibility_delays = kwargs["use_flexibility_delays"]
        flexibility_delta = kwargs["flexibility_delta"]
        self.transitions = {}
        self.transition_objects = {}
        self.variables = {}
        model = dp.Model()
        time_object = model.add_object_type(number=self.problem.horizon)
        item_type_object_with_dummy = model.add_object_type(
            number=self.problem.nb_items_type + 1
        )
        cumulative_delivery = [
            model.add_int_var(target=0) for _ in range(self.problem.nb_items_type)
        ]
        current_stock = [
            model.add_int_var(target=0) for _ in range(self.problem.nb_items_type)
        ]
        current_time_element = model.add_element_var(object_type=time_object, target=0)
        current_time = model.add_int_var(target=0)
        current_item = model.add_element_var(
            object_type=item_type_object_with_dummy, target=self.problem.nb_items_type
        )
        cumulative_demands = [
            model.add_int_table([int(x) for x in np.cumsum(self.problem.demands[item])])
            for item in range(self.problem.nb_items_type)
        ]
        change_over_cost = [
            self.problem.changeover_costs[item] + [0]
            for item in range(self.problem.nb_items_type)
        ] + [[0 for _ in range(self.problem.nb_items_type + 1)]]
        transitions_cost = model.add_int_table(change_over_cost)
        produced_current_step = model.add_int_var(target=0)

        for item in range(self.problem.nb_items_type):
            for quantity in range(
                1,
                min(
                    self.problem.total_demands_per_item[item] + 1,
                    self.problem.capacity_machine + 1,
                ),
            ):
                add_cost = 0
                if "changeover" in self.modified_params_objective_function.objectives:
                    index = self.modified_params_objective_function.objectives.index(
                        "changeover"
                    )
                    add_cost = (
                        self.modified_params_objective_function.weights[index]
                        * transitions_cost[current_item, item]
                    )
                tr = dp.Transition(
                    name=f"produce_{item}_{quantity}",
                    cost=add_cost + dp.IntExpr.state_cost(),
                    effects=[
                        (current_stock[item], current_stock[item] + quantity),
                        (current_item, item),
                        (produced_current_step, 1),
                    ],
                    preconditions=[produced_current_step == 0],
                )
                trans_name = f"produce_{item}_{quantity}"
                self.transitions[trans_name] = ("prod", item, quantity)
                self.transition_objects[trans_name] = tr
                model.add_transition(tr)
            for quantity in range(1, self.problem.total_demands_per_item[item] + 1):
                tr = dp.Transition(
                    name=f"deliver_{item}_{quantity}",
                    cost=dp.IntExpr.state_cost(),
                    effects=[
                        (current_stock[item], current_stock[item] - quantity),
                        (
                            cumulative_delivery[item],
                            cumulative_delivery[item] + quantity,
                        ),
                    ],
                    preconditions=[
                        current_stock[item] >= quantity,
                        cumulative_delivery[item] + quantity
                        <= cumulative_demands[item][current_time_element],
                    ],
                )
                trans_name = f"deliver_{item}_{quantity}"
                self.transitions[trans_name] = ("deliver", item, quantity)
                self.transition_objects[trans_name] = tr
                model.add_transition(tr)
        params = self.modified_params_objective_function
        objectives, weights = params.objectives, params.weights
        cost_expr = 0
        if "delays" in objectives or "stock" in objectives:
            cost_expr = dp.IntExpr(0)
            index_delays = None
            index_stock = None
            if "delays" in objectives:
                index_delays = objectives.index("delays")
            if "stock" in objectives:
                index_stock = objectives.index("stock")
            for item in range(self.problem.nb_items_type):
                if index_stock is not None:
                    cost_expr += (
                        weights[index_stock]
                        * current_stock[item]
                        * self.problem.stock_cost_per_type_per_time_per_unit[item]
                    )
                if index_delays is not None:
                    cost_expr += (
                        weights[index_delays]
                        * (
                            cumulative_demands[item][current_time_element]
                            - cumulative_delivery[item]
                        )
                        * self.problem.delay_cost_per_type_per_time_per_unit[item]
                    )
        precondition_advance_time = [current_time_element < self.problem.horizon - 1]
        if not self.problem.allow_delays and not relax_delays:
            for item in range(self.problem.nb_items_type):
                precondition_advance_time.append(
                    cumulative_demands[item][current_time_element]
                    - cumulative_delivery[item]
                    == 0
                )
                model.add_state_constr(
                    cumulative_demands[item][current_time_element]
                    - cumulative_delivery[item]
                    == 0
                )
        else:
            if use_flexibility_delays:
                for item in range(self.problem.nb_items_type):
                    model.add_state_constr(
                        cumulative_demands[item][current_time_element]
                        - cumulative_delivery[item]
                        <= flexibility_delta
                    )
                logger.info("Added flexibility delays")

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
                delays_prod = 0
                for item in range(self.problem.nb_items_type):
                    fun_[(item, i)] = model.add_int_state_fun(
                        dp.max(
                            0,
                            cumulative_demands[item][
                                dp.min(
                                    current_time_element + i, self.problem.horizon - 1
                                )
                            ]
                            - cumulative_demands[item][current_time_element]
                            - current_stock[item],
                        )
                    )
                    needed_production += fun_[(item, i)]

                # model.add_state_constr(needed_production <= (dp.min(current_time+i,
                #                                              self.problem.horizon-1)-current_time)
                #                       * self.problem.capacity_machine)
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
                            * self.problem.delay_cost_per_type_per_time_per_unit[item]
                            for item in range(self.problem.nb_items_type)
                        ]
                    )
                )

        model.add_dual_bound(0)
        self.model = model
        self.variables = {"current_time": current_time_element}

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        productions = []
        deliveries = []
        current_time = 0
        for t in sol.transitions:
            t: dp.Transition
            name = t.name
            if name == "advance_in_time":
                current_time += 1
            if name in self.transitions:
                key, item, quantity = self.transitions[name]
                prod_item = ProductionItem(
                    item_type=item, quantity=quantity, time=current_time
                )
                if key == "prod":
                    productions.append(prod_item)
                elif key == "deliver":
                    deliveries.append(
                        ProductionItem(
                            item_type=item, quantity=quantity, time=current_time
                        )
                    )
        sol = LotSizingSolution(
            problem=self.problem, productions=productions, deliveries=deliveries
        )
        if self.problem.known_bound is not None:
            logger.info(
                f"{self.aggreg_from_sol(sol) / self.problem.known_bound} relative perf"
            )
        return sol

    def set_warm_start(self, solution: LotSizingSolution) -> None:
        """
        Convert a LotSizingSolution to a sequence of DP transitions for warmstart.

        The DP model expects transitions in a specific order:
        - For each timestep: production(s), delivery(ies), advance_in_time
        - Final timestep: production(s), delivery(ies), finish
        """
        if self.model is None:
            self.init_model()

        # Sort productions and deliveries by time
        sorted_productions = sorted(
            solution.productions, key=lambda p: (p.time, p.item_type)
        )
        sorted_deliveries = sorted(
            solution.deliveries, key=lambda p: (p.time, p.item_type)
        )

        # Group by time
        actions_by_time = {}
        for t in range(self.problem.horizon):
            actions_by_time[t] = {
                "productions": [p for p in sorted_productions if p.time == t],
                "deliveries": [d for d in sorted_deliveries if d.time == t],
            }

        # Build transition sequence
        transition_names = []
        for t in range(self.problem.horizon):
            # Add production transitions
            for prod in actions_by_time[t]["productions"]:
                trans_name = f"produce_{prod.item_type}_{prod.quantity}"
                if trans_name not in self.transitions:
                    logger.warning(
                        f"Warmstart: transition '{trans_name}' not found in model"
                    )
                    return
                transition_names.append(trans_name)

            # Add delivery transitions
            for delivery in actions_by_time[t]["deliveries"]:
                trans_name = f"deliver_{delivery.item_type}_{delivery.quantity}"
                if trans_name not in self.transitions:
                    logger.warning(
                        f"Warmstart: transition '{trans_name}' not found in model"
                    )
                    return
                transition_names.append(trans_name)

            # Add time advancement (except for last timestep)
            if t < self.problem.horizon - 1:
                transition_names.append("advance_in_time")

        # Add finish transition
        transition_names.append("finish")

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


class DpSchedLotSizingSolver(DpSolver, WarmstartMixin):
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
    problem: LotSizingProblem
    transitions: dict  # Maps transition name -> (action_type, item, quantity)
    transition_objects: dict  # Maps transition name -> dp.Transition object
    variables: dict

    def __init__(
        self,
        problem: LotSizingProblem,
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
        for item in self.problem.items_range:
            demands = self.problem.demands[item]
            nb = 0
            for i in range(len(demands)):
                if demands[i] > 0:
                    # We assume it's 1 here
                    self.deadlines[(item, nb)] = i
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
        time_object = model.add_object_type(number=self.problem.horizon)
        all_items_object = model.add_object_type(number=len(all_items))
        item_type_object_with_dummy = model.add_object_type(
            number=self.problem.nb_items_type + 1
        )
        all_items_type = model.add_element_table(
            [all_items[i][0] for i in range(len(all_items))]
        )
        deadlines_table = model.add_int_table(
            [self.deadlines[all_items[i]] for i in range(len(all_items))]
        )
        # current_time_element = model.add_element_var(object_type=time_object,
        #                                             target=0)
        current_time = model.add_int_var(target=0)
        current_item = model.add_element_var(
            object_type=item_type_object_with_dummy, target=self.problem.nb_items_type
        )
        change_over_cost = [
            self.problem.changeover_costs[item] + [0]
            for item in range(self.problem.nb_items_type)
        ] + [[0 for _ in range(self.problem.nb_items_type + 1)]]
        transitions_cost = model.add_int_table(change_over_cost)
        scheduled = model.add_set_var(object_type=all_items_object, target=set())
        nb_skip_allowed = self.problem.horizon - len(all_items)
        nb_skip = model.add_int_var(target=0)
        for i in range(len(all_items)):
            model.add_state_constr(
                scheduled.contains(i)
                | (current_time <= self.deadlines[all_items[i]] + 10)
            )
        for i in range(len(all_items)):
            item, nb = all_items[i]
            add_cost = 0
            if "changeover" in self.modified_params_objective_function.objectives:
                index = self.modified_params_objective_function.objectives.index(
                    "changeover"
                )
                add_cost = (
                    self.modified_params_objective_function.weights[index]
                    * transitions_cost[current_item, item]
                )
            delay = 0
            if "delays" in self.modified_params_objective_function.objectives:
                index = self.modified_params_objective_function.objectives.index(
                    "delays"
                )
                delay = self.modified_params_objective_function.weights[index] * dp.max(
                    current_time - self.deadlines[all_items[i]], 0
                )
            advance = 0
            if "stock" in self.modified_params_objective_function.objectives:
                index = self.modified_params_objective_function.objectives.index(
                    "stock"
                )
                advance = self.modified_params_objective_function.weights[
                    index
                ] * dp.max(self.deadlines[all_items[i]] - current_time, 0)
            pred = item, nb - 1
            precond = []
            if nb - 1 > 0:
                precond = [scheduled.contains(item_to_index[pred])]
            tr = dp.Transition(
                name=f"produce_{i}",
                cost=add_cost
                + delay * self.problem.delay_cost_per_type_per_time_per_unit[item]
                + advance * self.problem.stock_cost_per_type_per_time_per_unit[item]
                + dp.IntExpr.state_cost(),
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
        params = self.modified_params_objective_function
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
            item_delay_cost = (
                delay * self.problem.delay_cost_per_type_per_time_per_unit[item]
            )
            # Only count if item is not yet scheduled
            delay_bound += (scheduled.contains(i)).if_then_else(0, item_delay_cost)

        # Dual Bound 2: Minimum changeover cost
        # If we have k different item types remaining, we need at least (k-1) changeovers
        unscheduled_item_types_count = dp.IntExpr(0)
        for item_type in range(self.problem.nb_items_type):
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
        if self.problem.nb_items_type > 1:
            min_changeover_cost = min(
                [
                    self.problem.changeover_costs[i][j]
                    for i in range(self.problem.nb_items_type)
                    for j in range(self.problem.nb_items_type)
                    if i != j
                ]
            )

        changeover_bound = dp.IntExpr(0)
        if (
            min_changeover_cost > 0
            and "changeover" in self.modified_params_objective_function.objectives
        ):
            idx = self.modified_params_objective_function.objectives.index("changeover")
            weight = self.modified_params_objective_function.weights[idx]
            # We need at least (k-1) changeovers for k different types
            num_changeovers_needed = dp.max(unscheduled_item_types_count - 1, 0)
            changeover_bound = weight * min_changeover_cost * num_changeovers_needed

        # Add combined dual bound
        model.add_dual_bound(delay_bound + changeover_bound)
        model.add_dual_bound(0)
        self.model = model
        # self.variables = {"current_time": current_time_element}

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        productions = []
        deliveries = []
        current_time = 0
        for t in sol.transitions:
            t: dp.Transition
            name = t.name
            if name == "advance_in_time":
                current_time += 1
            if name in self.transitions:
                key, item, nb, deadline = self.transitions[name]
                prod_item = ProductionItem(
                    item_type=item, quantity=1, time=current_time
                )
                productions.append(prod_item)
                deliveries.append(
                    ProductionItem(
                        item_type=item, quantity=1, time=max(deadline, current_time)
                    )
                )
                current_time += 1
        sol = LotSizingSolution(
            problem=self.problem, productions=productions, deliveries=deliveries
        )
        if self.problem.known_bound is not None:
            logger.info(
                f"{self.aggreg_from_sol(sol) / self.problem.known_bound} relative perf"
            )
        return sol

    def set_warm_start(self, solution: LotSizingSolution) -> None:
        """
        Convert a LotSizingSolution to a sequence of DP transitions for warmstart.

        The DP model expects transitions in a specific order:
        - For each timestep: production(s), delivery(ies), advance_in_time
        - Final timestep: production(s), delivery(ies), finish
        """
        if self.model is None:
            self.init_model()

        # Sort productions and deliveries by time
        sorted_productions = sorted(
            solution.productions, key=lambda p: (p.time, p.item_type)
        )
        sorted_deliveries = sorted(
            solution.deliveries, key=lambda p: (p.time, p.item_type)
        )

        # Group by time
        actions_by_time = {}
        for t in range(self.problem.horizon):
            actions_by_time[t] = {
                "productions": [p for p in sorted_productions if p.time == t],
                "deliveries": [d for d in sorted_deliveries if d.time == t],
            }

        # Build transition sequence
        transition_names = []
        for t in range(self.problem.horizon):
            # Add production transitions
            for prod in actions_by_time[t]["productions"]:
                trans_name = f"produce_{prod.item_type}_{prod.quantity}"
                if trans_name not in self.transitions:
                    logger.warning(
                        f"Warmstart: transition '{trans_name}' not found in model"
                    )
                    return
                transition_names.append(trans_name)

            # Add delivery transitions
            for delivery in actions_by_time[t]["deliveries"]:
                trans_name = f"deliver_{delivery.item_type}_{delivery.quantity}"
                if trans_name not in self.transitions:
                    logger.warning(
                        f"Warmstart: transition '{trans_name}' not found in model"
                    )
                    return
                transition_names.append(trans_name)

            # Add time advancement (except for last timestep)
            if t < self.problem.horizon - 1:
                transition_names.append("advance_in_time")

        # Add finish transition
        transition_names.append("finish")

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
