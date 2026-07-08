#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Iterable

from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar, LinearExpr

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    Item,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.backlog import (
    BacklogConstraintCpsat,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.changeover import (
    ChangeOverConstraintCpsat,
    ChangeoverModel,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.inventory import (
    InventoryConstraintCpsat,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.parallel_production import (
    ParallelProductionConstraintCpsat,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.production import (
    ProductionConstraintCpsat,
)
from discrete_optimization.lotsizing.production_solution import (
    DeliveryDecision,
    ProductionBasedSolution,
    ProductionDecision,
)

logger = logging.getLogger(__name__)


class GenericLotSizingCpsat(
    ProductionConstraintCpsat[Item],
    InventoryConstraintCpsat[Item],
    BacklogConstraintCpsat[Item],
    ChangeOverConstraintCpsat[Item],
    ParallelProductionConstraintCpsat[Item],
    WarmstartMixin,
):
    hyperparameters = [
        CategoricalHyperparameter(
            name="create_delivery_vars", choices=[True, False], default=True
        ),
        EnumHyperparameter(
            name="modeling_changeover",
            enum=ChangeoverModel,
            default=ChangeoverModel.SHORTEST_PATH_BASED,
        ),
    ]
    problem: GenericLotSizingProblem[Item]
    variables: dict
    production: dict[Item, list[IntVar]]
    production_binary: dict[Item, list[IntVar]]
    inventory: dict[Item, list[IntVar]]
    delivery: dict[Item, list[IntVar]]
    backlog: dict[Item, list[IntVar]]
    objectives: dict[str, LinearExpr]

    def set_warm_start(self, solution: ProductionBasedSolution) -> None:
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                self.cp_model.add_hint(
                    self.get_production_quantity_var(item, t),
                    int(solution.get_production_quantity(item, t)),
                )
                self.cp_model.add_hint(
                    self.get_delivery_var(item, t),
                    int(solution.get_delivery_quantity(item, t)),
                )
                self.cp_model.add_hint(
                    self.get_backlog_var(item, t),
                    int(solution.get_backlog_quantity(item, t)),
                )
                quantity = solution.get_production_quantity(item, t)
                self.cp_model.add_hint(
                    self.get_production_binary_var(item, t), 1 if quantity > 0 else 0
                )

    def init_vars_placeholder(self) -> None:
        self.variables = {}
        self.production = {}
        self.production_binary = {}
        self.inventory = {}
        self.delivery = {}
        self.backlog = {}

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        create_delivery_vars = kwargs["create_delivery_vars"]
        super().init_model(**kwargs)
        self.init_vars_placeholder()
        self.create_production_vars()
        self.create_inventory_vars()
        self.create_backlog_vars()
        if create_delivery_vars:
            self.create_delivery_vars()
        else:
            self.create_delivery_expr()
        self.create_constraint_inventory()
        self.create_constraint_backlog()
        self.create_constraint_production()
        self.create_constraint_parallel_production()
        self.create_objectives(**kwargs)

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.objectives.keys())

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        sol = res[-1][0]
        kpis = self.problem.evaluate(sol)
        return int(kpis[obj])

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.add(self.objectives[obj] <= value)]

    def create_objectives(self, **kwargs: Any) -> None:
        self.objectives = {}
        weights = {}
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "setup_cost":
                self.objectives[obj] = self.create_setup_cost()
            if obj == "production_cost":
                self.objectives[obj] = self.create_production_cost()
            if obj == "inventory_cost":
                self.objectives[obj] = self.create_inventory_cost()
            if obj == "backlog_cost":
                self.objectives[obj] = self.create_backlog_cost()
            if obj == "changeover_cost":
                if self.problem.allows_parallel_production():
                    self.objectives[obj] = 0
                else:
                    self.objectives[obj] = self.create_changeover_constraint_and_cost(
                        modeling=kwargs["modeling_changeover"]
                    )
            if obj == "unmet_demand":
                self.objectives[obj] = sum(
                    [self.get_unmet_demand(item) for item in self.problem.items_list]
                )
            weights[obj] = weight
        self.cp_model.Minimize(
            sum([self.objectives[obj] * weights[obj] for obj in self.objectives])
        )

    def get_unmet_demand(self, item: Item) -> Any:
        return int(self.problem.get_total_demand(item)) - sum(
            [self.get_delivery_var(item, period=t) for t in range(self.problem.horizon)]
        )

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> ProductionBasedSolution:
        for obj in self.objectives.keys():
            print(obj)
            logger.info(f"Objective {obj}: {cpsolvercb.value(self.objectives[obj])}")
        prods = []
        deliveries = []
        for t in range(self.problem.horizon):
            for item in self.problem.items_list:
                quantity_prod = cpsolvercb.value(
                    self.get_production_quantity_var(item, t)
                )
                if quantity_prod > 0:
                    prods.append(
                        ProductionDecision(
                            item=item, period=t, quantity=int(quantity_prod)
                        )
                    )
                delivery = cpsolvercb.value(self.get_delivery_var(item, t))
                if delivery > 0:
                    deliveries.append(
                        DeliveryDecision(item=item, period=t, quantity=int(delivery))
                    )
        return ProductionBasedSolution(
            problem=self.problem, productions=prods, deliveries=deliveries
        )

    def get_production_quantity_var(self, item: Item, period: int) -> Any:
        return self.production[item][period]

    def get_production_binary_var(self, item: Item, period: int) -> Any:
        return self.production_binary[item][period]

    def get_inventory_var(self, item: Item, period: int) -> Any:
        return self.inventory[item][period]

    def get_backlog_var(self, item: Item, period: int) -> Any:
        return self.backlog[item][period]

    def get_delivery_var(self, item: Item, period: int) -> Any:
        return self.delivery[item][period]

    def create_production_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            max_production_quantities = [
                self.problem.get_max_production_quantity(item=item, period=t)
                for t in range(self.problem.horizon)
            ]
            self.production[item] = [
                self.cp_model.new_int_var(
                    lb=0,
                    ub=min(total_demand, max_production_quantities[period]),
                    name=f"production_{item}_{period}",
                )
                for period in range(self.problem.horizon)
            ]
            if max(max_production_quantities) > 1:
                self.production_binary[item] = [
                    self.cp_model.new_bool_var(name=f"production_{item}_{period}")
                    for period in range(self.problem.horizon)
                ]
                for period in range(self.problem.horizon):
                    (
                        self.cp_model.add(
                            self.production[item][period] > 0
                        ).only_enforce_if(self.production_binary[item][period])
                    )
                    (
                        self.cp_model.add(
                            self.production[item][period] == 0
                        ).only_enforce_if(self.production_binary[item][period].Not())
                    )

            else:
                self.production_binary[item] = self.production[item]

    def create_delivery_expr(self) -> Any:
        for item in self.problem.items_list:
            self.delivery[item] = [None for t in range(self.problem.horizon)]
            for t in range(self.problem.horizon):
                if t == 0:
                    self.delivery[item][t] = self.get_production_quantity_var(
                        item=item, period=t
                    ) - self.get_inventory_var(item=item, period=t)
                else:
                    self.delivery[item][t] = (
                        self.get_inventory_var(item=item, period=t - 1)
                        + self.get_production_quantity_var(item=item, period=t)
                        - self.get_inventory_var(item=item, period=t)
                    )

    def create_inventory_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.inventory[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"inventory_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_delivery_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.delivery[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"delivery_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_backlog_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.backlog[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"backlog_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]
