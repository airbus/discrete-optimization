#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Generic MILP solver for lot sizing problems."""

import logging
from typing import Any, Callable

import gurobipy

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    InequalitySense,
    OrtoolsMathOptMilpSolver,
)
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    Item,
)
from discrete_optimization.lotsizing.generic_solver.milp.backlog import (
    BacklogConstraintMilp,
)
from discrete_optimization.lotsizing.generic_solver.milp.changeover import (
    ChangeOverConstraintMilp,
    ChangeoverModel,
)
from discrete_optimization.lotsizing.generic_solver.milp.inventory import (
    InventoryConstraintMilp,
)
from discrete_optimization.lotsizing.generic_solver.milp.parallel_production import (
    ParallelProductionConstraintMilp,
)
from discrete_optimization.lotsizing.generic_solver.milp.production import (
    ProductionConstraintMilp,
)
from discrete_optimization.lotsizing.production_solution import (
    DeliveryDecision,
    ProductionBasedSolution,
    ProductionDecision,
)

logger = logging.getLogger(__name__)


class GenericLotSizingMilp(
    ProductionConstraintMilp[Item],
    InventoryConstraintMilp[Item],
    BacklogConstraintMilp[Item],
    ChangeOverConstraintMilp[Item],
    ParallelProductionConstraintMilp[Item],
):
    """Generic MILP solver for lot sizing problems.

    This solver combines all constraint mixins to handle the full range of
    lot sizing problem variants. It mirrors the functionality of GenericLotSizingCpsat
    but uses MILP instead of CP-SAT.

    Features:
    - Production capacity constraints
    - Inventory tracking and stock limits
    - Backlog (delayed demand) support
    - Changeover costs for sequence-dependent setups
    - Parallel vs. exclusive production
    - Multi-objective support via weighted sum

    The solver can be configured via hyperparameters to create delivery variables
    or compute them as expressions, and to choose the changeover modeling approach.
    """

    hyperparameters = [
        CategoricalHyperparameter(
            name="create_delivery_vars",
            choices=[True, False],
            default=True,
        ),
        EnumHyperparameter(
            name="modeling_changeover",
            enum=ChangeoverModel,
            default=ChangeoverModel.FLOW_BASED,
        ),
    ]

    problem: GenericLotSizingProblem[Item]
    variables: dict
    production: dict[Item, list]
    production_binary: dict[Item, list]
    inventory: dict[Item, list]
    delivery: dict[Item, list]
    backlog: dict[Item, list]
    objectives: dict[str, Any]

    def init_vars_placeholder(self) -> None:
        """Initialize variable dictionaries."""
        self.variables = {}
        self.production = {}
        self.production_binary = {}
        self.inventory = {}
        self.delivery = {}
        self.backlog = {}
        self.objectives = {}

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the MILP model with variables and constraints.

        Args:
            **kwargs: Hyperparameters including:
                - create_delivery_vars: If True, create delivery variables;
                  otherwise compute as expressions
                - modeling_changeover: Changeover modeling approach
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        create_delivery_vars = kwargs["create_delivery_vars"]

        # Call parent init_model to create the empty model
        super().init_model(**kwargs)

        # Create empty MILP model
        self.model = self.create_empty_model("generic_lotsizing")

        # Initialize variable containers
        self.init_vars_placeholder()

        # Create decision variables
        self.create_production_vars()
        self.create_inventory_vars()
        self.create_backlog_vars()

        if create_delivery_vars:
            self.create_delivery_vars()
        else:
            self.create_delivery_expr()

        # Create constraints
        self.create_constraint_inventory()
        self.create_constraint_backlog()
        self.create_constraint_production()
        self.create_constraint_parallel_production()

        # Create and set objectives
        self.create_objectives(**kwargs)

    def create_objectives(self, **kwargs: Any) -> None:
        """Create objective function as weighted sum of individual objectives.

        Args:
            **kwargs: Must include modeling_changeover for changeover cost
        """
        self.objectives = {}
        objective_terms = []

        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "setup_cost":
                self.objectives[obj] = self.create_setup_cost()
                objective_terms.append(weight * self.objectives[obj])

            elif obj == "production_cost":
                self.objectives[obj] = self.create_production_cost()
                objective_terms.append(weight * self.objectives[obj])

            elif obj == "inventory_cost":
                self.objectives[obj] = self.create_inventory_cost()
                objective_terms.append(weight * self.objectives[obj])

            elif obj == "backlog_cost":
                self.objectives[obj] = self.create_backlog_cost()
                objective_terms.append(weight * self.objectives[obj])

            elif obj == "changeover_cost":
                if self.problem.allows_parallel_production():
                    self.objectives[obj] = 0
                else:
                    self.objectives[obj] = self.create_changeover_constraint_and_cost(
                        modeling=kwargs["modeling_changeover"]
                    )
                    objective_terms.append(weight * self.objectives[obj])

            elif obj == "unmet_demand":
                # Penalty for unmet demand
                unmet_terms = []
                for item in self.problem.items_list:
                    total_demand = int(self.problem.get_total_demand(item))
                    delivery_terms = [
                        self.get_delivery_var(item=item, period=t)
                        for t in range(self.problem.horizon)
                    ]
                    unmet = total_demand - self.construct_linear_sum(delivery_terms)
                    unmet_terms.append(unmet)
                self.objectives[obj] = self.construct_linear_sum(unmet_terms)
                objective_terms.append(weight * self.objectives[obj])

        # Set the objective function
        if objective_terms:
            self.set_model_objective(
                self.construct_linear_sum(objective_terms), minimize=True
            )

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> ProductionBasedSolution:
        """Extract solution from MILP model.

        Args:
            get_var_value_for_current_solution: Function to get variable values
            get_obj_value_for_current_solution: Function to get objective value

        Returns:
            ProductionBasedSolution with production and delivery decisions
        """
        prods = []
        deliveries = []

        for t in range(self.problem.horizon):
            for item in self.problem.items_list:
                # Extract production quantity
                quantity_prod = get_var_value_for_current_solution(
                    self.get_production_quantity_var(item, t)
                )
                if quantity_prod > 0:
                    prods.append(
                        ProductionDecision(
                            item=item, period=t, quantity=int(quantity_prod)
                        )
                    )

                # Extract delivery quantity
                delivery = get_var_value_for_current_solution(
                    self.get_delivery_var(item, t)
                )
                if delivery > 0:
                    deliveries.append(
                        DeliveryDecision(item=item, period=t, quantity=int(delivery))
                    )

        return ProductionBasedSolution(
            problem=self.problem, productions=prods, deliveries=deliveries
        )

    def get_production_quantity_var(self, item: Item, period: int) -> Any:
        """Get production quantity variable X_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Variable representing production quantity
        """
        return self.production[item][period]

    def get_production_binary_var(self, item: Item, period: int) -> Any:
        """Get setup binary variable Y_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Binary variable representing setup (1 if production > 0)
        """
        return self.production_binary[item][period]

    def get_inventory_var(self, item: Item, period: int) -> Any:
        """Get inventory variable I_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Variable representing inventory level
        """
        return self.inventory[item][period]

    def get_backlog_var(self, item: Item, period: int) -> Any:
        """Get backlog variable B_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Variable representing backlog quantity
        """
        return self.backlog[item][period]

    def get_delivery_var(self, item: Item, period: int) -> Any:
        """Get delivery variable D_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Variable or expression representing delivery quantity
        """
        return self.delivery[item][period]

    def create_production_vars(self):
        """Create production quantity and setup binary variables."""
        for item in self.problem.items_list:
            total_demand = int(self.problem.get_total_demand(item))
            max_production_quantities = []
            for t in range(self.problem.horizon):
                max_prod = self.problem.get_max_production_quantity(item=item, period=t)
                if max_prod == float("inf"):
                    max_production_quantities.append(total_demand)
                else:
                    max_production_quantities.append(int(max_prod))

            # Production quantity variables X_it
            self.production[item] = [
                self.add_integer_variable(
                    lb=0,
                    ub=min(total_demand, max_production_quantities[period]),
                    name=f"production_{item}_{period}",
                )
                for period in range(self.problem.horizon)
            ]

            # Setup binary variables Y_it
            # If max production > 1, we need separate binary variables
            # Otherwise, production variable itself is binary
            if max(max_production_quantities) > 1:
                self.production_binary[item] = [
                    self.add_binary_variable(name=f"setup_{item}_{period}")
                    for period in range(self.problem.horizon)
                ]

                # Link setup binary to production quantity
                # Y_it = 1 iff X_it > 0
                for period in range(self.problem.horizon):
                    prod_var = self.production[item][period]
                    setup_var = self.production_binary[item][period]

                    # If setup = 1, then production >= 1
                    self.add_linear_constraint_with_indicator(
                        binvar=setup_var,
                        binval=1,
                        lhs=prod_var,
                        sense=InequalitySense.GREATER_OR_EQUAL,
                        rhs=1,
                        penalty_coeff=total_demand,
                        name=f"setup_implies_prod_{item}_{period}",
                    )

                    # If setup = 0, then production = 0
                    self.add_linear_constraint_with_indicator(
                        binvar=setup_var,
                        binval=0,
                        lhs=prod_var,
                        sense=InequalitySense.EQUAL,
                        rhs=0,
                        penalty_coeff=total_demand,
                        name=f"no_setup_no_prod_{item}_{period}",
                    )
            else:
                # Production itself is binary, reuse it
                self.production_binary[item] = self.production[item]

    def create_delivery_expr(self) -> None:
        """Create delivery as expressions (not variables).

        Delivery is computed from inventory balance:
        D_it = I_i,t-1 + X_it - I_it
        """
        for item in self.problem.items_list:
            self.delivery[item] = [None for _ in range(self.problem.horizon)]
            for t in range(self.problem.horizon):
                if t == 0:
                    # D_i0 = X_i0 - I_i0
                    self.delivery[item][t] = self.get_production_quantity_var(
                        item=item, period=t
                    ) - self.get_inventory_var(item=item, period=t)
                else:
                    # D_it = I_i,t-1 + X_it - I_it
                    self.delivery[item][t] = (
                        self.get_inventory_var(item=item, period=t - 1)
                        + self.get_production_quantity_var(item=item, period=t)
                        - self.get_inventory_var(item=item, period=t)
                    )

    def create_inventory_vars(self):
        """Create inventory level variables I_it."""
        for item in self.problem.items_list:
            total_demand = int(self.problem.get_total_demand(item))
            self.inventory[item] = [
                self.add_integer_variable(
                    lb=0, ub=total_demand, name=f"inventory_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_delivery_vars(self):
        """Create delivery quantity variables D_it."""
        for item in self.problem.items_list:
            total_demand = int(self.problem.get_total_demand(item))
            self.delivery[item] = [
                self.add_integer_variable(
                    lb=0, ub=total_demand, name=f"delivery_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_backlog_vars(self):
        """Create backlog variables B_it."""
        for item in self.problem.items_list:
            total_demand = int(self.problem.get_total_demand(item))
            self.backlog[item] = [
                self.add_integer_variable(
                    lb=0, ub=total_demand, name=f"backlog_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]


class MathOptGenericLotSizingMilp(GenericLotSizingMilp[Item], OrtoolsMathOptMilpSolver):
    """Generic lot sizing MILP solver using OR-Tools MathOpt."""

    pass


class GurobiGenericLotSizingMilp(GenericLotSizingMilp[Item], GurobiMilpSolver):
    """Generic lot sizing MILP solver using Gurobi.

    This solver supports warm-starting from an initial solution via the
    convert_to_variable_values method, which maps production, inventory,
    and delivery decisions to Gurobi variable values.

    Example:
        >>> solver = GurobiGenericLotSizingMilp(problem)
        >>> solver.init_model()
        >>> result = solver.solve(
        ...     parameters_milp=params,
        ...     use_mipstart=True,
        ...     initial_solution=initial_solution
        ... )
    """

    def convert_to_variable_values(
        self, solution: ProductionBasedSolution
    ) -> dict[gurobipy.Var, float]:
        """Convert a solution to variable values for warm-starting.

        Args:
            solution: ProductionBasedSolution to convert

        Returns:
            Dictionary mapping Gurobi variables to their values
        """
        variable_values = {}

        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                # Production quantity
                prod_qty = solution.get_production_quantity(item, t)
                prod_var = self.get_production_quantity_var(item, t)
                if isinstance(prod_var, gurobipy.Var):
                    variable_values[prod_var] = float(prod_qty)

                # Setup binary (1 if production > 0, else 0)
                setup_var = self.get_production_binary_var(item, t)
                if isinstance(setup_var, gurobipy.Var):
                    variable_values[setup_var] = 1.0 if prod_qty > 0 else 0.0

                # Inventory level
                inv_qty = solution.get_inventory_level(item, t)
                inv_var = self.get_inventory_var(item, t)
                if isinstance(inv_var, gurobipy.Var):
                    variable_values[inv_var] = float(inv_qty)

                # Delivery quantity (only if it's a variable, not an expression)
                delivery_qty = solution.get_delivery_quantity(item, t)
                delivery_var = self.get_delivery_var(item, t)
                if isinstance(delivery_var, gurobipy.Var):
                    variable_values[delivery_var] = float(delivery_qty)

                # Backlog (if the problem supports it)
                if self.problem.is_backlog_allowed():
                    backlog_var = self.get_backlog_var(item, t)
                    if isinstance(backlog_var, gurobipy.Var):
                        # Compute backlog from solution
                        # Backlog = cumulative demand - cumulative delivery
                        cumulative_demand = sum(
                            self.problem.get_demand(item, period)
                            for period in range(t + 1)
                        )
                        cumulative_delivery = sum(
                            solution.get_delivery_quantity(item, period)
                            for period in range(t + 1)
                        )
                        backlog_qty = max(0, cumulative_demand - cumulative_delivery)
                        variable_values[backlog_var] = float(backlog_qty)

        # Add changeover variables if they exist
        if hasattr(self, "variables") and "changeover" in self.variables:
            # State-based changeover variables
            for var_name, var in self.variables.get("changeover", {}).items():
                if isinstance(var, gurobipy.Var):
                    # For changeover variables, we need to infer from production sequence
                    # This is complex, so we'll leave them unset for now
                    # Gurobi will handle them during MIP start
                    pass

        return variable_values
