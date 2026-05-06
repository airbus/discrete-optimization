#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Optional

from minizinc import Instance, Model, Solver

from discrete_optimization.generic_tools.cp_tools import CpSolverName
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.mzn_tools import (
    MinizincCpSolver,
    find_right_minizinc_solver_name,
)
from discrete_optimization.multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
)
from discrete_optimization.multibatching.solvers import MultibatchingSolver
from discrete_optimization.multibatching.solvers.solver_utils import (
    precompute_valid_links,
)

logger = logging.getLogger(__name__)

path_minizinc = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../minizinc/")
)


class CpMultibatchingSolver(MinizincCpSolver, MultibatchingSolver):
    """CP solver for the Multibatching problem using MiniZinc.

    This solver uses a flow formulation with network_flow constraints to model
    the multibatching problem. It computes the number of trips and the flow of
    each product on each transport link.

    Attributes:
        problem (MultibatchingProblem): The multibatching problem instance to solve.
        params_objective_function (ParamsObjectiveFunction): Parameters for objective function.
        cp_solver_name (CpSolverName): Backend CP solver to use with MiniZinc.
        silent_solve_error (bool): If True, raise warning instead of error on solve crashes.
    """

    hyperparameters = [
        EnumHyperparameter(
            name="cp_solver_name", enum=CpSolverName, default=CpSolverName.CHUFFED
        ),
        CategoricalHyperparameter(
            name="restrict_to_shortest_paths", choices=[True, False], default=False
        ),
        FloatHyperparameter(
            name="shortest_path_tolerance",
            depends_on=[("restrict_to_shortest_paths", [True])],
            default=0.1,
            low=0,
            high=5,
        ),
    ]

    problem: MultibatchingProblem

    def __init__(
        self,
        problem: MultibatchingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.model: Optional[Model] = None

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the MiniZinc model with problem data.

        Args:
            cp_solver_name (CpSolverName): CP solver to use (default: CHUFFED)
            **kwargs: Additional keyword arguments
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        cp_solver_name = kwargs["cp_solver_name"]

        # Load the MiniZinc model
        model_path = os.path.join(path_minizinc, "multibatching_flow.mzn")
        self.model = Model(model_path)

        # Create solver instance
        solver = Solver.lookup(find_right_minizinc_solver_name(cp_solver_name))
        instance = Instance(solver, self.model)

        # Set problem dimensions
        instance["nb_locations"] = self.problem.nb_locations
        instance["nb_products"] = self.problem.nb_products
        instance["nb_transport_links"] = self.problem.nb_transport_links

        # Set product characteristics
        instance["product_size"] = [int(p.size) for p in self.problem.products]

        # Set transport link characteristics
        instance["link_from"] = [
            self.problem.locations_to_index[tl.location_l1] + 1
            for tl in self.problem.transport_links
        ]
        instance["link_to"] = [
            self.problem.locations_to_index[tl.location_l2] + 1
            for tl in self.problem.transport_links
        ]
        instance["link_capacity"] = [
            int(tl.transport_type.capacity) for tl in self.problem.transport_links
        ]
        instance["link_distance"] = [
            int(tl.distance) for tl in self.problem.transport_links
        ]
        instance["link_cost"] = [
            int(tl.transport_type.cost) for tl in self.problem.transport_links
        ]
        instance["link_emissions"] = [
            int(tl.transport_type.emissions) for tl in self.problem.transport_links
        ]

        # Set net supply for each location and product (balance for network flow)
        # Positive = supply (source), Negative = demand (sink)
        net_supply = []
        for loc in self.problem.locations:
            net_supply_row = []
            for p in self.problem.products:
                supply_value = loc.net_supply.get(p, 0)
                net_supply_row.append(int(supply_value))
            net_supply.append(net_supply_row)
        instance["net_supply"] = net_supply

        # Apply shortest path heuristic if enabled
        use_shortest_path = kwargs["restrict_to_shortest_paths"]
        if use_shortest_path:
            sp_tolerance = kwargs["shortest_path_tolerance"]
            valid_links_map = precompute_valid_links(
                self.problem, tolerance=sp_tolerance
            )
            logger.info(
                f"Shortest path heuristic enabled with tolerance={sp_tolerance}"
            )
        else:
            valid_links_map = None

        # Set product compatibility with transport links
        product_can_use_link = []
        for link_idx, tl in enumerate(self.problem.transport_links):
            compatibility_row = []
            for p in self.problem.products:
                # Check basic compatibility
                can_use = (
                    tl.transport_type in p.valid_transports
                    and tl.transport_type.capacity >= p.size
                )
                # Apply shortest path filter if enabled
                if can_use and valid_links_map is not None:
                    can_use = link_idx in valid_links_map.get(p.id, set())

                compatibility_row.append(1 if can_use else 0)
            product_can_use_link.append(compatibility_row)
        instance["product_can_use_link"] = product_can_use_link

        # Set upper bounds for flows (total supply of each product)
        max_flow_per_product = [
            int(self.problem.get_total_supply(p)) for p in self.problem.products
        ]
        instance["max_flow_per_product"] = max_flow_per_product

        self.instance = instance

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> MultibatchingSolution:
        """Convert MiniZinc solution to MultibatchingSolution.

        Args:
            _output_item: MiniZinc output string (unused)
            **kwargs: Contains 'nb_trips' and 'flow' arrays from MiniZinc

        Returns:
            MultibatchingSolution: The solution object
        """
        nb_trips = kwargs["nb_trips"]
        flow = kwargs["flow"]
        # Reconstruct the solution from nb_trips and flow arrays
        list_flows = []

        for i, tl in enumerate(self.problem.transport_links):
            if nb_trips[i] > 0:
                # Create product packing for this link
                product_packing = {}
                total_flow = 0

                for p_idx, p in enumerate(self.problem.products):
                    flow_val = flow[i][p_idx]
                    if flow_val > 0:
                        # Average quantity per trip
                        avg_qty = flow_val / nb_trips[i]
                        product_packing[p] = avg_qty
                        total_flow += flow_val

                if product_packing and total_flow > 0:
                    list_flows.append(
                        PackingTransport(
                            transport_link=tl,
                            product_packing=product_packing,
                            nb_packing=int(nb_trips[i]),
                        )
                    )
        logger.info(f"Minizinc found solution...{kwargs['objective']}")
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)
