#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver_with_starting_solution import (
    SolverColoringWithStartingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True

import logging

logger = logging.getLogger(__name__)


class ToulbarColoringSolver(SolverColoringWithStartingSolution):
    def __init__(
        self,
        coloring_model: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        SolverColoringWithStartingSolution.__init__(self, coloring_model=coloring_model)
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.coloring_model, params_objective_function=params_objective_function
        )
        self.model: Optional[pytoulbar2.CFN] = None

    def get_range_value(
        self, index_node: int, nb_colors_on_subset: int, nb_colors_all: int
    ):
        node_name = self.coloring_model.nodes_name[index_node]
        if self.coloring_model.has_constraints_coloring:
            nodes = self.coloring_model.constraints_coloring.nodes_fixed()
            if node_name in nodes:
                value = self.coloring_model.constraints_coloring.color_constraint[
                    node_name
                ]
                return range(value, value + 1)
        in_subset = self.coloring_model.is_in_subset_index(index_node)
        ub_i = nb_colors_on_subset if in_subset else nb_colors_all
        return range(ub_i)

    def init_model(self, **kwargs: Any) -> None:
        number_nodes = self.coloring_model.number_of_nodes
        index_nodes_name = self.coloring_model.index_nodes_name
        nb_colors = kwargs.get("nb_colors", None)
        nb_colors_on_subset = kwargs.get("nb_colors_on_subset", nb_colors)
        if nb_colors is None:
            # Run greedy solver to get an upper bound of the bound.
            sol = self.get_starting_solution(**kwargs)
            nb_colors = int(self.coloring_model.evaluate(sol)["nb_colors"])
            nb_colors_all = self.coloring_model.count_colors_all_index(sol.colors)
            nb_colors_on_subset = self.coloring_model.count_colors(sol.colors)
            logger.info(f"{nb_colors_on_subset} colors found by the greedy method ")
        else:
            nb_colors_all = nb_colors
        # we don't have to have a very tight bound.
        Problem = pytoulbar2.CFN(nb_colors)
        Problem.AddVariable("max_color", range(nb_colors_on_subset))
        Problem.AddFunction(["max_color"], range(nb_colors_on_subset))
        range_map = {}
        for i in range(number_nodes):
            in_subset = self.coloring_model.is_in_subset_index(i)
            range_map[i] = self.get_range_value(
                index_node=i,
                nb_colors_on_subset=nb_colors_on_subset,
                nb_colors_all=nb_colors_all,
            )
            Problem.AddVariable(f"x_{i}", range_map[i])
            if in_subset:
                Problem.AddFunction(
                    [f"x_{i}", "max_color"],
                    [
                        10000 if val1 > val2 else 0
                        for val1 in range_map[i]
                        for val2 in range(nb_colors_on_subset)
                    ],
                )  # encode that x_{i}<=max_color.
                #  Problem.AddLinearConstraint([1, -1], [0, i+1], '>=', 0)  # max_color>x_{i} (alternative way ?)
        value_sequence_chain = kwargs.get("value_sequence_chain", False)
        # Warning : don't use this with special "constraints"
        if value_sequence_chain:
            hard_value_sequence_chain = kwargs.get("hard_value_sequence_chain", False)
            tolerance_delta_max = kwargs.get("tolerance_delta_max", 1)
            # play with how "fidele" should be the "max_x" variable
            for j in range(number_nodes):
                Problem.AddVariable(f"max_x_{j}", range(nb_colors_all))
            Problem.AddFunction([f"max_x_{0}"], [0] + [1000] * (nb_colors_all - 1))
            Problem.AddFunction([f"x_{0}"], [0] + [1000] * (nb_colors - 1))
            Problem.AddFunction(
                ["max_color", f"max_x_{number_nodes-1}"],
                [
                    1000 if val1 != val2 else 0
                    for val1 in range(nb_colors)
                    for val2 in range(nb_colors_all)
                ],
            )
            for j in range(1, number_nodes):
                Problem.AddFunction(
                    [f"max_x_{j-1}", f"max_x_{j}"],
                    [
                        10000 if val1 > val2 or val2 > val1 + tolerance_delta_max else 0
                        for val1 in range(nb_colors)
                        for val2 in range(nb_colors)
                    ],
                )  # Max is increasing but 1 by 1 only.
                Problem.AddFunction(
                    [f"max_x_{j}", f"x_{j}"],
                    [
                        10000 if val2 > val1 else 0
                        for val1 in range(nb_colors)
                        for val2 in range(nb_colors)
                    ],
                )
                Problem.AddFunction(
                    [f"max_x_{j - 1}", f"x_{j}"],
                    [
                        10000 if val2 > val1 + tolerance_delta_max else 0
                        for val1 in range(nb_colors)
                        for val2 in range(nb_colors)
                    ],
                )
                if hard_value_sequence_chain:
                    Problem.AddFunction(
                        [f"max_x_{j}", f"max_x_{j-1}", f"x_{j}"],
                        [
                            0 if val1 == max(val2, val3) else 10000
                            for val1 in range(nb_colors)
                            for val2 in range(nb_colors)
                            for val3 in range(nb_colors)
                        ],
                    )  # x_j <= max_x_{j}
                    Problem.AddFunction(
                        [f"max_x_{j-1}", f"x_{j}"],
                        [
                            10000 if val2 > val1 + 1 else 0
                            for val1 in range(nb_colors)
                            for val2 in range(nb_colors)
                        ],
                    )
        len_edges = len(self.coloring_model.graph.edges)
        index = 0
        costs = [
            10000 if val1 == val2 else 0
            for val1 in range(nb_colors_all)
            for val2 in range(nb_colors_all)
        ]
        costs_dict = self.default_costs_matrix(
            nb_colors_all=nb_colors_all, nb_colors_on_subset=nb_colors_on_subset
        )
        for e in self.coloring_model.graph.edges:
            if index % 100 == 0:
                logger.info(f"Nb edges introduced {index} / {len_edges}")
            index1 = index_nodes_name[e[0]]
            index2 = index_nodes_name[e[1]]
            costs_i1_i2 = self.get_costs_matrix(
                index1=index1, index2=index2, costs=costs_dict, range_map=range_map
            )
            Problem.AddFunction([f"x_{index1}", f"x_{index2}"], costs_i1_i2)
            index += 1
        self.model = Problem

    def default_costs_matrix(self, nb_colors_all: int, nb_colors_on_subset: int):
        costs = [
            10000 if val1 == val2 else 0
            for val1 in range(nb_colors_all)
            for val2 in range(nb_colors_all)
        ]
        costs_dict = {}
        if self.coloring_model.use_subset:
            costs_dict = {
                "out-out": costs,
                "in-out": [
                    10000 if val1 == val2 else 0
                    for val1 in range(nb_colors_on_subset)
                    for val2 in range(nb_colors_all)
                ],
                "out-in": [
                    10000 if val1 == val2 else 0
                    for val1 in range(nb_colors_all)
                    for val2 in range(nb_colors_on_subset)
                ],
                "in-in": [
                    10000 if val1 == val2 else 0
                    for val1 in range(nb_colors_on_subset)
                    for val2 in range(nb_colors_on_subset)
                ],
            }

        return costs_dict

    def get_costs_matrix(
        self,
        index1: int,
        index2: int,
        costs: Dict[str, List],
        range_map: Dict[int, Any],
    ):
        in_subset_index1 = self.coloring_model.is_in_subset_index(index1)
        in_subset_index2 = self.coloring_model.is_in_subset_index(index2)
        key = "out-out"
        if in_subset_index1 and in_subset_index2:
            key = "in-in"
        if not in_subset_index1 and in_subset_index2:
            key = "out-in"
        if in_subset_index1 and not in_subset_index2:
            key = "in-out"
        if not in_subset_index1 and not in_subset_index2:
            key = "out-out"
        if not self.coloring_model.has_constraints_coloring:
            return costs[key]
        nodes_fixed = self.coloring_model.constraints_coloring.nodes_fixed()
        node_index1 = self.coloring_model.index_to_nodes_name[index1]
        node_index2 = self.coloring_model.index_to_nodes_name[index2]
        if not (node_index1 in nodes_fixed or node_index2 in nodes_fixed):
            return costs[key]
        else:
            return [
                10000 if val1 == val2 else 0
                for val1 in range_map[index1]
                for val2 in range_map[index2]
            ]

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.model is None:
            self.init_model()
            if self.model is None:
                raise RuntimeError(
                    "self.model must not be None after self.init_model()."
                )
        logger.info("CFN Model init.")
        time_limit = kwargs.get("time_limit", 20)
        self.model.CFN.timer(time_limit)
        solution = self.model.Solve(showSolutions=1)
        logger.info(f"=== Solution === \n {solution}")
        rcpsp_sol = ColoringSolution(
            problem=self.coloring_model,
            colors=solution[0][1 : 1 + self.coloring_model.number_of_nodes],
        )
        fit = self.aggreg_sol(rcpsp_sol)
        return ResultStorage(
            list_solution_fits=[(rcpsp_sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
