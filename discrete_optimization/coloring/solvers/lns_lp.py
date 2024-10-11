"""Large neighborhood search + Linear programming toolbox for coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from collections.abc import Iterable
from enum import Enum
from typing import Any, Union

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers.greedy import GreedyColoringSolver
from discrete_optimization.coloring.solvers.lp import (
    GurobiColoringSolver,
    MathOptColoringSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_mip import (
    GurobiConstraintHandler,
    InitialSolution,
    OrtoolsMathOptConstraintHandler,
)
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class InitialColoringMethod(Enum):
    DUMMY = 0
    GREEDY = 1


class InitialColoring(InitialSolution):
    """Initial solution provider for lns algorithm.

    Attributes:
        problem (ColoringProblem): input coloring problem
        initial_method (InitialColoringMethod): the method to use to provide the initial solution.
    """

    def __init__(
        self,
        problem: ColoringProblem,
        initial_method: InitialColoringMethod,
        params_objective_function: ParamsObjectiveFunction,
    ):
        self.problem = problem
        self.initial_method = initial_method
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem, params_objective_function=params_objective_function
        )

    def get_starting_solution(self) -> ResultStorage:
        if self.initial_method == InitialColoringMethod.DUMMY:
            sol = self.problem.get_dummy_solution()
            fit = self.aggreg_from_sol(sol)
            return ResultStorage(
                mode_optim=self.params_objective_function.sense_function,
                list_solution_fits=[(sol, fit)],
            )
        else:
            solver = GreedyColoringSolver(
                problem=self.problem,
                params_objective_function=self.params_objective_function,
            )
            return solver.solve()


class _BaseFixColorsConstraintHandler(ConstraintHandler):
    """Base class for constraint builder used in LNS+LP for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    hyperparameters = [
        FloatHyperparameter("fraction_to_fix", low=0.0, high=1.0, default=0.9),
    ]

    def __init__(self, problem: ColoringProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix

    def adding_constraint_from_results_store(
        self,
        solver: Union[GurobiColoringSolver, MathOptColoringSolver],
        result_storage: ResultStorage,
        **kwargs: Any
    ) -> Iterable[Any]:
        subpart_color = set(
            random.sample(
                solver.nodes_name,
                int(self.fraction_to_fix * solver.number_of_nodes),
            )
        )
        dict_color_fixed = {}
        current_solution = result_storage.get_best_solution()
        if current_solution is None:
            raise ValueError(
                "result_storage.get_best_solution() " "should not be None."
            )
        if not isinstance(current_solution, ColoringSolution):
            raise ValueError(
                "result_storage.get_best_solution() " "should be a ColoringSolution."
            )
        if current_solution.colors is None:
            raise ValueError(
                "result_storage.get_best_solution().colors " "should not be None."
            )
        max_color = max(current_solution.colors)
        solver.set_warm_start(current_solution)
        for n in solver.nodes_name:
            current_node_color = current_solution.colors[solver.index_nodes_name[n]]
            if n in subpart_color and current_node_color <= max_color - 1:
                dict_color_fixed[n] = current_node_color
        colors_var = solver.variable_decision["colors_var"]
        lns_constraints = []
        for key in colors_var:
            n, c = key
            if n in dict_color_fixed:
                if c == dict_color_fixed[n]:
                    lns_constraints.append(
                        solver.add_linear_constraint(
                            colors_var[key] == 1, name=str((n, c))
                        )
                    )
                else:
                    lns_constraints.append(
                        solver.add_linear_constraint(
                            colors_var[key] == 0, name=str((n, c))
                        )
                    )
        return lns_constraints


class FixColorsGurobiConstraintHandler(
    GurobiConstraintHandler, _BaseFixColorsConstraintHandler
):
    """Constraint builder used in LNS+LP (using gurobi solver) for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def adding_constraint_from_results_store(
        self, solver: GurobiColoringSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Any]:
        constraints = (
            _BaseFixColorsConstraintHandler.adding_constraint_from_results_store(
                self, solver=solver, result_storage=result_storage, **kwargs
            )
        )
        solver.model.update()
        return constraints


class FixColorsMathOptConstraintHandler(
    OrtoolsMathOptConstraintHandler, _BaseFixColorsConstraintHandler
):
    """Constraint builder used in LNS+LP (using mathopt solver) for coloring problem.

    This constraint handler is pretty basic, it fixes a fraction_to_fix proportion of nodes color.

    Attributes:
        problem (ColoringProblem): input coloring problem
        fraction_to_fix (float): float between 0 and 1, representing the proportion of nodes to constrain.
    """

    def adding_constraint_from_results_store(
        self,
        solver: MathOptColoringSolver,
        result_storage: ResultStorage,
        **kwargs: Any
    ) -> Iterable[Any]:
        constraints = (
            _BaseFixColorsConstraintHandler.adding_constraint_from_results_store(
                self, solver=solver, result_storage=result_storage, **kwargs
            )
        )
        return constraints
